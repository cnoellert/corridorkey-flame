"""
convert.py  —  One-time converter: CorridorKey .pth → MLX .npz

Usage
-----
python convert.py /path/to/CorridorKey_v1.0.pth
# produces CorridorKey_v1.0.mlx.npz alongside the .pth

python convert.py /path/to/CorridorKey_v1.0.pth --quantize int8
# produces CorridorKey_v1.0.mlx.int8.npz

Key transformations
-------------------
• Key remapping (Bug 3 fix):
    _orig_mod.*            stripped  (torch.compile artifact)
    encoder.model.*   →   encoder.* (timm FeatureGetterNet wrapper)
• Conv2d weights: PyTorch [O, I, kH, kW] → MLX [O, kH, kW, I]
• BatchNorm folded into FoldedBN.weight / FoldedBN.bias  (Bug 4 fix)
    scale = gamma / sqrt(var + eps)
    bias  = beta - mean * scale
  Stored as .weight and .bias so FoldedBN loads without any remap.
• DropPath and head layers stripped (inference-only build)
• SHA-256 of source .pth embedded for cache invalidation
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Key remapping  (Bug 3 fix)
# ---------------------------------------------------------------------------

def _remap_key(key: str) -> str:
    """
    Strip torch.compile prefix, timm FeatureGetterNet wrapper, and fix
    nn.Sequential index naming (PyTorch .N. -> MLX .layers.N.).
      _orig_mod.encoder.model.blocks.0.norm1.weight -> encoder.blocks.0.norm1.weight
      _orig_mod.refiner.stem.0.weight -> refiner.stem.layers.0.weight
    """
    import re
    # 1. Strip torch.compile artifact
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod."):]
    # 2. Strip timm FeatureGetterNet: encoder.model.* -> encoder.*
    if key.startswith("encoder.model."):
        key = "encoder." + key[len("encoder.model."):]
    # 3. nn.Sequential: only remap known Sequential containers.
    #    MLX stores Sequential children under .layers.N, not .N.
    #    Only refiner.stem uses nn.Sequential in this model.
    #    We cannot use a blanket regex or it corrupts block indices.
    SEQ_PREFIXES = ("refiner.stem",)
    for pfx in SEQ_PREFIXES:
        needle = pfx + "."
        if needle in key:
            idx = key.index(needle) + len(needle)
            rest = key[idx:]
            # rest starts with a digit when it's a Sequential child index
            import re as _re
            rest = _re.sub(r'^(\d+)\.', lambda m: 'layers.' + m.group(1) + '.', rest)
            key = key[:idx] + rest
    return key


# ---------------------------------------------------------------------------
# Conv weight layout
# ---------------------------------------------------------------------------

def _is_conv_weight(key: str, shape: tuple) -> bool:
    return len(shape) == 4 and not key.endswith(".bias")


def _pt_conv_to_mlx(w: np.ndarray) -> np.ndarray:
    """[O, I, kH, kW] → [O, kH, kW, I]."""
    return w.transpose(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# BatchNorm folding  (Bug 4 fix)
# Outputs .weight and .bias directly — FoldedBN loads these by name.
# ---------------------------------------------------------------------------

class _BNFolder:
    _SUFFIXES = (".weight", ".bias", ".running_mean", ".running_var")

    def __init__(self) -> None:
        self._store: dict[str, dict[str, np.ndarray]] = {}

    def is_bn_key(self, key: str) -> bool:
        return ".bn" in key and any(key.endswith(s) for s in self._SUFFIXES)

    def collect(self, key: str, arr: np.ndarray) -> None:
        for suf in self._SUFFIXES:
            if key.endswith(suf):
                prefix = key[: -len(suf)]
                self._store.setdefault(prefix, {})[suf.lstrip(".")] = arr
                return

    def folded(self) -> dict[str, np.ndarray]:
        """Returns {prefix.weight: scale, prefix.bias: shifted_bias}."""
        out: dict[str, np.ndarray] = {}
        for prefix, d in self._store.items():
            if all(k in d for k in ("weight", "bias", "running_mean", "running_var")):
                eps   = 1e-5
                scale = d["weight"] / np.sqrt(d["running_var"] + eps)
                bias  = d["bias"] - d["running_mean"] * scale
                out[prefix + ".weight"] = scale.astype(np.float32)
                out[prefix + ".bias"]   = bias.astype(np.float32)
        return out

# ---------------------------------------------------------------------------
# Keys to drop
# ---------------------------------------------------------------------------

_DROP_PATTERNS = (
    "drop_path",
    ".num_batches_tracked",
    "head.",
    "encoder.model.head.",
)

def _should_drop(key: str) -> bool:
    return any(p in key for p in _DROP_PATTERNS)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, quantize: str | None = None) -> None:
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch required for conversion: pip install torch")

    print(f"[convert] Loading {src} …")
    ckpt: Any = torch.load(src, map_location="cpu", weights_only=True)

    # CorridorKey wraps the state_dict under 'state_dict' key
    state_dict: dict = ckpt.get("state_dict", ckpt)

    bn_folder  = _BNFolder()
    converted: dict[str, np.ndarray] = {}
    skipped = 0

    for raw_key, tensor in state_dict.items():
        # Remap key FIRST (Bug 3 fix)
        key = _remap_key(raw_key)

        if _should_drop(key):
            skipped += 1
            continue

        arr = tensor.float().numpy()

        # BN keys: collect for folding
        if bn_folder.is_bn_key(key):
            bn_folder.collect(key, arr)
            continue

        # Conv weight: transpose layout
        if _is_conv_weight(key, arr.shape):
            arr = _pt_conv_to_mlx(arr)

        converted[key] = arr.astype(np.float32)

    # Add folded BN (Bug 4 fix — outputs .weight/.bias, FoldedBN loads these directly)
    folded = bn_folder.folded()
    print(f"[convert] Folded {len(folded) // 2} BatchNorm layers")
    converted.update(folded)

    print(f"[convert] Tensors: {len(converted)} kept, {skipped} dropped")

    if quantize == "int8":
        converted = _quantize_int8(converted)
        print("[convert] int8 quantisation applied")

    # Embed source hash
    converted["__src_sha256__"] = np.array(list(_sha256(src).encode()), dtype=np.uint8)
    converted["__src_path__"]   = np.array(list(str(src).encode()),     dtype=np.uint8)

    print(f"[convert] Saving → {dst}")
    mx.savez(str(dst), **{k: mx.array(v) for k, v in converted.items()})
    print("[convert] Done.")


# ---------------------------------------------------------------------------
# int8 quantisation
# ---------------------------------------------------------------------------

def _quantize_int8(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in weights.items():
        if k.startswith("__"):
            out[k] = v
            continue
        is_weight = (
            (k.endswith(".weight") and v.ndim >= 2)
            and not any(s in k for s in (".bias", "_mean", "_var", "pos_embed", ".bn."))
        )
        if is_weight:
            abs_max = np.abs(v).max(axis=tuple(range(1, v.ndim)), keepdims=True)
            abs_max = np.clip(abs_max, 1e-8, None)
            scale   = (abs_max / 127.0).squeeze().astype(np.float32)
            q       = np.round(v / (abs_max / 127.0)).clip(-127, 127).astype(np.int8)
            out[k]              = q
            out[k + "_scale"]   = scale
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _dst_path(src: Path, quantize: str | None) -> Path:
    stem = src.stem
    return src.parent / (f"{stem}.mlx.{quantize}.npz" if quantize else f"{stem}.mlx.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CorridorKey .pth → MLX .npz")
    ap.add_argument("src", type=Path)
    ap.add_argument("--dst", type=Path, default=None)
    ap.add_argument("--quantize", choices=["int8"], default=None)
    args = ap.parse_args()
    src = args.src.expanduser().resolve()
    if not src.exists():
        sys.exit(f"File not found: {src}")
    convert(src, args.dst or _dst_path(src, args.quantize), quantize=args.quantize)


if __name__ == "__main__":
    main()
