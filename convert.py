"""
convert.py  —  One-time converter: CorridorKey .pth → MLX .npz

Usage
-----
python convert.py CorridorKey_v1.0.pth
# produces CorridorKey_v1.0.mlx.npz alongside the .pth

python convert.py CorridorKey_v1.0.pth --quantize int8
# produces CorridorKey_v1.0.mlx.int8.npz

Key transformations
-------------------
• Conv2d weights: PyTorch [O, I, kH, kW] → MLX [O, kH, kW, I]  (NHWC conv)
• BatchNorm running stats folded into scale/bias for fast inference
• DropPath, head, classifier layers stripped (inference-only)
• A SHA-256 of the source .pth is embedded so the inference script
  can detect when the source weights have changed.
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
# SHA-256 of source file
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Weight-shape normalisation helpers
# ---------------------------------------------------------------------------

def _is_conv_weight(key: str, shape: tuple) -> bool:
    """True if this is a 4-D convolution weight (not bias)."""
    return len(shape) == 4 and not key.endswith(".bias")


def _pt_conv_to_mlx(w: np.ndarray) -> np.ndarray:
    """[O, I, kH, kW] → [O, kH, kW, I]."""
    return w.transpose(0, 2, 3, 1)


def _fold_batchnorm(
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fold BatchNorm into affine params so inference avoids the running-stats
    memory bandwidth hit.

    Equivalent to:
      y = gamma * (x - mean) / sqrt(var + eps) + beta
      → scale = gamma / sqrt(var + eps)
        bias  = beta - mean * scale
    """
    scale = gamma / np.sqrt(running_var + eps)
    bias  = beta - running_mean * scale
    return scale.astype(np.float32), bias.astype(np.float32)


# ---------------------------------------------------------------------------
# BatchNorm tracker  (to fold paired gamma/beta/mean/var together)
# ---------------------------------------------------------------------------

class _BNFolder:
    """
    Collects all BatchNorm tensors keyed by their shared prefix, then folds
    them into scale/bias pairs that inference.py loads as regular Linear-style
    affine params.
    """
    _SUFFIXES = (".weight", ".bias", ".running_mean", ".running_var")

    def __init__(self) -> None:
        self._store: dict[str, dict[str, np.ndarray]] = {}

    def is_bn_key(self, key: str) -> bool:
        return any(key.endswith(s) for s in self._SUFFIXES) and ".bn" in key

    def collect(self, key: str, arr: np.ndarray) -> None:
        for suf in self._SUFFIXES:
            if key.endswith(suf):
                prefix = key[: -len(suf)]
                self._store.setdefault(prefix, {})[suf.lstrip(".")] = arr
                return

    def folded(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for prefix, d in self._store.items():
            if all(k in d for k in ("weight", "bias", "running_mean", "running_var")):
                scale, bias = _fold_batchnorm(
                    d["weight"], d["bias"], d["running_mean"], d["running_var"]
                )
                out[prefix + ".scale"] = scale
                out[prefix + ".bias"]  = bias
        return out


# ---------------------------------------------------------------------------
# Keys to drop (not needed at inference)
# ---------------------------------------------------------------------------
_DROP_PATTERNS = (
    "drop_path",       # DropPath — identity at inference
    "head.",           # ImageNet classifier head
    "encoder.model.head.",
    ".num_batches_tracked",
)

def _should_drop(key: str) -> bool:
    return any(p in key for p in _DROP_PATTERNS)


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, quantize: str | None = None) -> None:
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch required for conversion: pip install torch")

    print(f"[convert] Loading {src} …")
    ckpt: Any = torch.load(src, map_location="cpu", weights_only=True)

    # CorridorKey stores the state_dict directly (no 'model' wrapper)
    state_dict: dict = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)

    bn_folder = _BNFolder()
    converted: dict[str, np.ndarray] = {}
    skipped = 0

    for key, tensor in state_dict.items():
        if _should_drop(key):
            skipped += 1
            continue

        arr = tensor.float().numpy()

        if bn_folder.is_bn_key(key):
            bn_folder.collect(key, arr)
            continue

        if _is_conv_weight(key, arr.shape):
            arr = _pt_conv_to_mlx(arr)

        converted[key] = arr.astype(np.float32)

    # Fold BatchNorm
    folded = bn_folder.folded()
    print(f"[convert] Folded {len(folded)//2} BatchNorm layers")
    converted.update(folded)

    print(f"[convert] Tensors: {len(converted)} kept, {skipped} dropped")

    # Optional int8 quantisation of Linear / Conv weights
    if quantize == "int8":
        converted = _quantize_int8(converted)
        print("[convert] int8 quantisation applied to weight tensors")

    # Embed source hash for staleness detection
    converted["__src_sha256__"] = np.array(list(_sha256(src).encode()), dtype=np.uint8)
    converted["__src_path__"]   = np.array(list(str(src).encode()),     dtype=np.uint8)

    print(f"[convert] Saving → {dst}")
    mx.savez(str(dst), **{k: mx.array(v) for k, v in converted.items()})
    print("[convert] Done.")


# ---------------------------------------------------------------------------
# int8 quantisation pass
# ---------------------------------------------------------------------------

def _quantize_int8(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Symmetric per-output-channel int8 for weight tensors (not biases/norms).
    Stores: key          → int8 weights
            key + _scale → float32 per-channel scale  (shape [O])
    Biases and normalisation params stay float32.
    """
    out: dict[str, np.ndarray] = {}
    for k, v in weights.items():
        if k.startswith("__"):
            out[k] = v
            continue
        # Quantise weights only (not biases, scales, running stats)
        is_weight = (
            (k.endswith(".weight") or (len(v.shape) == 4 and "conv" in k))
            and v.ndim >= 2
            and not any(s in k for s in (".bias", "_scale", "_mean", "_var"))
        )
        if is_weight:
            axis = 0
            abs_max = np.abs(v).max(axis=tuple(range(1, v.ndim)), keepdims=True)
            abs_max = np.clip(abs_max, 1e-8, None)
            scale = (abs_max / 127.0).squeeze().astype(np.float32)
            q = np.round(v / (abs_max / 127.0)).clip(-127, 127).astype(np.int8)
            out[k] = q
            out[k + "_scale"] = scale
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _dst_path(src: Path, quantize: str | None) -> Path:
    stem = src.stem
    if quantize:
        return src.parent / f"{stem}.mlx.{quantize}.npz"
    return src.parent / f"{stem}.mlx.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CorridorKey .pth → MLX .npz")
    ap.add_argument("src", type=Path, help="Source .pth checkpoint")
    ap.add_argument("--dst", type=Path, default=None, help="Output path (default: auto)")
    ap.add_argument(
        "--quantize",
        choices=["int8"],
        default=None,
        help="Apply int8 weight quantisation",
    )
    args = ap.parse_args()

    src: Path = args.src.expanduser().resolve()
    if not src.exists():
        sys.exit(f"File not found: {src}")

    dst: Path = args.dst or _dst_path(src, args.quantize)
    convert(src, dst, quantize=args.quantize)


if __name__ == "__main__":
    main()
