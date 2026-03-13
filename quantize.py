"""
quantize.py  —  Post-hoc int8 quantisation of an already-converted .mlx.npz

Usage
-----
python quantize.py CorridorKey_v1.0.mlx.npz
# produces CorridorKey_v1.0.mlx.int8.npz

This is a standalone alternative to passing --quantize int8 at conversion
time. Useful if you already have a float32 .npz and want to try the int8
variant without re-running the full conversion.

Strategy
--------
Symmetric per-output-channel int8 for all weight tensors (2D Linear and
4D Conv). Biases, norms, and positional embeddings stay float32.

At inference, each quantised layer dequantises on the fly:
    w_fp32 = w_int8 * scale[output_channel]

MLX's built-in mx.dequantize() handles this with a fused Metal kernel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def _is_quantisable(key: str, arr: np.ndarray) -> bool:
    """Weight tensors only — not biases, norms, pos_embed, metadata."""
    skip_suffixes = (".bias", "_scale", "_mean", "_var", "pos_embed", "__")
    if any(key.endswith(s) or key.startswith(s) for s in skip_suffixes):
        return False
    if "running" in key or "num_batches" in key:
        return False
    return arr.ndim >= 2


def quantize_npz(src: Path, dst: Path) -> None:
    print(f"[quantize] Loading {src} …")
    raw = dict(mx.load(str(src)))

    out: dict[str, mx.array] = {}
    n_quantised = 0

    for key, val in raw.items():
        arr = np.array(val, dtype=np.float32)

        if not _is_quantisable(key, arr):
            out[key] = mx.array(arr)
            continue

        # Symmetric per-output-channel (axis 0)
        abs_max = np.abs(arr).max(axis=tuple(range(1, arr.ndim)), keepdims=True)
        abs_max = np.clip(abs_max, 1e-8, None)
        scale   = (abs_max / 127.0).squeeze().astype(np.float32)
        q       = np.round(arr / (abs_max / 127.0)).clip(-127, 127).astype(np.int8)

        out[key]            = mx.array(q)
        out[key + "_scale"] = mx.array(scale)
        n_quantised += 1

    print(f"[quantize] Quantised {n_quantised} weight tensors → int8")

    # Preserve metadata
    for k in raw:
        if k.startswith("__") and k not in out:
            out[k] = raw[k]

    print(f"[quantize] Saving → {dst}")
    mx.savez(str(dst), **out)

    # Report size reduction
    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    print(f"[quantize] {src_mb:.0f} MB  →  {dst_mb:.0f} MB  ({100*(1-dst_mb/src_mb):.0f}% reduction)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc int8 quantisation of MLX weights")
    ap.add_argument("src", type=Path, help="Source .mlx.npz (float32)")
    ap.add_argument("--dst", type=Path, default=None, help="Output path (default: auto .int8.npz)")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    if not src.exists():
        sys.exit(f"File not found: {src}")

    if args.dst:
        dst = args.dst
    else:
        # CorridorKey_v1.0.mlx.npz → CorridorKey_v1.0.mlx.int8.npz
        name = src.name.replace(".mlx.npz", ".mlx.int8.npz")
        if name == src.name:
            name = src.stem + ".int8.npz"
        dst = src.parent / name

    quantize_npz(src, dst)


if __name__ == "__main__":
    main()
