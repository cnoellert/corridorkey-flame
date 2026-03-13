# corridorkey-mlx

Pure MLX inference port of [CorridorKey](https://github.com/nikopueringer/CorridorKey) — GreenFormer neural green screen keying, running natively on Apple Silicon with no PyTorch dependency at inference time.

## Architecture

```
GreenFormer
├── HieraEncoder       (Hiera Base Plus, 4-ch input, NHWC)
├── DecoderHead ×2     (alpha + foreground, MLP-mixer style)
└── CNNRefinerModule   (dilated residual, receptive field ~65px)
```

## Setup

```bash
pip install mlx numpy OpenEXR
# PyTorch only needed for one-time weight conversion:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

```bash
# First run auto-converts CorridorKey_v1.0.pth → .mlx.npz (one time, ~30s)
python inference.py frames/*.exr --model /path/to/CorridorKey_v1.0.pth --out-dir ./keyed

# Optional: int8 weights (~6 GB instead of ~23 GB, opens up 24 GB machines)
python inference.py frames/*.exr --model /path/to/CorridorKey_v1.0.pth --quantize int8

# If you already have a .mlx.npz:
python inference.py frames/*.exr --model CorridorKey_v1.0.mlx.npz

# Manual conversion
python convert.py CorridorKey_v1.0.pth [--quantize int8]

# Post-hoc int8 on existing float32 .npz
python quantize.py CorridorKey_v1.0.mlx.npz
```

## File layout

```
corridorkey-mlx/
├── model.py       # GreenFormer in mlx.nn (no PyTorch)
├── convert.py     # .pth → .mlx.npz (PyTorch required, run once)
├── inference.py   # CLI batch processor + Option-C auto-convert
├── quantize.py    # Post-hoc int8 quantisation pass
└── requirements.txt
```

## Weight conversion details

| Transform | Reason |
|---|---|
| Conv2d `[O,I,kH,kW]` → `[O,kH,kW,I]` | MLX NHWC conv layout |
| BatchNorm folded into affine scale/bias | Removes running-stats memory access at inference |
| DropPath / classifier head dropped | Inference-only build |
| SHA-256 embedded in .npz | Cache invalidation when .pth is updated |

## Performance notes (M4 Max, 128 GB)

- float32 path: ~23 GB weights, numerically stable (MPS float16 causes NaN in refiner)
- int8 path: ~6 GB weights, ~2× throughput on conv layers via MLX dequantise kernels
- Transformer backbone (linear/matmul) sees the largest MLX gains over MPS
- CNN refiner conv ops are currently the bottleneck; improves with MLX conv kernel updates

## Checkpoint

Download `CorridorKey_v1.0.pth` from HuggingFace:
```
huggingface-cli download nikopueringer/CorridorKey_v1.0 CorridorKey_v1.0.pth
```
