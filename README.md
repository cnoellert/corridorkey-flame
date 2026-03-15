# corridorkey-flame

Flame PyBox integration for [CorridorKey](https://github.com/cnoellert/comfyui-corridorkey) — a neural green screen keyer based on GreenFormer/Hiera.

Runs on **macOS Apple Silicon** (MLX) and **Linux CUDA** (Rocky Linux / Ubuntu). Platform is detected automatically.

---

## Requirements

| Platform | Hardware | Software |
|----------|----------|----------|
| macOS | Apple Silicon (M1–M4) | Miniconda, Flame 2025+ |
| Linux | NVIDIA GPU (RTX/A-series) | Miniconda, CUDA 12.x, Flame 2025+ |

> **Linux note:** Do not run ComfyUI or other GPU-heavy processes alongside Flame. The daemon uses CPU offload (model lives in system RAM, moves to GPU per-frame) but Flame itself holds ~20GB of VRAM on a 24GB card.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/cnoellert/corridorkey-flame.git
cd corridorkey-flame
```

### 2. Run the installer

```bash
bash install.sh
```

Optionally provide weights path directly:

```bash
bash install.sh --weights /path/to/CorridorKey_v1.0.pth
```

The installer will:
- Detect platform (macOS → MLX env, Linux → CUDA env)
- Create the conda environment (`corridorkey-mlx` or `corridorkey-cuda`)
- Install Python dependencies (auto-detects CUDA version on Linux)
- Create `/opt/corridorkey/{models,pybox,reference}/`
- Copy pybox and reference inference code into place
- Copy weights if provided

### 3. Copy weights (if not done above)

| Platform | File |
|----------|------|
| macOS | `CorridorKey_v1.0.mlx.npz` → `/opt/corridorkey/models/` |
| Linux | `CorridorKey_v1.0.pth` → `/opt/corridorkey/models/` |

### 4. Add to Flame

In Flame Batch, add a **PyBox** node and point it at:

```
/opt/corridorkey/pybox/corridorkey_pybox.py
```

---

## Inputs / Outputs

| Pin | Description |
|-----|-------------|
| `IN_PLATE` | RGB plate (EXR, scene-linear or sRGB) |
| `IN_MATTE` | Rough matte / holdout mask (EXR) |
| `OUT_FG` | Keyed foreground RGBA (EXR) |
| `OUT_ALPHA` | Alpha channel RGB (EXR) |

---

## PyBox Parameters

**Model page**
- `Weights` — path to model weights file
- `Quantized` — enable quantized inference (macOS only, reduces memory)

**Settings page**
- `Add sRGB Gamma` — enable if input is scene-linear (converts to sRGB before inference, back to linear after)
- `Despill Strength` — green spill suppression (0–1)
- `Despeckle` — minimum alpha island area to remove (0–2000 px²)

---

## Troubleshooting

**`EnvironmentNameNotFound: corridorkey-mlx`** — Old pybox file installed. Run `git pull && make -C pybox install`.

**`CUDA out of memory`** — ComfyUI or another GPU process is running alongside Flame. Kill it and retry.

**Daemon not starting** — Check `/tmp/corridorkey_daemon.log` for the full error.

**Frames not updating** — Check `/tmp/corridorkey_ready` exists (daemon loaded). If not, the model is still loading.
