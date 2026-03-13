"""
test_frame.py — single-frame inference test for corridorkey-mlx

Usage
-----
# Maskless (trimap = 0.5 everywhere)
python test_frame.py /path/to/frame.exr

# With trimap
python test_frame.py /path/to/frame.exr --trimap /path/to/trimap.exr

# Control tile size (default 1024 — good up to ~6K with overlap)
python test_frame.py /path/to/frame.exr --tile 512

# Save a composite-over-black preview alongside the key
python test_frame.py /path/to/frame.exr --preview

Supports: EXR (linear), TIFF (16/32-bit), PNG/JPG (auto gamma-decoded to linear)
Outputs:  <stem>_alpha.exr, <stem>_fg.exr, <stem>_key.exr (premult RGBA)
          <stem>_preview.png  (if --preview)
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# I/O helpers — EXR, TIFF, PNG/JPG
# ---------------------------------------------------------------------------

def _read_image(path: Path) -> np.ndarray:
    """
    Returns float32 [H, W, 3] RGB in linear light, range 0-1 (clamped for display,
    unclamped for HDR EXR so the model sees full linear values).
    """
    ext = path.suffix.lower()
    if ext == '.exr':
        import OpenEXR, Imath
        f = OpenEXR.InputFile(str(path))
        dw = f.header()['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = f.header().get('channels', {})
        def _chan(name):
            if name in channels:
                return np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(H, W)
            return np.zeros((H, W), dtype=np.float32)
        R, G, B = _chan('R'), _chan('G'), _chan('B')
        return np.stack([R, G, B], axis=-1)

    elif ext in ('.tif', '.tiff'):
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cv2 could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Normalise 16-bit to 0-1
        if img.max() > 1.0:
            img /= 65535.0 if img.max() <= 65535.0 else img.max()
        return img

    else:  # PNG / JPG — sRGB, decode gamma
        from PIL import Image
        img = np.array(Image.open(str(path)).convert('RGB')).astype(np.float32) / 255.0
        # sRGB → linear
        img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
        return img


def _read_trimap(path: Path, H: int, W: int) -> np.ndarray:
    """Returns float32 [H, W, 1], values 0-1."""
    ext = path.suffix.lower()
    if ext == '.exr':
        import OpenEXR, Imath
        f = OpenEXR.InputFile(str(path))
        dw = f.header()['dataWindow']
        tW = dw.max.x - dw.min.x + 1
        tH = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        ch = f.header().get('channels', {})
        name = 'A' if 'A' in ch else ('R' if 'R' in ch else next(iter(ch)))
        arr = np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(tH, tW)
    else:
        from PIL import Image
        arr = np.array(Image.open(str(path)).convert('L')).astype(np.float32) / 255.0
    if arr.shape[:2] != (H, W):
        import cv2
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    return arr[:, :, None]


def _write_exr(path: Path, data: np.ndarray) -> None:
    """data: float32 [H, W, C] where C is 1, 3, or 4."""
    import OpenEXR, Imath
    H, W, C = data.shape
    header = OpenEXR.Header(W, H)
    names = {1: ['Y'], 3: ['R','G','B'], 4: ['R','G','B','A']}[C]
    header['channels'] = {n: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for n in names}
    f = OpenEXR.OutputFile(str(path), header)
    f.writePixels({n: data[:,:,i].tobytes() for i, n in enumerate(names)})
    f.close()


def _write_preview(path: Path, rgba_linear: np.ndarray) -> None:
    """Composite premult RGBA over black, tonemap, save PNG."""
    from PIL import Image
    rgb = rgba_linear[:, :, :3]  # premult — already composited over black
    # Clamp and linear → sRGB
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb**(1/2.4) - 0.055)
    rgb8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb8).save(str(path))
    print(f"  Preview saved: {path.name}")


def _apply_garbage_matte(result: np.ndarray, matte_path: Path,
                          H: int, W: int, dilation: int = 15) -> np.ndarray:
    """
    Load garbage matte, dilate, multiply against alpha channel of result.
    result: premult RGBA [H, W, 4]
    Returns same shape with alpha (and premult RGB) masked.
    """
    import cv2
    # Load matte as single channel 0-1
    ext = matte_path.suffix.lower()
    if ext == '.exr':
        import OpenEXR, Imath
        f = OpenEXR.InputFile(str(matte_path))
        dw = f.header()['dataWindow']
        mW = dw.max.x - dw.min.x + 1
        mH = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        ch = f.header().get('channels', {})
        name = 'A' if 'A' in ch else ('R' if 'R' in ch else next(iter(ch)))
        arr = np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(mH, mW)
    else:
        from PIL import Image
        arr = np.array(Image.open(str(matte_path)).convert('L')).astype(np.float32) / 255.0

    # Resize to frame dims if needed
    if arr.shape[:2] != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)

    # Dilate to be safe (avoids hard garbage matte edge cutting into subject)
    if dilation > 0:
        ksize = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        arr = cv2.dilate(arr, kernel)

    mask = arr[:, :, None]  # [H, W, 1]

    # Multiply alpha and premult RGB by mask
    out = result.copy()
    out[:, :, :3] *= mask   # premult RGB
    out[:, :, 3:4] *= mask  # alpha
    return out


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def _infer_tile(model, tile_rgba: np.ndarray) -> np.ndarray:
    """tile_rgba: float32 [H, W, 4] → premult RGBA float32 [H, W, 4]."""
    x = mx.array(tile_rgba[None])       # [1, H, W, 4]
    out = model(x)
    mx.eval(out['alpha'], out['fg'])
    alpha = np.array(out['alpha'][0])   # [H, W, 1]
    fg    = np.array(out['fg'][0])      # [H, W, 3]
    return np.concatenate([fg * alpha, alpha], axis=-1)


def _pad32(x: np.ndarray):
    """Pad H, W to next multiple of 32. Returns (padded, pH, pW)."""
    H, W = x.shape[:2]
    pH = (32 - H % 32) % 32
    pW = (32 - W % 32) % 32
    return np.pad(x, ((0, pH), (0, pW), (0, 0)), mode='reflect'), pH, pW


def infer_frame(model, rgb: np.ndarray, trimap: np.ndarray | None,
                tile_size: int = 1024, overlap: int = 128) -> np.ndarray:
    """
    Full-resolution tiled inference.
    rgb:    [H, W, 3] float32 linear
    trimap: [H, W, 1] float32 0-1, or None
    Returns premult RGBA [H, W, 4] float32.
    """
    H, W = rgb.shape[:2]
    if trimap is None:
        trimap = np.full((H, W, 1), 0.5, dtype=np.float32)

    rgba_in = np.concatenate([rgb, trimap], axis=-1)  # [H, W, 4]

    # Fast path: fits in one tile
    if H <= tile_size and W <= tile_size:
        padded, pH, pW = _pad32(rgba_in)
        result = _infer_tile(model, padded)
        return result[:H, :W]

    # Tiled path with Hanning-window blending
    output  = np.zeros((H, W, 4), dtype=np.float64)
    weights = np.zeros((H, W, 1), dtype=np.float64)
    step    = tile_size - overlap

    ys = list(range(0, max(1, H - tile_size + 1), step))
    xs = list(range(0, max(1, W - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < H:
        ys.append(max(0, H - tile_size))
    if not xs or xs[-1] + tile_size < W:
        xs.append(max(0, W - tile_size))
    ys = sorted(set(ys)); xs = sorted(set(xs))

    win_1d_y = np.hanning(tile_size).reshape(-1, 1).astype(np.float64)
    win_1d_x = np.hanning(tile_size).reshape(1, -1).astype(np.float64)
    win = (win_1d_y * win_1d_x)[:, :, None]  # [T, T, 1]

    total = len(ys) * len(xs)
    done  = 0
    for y in ys:
        y2 = min(y + tile_size, H)
        y1 = max(0, y2 - tile_size)
        for x in xs:
            x2 = min(x + tile_size, W)
            x1 = max(0, x2 - tile_size)
            tile = rgba_in[y1:y2, x1:x2]
            th, tw = tile.shape[:2]
            padded, pH, pW = _pad32(tile)
            result = _infer_tile(model, padded)[:th, :tw]
            w = win[:th, :tw]
            output [y1:y2, x1:x2] += result * w
            weights[y1:y2, x1:x2] += w
            done += 1
            print(f"  Tile {done}/{total} ({y1},{x1})→({y2},{x2})", end='\r', flush=True)

    print()
    return (output / np.maximum(weights, 1e-8)).astype(np.float32)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='CorridorKey MLX single-frame test')
    ap.add_argument('frame',             type=Path,            help='Input frame (EXR/TIFF/PNG/JPG)')
    ap.add_argument('--trimap',         type=Path, default=None, help='Optional trimap')
    ap.add_argument('--garbage-matte',  type=Path, default=None, help='Garbage matte EXR/PNG')
    ap.add_argument('--gm-dilation',    type=int,  default=15,   help='Garbage matte dilation px (default 15)')
    ap.add_argument('--model',           type=Path,
                    default=Path('/Users/cnoellert/ComfyUI/models/corridorkey/CorridorKey_v1.0.pth'),
                    help='.pth or .mlx.npz checkpoint')
    ap.add_argument('--tile',            type=int,  default=1024,  help='Tile size (default 1024)')
    ap.add_argument('--overlap',         type=int,  default=128,   help='Tile overlap px (default 128)')
    ap.add_argument('--out-dir',         type=Path, default=None,  help='Output dir (default: alongside input)')
    ap.add_argument('--preview',         action='store_true',      help='Save PNG composite-over-black preview')
    ap.add_argument('--quantize',        choices=['int8'], default=None)
    args = ap.parse_args()

    frame_path = args.frame.expanduser().resolve()
    out_dir    = (args.out_dir or frame_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem       = frame_path.stem

    # ---- Auto-convert weights (Option C) -----------------------------------
    model_path = args.model.expanduser().resolve()
    if model_path.suffix != '.npz':
        from inference import _ensure_converted
        npz_path = _ensure_converted(model_path, args.quantize)
    else:
        npz_path = model_path

    # ---- Load model --------------------------------------------------------
    from model import GreenFormer
    print(f"[test] Building GreenFormer …")
    gf = GreenFormer()
    weights = {k: v for k, v in dict(mx.load(str(npz_path))).items()
               if not k.startswith('__')}
    gf.load_weights(list(weights.items()))
    mx.eval(gf.parameters())
    gf.eval()
    print(f"[test] Weights loaded ({len(weights)} tensors)")

    # ---- Read input --------------------------------------------------------
    print(f"[test] Reading {frame_path.name} …")
    rgb = _read_image(frame_path)
    H, W = rgb.shape[:2]
    print(f"[test] Frame: {W}×{H}")

    trimap = None
    if args.trimap:
        trimap = _read_trimap(args.trimap.expanduser().resolve(), H, W)
        print(f"[test] Trimap loaded")
    else:
        print(f"[test] No trimap — using maskless (all 0.5)")

    # ---- Inference ---------------------------------------------------------
    print(f"[test] Inferring at tile={args.tile} overlap={args.overlap} …")
    t0 = time.time()
    result = infer_frame(gf, rgb, trimap, tile_size=args.tile, overlap=args.overlap)
    elapsed = time.time() - t0
    print(f"[test] Done: {elapsed:.2f}s")

    alpha = result[:, :, 3:4]
    fg    = result[:, :, :3]  # premult FG

    # Apply garbage matte (post-inference, dilated)
    if args.garbage_matte:
        gm_path = args.garbage_matte.expanduser().resolve()
        print(f"[test] Applying garbage matte: {gm_path.name} (dilation={args.gm_dilation}px)")
        result = _apply_garbage_matte(result, gm_path, H, W, dilation=args.gm_dilation)

    # Stats
    print(f"[test] Alpha: min={float(alpha.min()):.4f}  max={float(alpha.max()):.4f}  "
          f"mean={float(alpha.mean()):.4f}")

    # ---- Write outputs -----------------------------------------------------
    alpha_path   = out_dir / f"{stem}_alpha.exr"
    fg_path      = out_dir / f"{stem}_fg.exr"
    key_path     = out_dir / f"{stem}_key.exr"

    print(f"[test] Writing outputs to {out_dir} …")
    _write_exr(alpha_path, alpha)
    _write_exr(fg_path,    np.concatenate([fg, alpha], axis=-1))
    _write_exr(key_path,   result)

    print(f"  {alpha_path.name}")
    print(f"  {fg_path.name}")
    print(f"  {key_path.name}")

    if args.preview:
        _write_preview(out_dir / f"{stem}_preview.png", result)

    print("[test] Complete.")


if __name__ == '__main__':
    main()
