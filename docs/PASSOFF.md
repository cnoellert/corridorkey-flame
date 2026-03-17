# CorridorKey Flame PyBox — Session Passoff
**Date:** 2026-03-17 (fully working on Mac MLX + Linux CUDA)
**Repo:** `git@github.com:cnoellert/corridorkey-flame.git`

---

## Status: WORKING — Mac and Linux ✅

The scrubbing flood is fixed. The root cause and full architecture are
documented in `docs/PYBOX_ARCHITECTURE.md`. Read that before touching any
PyBox code.

**The fix in one line:**
```python
if not self.is_processing():
    return
```

`is_processing()` is a `pybox_v1` API method that returns `True` only when
Flame has committed to rendering a frame and is actively waiting for output EXRs.
During scrubbing, timeline navigation, and UI ticks it is `False`. We simply
return immediately in those cases and block (via `_send_frame()`) only when
Flame is actually waiting.

**Source of this discovery:** `BLG.py` from FilmLight's BLG-for-Flame product.
Their `execute()` uses the identical guard followed by a 45-second blocking
HTTP request. It has never had a scrubbing problem.

---

## What Works

- **Mac (MLX daemon):** Single frame ✅. Frame scrubbing ✅ (fixed this session).
- **Linux/Rocky (CUDA daemon):** Single frame ✅. Frame scrubbing ✅ (fixed this session).
- Model loads, runs inference, writes EXRs correctly on both platforms.
- CPU offload on CUDA works — model on CPU between frames, VRAM clears.
- `--quantized` flag: functional on MLX, accepted/ignored on CUDA.
- Atomic EXR writes (`.tmp` → `rename`) prevent partial-file reads.

---

## What Changed This Session

### `pybox/corridorkey_pybox.py` — full rewrite

The previous async trigger-fire approach was replaced with:
1. `if not self.is_processing(): return` guard at top of execute()
2. `_send_frame()` blocking call (already existed, now actually used)
3. Removed: `done_frame` sentinel, `last_frame`/`last_params` file tracking,
   stale-EXR deletion logic, async trigger fire, `results_valid` check.
   None of that is needed when `is_processing()` gates everything.

The module docstring in `corridorkey_pybox.py` explains the architecture.
`docs/PYBOX_ARCHITECTURE.md` has the full story including the failed history.

### `pybox/corridorkey_daemon_mlx.py` — trigger unlink fix

The bare `os.unlink(trigger)` after the try/except block was a duplicate —
it crashed the daemon with `FileNotFoundError` on every frame because the
trigger had already been consumed. Removed the duplicate. The remaining
unlink is now solely inside `try/except OSError`.

### `pybox/corridorkey_daemon_cuda.py` — same trigger unlink fix

Identical bug: bare `os.unlink(trigger)` not wrapped in try/except. The
trigger can disappear between `os.path.exists()` and `os.unlink()` if
the handler's `_cleanup_sentinels()` runs in that window. Wrapped in
`try/except OSError`.

### `docs/PYBOX_ARCHITECTURE.md` — new file

Comprehensive architecture document covering:
- How Flame calls `execute()` and what `is_processing()` means
- The BLG reference and how we found the answer
- Full system diagram (handler → IPC → daemon)
- All critical rules with explanations
- Failed approaches and why they failed (do not re-try)
- Reinstall and debugging commands
- Known backlog

---

## Critical Rules — Do Not Violate

(Full explanations in `docs/PYBOX_ARCHITECTURE.md`)

1. **`is_processing()` guard must stay** at top of execute() — do not remove
2. **`try/finally` must wrap CUDA inference** — CPU offload must always run
3. **No OOM retry loops on CUDA** — causes kernel panic / hard system crash
4. **`PYTORCH_CUDA_ALLOC_CONF` before `import torch`** — top of daemon file
5. **Do not add `max_split_size_mb`** — causes per-frame VRAM spikes
6. **Wait 0.5 s after `pkill` before spawning** — pkill is async
7. **Do not modify `reference/utils/inference.py`** — restored to 530e607, leave it
8. **`--quantized` must be in CUDA argparse** — undeclared args cause silent exit
9. **`os.unlink(trigger)` must be in `try/except OSError`** in both daemons —
   `_cleanup_sentinels()` in the handler can race and delete the trigger
   between `os.path.exists()` and `os.unlink()`, causing a `FileNotFoundError`
   that crashes the daemon and requires a full respawn

---

## Install / Update Commands

**Mac:**
```bash
cd ~/Documents/GitHub/corridorkey-flame
git pull
make -C pybox install
pkill -f corridorkey_daemon_mlx
rm -f /tmp/corridorkey_*.json* /tmp/corridorkey_ready /tmp/corridorkey_trigger \
       /tmp/corridorkey_done /tmp/corridorkey_error
```

**Rocky Linux:**
```bash
cd /opt/corridorkey-flame
git pull
make -C pybox install
make -C pybox install-ref   # only if reference/utils/ changed
pkill -f corridorkey_daemon_cuda
rm -f /tmp/corridorkey_*.json* /tmp/corridorkey_ready /tmp/corridorkey_trigger \
       /tmp/corridorkey_done /tmp/corridorkey_error
```

**Clean reinstall:**
```bash
sudo rm -rf /opt/corridorkey
bash install.sh
```

---

## Known Backlog

- **ONNX export of GreenFormer** — eliminate daemon/IPC entirely.
  PyTorch `.pth` → ONNX → in-process inference.
  ~1 day estimate. Key risk: Hiera windowed attention tracing.
  See `docs/PYBOX_ARCHITECTURE.md` for details.
