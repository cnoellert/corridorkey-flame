# CorridorKey Flame PyBox — Session Passoff
**Date:** 2026-03-16  
**Repo:** `git@github.com:cnoellert/corridorkey-flame.git`  
**Last commit:** `a205197`

---

## What Works

- **Mac (MLX daemon):** Single frame result works. Matte output works. Frame scrubbing is BROKEN (see below).
- **Linux/Rocky (CUDA daemon):** Single frame result works. VRAM/OOM is stable. Frame scrubbing is BROKEN (same issue).
- Model loads, runs inference, writes EXRs correctly on both platforms.
- CPU offload on CUDA works — model returns to CPU between frames, VRAM clears.
- `--quantized` flag accepted on both platforms (no-op on CUDA, functional on MLX).

---

## The Unsolved Problem: Frame Scrubbing

### Symptom
Changing frames floods the Flame shell with:
```
Unable to open file '/tmp/corridorkey_out_fg.exr': No such file or directory
Node "pybox1" must have output Result0
(repeats until schematic view)
```
Frame never draws. Must switch to schematic, change frame, then switch back to result.

### Root Cause (understood)
Flame calls `execute()` on every frame it passes through during scrubbing.  
The PyBox uses an async IPC pattern (fire trigger, daemon processes, return EXR).  
This creates a fundamental conflict:

- **If `execute()` blocks** waiting for inference (~3s on Mac): Flame aborts the PyBox process ("PYBOX process aborted"). Flame has a hard timeout on `execute()`.
- **If `execute()` returns immediately** (async): Flame tries to read the EXR before it exists, errors, retries, and floods the shell. The EXR on disk may belong to the wrong frame.

### What Was Tried (all failed)
1. **Debounce in daemon** — daemon waits N ms after trigger before processing. Broken because handler keeps writing triggers faster than the debounce window.
2. **Idle-for-300ms debounce** — wait until no trigger for 300ms. Same race condition — daemon unlinks trigger, handler writes new one in the gap.
3. **INFERRING lockfile** — block handler from writing new triggers while daemon is busy. Broke first-frame render entirely.
4. **done_frame sentinel** — daemon writes which frame it processed, handler checks if EXR is valid for current frame. Helped stale result issue but didn't fix flooding.
5. **Delete EXRs before firing trigger** — so Flame errors cleanly on current call and retries. Still floods because Flame retries faster than inference completes.
6. **Always block in execute()** — worked conceptually but Flame's timeout killed it.

### What Needs Investigation
The question is: **how do other PyBoxes with slow operations handle this?**

Options to investigate:
1. **`set_notice_msg()` as a stall** — some PyBoxes return a notice/warning and Flame stops retrying until user interaction. Investigate if there's a way to tell Flame "not ready yet, stop retrying."
2. **Return a placeholder EXR** — write a 1x1 black EXR immediately so Flame has *something* to read. Handler returns, Flame draws black, daemon finishes and overwrites with real result. Flame may or may not re-read.
3. **Flame PyBox API research** — check if there's a `set_busy()`, `request_update()`, or similar API call that signals Flame to wait and retry on its own schedule rather than hammering execute().
4. **Check the Wiretap PyBox** — Chris wrote ComfyUI-WiretapBrowser nodes. How does that handle async results?
5. **Single-frame-only mode** — disable scrubbing support entirely (document it), only process on explicit Result button click.

---

## Current Code State

### Handler: `pybox/corridorkey_pybox.py`
- Spawns daemon on first use, kills/respawns on weights or quantized change
- `execute()` fires trigger async (no blocking), returns immediately
- Checks `done_frame` sentinel to validate EXR belongs to current frame
- `_send_frame()` (blocking) exists but causes Flame timeout if used directly

### CUDA Daemon: `pybox/corridorkey_daemon_cuda.py`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set at module level (before torch import)
- Model loads to CPU, moves to CUDA for inference, returns to CPU in `try/finally`
- `try/finally` wraps inference — CPU offload is guaranteed even on error
- No retry loop (caused kernel panic previously)
- `--quantized` accepted, ignored (no CUDA quantized weights exist)

### MLX Daemon: `pybox/corridorkey_daemon_mlx.py`  
- Functional int8 dequantization on load when `--quantized`
- Same IPC contract as CUDA daemon

### Reference: `reference/utils/inference.py`
- Restored to original working state (530e607)
- `channels_last` on both MPS and CUDA (original, working)
- No float16 input cast (original)
- Do NOT modify this file again without careful testing

---

## Critical Bugs Fixed This Session (do not re-introduce)

### 1. `try/finally` must wrap the actual inference call
```python
# WRONG -- finally never runs on exception
for attempt in range(2):
    try: result = engine.process_frame_tensor(...)
    except OOM: retry()
try: pass
finally: engine.model.to(cpu)  # ← NEVER RUNS if inference raised

# CORRECT
try:
    result = engine.process_frame_tensor(...)
finally:
    engine.model.to(cpu)  # ← ALWAYS RUNS
```

### 2. No OOM retry loops
Retrying on CUDA OOM while model state is uncertain causes kernel panic and hard system crash.

### 3. PYTORCH_CUDA_ALLOC_CONF must be set before torch import
Setting it inside main() after `import torch` has no effect — CUDA allocator is frozen at first import.

### 4. max_split_size_mb causes spikes on every frame
Prevents large contiguous allocations. Do not add this.

### 5. --quantized must be declared in CUDA daemon argparse
Undeclared args cause silent SystemExit on daemon spawn.

### 6. pkill is async — wait after kill
After `_kill_daemon()`, wait 0.5s and clean sentinels before spawning replacement.

### 7. Do not modify reference/utils/inference.py
All channels_last and float16 cast changes to inference.py made things worse, not better. The original code handles this correctly via autocast.

---

## Install / Update Commands

**Mac:**
```bash
cd ~/Documents/GitHub/corridorkey-flame
git pull
make -C pybox install
pkill -f corridorkey_daemon_mlx
rm -f /tmp/corridorkey_params.json.* /tmp/corridorkey_ready /tmp/corridorkey_error
```

**Rocky Linux:**
```bash
cd /opt/corridorkey-flame
git pull
make -C pybox install
make -C pybox install-ref   # only needed if reference/utils/ changed
pkill -f corridorkey_daemon_cuda
rm -f /tmp/corridorkey_params.json.* /tmp/corridorkey_ready /tmp/corridorkey_error
```

**Clean reinstall (either platform):**
```bash
sudo rm -rf /opt/corridorkey
bash install.sh
```

---

## Install Layout
```
/opt/corridorkey/
├── models/
│   ├── CorridorKey_v1.0.mlx.npz   (Mac)
│   └── CorridorKey_v1.0.pth        (Linux)
├── mlx/                             (Mac only)
├── reference/                       (Linux only)
│   ├── CorridorKeyModule/
│   └── utils/
└── pybox/
    ├── corridorkey_pybox.py
    ├── corridorkey_daemon_mlx.py
    └── corridorkey_daemon_cuda.py
```
