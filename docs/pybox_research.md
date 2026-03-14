# Flame PyBox Research Notes
_March 2026 — corridorkey-mlx integration planning_

---

## What Is PyBox

PyBox is a **batch compositing node** in Autodesk Flame. It lives in the batch schematic
alongside Comp, Action, Matchbox, etc. and has proper input/output sockets that wire to
other nodes. It is NOT a hook or a right-click script — it is a first-class node in the
compositing graph.

Flame invokes the handler script once per frame during render. The handler reads input
frames from disk (written by Flame), processes them, and writes output frames back to
disk (read back by Flame).

---

## File Locations

```
/opt/Autodesk/presets/2026.2.1/pybox/          # example handlers (bundled)
/opt/Autodesk/presets/2026.2.1/shared/pybox/   # pybox_v1.py module lives here
```

Flame's embedded Python needs to be able to `import pybox_v1`. The shared/pybox path
is on the embedded Python's sys.path automatically.

Custom handlers can live anywhere accessible on disk; you point the node at them via
the Pybox node UI or:
```python
flame.batch.create_node("Pybox", "/path/to/handler.py")
```

---

## The Handler Contract

Every handler is a Python file that:

1. Imports `pybox_v1 as pybox`
2. Defines a class inheriting from `pybox.BaseClass`
3. Implements four methods: `initialize`, `setup_ui`, `execute`, `teardown`
4. At module level: instantiates the class with `argv[0]`, calls `dispatch()`,
   then `write_to_disk(argv[0])`

Flame passes a JSON state file as `argv[0]`. The base class reads it in `__init__`,
and `write_to_disk` saves any mutations back. `dispatch()` routes to whichever
method matches the current `state_id`.

### State machine

```
initialize → setup_ui → execute → (execute on every frame) → teardown
```

Each method must call `self.set_state_id("next_state")` before returning so Flame
knows where to go on the next invocation.


---

## pybox_v1.BaseClass API

### Image format
```python
self.set_img_format("exr")   # "jpeg", "exr", "sgi", "tga", "tiff"
```
Determines what format Flame writes the input frames in. Always use `"exr"` for VFX.

### Input sockets
```python
self.remove_in_sockets()                          # clear defaults (Front/Undefined/Matte)
self.add_in_socket("Front", "/tmp/ck_plate.exr")  # plate
self.add_in_socket("Matte", "/tmp/ck_matte.exr")  # alpha hint / garbage matte
self.get_in_socket_path(0)                        # path Flame wrote the frame to
```
Valid socket type strings: `Front`, `Back`, `Matte`, `3DMotion`, `Background`,
`MotionVector`, `Normal`, `Position`, `Uv`, `ZDepth`, `undefined`

### Output sockets
```python
self.remove_out_sockets()
self.add_out_socket("Result",   "/tmp/ck_fg.exr")
self.add_out_socket("OutMatte", "/tmp/ck_alpha.exr")
self.get_out_socket_path(0)
```
Valid socket type strings: `Result`, `OutMatte`, `Out3DMotion`, `OutBackground`,
`OutMotionVector`, `OutNormal`, `OutPosition`, `OutUv`, `OutZDepth`, `undefined`

### Metadata available in execute()
```python
self.get_frame()          # int — current frame number
self.get_width()          # int — from batch defaults
self.get_height()         # int
self.get_bit_depth()      # str
self.get_colour_space()   # str — project colour space string
self.get_framerate()      # str
self.get_shot_name()      # str
self.get_project()        # str
```

### UI elements
```python
pybox.create_float_numeric("Despill Strength", value=1.0, default=1.0, min=0.0, max=1.0, row=0, col=0)
pybox.create_toggle_button("Input is sRGB", value=True, default=True, row=1, col=0)
pybox.create_popup("Mode", ["sRGB", "Linear"], value=0, row=2, col=0)
pybox.create_file_browser("Weights Path", "/path/to/weights.npz", "npz", "/")
pybox.create_page("CorridorKey", "Settings")   # tab name + column headers

self.add_global_elements(elem, ...)    # persist across frames — use for settings
self.add_render_elements(elem, ...)    # per-render, animatable via channel editor
self.set_ui_pages(page)

self.get_global_element_value("Despill Strength")   # read current slider value
self.get_ui_changes()                               # list of elements changed since last call
```
UI grid: rows 0–4, cols 0–3, pages 0–5.


---

## The Critical Constraint: Embedded Python

Flame runs the handler in its **own embedded Python interpreter**, NOT your conda env.
- `import mlx` will fail
- `import torch` will fail
- `import numpy`, `import cv2`, stdlib: generally available
- Any package installed into Flame's Python is available

**Solution: daemon pattern**

`initialize()` spawns a persistent subprocess running in the `corridorkey-mlx` conda
env with the model already loaded, listening on named pipes (FIFOs).
`execute()` signals it per-frame and waits for the output.
`teardown()` kills it.

This is exactly the pattern used by the bundled `nuke_px.py` example and by
`Ls_nukeBLG.py` in lcrs/pyboxes.

### Daemon lifecycle

```
initialize()
  ├─ create FIFOs: /tmp/ck_cmd, /tmp/ck_done
  ├─ spawn: conda run -n corridorkey-mlx python corridorkey_daemon.py &
  └─ daemon loads GreenFormer model (~2s warmup), opens FIFOs

execute() — called once per frame during render
  ├─ Flame has already written plate EXR to in_socket[0] path
  ├─ Flame has already written matte EXR to in_socket[1] path
  ├─ handler writes JSON params + frame number → cmd FIFO
  ├─ daemon: runs infer_frame(), writes FG + alpha EXRs to out socket paths
  ├─ daemon: signals done → done FIFO
  └─ handler unblocks, returns (Flame reads EXRs from out socket paths)

teardown()
  └─ handler sends kill signal, daemon exits
```

Net cost per frame: ~3s inference. Model loads **once** — no reload overhead after warmup.

---

## Input/Output Socket Paths

Set in `initialize()` as fixed temp paths. Flame writes the input EXR to
`in_socket[n].path` *before* calling `execute()`. Handler (via daemon) must write the
result EXR to `out_socket[n].path` *before* `execute()` returns.

Use `/tmp` with a unique prefix to avoid collisions:
```python
PREFIX = "/tmp/corridorkey_"
self.add_in_socket("Front", PREFIX + "in_plate.exr")
self.add_in_socket("Matte", PREFIX + "in_matte.exr")
self.add_out_socket("Result",   PREFIX + "out_fg.exr")
self.add_out_socket("OutMatte", PREFIX + "out_alpha.exr")
```

---

## UI Responsiveness Note

From field experience (vfxrnd.com): "performance in Pybox is pretty abysmal in
its responsiveness." Every parameter change triggers a re-execute() call.

**Mitigation:**
- Check `self.get_ui_changes()` at the top of `execute()`.
- If changes are UI-only (no new frame render), skip inference and return early.
- Add an explicit "Reprocess" toggle that the artist uses to re-run inference
  on the current frame with updated params.

This makes the node feel responsive for parameter tweaking without burning 3s
per slider drag.


---

## Planned CorridorKey PyBox Architecture

### Two-file structure

```
corridorkey_pybox.py    — handler (runs in Flame's embedded Python, thin IPC glue)
corridorkey_daemon.py   — inference server (runs in corridorkey-mlx conda env)
```

### Handler sketch: `corridorkey_pybox.py`

```python
import sys, os, json
import pybox_v1 as pybox

CMD_FIFO  = "/tmp/corridorkey_cmd"
DONE_FIFO = "/tmp/corridorkey_done"
IN_PLATE  = "/tmp/corridorkey_in_plate.exr"
IN_MATTE  = "/tmp/corridorkey_in_matte.exr"
OUT_FG    = "/tmp/corridorkey_out_fg.exr"
OUT_ALPHA = "/tmp/corridorkey_out_alpha.exr"
CONDA_ENV = "corridorkey-mlx"
DAEMON    = "/path/to/corridorkey_daemon.py"

class CorridorKeyBox(pybox.BaseClass):

    def initialize(self):
        self.set_img_format("exr")
        self.remove_in_sockets()
        self.add_in_socket("Front", IN_PLATE)
        self.add_in_socket("Matte", IN_MATTE)
        self.remove_out_sockets()
        self.add_out_socket("Result",   OUT_FG)
        self.add_out_socket("OutMatte", OUT_ALPHA)
        for f in (CMD_FIFO, DONE_FIFO):
            if os.path.exists(f): os.unlink(f)
            os.mkfifo(f)
        os.system(f"conda run -n {CONDA_ENV} python {DAEMON} &")
        self.set_state_id("setup_ui")
        self.setup_ui()

    def setup_ui(self):
        self.add_global_elements(
            pybox.create_float_numeric("Despill Strength", 1.0, 1.0, 0.0, 1.0, row=0, col=0),
            pybox.create_float_numeric("Refiner Scale",    1.0, 1.0, 0.0, 3.0, row=1, col=0),
            pybox.create_toggle_button("Input is sRGB",    True,  True,  row=2, col=0),
            pybox.create_toggle_button("Despeckle",        False, False, row=3, col=0),
            pybox.create_toggle_button("Reprocess",        False, False, row=4, col=0),
        )
        self.set_ui_pages(pybox.create_page("CorridorKey", "Settings"))
        self.set_state_id("execute")

    def execute(self):
        changes = self.get_ui_changes()
        reprocess = self.get_global_element_value("Reprocess")
        if changes and not reprocess:
            return   # UI-only change, skip inference
        if reprocess:
            self.set_global_element_value("Reprocess", False)
        params = {
            "frame":            self.get_frame(),
            "despill_strength": self.get_global_element_value("Despill Strength"),
            "refiner_scale":    self.get_global_element_value("Refiner Scale"),
            "input_is_srgb":    self.get_global_element_value("Input is sRGB"),
            "despeckle":        self.get_global_element_value("Despeckle"),
        }
        open(CMD_FIFO,  "w").write(json.dumps(params))
        open(DONE_FIFO, "r").close()   # block until daemon signals done

    def teardown(self):
        try:
            open(CMD_FIFO, "w").write(json.dumps({"quit": True}))
        except Exception:
            pass

p = CorridorKeyBox(sys.argv[1])
p.dispatch()
p.write_to_disk(sys.argv[1])
```

### Daemon sketch: `corridorkey_daemon.py`

```python
# Runs in corridorkey-mlx conda env. Loaded ONCE, serves frames via FIFO.
import json, sys
sys.path.insert(0, "/path/to/corridorkey-mlx")
from model import GreenFormer
from inference import infer_frame   # existing function from our repo

# load model once
model = GreenFormer(...)
# ... load weights ...

CMD_FIFO  = "/tmp/corridorkey_cmd"
DONE_FIFO = "/tmp/corridorkey_done"
IN_PLATE  = "/tmp/corridorkey_in_plate.exr"
IN_MATTE  = "/tmp/corridorkey_in_matte.exr"
OUT_FG    = "/tmp/corridorkey_out_fg.exr"
OUT_ALPHA = "/tmp/corridorkey_out_alpha.exr"

while True:
    params = json.loads(open(CMD_FIFO, "r").read())
    if params.get("quit"):
        break
    infer_frame(
        model, IN_PLATE, IN_MATTE,
        out_fg=OUT_FG, out_alpha=OUT_ALPHA,
        despill_strength=params["despill_strength"],
        refiner_scale=params["refiner_scale"],
        input_is_srgb=params["input_is_srgb"],
        despeckle=params["despeckle"],
    )
    open(DONE_FIFO, "w").close()   # signal handler
```

---

## Reference Material

| Resource | Location |
|----------|----------|
| `pybox_v1.py` (full source, authoritative) | `/opt/Autodesk/presets/2026.2.1/shared/pybox/pybox_v1.py` |
| Bundled examples | `/opt/Autodesk/presets/2026.2.1/pybox/*.py` |
| Best FIFO daemon example | `/opt/Autodesk/presets/2026.2.1/pybox/nuke_px.py` |
| External examples | https://github.com/lcrs/pyboxes |
| Autodesk docs (2026 URL 404s) | https://help.autodesk.com/view/FLAME/2023/ENU/?guid=Flame_API_Pybox_Documentation_html |

---

## Decision Log

| Option | Verdict | Reason |
|--------|---------|--------|
| PyBox  | ✅ Preferred | Native batch node; Python glue; daemon pattern proven; no compile cycle |
| OpenFX | Later / optional | C++ required; cross-platform distribution benefit; CUDA stack available on Rocky Linux if needed |
