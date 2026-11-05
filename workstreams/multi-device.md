# Multi-device input

> **Status:** active · **Branch:** `agent/multi-device` (from `agent/tui`) · **Updated:** 2026-11-05
> **Now:** two-camera backend done; scene tile not yet seen live
> **Next:** look at the "Camera 2" tile in the Multimodal 3D Scene (needs a click on "Show Multimodal 3D Scene"); decide whether IR gets its own readout
> **Blocked:** none
> **Resume:** `copilot --resume 09d6b83c-9ae7-464e-ab82-669fba254e45`
> **Run:** `python main.py`, then select both cameras on the deck's `camera` knob

**Essence.** The live device switch selects one camera. This stream lets several
devices run at once, each feeding its own tile. First target: the laptop's IR and
EO cameras side by side.

**Present.** `CameraDevices` (capacity 2): first selection feeds the canvas, second is
`orchestrator.secondary_camera_source`, observation-only, shown as scene source
`camera_secondary` ("Camera 2"). Tested: unit tests (`tests/tui/test_camera_devices.py`); live, with
`main.py --no-tui` the deck accepted both cameras and /dev/video0 and /dev/video2 were open together;
`CameraFrameSource` on both returned live EO and IR frames (IR needs ~1 s of warm-up).
Node map: video0 = EO (MJPG/YUYV), video2 = IR (GREY); video1/3 are metadata.


**Open.**
- Which `/dev/video*` nodes are IR vs EO (one USB device exposes video0-3; metadata nodes are not capture nodes).
- `sources/pose/frame_source.py` opens its own `VideoCapture`; the device is exclusive, so it must share the source.
- Per-tile source binding does not exist; tiles are view-level.

**Future.** Multiple audio inputs/outputs after cameras.

**Notes.** Single-device limits (from the TUI agent, efa7117b):
- `sources/camera.py`: `CameraFrameSource` holds one `camera_index`.
- `sources/orchestrator.py` ~111 builds one camera from `config.camera_frame.camera_index`.
- `tui/deck_devices.py` ~101: camera `DeviceControl` has `capacity=1`, `apply=camera.set_camera_index`.
- Plan: source registry keyed by device; per-tile source name; `DeviceControl.capacity` raised and `apply` takes a list.
- Coordination: avoid `tui/app.py`, `tui/keys.py` (TUI agent active), `tui/control.py`, `checks*`, `live-test-agent.md`. Leave the user's `main.py` window-size edit alone. Rebase on `agent/tui`, not `main`.
