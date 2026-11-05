"""Serves a running Qt visualizer's parameters to the TUI deck over a local socket.

JSON lines. Requests: {"op": "controls"} | {"op": "nudge", "name": n, "steps": k}
                      | {"op": "set", "name": n, "value": v} | {"op": "select", "name": n, "values": [v, ...]}
                      | {"op": "save"} | {"op": "views", "views": [[name, rows, cols], ...]}
Every reply is {"controls": [...]} describing the current state of each control, or {"error": ...}.
A views reply is {"views": {name: {"shape": [h, w, 3], "data": base64 uint8} | null}}: the images the TUI's
view functions draw from the window's own graph, null while a node has no data.
Requests run on the Qt event loop thread, the thread that steps the engine.
"""

import base64
import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
from PyQt5.QtNetwork import QLocalServer

from audioviz.tui import views
from audioviz.tui.deck_devices import device_controls
from audioviz.tui.mixer import CONTROLS, Control

MAX_PREVIEW_PX = 160  # longest side of a served view; tiles resample to their cells anyway
SOCKET = Path(os.environ.get("AUDIOVIZ_DECK_SOCKET") or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "audioviz-deck.sock")
DECK_CONFIG = Path(os.environ.get("AUDIOVIZ_DECK_CONFIG") or Path(__file__).resolve().parents[3] / "config" / "deck.json")
LIVE_CONTROLS = ("evidence gain", "camera gain", "damping", "speed", "inference rate", "canvas rate", "learning", "learning rate")


def window_control(name: str, widget) -> Control:
    """Shows or hides one of the app's Qt windows; hidden windows keep running."""
    def show(_target, value) -> None:
        widget.show() if value else widget.hide()
    return Control(f"{name} window", 0.0, 1.0, lambda _target: widget.isVisible(), show, toggle=True)


def query_audio_devices() -> list[dict]:
    try:
        import sounddevice as sd

        return list(sd.query_devices())
    except Exception:  # no audio backend: the audio knobs are simply absent
        return []


class WindowTarget:
    """Presents a RippleWaveVisualizer with the attributes the mixer controls read and write."""

    def __init__(self, window) -> None:
        self.window = window

    @property
    def engine(self):
        return self.window.engine

    @property
    def orchestrator(self):
        return self.window.source_orchestrator

    # the rest of what the view functions read from a GraphRuntime
    @property
    def processor(self):
        return self.window.processor

    @property
    def visual_prediction(self):
        return self.window.visual_prediction

    @property
    def audio_readout(self):
        return self.window.source_orchestrator.audio_readout

    show_camera = True

    @property
    def camera(self):
        return _SensorFrames(self.orchestrator.sensors["camera 1"])


class _SensorFrames:
    """The runtime's camera interface, backed by the window's camera 1 observation."""

    def __init__(self, sensor) -> None:
        self.sensor = sensor

    def rgb(self) -> np.ndarray | None:
        observation = self.sensor.observation
        return None if observation is None else (np.clip(observation, 0.0, 1.0) * 255).astype(np.uint8)


def render_view(target, name: str, rows: int, cols: int) -> dict | None:
    image = views.VIEWS[name](target, rows, cols)
    if image is None:
        return None
    stride = max(1, -(-max(image.shape[:2]) // MAX_PREVIEW_PX))
    image = np.ascontiguousarray(image[::stride, ::stride], dtype=np.uint8)
    return {"shape": list(image.shape), "data": base64.b64encode(image.tobytes()).decode()}


def describe_control(control, target) -> dict:
    description = {
        "name": control.name,
        "position": control.position(target),
        "text": control.format(target),
        "choices": list(control.choices) if control.choices else None,
        "toggle": control.toggle,
    }
    if hasattr(control, "selected"):  # a device list: the checked entries and how many may be checked
        description.update(selected=control.selected(target), capacity=control.capacity)
    return description


class DeckServer:
    def __init__(self, window, path: Path = SOCKET, config_path: Path = DECK_CONFIG, windows: dict | None = None) -> None:
        self.target = WindowTarget(window)
        self.config_path = config_path
        self.controls = {c.name: c for c in CONTROLS if c.name in LIVE_CONTROLS}
        self.controls["speed"] = replace(self.controls["speed"], low=10.0, high=3000.0, log=True)  # the window runs in m/s
        if getattr(window, "processor", None) is not None:
            self.controls.update(device_controls(window, query_audio_devices()))
        for name, widget in (windows or {}).items():
            self.controls[f"{name} window"] = window_control(name, widget)
        self.server = QLocalServer()
        QLocalServer.removeServer(str(path))
        if not self.server.listen(str(path)):
            raise OSError(f"deck socket {path}: {self.server.errorString()}")
        self.server.newConnection.connect(self._accept)
        self.connections = []  # Python must keep the accepted sockets' wrappers alive while they serve
        self.load()

    def _accept(self) -> None:
        connection = self.server.nextPendingConnection()
        self.connections.append(connection)
        connection.readyRead.connect(partial(self._serve, connection))
        connection.disconnected.connect(partial(self.connections.remove, connection))

    def _serve(self, connection) -> None:
        while connection.canReadLine():
            try:
                reply = self.handle(json.loads(bytes(connection.readLine()).decode()))
            except Exception as error:  # a bad request must not drop the connection
                reply = {"error": repr(error)}
            connection.write(json.dumps(reply).encode() + b"\n")

    def handle(self, request: dict) -> dict:
        op = request["op"]
        if op == "views":
            return {"views": {name: self.view(name, rows, cols) for name, rows, cols in request["views"]}}
        saved = None
        if op in ("nudge", "set", "select"):
            control = self.controls[request["name"]]
            if op == "nudge":
                control.nudge(self.target, int(request["steps"]))
            elif op == "select":
                control.select(self.target, list(request["values"]))
            else:
                control.set(self.target, request["value"])
        elif op == "save":
            saved = self.save()
        elif op != "controls":
            raise ValueError(f"unknown op {op!r}")
        reply = {
            "controls": [describe_control(c, self.target) for c in self.controls.values()],
            "sensors": [{"id": s.sensor_id, "kind": s.kind, "enabled": s.enabled} for s in self.sensors()],
        }
        if saved:
            reply["saved"] = saved
        return reply

    def sensors(self) -> list:
        orchestrator = self.target.orchestrator
        return list(getattr(orchestrator, "sensors", {}).values())

    def view(self, name: str, rows: int, cols: int) -> dict | None:
        try:
            return render_view(self.target, name, int(rows), int(cols))
        except Exception:  # a node the window does not have yet, or a view that needs the local graph
            return None

    def save(self) -> str:
        values = {name: control.get(self.target) for name, control in self.controls.items()}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(values, indent=2) + "\n")
        return str(self.config_path)

    def load(self) -> None:
        """Applies the saved values; entries the window no longer offers, or whose device fails to open, are skipped."""
        if not self.config_path.exists():
            return
        for name, value in json.loads(self.config_path.read_text()).items():
            control = self.controls.get(name)
            if control is None:
                continue
            if control.choices and value not in control.choices:
                value = next((c for c in control.choices if c.split(": ", 1)[-1] == str(value).split(": ", 1)[-1]), None)
            try:
                if value is not None:
                    control.set(self.target, value)
            except Exception as error:  # an unplugged device must not stop the window starting
                print(f"deck config: {name}={value!r} skipped: {error}")
