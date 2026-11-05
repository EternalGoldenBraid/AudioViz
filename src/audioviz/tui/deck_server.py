"""Serves a running Qt visualizer's parameters to the TUI deck over a local socket.

JSON lines. Requests: {"op": "controls"} | {"op": "nudge", "name": n, "steps": k}
                      | {"op": "set", "name": n, "value": v} | {"op": "select", "name": n, "values": [v, ...]}
                      | {"op": "save"}
Every reply is {"controls": [...]} describing the current state of each control, or {"error": ...}.
Requests run on the Qt event loop thread, the thread that steps the engine.
"""

import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

from PyQt5.QtNetwork import QLocalServer

from audioviz.tui.deck_devices import device_controls
from audioviz.tui.mixer import CONTROLS, Control

SOCKET = Path(os.environ.get("AUDIOVIZ_DECK_SOCKET") or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "audioviz-deck.sock")
DECK_CONFIG = Path(os.environ.get("AUDIOVIZ_DECK_CONFIG") or Path(__file__).resolve().parents[3] / "config" / "deck.json")
LIVE_CONTROLS = ("evidence gain", "camera gain", "damping", "speed", "inference rate", "canvas rate", "learning", "learning rate")


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
    def __init__(self, window, path: Path = SOCKET, config_path: Path = DECK_CONFIG) -> None:
        self.target = WindowTarget(window)
        self.config_path = config_path
        self.controls = {c.name: c for c in CONTROLS if c.name in LIVE_CONTROLS}
        self.controls["speed"] = replace(self.controls["speed"], low=10.0, high=3000.0, log=True)  # the window runs in m/s
        if getattr(window, "processor", None) is not None:
            self.controls.update(device_controls(window, query_audio_devices()))
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
        reply = {"controls": [describe_control(c, self.target) for c in self.controls.values()]}
        if saved:
            reply["saved"] = saved
        return reply

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
