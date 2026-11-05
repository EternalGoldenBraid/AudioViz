"""Serves a running Qt visualizer's parameters to the TUI deck over a local socket.

JSON lines. Requests: {"op": "controls"} | {"op": "nudge", "name": n, "steps": k}
                      | {"op": "set", "name": n, "value": v}
Every reply is {"controls": [...]} describing the current state of each control, or {"error": ...}.
Requests run on the Qt event loop thread, the thread that steps the engine.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

from PyQt5.QtNetwork import QLocalServer

from audioviz.tui.mixer import CONTROLS, Control

SOCKET = Path(os.environ.get("AUDIOVIZ_DECK_SOCKET") or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "audioviz-deck.sock")
LIVE_CONTROLS = ("evidence gain", "camera gain", "damping", "speed", "inference rate", "canvas rate", "learning", "learning rate")


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


def describe_control(control: Control, target) -> dict:
    return {
        "name": control.name,
        "position": control.position(target),
        "text": control.format(target),
        "choices": list(control.choices) if control.choices else None,
        "toggle": control.toggle,
    }


class DeckServer:
    def __init__(self, window, path: Path = SOCKET) -> None:
        self.target = WindowTarget(window)
        self.controls = {c.name: c for c in CONTROLS if c.name in LIVE_CONTROLS}
        self.controls["speed"] = replace(self.controls["speed"], low=10.0, high=3000.0, log=True)  # the window runs in m/s
        self.server = QLocalServer()
        QLocalServer.removeServer(str(path))
        if not self.server.listen(str(path)):
            raise OSError(f"deck socket {path}: {self.server.errorString()}")
        self.server.newConnection.connect(self._accept)

    def _accept(self) -> None:
        connection = self.server.nextPendingConnection()
        connection.readyRead.connect(lambda: self._serve(connection))

    def _serve(self, connection) -> None:
        while connection.canReadLine():
            try:
                reply = self.handle(json.loads(bytes(connection.readLine()).decode()))
            except Exception as error:  # a bad request must not drop the connection
                reply = {"error": repr(error)}
            connection.write(json.dumps(reply).encode() + b"\n")

    def handle(self, request: dict) -> dict:
        op = request["op"]
        if op in ("nudge", "set"):
            control = self.controls[request["name"]]
            if op == "nudge":
                control.nudge(self.target, int(request["steps"]))
            else:
                control.set(self.target, request["value"])
        elif op != "controls":
            raise ValueError(f"unknown op {op!r}")
        return {"controls": [describe_control(c, self.target) for c in self.controls.values()]}
