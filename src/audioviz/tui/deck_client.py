"""Deck side of the deck socket: knobs that read and write a running visualizer window.

DeckClient keeps the last reply of the server; RemoteControl is a mixer Control look-alike over it,
so knob tiles work the same against a local runtime or a remote window.
"""

import base64
import json
import socket
import threading
from pathlib import Path

import numpy as np

from audioviz.tui.deck_server import SOCKET

TIMEOUT_S = 3.0  # the window answers between engine steps, which take a few hundred ms on large grids
STARTUP_TIMEOUT_S = 60.0  # the first answer waits for the window to finish building its graph


class DeckClient:
    def __init__(self, path: Path = SOCKET) -> None:
        self._lock = threading.Lock()
        self._socket = socket.socket(socket.AF_UNIX)
        self._socket.settimeout(STARTUP_TIMEOUT_S)  # the window listens before its event loop runs
        self._socket.connect(str(path))
        self._reader = self._socket.makefile("rb")
        self.state: dict[str, dict] = {}
        self.sensors: dict[str, dict] = {}  # sensor id -> {id, kind, enabled}
        self.max_px = 160  # longest side of the images the window sends
        self.previews: dict[str, np.ndarray] = {}
        self.wanted: dict[str, tuple[int, int]] = {}
        self.refresh()
        self._socket.settimeout(TIMEOUT_S)

    def _call(self, request: dict) -> dict:
        with self._lock:  # one request in flight; the poller thread and key handlers share the stream
            self._socket.sendall(json.dumps(request).encode() + b"\n")
            reply = json.loads(self._reader.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        if "controls" in reply:
            self.state = {c["name"]: c for c in reply["controls"]}
            self.sensors = {s["id"]: s for s in reply.get("sensors", [])}
        return reply

    def refresh(self) -> None:
        self._call({"op": "controls"})

    def preview(self, name: str, rows: int, cols: int) -> "np.ndarray | None":
        """The last image fetched for a view; asking for it also schedules it for the next fetch_previews."""
        self.wanted[name] = (rows, cols)
        return self.previews.get(name)

    def fetch_previews(self) -> None:
        """One round trip for every view asked for since the last fetch."""
        wanted, self.wanted = self.wanted, {}
        if not wanted:
            return
        reply = self._call({"op": "views", "max_px": self.max_px,
                            "views": [[name, rows, cols] for name, (rows, cols) in wanted.items()]})
        for name, image in reply["views"].items():
            if image is None:
                self.previews.pop(name, None)
            else:
                self.previews[name] = np.frombuffer(base64.b64decode(image["data"]), dtype=np.uint8).reshape(image["shape"])

    def save(self) -> str:
        """Writes the window's current knob values to its deck config file; returns the path."""
        return self._call({"op": "save"})["saved"]

    def select(self, name: str, values: list[str]) -> None:
        self._call({"op": "select", "name": name, "values": values})

    def nudge(self, name: str, steps: int) -> None:
        self._call({"op": "nudge", "name": name, "steps": steps})

    def controls(self) -> list["RemoteControl"]:
        return [RemoteControl(name, self) for name in self.state]


class RemoteControl:
    """Same surface the knob tiles use on a Control; the context argument is the client."""

    def __init__(self, name: str, client: DeckClient) -> None:
        self.name = name
        self.client = client

    @property
    def toggle(self) -> bool:
        return self.client.state[self.name]["toggle"]

    def position(self, context=None) -> float:
        return self.client.state[self.name]["position"]

    @property
    def is_picker(self) -> bool:
        """A device list: Enter opens a checklist instead of entering the knob."""
        return "selected" in self.client.state[self.name]

    @property
    def choices(self) -> list[str]:
        return self.client.state[self.name]["choices"] or []

    @property
    def selected(self) -> list[str]:
        return self.client.state[self.name]["selected"]

    @property
    def capacity(self) -> int:
        return self.client.state[self.name]["capacity"]

    def select(self, values: list[str]) -> None:
        self.client.select(self.name, values)

    def format(self, context=None) -> str:
        return self.client.state[self.name]["text"]

    def nudge(self, context, steps: int) -> None:
        self.client.nudge(self.name, steps)

    def get(self, context=None):
        return self.client.state[self.name]["text"]

    def set(self, context, value) -> None:
        pass  # restoring a saved state never writes to the live window
