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
        self.path = path
        self._socket = None
        self._reader = None
        self.state: dict[str, dict] = {}
        self.sensors: dict[str, dict] = {}  # sensor id -> {id, kind, enabled}
        self.max_px = 160  # longest side of the images the window sends
        self.previews: dict[str, np.ndarray] = {}
        self.wanted: dict[str, tuple[int, int]] = {}
        self._connect(STARTUP_TIMEOUT_S)
        try:
            self.refresh()
        except (OSError, RuntimeError, ValueError):
            self.close()
            raise
        self._socket.settimeout(TIMEOUT_S)

    def _connect(self, timeout: float = TIMEOUT_S) -> None:
        connection = socket.socket(socket.AF_UNIX)
        connection.settimeout(timeout)
        try:
            connection.connect(str(self.path))
        except OSError:
            connection.close()
            raise
        self._socket = connection
        self._reader = connection.makefile("rb")

    def _disconnect(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _call(self, request: dict) -> dict:
        # Serialize both the exchange and publication so old polls cannot replace newer edits.
        with self._lock:
            if self._socket is None:
                self._connect()
            try:
                self._socket.sendall(json.dumps(request).encode() + b"\n")
                line = self._reader.readline()
                if not line:
                    raise ConnectionError("The window closed its deck connection.")
                reply = json.loads(line)
            except (OSError, ValueError):
                # A timed-out buffered reader cannot be reused. Reconnect on the next request;
                # never replay an edit whose outcome is unknown.
                self._disconnect()
                raise
            if "error" in reply:
                raise RuntimeError(reply["error"])
            if "controls" in reply:
                self.state = {c["name"]: c for c in reply["controls"]}
                self.sensors = {s["id"]: s for s in reply.get("sensors", [])}
            return reply

    def refresh(self) -> None:
        self._call({"op": "controls"})

    def close(self) -> None:
        with self._lock:
            self._disconnect()

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

    def save(self, path: Path | None = None, state: dict | None = None) -> str:
        """Writes current runtime values and optional TUI state to one config."""
        request = {"op": "save"}
        if path is not None:
            request["path"] = str(path)
        if state is not None:
            request["state"] = state
        return self._call(request)["saved"]

    def load(self, path: Path) -> None:
        self._call({"op": "load", "path": str(path)})

    def reset(self) -> None:
        self._call({"op": "reset"})

    def rescan(self) -> None:
        """Asks the window to re-list its devices; audio streams are briefly reopened."""
        self._call({"op": "rescan"})

    def select(self, name: str, values: list[str]) -> None:
        self._call({"op": "select", "name": name, "values": values})

    def nudge(self, name: str, steps: int) -> None:
        self._call({"op": "nudge", "name": name, "steps": steps})

    def set(self, name: str, value) -> None:
        self._call({"op": "set", "name": name, "value": value})

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

    @property
    def category(self) -> str | None:
        return self.client.state[self.name].get("category")

    @property
    def description(self) -> str:
        return self.client.state[self.name].get("description", "")

    @property
    def label(self) -> str | None:
        return self.client.state[self.name].get("label")

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
        state = self.client.state[self.name]
        return state.get("value", state["text"])

    def set(self, context, value) -> None:
        pass  # restoring a saved state never writes to the live window
