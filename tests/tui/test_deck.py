import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5.QtNetwork")
from PyQt5.QtCore import QCoreApplication, QTimer

from audioviz.tui.deck_client import DeckClient
from audioviz.tui.deck_server import DeckServer


class FakeEngine:
    def __init__(self):
        self.damping, self.speed = 0.98, 1.0

    def set_damping(self, value):
        self.damping = value

    def set_speed(self, value):
        self.speed = value


def test_client_knobs_turn_the_window_engine_over_the_socket(tmp_path):
    app = QCoreApplication.instance() or QCoreApplication([])
    window = SimpleNamespace(engine=FakeEngine(), source_orchestrator=None)
    path = tmp_path / "deck.sock"
    server = DeckServer(window, path)
    server.controls = {n: c for n, c in server.controls.items() if n in ("damping", "speed")}

    seen = {}

    def client_side():  # blocking client; the Qt loop on the main thread serves it
        client = DeckClient(path)
        speed = next(c for c in client.controls() if c.name == "speed")
        before = speed.format(client)
        speed.nudge(client, 3)
        seen.update(before=before, after=speed.format(client), names=sorted(client.state))
        QTimer.singleShot(0, app.quit)

    thread = threading.Thread(target=client_side)
    QTimer.singleShot(0, thread.start)
    QTimer.singleShot(5000, app.quit)
    app.exec_()
    thread.join()

    assert seen["names"] == ["damping", "speed"]
    assert window.engine.speed > 1.0
    assert seen["after"] != seen["before"]


def test_saved_knobs_are_applied_to_the_next_window(tmp_path):
    config = tmp_path / "deck.json"
    first = SimpleNamespace(engine=FakeEngine(), source_orchestrator=None)
    server = DeckServer(first, tmp_path / "a.sock", config)
    server.controls = {"damping": server.controls["damping"]}
    server.controls["damping"].set(server.target, 0.95)

    reply = server.handle({"op": "save"})

    assert reply["saved"] == str(config)
    second = SimpleNamespace(engine=FakeEngine(), source_orchestrator=None)
    DeckServer(second, tmp_path / "b.sock", config)  # applies the saved file on startup
    assert second.engine.damping == 0.95
