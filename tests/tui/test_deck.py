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


class FakeProcessor:
    input_channels, output_channels = 1, 1

    def __init__(self):
        self.opened = []

    def set_input_device(self, index):
        if index == 2:
            raise RuntimeError("busy")
        self.opened.append(index)

    def set_output_device(self, index):
        pass


DEVICES = [
    {"name": "hdmi", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "usb mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "busy mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "pro mic", "max_input_channels": 2, "max_output_channels": 0},
]


def test_device_knob_steps_past_a_device_that_fails_and_saved_names_survive_reindexing(tmp_path, monkeypatch):
    from audioviz.tui import deck_devices, deck_server

    monkeypatch.setattr(deck_server, "query_audio_devices", lambda: DEVICES)
    monkeypatch.setattr(deck_devices, "camera_choices", lambda: ())
    processor = FakeProcessor()
    camera = SimpleNamespace(camera_index=0)
    window = SimpleNamespace(
        engine=FakeEngine(), source_orchestrator=SimpleNamespace(camera_source=camera), processor=processor)
    config = tmp_path / "deck.json"
    server = DeckServer(window, tmp_path / "a.sock", config)
    knob = server.controls["audio in"]
    assert knob.choices == ("1: usb mic", "2: busy mic", "3: pro mic")
    assert knob.get(server.target) == "off"  # no stream open yet, so nothing for a save to reopen

    knob.set(server.target, "1: usb mic")
    for _ in range(2):  # the failing step still moves the selection, so the next step is reachable
        try:
            knob.nudge(server.target, 1)
        except RuntimeError:
            pass
    assert knob.get(server.target) == "3: pro mic"
    assert processor.opened == [1, 3]

    # The saved name is found again after the devices were renumbered.
    config.write_text('{"audio in": "7: pro mic"}')
    server.load()
    assert processor.opened == [1, 3, 3]
