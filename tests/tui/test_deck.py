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
        self.input_stream = SimpleNamespace(device=index)
    def set_output_device(self, index):
        pass


DEVICES = [
    {"name": "hdmi", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "usb mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "busy mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "pro mic", "max_input_channels": 2, "max_output_channels": 0},
]


def test_choosing_a_device_replaces_the_previous_one_and_a_failed_open_changes_nothing(tmp_path, monkeypatch):
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
    server.controls = {"audio in": knob}  # the fake window has no orchestrator for the mixer knobs
    assert knob.choices == ("1: usb mic", "2: busy mic", "3: pro mic")
    assert knob.selected() == []  # no stream open yet, so nothing for a save to reopen

    # Two devices checked at once: capacity is 1, so only the later one opens.
    server.handle({"op": "select", "name": "audio in", "values": ["1: usb mic", "3: pro mic"]})
    assert (processor.opened, knob.selected()) == ([3], ["3: pro mic"])

    with pytest.raises(RuntimeError):  # the fake refuses device 2
        server.handle({"op": "select", "name": "audio in", "values": ["2: busy mic"]})
    assert knob.selected() == ["3: pro mic"]

    # The saved name is found again after the devices were renumbered.
    config.write_text('{"audio in": "7: pro mic"}')
    server.load()
    assert processor.opened == [3, 3]
    described = {c["name"]: c for c in server.handle({"op": "controls"})["controls"]}
    assert (described["audio in"]["selected"], described["audio in"]["capacity"]) == (["3: pro mic"], 1)


class StubDeck:
    """Stands in for DeckClient: one device list whose selection the test can read back."""

    def __init__(self):
        self.chosen = ["1: usb mic"]
        self.sent = []
        self.max_px = 160
        self.sensors = {"camera 1": {"id": "camera 1", "kind": "camera", "enabled": True},
                        "camera 2": {"id": "camera 2", "kind": "camera", "enabled": False}}

    @property
    def state(self):
        return {"audio in": {
            "name": "audio in", "position": 0.0, "text": ", ".join(self.chosen), "toggle": False,
            "choices": ["1: usb mic", "3: pro mic"], "selected": list(self.chosen), "capacity": 1}}

    def controls(self):
        from audioviz.tui.deck_client import RemoteControl

        return [RemoteControl("audio in", self)]

    def refresh(self):
        pass

    def preview(self, name, rows, cols):
        return None

    def fetch_previews(self):
        pass

    def select(self, name, values):
        self.sent.append(values)
        self.chosen = values


def test_enter_on_a_device_knob_opens_a_checklist_that_replaces_the_selection(tmp_path, monkeypatch):
    import asyncio

    from audioviz.tui import app as app_module
    from audioviz.tui.device_checklist import DeviceChecklist

    deck = StubDeck()
    monkeypatch.setattr(app_module, "DeckClient", lambda: deck)

    async def run():
        app = app_module.AudioVizApp(audio="synthetic", camera=False, state_dir=tmp_path, autosave=False, attach=True)
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#knob-audio-in").focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, DeviceChecklist)

            await pilot.press("j", "space")  # check the second device; capacity 1 unchecks the first
            assert app.screen.checked == ["3: pro mic"]
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert not isinstance(app.screen, DeviceChecklist)
            assert deck.sent == [["3: pro mic"]]

    asyncio.run(run())


def test_window_knobs_show_and_hide_windows_and_a_saved_hidden_window_starts_hidden(tmp_path):
    class Widget:
        visible = True

        def isVisible(self):
            return self.visible

        def show(self):
            self.visible = True

        def hide(self):
            self.visible = False

    config = tmp_path / "deck.json"
    first, spectrogram = SimpleNamespace(engine=FakeEngine(), source_orchestrator=None), Widget()
    server = DeckServer(first, tmp_path / "a.sock", config, windows={"spectrogram": spectrogram})
    server.controls = {n: c for n, c in server.controls.items() if n == "spectrogram window"}

    reply = server.handle({"op": "nudge", "name": "spectrogram window", "steps": -1})["controls"][0]
    assert (spectrogram.visible, reply["text"], reply["toggle"]) == (False, "off", True)  # hidden, not closed
    server.handle({"op": "save"})

    second, reopened = SimpleNamespace(engine=FakeEngine(), source_orchestrator=None), Widget()
    DeckServer(second, tmp_path / "b.sock", config, windows={"spectrogram": reopened})
    assert reopened.visible is False
    server.handle({"op": "nudge", "name": "spectrogram window", "steps": 1})
    assert spectrogram.visible is True


def test_the_grid_knob_chooses_the_next_launchs_grid_and_leaves_the_running_one_alone(tmp_path):
    from audioviz.tui.deck_server import saved_grid

    config = tmp_path / "deck.json"
    window = SimpleNamespace(engine=FakeEngine(), source_orchestrator=None, resolution=(100, 200))
    server = DeckServer(window, tmp_path / "a.sock", config)
    server.controls = {n: c for n, c in server.controls.items() if n == "grid at launch"}

    reply = server.handle({"op": "nudge", "name": "grid at launch", "steps": 1})["controls"]
    chosen = next(c for c in reply if c["name"] == "grid at launch")["text"]
    assert chosen != "100x200" and window.resolution == (100, 200)
    assert saved_grid(config) is None  # nothing is stored until the deck saves
    server.handle({"op": "save"})
    assert saved_grid(config) == tuple(int(n) for n in chosen.split("x"))


def test_the_window_serves_the_tui_views_downsampled_and_the_client_decodes_them():
    from audioviz.tui.graph_runtime import GraphRuntime

    runtime = GraphRuntime(camera=False)
    runtime.rebuild("300x400")  # larger than the preview cap, so the server strides
    server = DeckServer.__new__(DeckServer)
    server.target = runtime  # a GraphRuntime has every attribute the views read from a window
    client = DeckClient.__new__(DeckClient)
    client.previews, client.wanted, client.max_px = {}, {}, 160
    client._call = server.handle

    assert client.preview("canvas", 40, 80) is None  # the first ask only schedules the fetch
    client.preview("camera_2", 40, 80)  # no observation yet: null
    client.preview("no-such-view", 40, 80)
    client.fetch_previews()

    canvas = client.preview("canvas", 40, 80)
    assert canvas.dtype == "uint8" and canvas.shape == (100, 134, 3)  # 300x400 strided by 3
    assert client.previews.keys() == {"canvas"}
    assert client.wanted == {"canvas": (40, 80)}  # asking again schedules the next round only for what tiles show


def test_rescan_relists_audio_devices_plugged_in_after_start(tmp_path, monkeypatch):
    from audioviz.tui import deck_devices, deck_server

    listing = {"devices": DEVICES[:2]}
    monkeypatch.setattr(deck_server, "query_audio_devices", lambda: listing["devices"])
    monkeypatch.setattr(deck_devices, "camera_choices", lambda: ())
    processor = FakeProcessor()
    processor.rescanned = 0
    processor.rescan_devices = lambda: setattr(processor, "rescanned", processor.rescanned + 1)
    window = SimpleNamespace(
        engine=FakeEngine(), source_orchestrator=SimpleNamespace(camera_source=SimpleNamespace(camera_index=0)),
        processor=processor)
    server = DeckServer(window, tmp_path / "a.sock", tmp_path / "deck.json")
    server.controls = {"audio in": server.controls["audio in"]}  # the fake window has no orchestrator for the other knobs
    assert server.controls["audio in"].choices == ("1: usb mic",)
    listing["devices"] = DEVICES  # a microphone is plugged in
    server.handle({"op": "rescan"})
    assert processor.rescanned == 1
    assert server.controls["audio in"].choices == ("1: usb mic", "2: busy mic", "3: pro mic")
