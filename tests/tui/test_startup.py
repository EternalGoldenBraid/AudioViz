import asyncio
import json
import sys

import pytest
from textual.widgets import Static

from audioviz.tui import app as app_module, control
from audioviz.tui.device_checklist import DeviceChecklist
from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.mixer import Control
from audioviz.tui.startup import StartupPanel, WindowSession
from audioviz.utils.audio_devices import make_device_config


DEVICES = [
    {"name": "Test microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000},
    {"name": "Test headphones", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
]


class FakeDeck:
    max_px = 160
    sensors = {}
    closed = False

    def controls(self):
        return [Control("gain", 0, 1, lambda _: 0.5, lambda *_: None)]

    def refresh(self):
        pass

    def fetch_previews(self):
        pass

    def preview(self, *args):
        return None

    def close(self):
        self.closed = True


@pytest.fixture
def startup(tmp_path, monkeypatch):
    from audioviz.tui import startup as startup_module

    monkeypatch.setattr(control, "SOCKET", tmp_path / "control.sock")
    monkeypatch.setattr(startup_module.sd, "query_devices", lambda: DEVICES)
    monkeypatch.setattr(startup_module.sd, "check_input_settings", lambda **_: None)
    monkeypatch.setattr(startup_module.sd, "check_output_settings", lambda **_: None)
    monkeypatch.setattr(app_module, "GraphRuntime", lambda **kw: GraphRuntime(resolution=16, **kw))
    return WindowSession(
        ["unused"], tmp_path / "config/audio.json",
        socket_path=tmp_path / "deck.sock", log_path=tmp_path / "main.log",
    )


async def wait_for(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("saved", ["missing", "valid", "unavailable", "malformed"])
def test_tui_is_first_and_reuses_or_repairs_saved_devices(startup, tmp_path, monkeypatch, saved):
    config = make_device_config(DEVICES, 0, 1)
    if saved != "missing":
        startup.config_file.parent.mkdir()
        startup.config_file.write_text("{" if saved == "malformed" else json.dumps(
            config if saved == "valid" else {**config, "output_device_name": "Unplugged headphones"}
        ))
    connected = []
    deck = FakeDeck()
    app = app_module.AudioVizApp(startup=startup, state_dir=tmp_path, autosave=False, hot_reload=False)
    assert not hasattr(app, "runtime")

    async def connect():
        assert app.is_running
        assert app.query_one(StartupPanel).is_mounted
        assert not hasattr(app, "runtime")
        assert json.loads(startup.config_file.read_text()) == config
        connected.append(True)
        startup.deck = deck
        return deck

    monkeypatch.setattr(startup, "connect", connect)

    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            if saved != "valid":
                await wait_for(lambda: isinstance(app.screen, DeviceChecklist))
                assert not connected
                assert not hasattr(app, "runtime")
                await pilot.press("space", "enter")
                await wait_for(lambda: isinstance(app.screen, DeviceChecklist) and "output" in app.screen.title_text)
                await pilot.press("space", "enter")
            await wait_for(lambda: app.deck_ready and not app._startup_busy)
            assert connected == [True]
            assert app.runtime._thread is None
            assert app.runtime.processor.frame_counter == 0
            assert app.query_one("#canvas").is_mounted
        assert deck.closed

    asyncio.run(run())


def test_cancel_setup_keeps_tui_idle_and_configuration_unwritten(startup, tmp_path, monkeypatch):
    async def unexpected_connect():
        pytest.fail("Cancelled setup must not start a window")

    monkeypatch.setattr(startup, "connect", unexpected_connect)
    app = app_module.AudioVizApp(startup=startup, state_dir=tmp_path, autosave=False, hot_reload=False)

    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            await wait_for(lambda: isinstance(app.screen, DeviceChecklist))
            await pilot.press("escape")
            await wait_for(lambda: not app._startup_busy)
            assert not app.deck_ready
            assert not hasattr(app, "runtime")
            assert not startup.config_file.exists()
            assert "paused" in str(app.query_one("#startup-status", Static).render())
            await pilot.press("a")
            await wait_for(lambda: isinstance(app.screen, DeviceChecklist))
            await pilot.press("escape", "q")

    asyncio.run(run())


def test_failed_child_leaves_error_visible_and_retry_attaches(startup, tmp_path, monkeypatch):
    startup.save_config(make_device_config(DEVICES, 0, 1))
    attempts = []

    async def connect():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("Output device is busy")
        startup.deck = FakeDeck()
        return startup.deck

    monkeypatch.setattr(startup, "connect", connect)
    app = app_module.AudioVizApp(startup=startup, state_dir=tmp_path, autosave=False, hot_reload=False)

    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            await wait_for(lambda: attempts and not app._startup_busy)
            assert not app.deck_ready
            assert "Output device is busy" in str(app.query_one("#startup-status", Static).render())
            await pilot.press("r")
            await wait_for(lambda: app.deck_ready and not app._startup_busy)
            assert len(attempts) == 2
        app.refresh_tiles()
        app.refresh_deck()

    asyncio.run(run())


@pytest.mark.parametrize("command,timeout,expected", [
    (["-c", "import sys; print('device failed', flush=True); sys.exit(3)"], 5, "device failed"),
    (["-c", "import time; time.sleep(60)"], 0.15, "startup failed"),
])
def test_failed_or_timed_out_child_is_reaped_with_visible_error(tmp_path, command, timeout, expected):
    session = WindowSession(
        [sys.executable, *command], None, socket_path=tmp_path / "deck.sock",
        log_path=tmp_path / "main.log", timeout=timeout,
    )

    async def run():
        with pytest.raises(RuntimeError, match=expected):
            await session.connect()
        assert session.process is None
        assert session._log is None
        assert not session.socket_path.exists()

    asyncio.run(run())


def test_incompatible_saved_rate_is_rejected_before_opening_streams(startup, monkeypatch):
    from audioviz.tui import startup as startup_module

    def reject(**kwargs):
        raise startup_module.sd.PortAudioError("Invalid sample rate")

    monkeypatch.setattr(startup_module.sd, "check_output_settings", reject)
    with pytest.raises(ValueError, match="Test headphones cannot use 48000 Hz"):
        startup.validate_config(make_device_config(DEVICES, 0, 1))


def test_real_child_attaches_without_stdin_and_is_stopped_on_tui_quit(startup, tmp_path):
    startup.save_config(make_device_config(DEVICES, 0, 1))
    backend = """
import json, socket, sys
from pathlib import Path
assert sys.stdin.read() == ""
assert json.loads(Path(sys.argv[2]).read_text())["input_device_index"] == 0
with socket.socket(socket.AF_UNIX) as server:
    server.bind(sys.argv[1])
    server.listen()
    connection, _ = server.accept()
    with connection, connection.makefile("rb") as reader:
        for line in reader:
            request = json.loads(line)
            if request["op"] == "views":
                response = {"views": {v[0]: None for v in request["views"]}}
            else:
                response = {"controls": [{
                    "name": "gain", "text": "0.5", "position": 0.5,
                    "toggle": False, "choices": None,
                }], "sensors": []}
            connection.sendall(json.dumps(response).encode() + b"\\n")
"""
    startup.command = [sys.executable, "-c", backend, str(startup.socket_path), str(startup.config_file)]
    app = app_module.AudioVizApp(startup=startup, state_dir=tmp_path, autosave=False, hot_reload=False)

    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            await wait_for(lambda: app.deck_ready and not app._startup_busy)
            process = startup.process
            assert process.poll() is None
            assert app.attached
            assert app.runtime._thread is None
            await pilot.press("colon", "q", "enter")
        assert process.poll() is not None
        assert not startup.socket_path.exists()

    asyncio.run(run())


def test_tui_can_quit_while_child_is_still_starting(startup, tmp_path):
    startup.save_config(make_device_config(DEVICES, 0, 1))
    startup.command = [sys.executable, "-c", "import time; time.sleep(60)"]
    app = app_module.AudioVizApp(startup=startup, state_dir=tmp_path, autosave=False, hot_reload=False)

    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            await wait_for(lambda: startup.process is not None)
            process = startup.process
            assert not app.deck_ready
            await pilot.press("q")
        assert process.poll() is not None
        assert startup.process is None

    asyncio.run(run())


def test_file_playback_startup_does_not_offer_or_query_audio_setup(startup, tmp_path, monkeypatch):
    from textual.widgets import OptionList
    from audioviz.tui import startup as startup_module

    startup.config_file = None

    def unexpected_devices():
        pytest.fail("File playback must not run live-audio device setup")

    async def connect():
        raise RuntimeError("Missing audio file")

    monkeypatch.setattr(startup_module.sd, "query_devices", unexpected_devices)
    monkeypatch.setattr(startup, "connect", connect)
    app = app_module.AudioVizApp(startup=startup, state_dir=tmp_path, autosave=False, hot_reload=False)

    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            await wait_for(lambda: "Missing audio file" in str(app.query_one("#startup-status", Static).render()))
            await pilot.pause()
            assert app.query_one(OptionList).option_count == 2
            assert "Missing audio file" in str(app.query_one("#startup-status", Static).render())

    asyncio.run(run())


def test_long_device_list_keeps_hints_and_final_choice_visible(startup, tmp_path, monkeypatch):
    from textual.widgets import OptionList
    from audioviz.tui import startup as startup_module

    devices = [
        {"name": f"Microphone {i:02}", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000}
        for i in range(30)
    ]
    monkeypatch.setattr(startup_module.sd, "query_devices", lambda: devices)
    app = app_module.AudioVizApp(startup=startup, state_dir=tmp_path, autosave=False, hot_reload=False)

    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            await wait_for(lambda: isinstance(app.screen, DeviceChecklist))
            await pilot.press("end")
            await pilot.pause()
            screen = app.screen
            assert screen.query_one(OptionList).highlighted == 29
            assert screen.query_one("#capacity").region.bottom < screen.children[0].region.bottom
            assert "Microphone&#160;29" in app.export_screenshot() or "Microphone 29" in app.export_screenshot()
            await pilot.press("escape", "q")

    asyncio.run(run())
