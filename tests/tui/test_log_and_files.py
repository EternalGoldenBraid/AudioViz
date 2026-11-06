import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Input, OptionList, Static

from audioviz.tui import app as app_module, control
from audioviz.tui.app import AudioVizApp, LogTile
from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.deck_server import DeckServer
from audioviz.tui.deck_client import RemoteControl
from audioviz.tui.preset_picker import PresetPicker


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "SOCKET", tmp_path / "control.sock")
    monkeypatch.setattr(app_module, "GraphRuntime", lambda **kw: GraphRuntime(resolution=16, **kw))
    app = AudioVizApp(audio="synthetic", camera=False, state_dir=tmp_path, autosave=False, hot_reload=False)
    monkeypatch.setattr(app.runtime, "start", lambda: None)

    def screenshot(name, directory):
        path = directory / f"{name}.svg"
        path.write_text("<svg/>")
        return str(path)

    monkeypatch.setattr(app, "screenshot", screenshot)
    return app


def test_log_navigation_keeps_position_while_entries_arrive_and_g_resumes_following(app):
    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            log = app.query_one(LogTile)
            log.focus()
            await pilot.pause()
            app.toggle_zoom()
            for i in range(100):
                log.write(f"entry {i:03}")
            await pilot.pause()
            assert log.scroll_y == log.max_scroll_y
            assert app.screen.get_widget_at(5, 5)[0] is log
            await pilot.press("enter", "k")
            await pilot.pause()
            assert app.mode == "LOG"
            held = log.scroll_y
            assert held < log.max_scroll_y
            for i in range(100, 120):
                log.write(f"entry {i:03}")
            await pilot.pause()
            assert log.scroll_y == held
            assert not log.auto_scroll
            await pilot.press("g", "g")
            await pilot.pause()
            assert log.scroll_y == 0
            await pilot.press("ctrl+f")
            await pilot.pause()
            assert log.scroll_y > 0
            await pilot.press("G")
            await pilot.pause()
            assert log.auto_scroll
            assert log.scroll_y == log.max_scroll_y
            log.write("latest entry")
            await pilot.pause()
            assert log.scroll_y == log.max_scroll_y
            await pilot.press("escape")
            assert app.mode == "NORMAL"

    asyncio.run(run())


def test_attached_cache_restore_never_applies_runtime_but_explicit_load_does(app, tmp_path):
    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            path = tmp_path / "attached.json"
            path.write_text(json.dumps({"backend": "window", "mixer": {"damping": 0.9}}))
            calls = []
            client = SimpleNamespace(
                state={"damping": {"text": "0.95", "value": 0.95, "toggle": False, "choices": None}},
                sensors={}, load=lambda selected: calls.append(selected),
            )
            app.attached = True
            app.deck = client
            app.mixer_controls = [RemoteControl("damping", client)]
            app.mixer_context = client
            app.load_state(path, quiet=True)
            assert calls == [] and app.mixer_controls[0].get() == 0.95
            app.run_command("load", [str(path)])
            assert calls == [path.resolve()]

    asyncio.run(run())


def test_write_and_load_roundtrip_has_persistent_status_and_quoted_path_completion(app, tmp_path):
    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            target = tmp_path / "saved sessions" / "test.json"
            app.runtime.engine.set_damping(0.94)
            app.tile_background = "#10202a"
            expected = app.snapshot()
            await pilot.press("colon")
            command = app.query_one(Input)
            command.value = "w " + shlex.quote(str(target))
            await pilot.press("enter")
            await pilot.pause()
            assert json.loads(target.read_text()) == expected
            assert not target.with_suffix(".svg").exists()
            assert str(target) in str(app.query_one("#mode", Static).render())
            app.query_one(LogTile).write("unrelated background entry")
            app.refresh_tiles()
            assert str(target) in str(app.query_one("#mode", Static).render())
            app.runtime.engine.set_damping(0.99)
            app.set_tile_background(None)
            await pilot.press("colon")
            command.value = "load " + shlex.quote(str(target.parent / "te"))
            await pilot.press("tab")
            assert shlex.split(command.value) == ["load", str(target)]
            await pilot.press("enter")
            await pilot.pause()
            assert app.runtime.engine.damping == 0.94
            assert app.tile_background == "#10202a"
            assert "Loaded config" in str(app.query_one("#mode", Static).render())

    asyncio.run(run())


def test_load_picker_discovers_only_project_presets_and_cancel_keeps_settings(app, tmp_path):
    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            app.runtime.engine.set_damping(0.94)
            app.set_tile_background("#10202a")
            app.run_command("w", ["alpha"])
            alpha = app.config_path
            app.runtime.engine.set_damping(0.99)
            app.set_tile_background(None)
            app.run_command("w", ["config/nested/beta"])
            beta = app.config_path
            for name in ("audio_devices.json", "audio_devices.example.json"):
                (app.saves_dir / name).write_text("{}")
            (tmp_path / "outside.json").write_text("{}")
            await pilot.press("colon")
            app.query_one("#command", Input).value = "load"
            await pilot.press("enter")
            await pilot.pause()
            picker = app.screen
            assert isinstance(picker, PresetPicker)
            assert picker.paths == [alpha, beta]
            assert picker.query_one(OptionList).highlighted == 1
            await pilot.press("k", "enter")
            await pilot.pause()
            assert not isinstance(app.screen, PresetPicker)
            assert app.config_path == alpha
            assert app.runtime.engine.damping == 0.94
            assert app.tile_background == "#10202a"
            assert f"Loaded config: {alpha}" in app.status_message
            app.run_command("load", [])
            await pilot.pause()
            await pilot.press("j", "escape")
            await pilot.pause()
            assert app.config_path == alpha
            assert app.runtime.engine.damping == 0.94
            assert app.tile_background == "#10202a"

    asyncio.run(run())


def test_load_picker_empty_and_invalid_presets_report_errors_without_changing_runtime(app):
    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            app.run_command("load", [])
            assert "No saved settings" in app.status_message
            assert not isinstance(app.screen, PresetPicker)
            assert not app.saves_dir.exists()
            before = app.runtime.engine.damping
            original = app.config_path
            app.saves_dir.mkdir()
            (app.saves_dir / "broken.json").write_text("{")
            app.run_command("load", [])
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert "Could not load" in app.status_message
            assert "broken.json" in app.status_message
            assert app.config_path == original
            assert app.runtime.engine.damping == before
            (app.saves_dir / "window.json").write_text(json.dumps({
                "backend": "window", "mixer": {"damping": 0.8},
            }))
            app.run_command("load", ["window"])
            assert "not standalone" in app.status_message
            assert app.config_path == original
            assert app.runtime.engine.damping == before

    asyncio.run(run())


def test_short_names_complete_in_save_directory_and_missing_files_report_failure(app):
    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            app.run_command("w", ["alpha.json"])
            app.run_command("w", ["alpine"])
            await pilot.pause()
            assert app.session_path("alpha.json") == app.saves_dir / "alpha.json"
            assert not (app.saves_dir / "alpha.json.json").exists()
            await pilot.press("colon")
            command = app.query_one(Input)
            command.value = "e al"
            await pilot.press("tab")
            first = shlex.split(command.value)[1]
            await pilot.press("tab")
            second = shlex.split(command.value)[1]
            assert {Path(first).name, Path(second).name} == {"alpha.json", "alpine.json"}
            await pilot.press("shift+tab")
            assert shlex.split(command.value)[1] == first
            await pilot.press("escape")
            app.run_command("e", ["missing"])
            assert "No saved config" in str(app.query_one("#mode", Static).render())
            assert app.saves_dir == app.project_dir / "config"
            assert app.recordings_dir() == app.project_dir / "recordings"
            assert app.session_path("config/nested/example") == app.project_dir / "config/nested/example.json"

    asyncio.run(run())


def test_window_config_save_persists_and_reloads_actual_control_values(app, tmp_path):
    def engine():
        result = SimpleNamespace(damping=0.98, speed=1.0)
        result.set_damping = lambda v: setattr(result, "damping", v)
        result.set_speed = lambda v: setattr(result, "speed", v)
        return result

    config_path = tmp_path / "window-config.json"
    window = SimpleNamespace(engine=engine(), source_orchestrator=None)
    server = DeckServer(window, tmp_path / "first.sock", config_path)
    server.controls = {name: c for name, c in server.controls.items() if name in ("damping", "speed")}
    window.engine.damping = 0.95
    window.engine.speed = 123.0

    async def run():
        async with app.run_test(size=(100, 36)) as pilot:
            app.attached = True
            app.mixer_controls = tuple(server.controls.values())
            app.mixer_context = server.target
            app.deck = SimpleNamespace(
                save=lambda path, state: server.handle({"op": "save", "path": str(path), "state": state})["saved"],
                load=lambda path: server.handle({"op": "load", "path": str(path)}),
                sensors={},
            )
            app.run_command("w", [str(config_path)])
            window.engine.damping = 0.9
            app.run_command("e", [str(config_path)])
            assert window.engine.damping == 0.95
            app.run_command("save", [])
            await pilot.pause()
            assert json.loads(config_path.read_text())["mixer"] == {"damping": 0.95, "speed": 123.0}
            assert str(config_path) in str(app.query_one("#mode", Static).render())
            restored_window = SimpleNamespace(engine=engine(), source_orchestrator=None)
            restored = DeckServer(restored_window, tmp_path / "second.sock", config_path)
            assert restored_window.engine.damping == 0.95
            assert restored_window.engine.speed == 123.0
            restored.server.close()
        server.server.close()

    asyncio.run(run())
