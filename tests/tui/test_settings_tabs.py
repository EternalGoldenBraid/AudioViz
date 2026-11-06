import asyncio
import json
import tomllib

from audioviz.tui.app import AudioVizApp, KnobTile, LogTile
from audioviz.tui.keys import default_toml

from test_log_and_files import app


def test_tabs_share_live_controls_and_return_to_overview_without_rebuilding(app):
    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            engine = app.runtime.engine
            await pilot.press("tab")
            await pilot.pause()
            assert app.workspace == "audio"
            assert isinstance(app.focused, KnobTile)
            await pilot.press("tab")
            await pilot.pause()
            assert app.workspace == "wave"
            damping = app.focused
            assert damping.control.name == "damping"
            before = engine.damping
            await pilot.press("enter", "h", "escape")
            assert engine.damping < before
            await pilot.press("l")
            assert app.focused.control.name == "speed"
            app.run_command("w", ["session"])
            assert json.loads(app.session_path("session").read_text())["tab"] == "wave"
            app.run_command("tab", ["learning"])
            await pilot.pause()
            assert app.focused.control.name == "learning"
            app.run_command("e", ["session"])
            await pilot.pause()
            assert app.workspace == "wave"
            await pilot.press("shift+tab", "shift+tab")
            await pilot.pause()
            assert app.workspace == "overview"
            assert app.focused.id == "canvas"
            assert app.runtime.engine is engine

    defaults = tomllib.loads(default_toml())
    assert defaults["tabs"] == {"tab_next": ["tab"], "tab_prev": ["shift+tab"]}
    assert defaults["navigate"]["next_tile"] == ["w"]
    assert defaults["navigate"]["prev_tile"] == ["b"]
    asyncio.run(run())


def test_tab_switching_leaves_edit_and_log_modes_and_wraps(app):
    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("w", "b")
            assert app.focused.id == "canvas"
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.workspace == "interface"
            await pilot.press("tab")
            await pilot.pause()
            assert app.workspace == "overview"
            app.query_one(LogTile).focus()
            await pilot.press("enter")
            assert app.mode == "LOG"
            await pilot.press("tab")
            await pilot.pause()
            assert app.workspace == "audio" and app.mode == "NORMAL"
            knob = app.focused
            await pilot.press("enter")
            assert knob.editing
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.workspace == "overview" and app.mode == "NORMAL"
            assert not knob.editing

    asyncio.run(run())


def test_project_config_restores_the_selected_tab_on_a_new_launch(app):
    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.select_workspace("learning")
            await pilot.pause()
            app.run_command("w", [])
            assert json.loads(app.config_path.read_text())["tab"] == "learning"
        restarted = AudioVizApp(
            audio="synthetic", camera=False, state_dir=app.project_dir,
            autosave=False, hot_reload=False,
        )
        restarted.runtime.start = lambda: None
        async with restarted.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert restarted.workspace == "learning"
            assert restarted.focused.control.name == "learning"
            assert restarted.config_path == app.project_dir / "config" / "deck.json"

    asyncio.run(run())
