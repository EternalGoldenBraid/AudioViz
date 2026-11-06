import asyncio
import json

from test_log_and_files import app


def test_incremental_terminal_resizes_preserve_tile_edges_and_round_trip(app):
    async def run():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            before = app.snapshot()["placements"]
            for width in range(119, 79, -1):
                await pilot.resize_terminal(width, 24 + round((width - 80) * 0.4))
                app.fit_if_resized()
            await pilot.pause()
            width, height = app.stage_size()
            stage = app.query_one("#stage").region
            for name, placement in app.placements.items():
                assert 0 <= placement.x < placement.x + placement.width <= width, name
                assert 0 <= placement.y < placement.y + placement.height <= height, name
                if app.visible(name):
                    region = app.query_one(f"#{name}").region
                    assert (region.x, region.y, region.width, region.height) == (
                        stage.x + placement.x, stage.y + placement.y,
                        placement.width, placement.height,
                    ), name
            for left, right in (("canvas", "camera_1"), ("camera_1", "visual_prediction")):
                assert app.placements[left].x + app.placements[left].width == app.placements[right].x
            await pilot.resize_terminal(120, 40)
            app.fit_if_resized()
            await pilot.pause()
            assert app.snapshot()["placements"] == before

    asyncio.run(run())


def test_resize_while_zoomed_scales_the_saved_manual_layout_and_stack(app, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)

    async def run():
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#canvas").focus()
            await pilot.pause()
            app.resize_tile(6, 2)
            app.placements["canvas"].opacity = 0.6
            before = app.snapshot()
            width, _ = app.layout_size
            terminal_size = app.size
            app.toggle_zoom()
            await pilot.resize_terminal(80, 24)
            app.fit_if_resized()
            await pilot.pause()
            stage = app.query_one("#stage").size
            assert app.query_one("#canvas").region.size == stage
            app.run_command("w", ["zoomed"])
            saved = json.loads(app.session_path("zoomed").read_text())
            expected_width = round(before["placements"]["canvas"]["width"] * 80 / width)
            assert saved["size"] == list(app.layout_size)
            assert saved["placements"]["canvas"]["width"] == expected_width
            app.toggle_zoom()
            assert app.placements["canvas"].width == expected_width
            assert app.placements["canvas"].opacity == 0.6
            app.select_workspace("audio")
            await pilot.resize_terminal(*terminal_size)
            app.fit_if_resized()
            app.select_workspace("overview")
            await pilot.pause()
            assert app.snapshot()["placements"] == before["placements"]
            assert app.stack == before["stack"]
            app.toggle_immersive()
            await pilot.pause()
            assert app.layout_size == app.stage_size()
            app.toggle_immersive()
            await pilot.pause()
            assert app.snapshot()["placements"] == before["placements"]
            assert app.stack == before["stack"]

    asyncio.run(run())


def test_resize_events_and_saved_viewport_work_without_preview_polling(app, monkeypatch):
    monkeypatch.setattr(app, "refresh_tiles", lambda: None)

    async def run():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.placements["canvas"].width += 3
            before = app.snapshot()
            app.run_command("w", ["layout"])
            saved = json.loads(app.session_path("layout").read_text())
            assert saved["size"] == list(app.stage_size())
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app.layout_size == app.stage_size()
            app.query_one("#canvas").focus()
            await pilot.pause()
            app.toggle_zoom()
            app.run_command("e", ["layout"])
            await pilot.pause()
            assert app.zoomed is None
            assert app.layout_size == app.stage_size()
            assert app.placements["canvas"].width == round(
                before["placements"]["canvas"]["width"] * 80 / 120
            )
            await pilot.resize_terminal(120, 40)
            await pilot.pause()
            assert app.snapshot()["placements"] == before["placements"]

    asyncio.run(run())
