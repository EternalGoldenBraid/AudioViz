import numpy as np
import pytest

from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.views import VIEWS


@pytest.fixture(scope="module")
def runtime():
    runtime = GraphRuntime(camera=False)
    for _ in range(30):
        runtime.step()
    return runtime


@pytest.mark.parametrize("name", sorted(set(VIEWS) - {"camera_1", "camera_2", "camera_2_prediction", "camera_2_error", "cochlea"}))
def test_every_view_renders_an_rgb_image(runtime, name):
    image = VIEWS[name](runtime, 20, 40)
    assert image is not None
    assert image.ndim == 3 and image.shape[-1] == 3
    assert image.dtype == np.uint8


def test_camera_view_shows_the_latest_frame_as_rgb(runtime):
    assert VIEWS["camera_1"](runtime, 20, 40) is None  # no device in this runtime
    runtime.show_camera = True
    runtime.camera._frame = np.zeros((4, 6, 3), np.uint8)
    runtime.camera._frame[..., 0] = 200  # blue channel in BGR
    try:
        assert VIEWS["camera_1"](runtime, 20, 40)[0, 0].tolist() == [0, 0, 200]
    finally:
        runtime.camera._frame = None


def test_a_sensor_view_shows_that_sensors_observation(runtime):
    camera_2 = runtime.orchestrator.sensors["camera 2"]
    assert VIEWS["camera_2"](runtime, 20, 40) is None
    camera_2.observation = np.full((3, 4, 3), 0.5, np.float32)
    assert VIEWS["camera_2"](runtime, 20, 40)[0, 0].tolist() == [127, 127, 127]


def test_bar_views_fill_the_requested_pixel_grid(runtime):
    for name in ("prediction", "error", "waveform"):
        assert VIEWS[name](runtime, 20, 40).shape == (20, 40, 3)


def test_views_report_no_data_before_first_block():
    runtime = GraphRuntime(camera=False)
    assert VIEWS["spectrogram"](runtime, 20, 40) is None
    assert VIEWS["prediction"](runtime, 20, 40) is None


def test_old_saved_layout_keeps_the_new_cochlea_tile(tmp_path, monkeypatch):
    import asyncio
    from dataclasses import asdict
    from audioviz.tui.app import AudioVizApp

    class PreviewApp(AudioVizApp):
        async def _serve_control(self):
            pass

    app = PreviewApp(audio="synthetic", camera=False, state_dir=tmp_path, autosave=False, hot_reload=False)
    monkeypatch.setattr(app.runtime, "start", lambda: None)
    async def restore():
        async with app.run_test(size=(120, 42)):
            current = dict(app.placements)
            old = {name: asdict(value) for name, value in current.items() if name != "cochlea"}
            app.restore({"placements": old, "stack": list(old)})
            assert app.placements == current
            assert "cochlea" in app.stack
            app.apply_layout()
            assert app.query_one("#cochlea").border_subtitle == "index / time > | -80..0 dB"
            assert app.query_one("#waveform").border_subtitle == "fixed +/-1 | time >"

    asyncio.run(restore())


def test_synthetic_feed_advances_the_processor(runtime):
    assert runtime.processor.frame_counter >= 30


def test_render_cells_modes_cover_the_cell_grid():
    import numpy as np
    from audioviz.tui.halfblock import RENDER_MODES, render_cells

    image = np.random.default_rng(0).integers(0, 255, (40, 60, 3)).astype(np.uint8)
    for mode in RENDER_MODES:
        cells = render_cells(image, 10, 20, mode)
        assert cells.glyphs.shape[:2] == (10, 20)


def test_mixer_nudge_round_trips_through_runtime():
    from audioviz.tui.graph_runtime import GraphRuntime
    from audioviz.tui.mixer import CONTROLS

    runtime = GraphRuntime(resolution=16, camera=False)
    damping = next(c for c in CONTROLS if c.name == "damping")
    before = damping.get(runtime)
    damping.nudge(runtime, -5)
    assert damping.get(runtime) < before
    runtime.rebuild(24)
    assert runtime.engine.damping < before


def _run(runtime, steps=60):
    for _ in range(steps):
        runtime.step()


def test_evidence_vocoder_reproduces_the_input_tone():
    import numpy as np
    from audioviz.tui.audio_feed import BLOCK_SIZE, SAMPLE_RATE
    from audioviz.tui.graph_runtime import GraphRuntime

    runtime = GraphRuntime(resolution=16, camera=False)
    runtime.feed.use_tone(880.0)
    runtime.out.set_mode("evidence", speakers=False)
    _run(runtime)
    block = runtime.out.last_block
    spectrum = np.abs(np.fft.rfft(block * np.hanning(BLOCK_SIZE)))
    peak = np.fft.rfftfreq(BLOCK_SIZE, 1 / SAMPLE_RATE)[spectrum.argmax()]
    bands = runtime.out.vocoder.freqs
    nearest = bands[np.abs(bands - 880.0).argmin()]
    assert abs(peak - nearest) < 60.0, (peak, nearest)


def test_prediction_vocoder_sounds_the_model_output():
    import numpy as np
    from audioviz.tui.graph_runtime import GraphRuntime

    runtime = GraphRuntime(resolution=16, camera=False)
    runtime.feed.use_tone(880.0)
    runtime.out.set_mode("prediction", speakers=False)
    _run(runtime)
    assert np.isfinite(runtime.out.last_block).all()
    assert np.abs(runtime.out.last_block).max() > 1e-4


def test_spatial_focus_picks_nearest_neighbour_in_each_direction():
    from audioviz.tui.app import AudioVizApp

    app = AudioVizApp.__new__(AudioVizApp)
    from audioviz.tui.app import Placement
    app.groups, app.pages, app.hidden = [], {}, set()
    app.sensors_enabled = lambda: {}
    app.placements = {
        "a": Placement(0, 0, 10, 5), "b": Placement(10, 0, 10, 5),
        "c": Placement(0, 5, 10, 5), "far": Placement(30, 0, 10, 5),
    }
    assert app.neighbour("a", 1, 0) == "b"
    assert app.neighbour("a", 0, 1) == "c"
    assert app.neighbour("a", -1, 0) is None
    assert app.neighbour("b", 0, 1) == "c"


def _headless(tmp_path, scenario):
    import asyncio
    from audioviz.tui.app import AudioVizApp

    async def run():
        app = AudioVizApp(audio="synthetic", camera=False, state_dir=tmp_path, autosave=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.5)
            await scenario(app, pilot)
    asyncio.run(run())


def test_grouped_tiles_move_together_and_visual_mode_groups_neighbours(tmp_path):
    async def scenario(app, pilot):
        app.query_one("#canvas").focus()
        await pilot.press("v", "l", "g")  # canvas + its right neighbour
        group = app.groups[-1]  # presets (mixer, settings) come first
        assert group[0] == "canvas" and len(group) == 2
        before = {n: (app.placements[n].x, app.placements[n].y) for n in app.placements}
        await pilot.press("J")
        for name in app.placements:
            moved = app.placements[name].y - before[name][1]
            assert moved == (1 if name in group else 0), name
        await pilot.press("U")
        assert group not in app.groups
    _headless(tmp_path, scenario)


def test_state_round_trips_through_write_and_load(tmp_path):
    async def scenario(app, pilot):
        app.run_command("group", ["canvas", "knob-damping"])
        app.run_command("theme", ["amber"])
        app.placements["log"].x = 7
        app.run_command("w", ["t"])
        await pilot.pause(0.5)
        assert (tmp_path / "saves" / "t.json").exists()
        app.run_command("ungroup", [])
        app.run_command("theme", ["black"])
        app.placements["log"].x = 0
        app.run_command("e", ["t"])
        assert ["canvas", "knob-damping"] in app.groups and app.theme == "amber" and app.placements["log"].x == 7
    _headless(tmp_path, scenario)


def test_choice_control_steps_through_values():
    from audioviz.tui.mixer import Control

    box = {"v": "b"}
    control = Control("c", 0, 0, lambda _: box["v"], lambda _, v: box.update(v=v), choices=("a", "b", "c"))
    control.nudge(None, 1)
    control.nudge(None, 1)
    assert box["v"] == "c" and control.position(None) == 1.0
    control.nudge(None, -1)
    assert box["v"] == "b"


def test_hot_reload_picks_up_an_edited_view_and_survives_a_syntax_error():
    import os
    from pathlib import Path
    from audioviz.tui import views
    from audioviz.tui.hot_reload import HotReload

    path = Path(views.__file__)
    original = path.read_text()
    reloader = HotReload()
    try:
        path.write_text(original + "\nMARKER = 1\n")
        assert reloader.poll() == ["views"] and views.MARKER == 1

        path.write_text(original + "\ndef broken(:\n")
        with pytest.raises(SyntaxError):
            reloader.poll()
        assert views.MARKER == 1 and reloader.poll() == []  # old code stays live, no retry storm
    finally:
        path.write_text(original)
        os.utime(path)
        reloader.poll()


def test_knob_tile_turns_directly_and_slides_with_h_l_once_entered(tmp_path):
    async def scenario(app, pilot):
        before = app.runtime.engine.damping
        knob = app.query_one("#knob-damping")
        knob.focus()
        await pilot.press("minus", "left")  # turn without entering
        assert app.runtime.engine.damping < before
        low = app.runtime.engine.damping
        await pilot.press("h")  # not entered: h walks to another knob and turns nothing
        assert app.focused.id != "knob-damping" and app.runtime.engine.damping == low
        knob.focus()
        await pilot.press("enter")
        assert knob.editing and app.mode == "EDIT"
        await pilot.press("l", "l", "l", "j")  # entered: h/l slide, every other key is swallowed
        assert app.runtime.engine.damping > low and app.focused is knob
        await pilot.press("escape")
        assert not knob.editing and app.mode == "NORMAL"
        await pilot.press("j")
        assert app.focused.id != "knob-damping" and app.focused.id.startswith("knob-")
        assert any("knob-damping" in g and "knob-speed" in g for g in app.groups)  # mixer preset group
    _headless(tmp_path, scenario)


def test_camera_off_releases_the_device_and_clears_the_tile():
    from audioviz.tui.camera_feed import CameraFeed

    class Device:
        released = False
        def release(self):
            self.released = True

    feed = CameraFeed()
    feed._capture, feed._frame = Device(), np.ones((2, 2, 3), np.uint8)
    device = feed._capture
    assert feed.isOpened() and feed.rgb() is not None
    feed.release()
    assert device.released and not feed.isOpened() and feed.rgb() is None


def test_camera_view_and_canvas_feed_share_the_device():
    runtime = GraphRuntime(camera=False)
    opened = []
    runtime.camera.open = lambda: opened.append(1) or runtime.camera.isOpened()
    runtime.camera._capture = object()  # pretends the device is open
    runtime.camera.release = lambda: setattr(runtime.camera, "_capture", None)
    runtime.set_canvas_feed(True)
    runtime.set_camera(False)
    assert runtime.camera.isOpened()  # the canvas still uses the device
    runtime.set_canvas_feed(False)
    assert not runtime.camera.isOpened()


def test_rising_focuses_the_whole_group_and_navigation_steps_between_groups(tmp_path):
    async def scenario(app, pilot):
        app.query_one("#knob-damping").focus()
        await pilot.press("escape")  # rise: the mixer group becomes the focused node
        assert app.group_focus == app.members("knob-damping")
        before = app.placements["knob-damping"].x
        await pilot.press("L")
        assert app.placements["knob-damping"].x == before + app.MOVE_STEP[0]  # the whole group moves
        await pilot.press("l")
        assert set(app.group_focus).isdisjoint(app.members("knob-damping"))  # stepped to the next node
        await pilot.press("enter")  # dive: back to single tiles
        assert app.group_focus == []
    _headless(tmp_path, scenario)


def test_a_paged_group_shows_one_page_of_members_in_shared_slots(tmp_path):
    async def scenario(app, pilot):
        group = app.members("knob-damping")
        size = app.pages[group[0]]["size"]
        assert len(group) > size
        assert [n for n in group if app.visible(n)] == group[:size]
        first_slots = [(app.placements[n].x, app.placements[n].y) for n in group[:size]]
        app.query_one("#knob-damping").focus()
        await pilot.press("n")
        assert [n for n in group if app.visible(n)] == group[size:2 * size]
        assert [(app.placements[n].x, app.placements[n].y) for n in group[size:2 * size]] == first_slots[:len(group) - size]
        assert app.focused.id in group[size:]
        await pilot.press("p")
        assert [n for n in group if app.visible(n)] == group[:size]
    _headless(tmp_path, scenario)


def test_keys_file_rebinds_actions_and_reports_what_it_cannot_use(tmp_path):
    from audioviz.tui import keys

    defaults = {(b.section, b.action): b.keys for b in keys.DEFAULTS}
    assert len(keys.keymap(keys.DEFAULTS)) == sum(len(k) for k in defaults.values())  # no key bound twice per mode

    path = tmp_path / "keys.toml"
    path.write_text('[view]\nzoom = "x"\n[navigate]\nnext_tile = ["tab", "]"]\nbogus = "q"\n[app]\nquit = 3\n')
    overrides, problems = keys.load_overrides(path)
    table = keys.keymap(keys.bindings(overrides))
    assert table[("normal", "x")] == "zoom" and ("normal", "z") not in table
    assert table[("normal", "right_square_bracket")] == "next_tile"  # symbols as shown by `?` are accepted
    assert len(problems) == 2

    path.write_text(keys.default_toml())  # the generated file is a valid no-op override
    assert keys.keymap(keys.bindings(keys.load_overrides(path)[0])) == keys.keymap(keys.DEFAULTS)


def test_question_mark_lists_bindings_and_a_rebound_key_takes_effect(tmp_path):
    async def scenario(app, pilot):
        await pilot.press("question_mark")
        assert type(app.screen).__name__ == "KeysScreen"
        await pilot.press("escape")
        assert type(app.screen).__name__ != "KeysScreen"
        app.keys_path.write_text('[view]\nzoom = "x"\n')
        app.run_command("keys", ["reload"])
        await pilot.press("x")
        assert app.zoomed
    _headless(tmp_path, scenario)


def test_layout_refits_to_the_terminal_keeping_knob_height_and_the_bottom_edge(tmp_path):
    async def scenario(app, pilot):
        knob, view = "knob-damping", "canvas"
        before = {n: vars(p).copy() for n, p in app.placements.items()}
        old_w, old_h = app.layout_size
        await pilot.resize_terminal(old_w * 2, old_h + 21)  # one row is the mode bar
        await pilot.pause(0.2)
        new_w, new_h = app.layout_size
        assert (new_w, new_h) == (old_w * 2, old_h + 20)
        assert app.placements[knob].height == before[knob]["height"]
        assert app.placements[knob].y + app.placements[knob].height == before[knob]["y"] + before[knob]["height"] + 20
        assert app.placements[view].width == before[view]["width"] * 2
        assert app.placements[view].height > before[view]["height"]
    _headless(tmp_path, scenario)


def test_resolution_accepts_a_square_size_or_rows_by_cols_and_keeps_cells_square():
    from audioviz.tui.graph_runtime import parse_resolution
    assert parse_resolution(64) == (64, 64) and parse_resolution("100x200") == (100, 200)
    with pytest.raises(ValueError):
        parse_resolution("4x200")
    runtime = GraphRuntime(resolution=16, camera=False)
    runtime.rebuild("24x48")
    runtime.step()
    assert runtime.resolution == (24, 48)
    assert runtime.engine.get_field_numpy().shape[:2] == (24, 48)
    assert runtime.engine.dx == runtime.engine.dy


def test_dd_hides_a_tile_u_restores_it_and_the_tile_list_hides_whole_groups(tmp_path):
    from audioviz.tui import app as app_module
    from audioviz.tui.tile_list import TileList

    app_module.SEQUENCE_SECONDS = 30.0
    default = set(app_module.HIDDEN_BY_DEFAULT)
    async def scenario(app, pilot):
        app.query_one("#canvas").focus()
        await pilot.press("d", "x")  # `d` followed by another key is not a hide
        assert app.hidden == default
        app.query_one("#canvas").focus()
        await pilot.press("d", "d")  # the pilot idles about 2 s per key, so the sequence timeout is widened above
        assert app.hidden == default | {"canvas"} and not app.query_one("#canvas").display
        assert app.focused.id != "canvas"  # focus moved to a tile that is still shown

        await pilot.press("u")
        assert app.hidden == default and app.focused.id == "canvas"

        await pilot.press("T")  # the tile list: the mixer section row toggles all its knobs
        assert isinstance(app.screen, TileList)
        rows = [label for label, _, section in app.screen.rows]
        await pilot.press(*["j"] * rows.index("mixer"), "space", "enter")
        assert set(app.members("knob-damping")) <= app.hidden
        assert app.query_one("#canvas").display

        await pilot.press("u")  # one undo brings back everything the list hid
        assert app.hidden == default

        app.hide_tiles([n for n in app.placements if n != "log"])
        app.hide_tiles(["log"])
        assert app.hidden == set(app.placements) - {"log"}  # the last tile cannot be hidden

        app.hidden.clear()
        app.hidden |= {"canvas", "spectrogram"}
        saved = app.snapshot()
        app.hidden.clear()
        app.restore(saved)
        assert app.hidden == {"canvas", "spectrogram"}
    _headless(tmp_path, scenario)


def test_a_sensor_tile_is_shown_only_while_its_sensor_is_enabled(tmp_path):
    async def scenario(app, pilot):
        sensor = app.runtime.orchestrator.sensors["camera 2"]
        assert not app.query_one("#camera_2").display  # camera 2 is off, so its tile is not shown

        sensor.capture = type("NoFrames", (), {"read": lambda self: (False, None)})()  # open, delivering nothing
        sensor.enabled = True  # the graph now uses it; the tile appears without any key press
        app.apply_layout()
        assert app.query_one("#camera_2").display
        assert not app.query_one("#camera_2_error").display  # camera 2's other views start hidden until the tile list shows them
        app.hidden.discard("camera_2_error")
        app.apply_layout()
        assert app.query_one("#camera_2_error").display
        sensor.enabled = False
        app.apply_layout()
        assert not app.query_one("#camera_2_error").display  # and they follow their sensor off again
    _headless(tmp_path, scenario)


def test_tile_list_stays_inside_a_short_terminal_so_it_can_scroll():
    import asyncio
    from textual.app import App
    from textual.widgets import OptionList
    from audioviz.tui.tile_list import TileList

    sections = [("tiles", [f"tile-{i}" for i in range(40)])]

    async def run():
        app = App()
        async with app.run_test(size=(80, 20)) as pilot:
            app.push_screen(TileList(sections, set()))
            await pilot.pause()
            options = app.screen.query_one(OptionList)
            assert options.region.height <= 20 * 0.8
            for _ in range(39):
                await pilot.press("j")
            await pilot.pause()
            assert options.scroll_y > 0

    asyncio.run(run())


def test_record_writes_the_shown_tiles_and_stops_on_the_second_command(tmp_path):
    import cv2

    async def scenario(app, pilot):
        await pilot.pause(0.5)
        before = len(app.recorded_tiles())
        app.hide_tiles(["spectrogram"])
        target = tmp_path / "clip.mp4"
        app.run_command("record", [str(target)])
        await pilot.pause(0.8)
        assert 0 < len(app.recorded_tiles()) == before - 1
        app.run_command("record", [])
        assert target.exists()
        video = cv2.VideoCapture(str(target))
        assert video.get(cv2.CAP_PROP_FRAME_COUNT) >= 3
    _headless(tmp_path, scenario)


@pytest.mark.parametrize("fail_capture", [False, True])
def test_recording_is_finalized_on_exit_or_capture_failure(tmp_path, monkeypatch, fail_capture):
    import cv2

    target = tmp_path / "finalized.mp4"
    recordings = []

    async def scenario(app, pilot):
        app.run_command("record", [str(target)])
        await pilot.pause(0.5)
        recorder = app.recorder
        assert recorder.frames > 0
        recordings.append(recorder)
        if fail_capture:
            def fail(tiles):
                raise RuntimeError("capture failed")

            monkeypatch.setattr(recorder, "capture", fail)
            app.record_frame()
            assert app.recorder is None

    _headless(tmp_path, scenario)
    assert recordings[0].writer is None
    video = cv2.VideoCapture(str(target))
    try:
        assert video.isOpened() and video.read()[0]
    finally:
        video.release()


def test_slash_filters_the_key_list_and_escape_steps_back(tmp_path):
    from textual.widgets import Static

    async def scenario(app, pilot):
        def rows():
            return "\n".join(str(w.render()) for w in app.screen.query("Grid > Static"))
        await pilot.press("question_mark")
        everything = rows()
        await pilot.press("slash", "z", "o", "o", "m")
        await pilot.pause()
        assert app.screen.query_one("#search").value == "zoom" and 0 < len(rows()) < len(everything)
        assert "zoom" in rows().lower()
        await pilot.press("escape")  # drops the filter, stays in the list
        await pilot.pause()
        assert type(app.screen).__name__ == "KeysScreen" and rows() == everything
        await pilot.press("escape")
        assert type(app.screen).__name__ != "KeysScreen"
    _headless(tmp_path, scenario)


def test_record_tile_prompts_for_a_path_blinks_while_recording_and_stops_on_enter(tmp_path):
    async def scenario(app, pilot):
        tile = app.query_one("#record")
        await pilot.pause(0.3)
        assert "not recording" in str(tile.render())
        tile.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert type(app.screen).__name__ == "RecordPrompt"
        box = app.screen.query_one("#record-path")
        box.value = str(tmp_path / "clips" / "run.mp4")
        await pilot.press("enter")
        await pilot.pause(0.6)
        assert app.recorder is not None and "REC" in str(tile.render()) and "run.mp4" in str(tile.render())
        await pilot.press("enter")
        assert app.recorder is None and (tmp_path / "clips" / "run.mp4").exists()
    _headless(tmp_path, scenario)


def test_path_suggester_completes_folders_and_recordings(tmp_path):
    import asyncio
    from audioviz.tui.record_tile import PathSuggester

    (tmp_path / "runs").mkdir()
    (tmp_path / "run1.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("")
    suggest = lambda value: asyncio.run(PathSuggester().get_suggestion(value))
    assert suggest(f"{tmp_path}/ru") == f"{tmp_path}/run1.mp4" or suggest(f"{tmp_path}/ru") == f"{tmp_path}/runs/"
    assert suggest(f"{tmp_path}/no") is None
    assert suggest(f"{tmp_path}/") in {f"{tmp_path}/run1.mp4", f"{tmp_path}/runs/"}


def test_shift_arrows_resize_and_zoom_is_just_a_big_resize_that_restores(tmp_path):
    async def scenario(app, pilot):
        tile = app.query_one("#canvas")
        tile.focus()
        before = vars(app.placements["canvas"]).copy()
        depth = app.stack.index("canvas")
        await pilot.press("shift+right", "shift+down")
        grown = app.placements["canvas"]
        assert (grown.width, grown.height) == (before["width"] + 2, before["height"] + 1)
        await pilot.press("shift+left", "shift+up")
        assert vars(app.placements["canvas"]) == before
        app.resize_tile(-500, -500)
        assert (app.placements["canvas"].width, app.placements["canvas"].height) == (6, 3)
        app.placements["canvas"].width, app.placements["canvas"].height = before["width"], before["height"]
        await pilot.press("z")
        stage = app.query_one("#stage").size
        zoomed = app.placements["canvas"]
        assert (zoomed.x, zoomed.y, zoomed.width, zoomed.height) == (0, 0, stage.width, stage.height)
        assert app.stack[-1] == "canvas" and tile.region.width == stage.width
        assert app.snapshot()["placements"]["canvas"] == before  # a save while zoomed keeps the real placement
        await pilot.press("z")
        assert vars(app.placements["canvas"]) == before and app.stack.index("canvas") == depth
    _headless(tmp_path, scenario)
