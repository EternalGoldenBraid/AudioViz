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


@pytest.mark.parametrize("name", sorted(set(VIEWS) - {"camera"}))
def test_every_view_renders_an_rgb_image(runtime, name):
    image = VIEWS[name](runtime, 20, 40)
    assert image is not None
    assert image.ndim == 3 and image.shape[-1] == 3
    assert image.dtype == np.uint8


def test_camera_view_shows_the_latest_frame_as_rgb(runtime):
    assert VIEWS["camera"](runtime, 20, 40) is None  # no device in this runtime
    runtime.camera._frame = np.zeros((4, 6, 3), np.uint8)
    runtime.camera._frame[..., 0] = 200  # blue channel in BGR
    try:
        assert VIEWS["camera"](runtime, 20, 40)[0, 0].tolist() == [0, 0, 200]
    finally:
        runtime.camera._frame = None


def test_bar_views_fill_the_requested_pixel_grid(runtime):
    for name in ("prediction", "error", "waveform"):
        assert VIEWS[name](runtime, 20, 40).shape == (20, 40, 3)


def test_views_report_no_data_before_first_block():
    runtime = GraphRuntime(camera=False)
    assert VIEWS["spectrogram"](runtime, 20, 40) is None
    assert VIEWS["prediction"](runtime, 20, 40) is None


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
        (group,) = app.groups
        assert group[0] == "canvas" and len(group) == 2
        before = {n: (app.placements[n].x, app.placements[n].y) for n in app.placements}
        await pilot.press("J")
        for name in app.placements:
            moved = app.placements[name].y - before[name][1]
            assert moved == (1 if name in group else 0), name
        await pilot.press("U")
        assert app.groups == []
    _headless(tmp_path, scenario)


def test_state_round_trips_through_write_and_load(tmp_path):
    async def scenario(app, pilot):
        app.run_command("group", ["canvas", "mixer"])
        app.run_command("theme", ["amber"])
        app.placements["log"].x = 7
        app.run_command("w", ["t"])
        await pilot.pause(0.5)
        assert (tmp_path / "saves" / "t.json").exists()
        app.run_command("ungroup", [])
        app.run_command("theme", ["black"])
        app.placements["log"].x = 0
        app.run_command("e", ["t"])
        assert app.groups == [["canvas", "mixer"]] and app.theme == "amber" and app.placements["log"].x == 7
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
