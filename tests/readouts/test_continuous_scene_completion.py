import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "continuous_scene_completion", _SCRIPTS / "continuous_scene_completion.py",
)
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


def test_patterns_avoid_the_passive_zero_equilibrium_everywhere():
    """A uniform band at exactly 0 would coincide with undriven decay and carry

    no observation error at all, regardless of training; values must stay off zero.
    """
    shape = (16, 16)
    for name in completion.PATTERNS:
        image = completion.pattern_image(name, shape)
        assert not np.any(image == 0.)


def test_raster_fixations_disks_cover_the_entire_world():
    shape = (16, 16)
    radius = 3
    fovea = completion.WorldAnchoredFovea(shape, radius=radius)
    coverage = np.zeros(shape, dtype=bool)
    for gaze in completion.raster_fixations(shape, radius):
        fovea.set_gaze(gaze)
        coverage |= fovea.observation_mask
    assert coverage.all()


def test_clamp_strength_blends_only_observed_coordinates_toward_evidence():
    graph = completion.plain_graph((2, 2), hidden=2, inference_rate=2., tau=1 / 3)
    graph.forecast(np.zeros((2, 2, 3)), .1)
    graph.condition(np.full((2, 2, 3), .8))
    before = graph.sensory_potential[graph.visual_slice].copy()
    observed = np.array([[True, False], [False, False]])
    evidence = np.full((2, 2, 3), .8)

    completion.apply_clamp_strength(graph, observed, evidence, 0.)
    np.testing.assert_array_equal(graph.sensory_potential[graph.visual_slice], before)

    completion.apply_clamp_strength(graph, observed, evidence, 1.)
    after = graph.sensory_potential[graph.visual_slice].reshape(2, 2, 3)
    before_grid = before.reshape(2, 2, 3)
    np.testing.assert_allclose(after[0, 0], .8)
    np.testing.assert_array_equal(after[0, 1], before_grid[0, 1])
    np.testing.assert_array_equal(after[1, 0], before_grid[1, 0])
    np.testing.assert_array_equal(after[1, 1], before_grid[1, 1])


def test_plain_graph_tau_overrides_the_default_memory_time_constant():
    default = completion.plain_graph((1, 1), hidden=1)
    slower = completion.plain_graph((1, 1), hidden=1, tau=3.)
    assert default.memory_time_constant == pytest.approx(1 / 3)
    assert slower.memory_time_constant == pytest.approx(3.)


def test_evaluate_hidden_trace_has_one_recorded_vector_per_frame_and_matches_the_graph():
    shape = (4, 4)
    graph = completion.plain_graph(shape, hidden=3, inference_rate=.3, learning_rate=.05, tau=1 / 3)
    fovea = completion.WorldAnchoredFovea(shape, radius=1)
    rng = np.random.default_rng(0)
    results, observed = completion.evaluate(
        graph, fovea, shape, patch_gaze=(.5, .5), dt=1 / 30, settle_frames=5,
        rng=rng, sigma=0., clamp_strength=.5,
    )
    for name, case in results.items():
        assert len(case["hidden_trace"]) == 5
        for vector in case["hidden_trace"]:
            assert vector.shape == (3,)
    # Replaying evaluate deterministically (fresh rng, sigma=0) must reproduce the same final trace vector.
    rng_replay = np.random.default_rng(0)
    graph_replay = completion.plain_graph(shape, hidden=3, inference_rate=.3, learning_rate=.05, tau=1 / 3)
    replayed, _ = completion.evaluate(
        graph_replay, fovea, shape, patch_gaze=(.5, .5), dt=1 / 30, settle_frames=5,
        rng=rng_replay, sigma=0., clamp_strength=.5,
    )
    for name in completion.PATTERNS:
        np.testing.assert_array_equal(replayed[name]["hidden_trace"][-1], results[name]["hidden_trace"][-1])


def test_evaluate_prediction_trace_last_frame_matches_the_scalar_prediction():
    shape = (4, 4)
    graph = completion.plain_graph(shape, hidden=2, inference_rate=.3, learning_rate=.05, tau=1 / 3)
    fovea = completion.WorldAnchoredFovea(shape, radius=1)
    rng = np.random.default_rng(0)
    results, _ = completion.evaluate(
        graph, fovea, shape, patch_gaze=(.5, .5), dt=1 / 30, settle_frames=4,
        rng=rng, sigma=0., clamp_strength=.6,
    )
    for name, case in results.items():
        assert len(case["prediction_trace"]) == 4
        np.testing.assert_array_equal(case["prediction_trace"][-1], case["prediction"])


def test_playback_atlas_tiles_every_frame_for_every_pattern(tmp_path):
    shape = (4, 4)
    completion.run(
        tmp_path, shape=shape, radius=1, hidden=2, epochs=1, hold_frames=1,
        settle_frames=3, learning_rate=.05, sigma=0., tau=1 / 3, clamp_strength=.5, label="t",
    )
    import json
    from PIL import Image

    summary = json.loads((tmp_path / "t-summary.json").read_text())
    atlas_info = summary["playback_atlas"]
    assert atlas_info["columns"] == 3
    assert atlas_info["rows"] == len(completion.PATTERNS)
    with Image.open(tmp_path / "t-playback.png") as atlas:
        assert atlas.size == (3 * shape[1], len(completion.PATTERNS) * shape[0])
    for name in completion.PATTERNS:
        assert (tmp_path / f"pattern-{name}.png").exists()


def test_run_grid_covers_every_curated_clamp_and_tau_combination(tmp_path):
    index = completion.run_grid(
        tmp_path, shape=(4, 4), radius=1, hidden=2, epochs=1, hold_frames=1,
        settle_frames=2, learning_rate=.05, sigma=0.,
    )
    labels = {config["label"] for config in index["configs"]}
    assert labels == {
        f"completion-{clamp}-{tau}"
        for clamp in completion.GRID for tau in completion.GRID_TAU
    }
    for config in index["configs"]:
        assert (tmp_path / f"{config['label']}-summary.json").exists()
        assert (tmp_path / f"{config['label']}-playback.png").exists()
    # Section highlights read back the just-written playback/summary files, without rerunning inference.
    for name in completion.PATTERNS:
        assert (tmp_path / f"highlight-patch-target-{name}.png").exists()
        for label in ("completion-soft-small", "completion-hard-small"):
            assert (tmp_path / f"highlight-patch-{label}-{name}.png").exists()
            assert (tmp_path / f"highlight-scene-{label}-{name}.png").exists()


def test_summary_json_matches_the_frontend_contract(tmp_path):
    """The explorer component reads these exact keys; a rename here breaks it silently."""
    import json

    completion.run(
        tmp_path, shape=(4, 4), radius=1, hidden=2, epochs=1, hold_frames=1,
        settle_frames=3, learning_rate=.05, sigma=0., tau=1 / 3, clamp_strength=.5, label="t",
    )
    summary = json.loads((tmp_path / "t-summary.json").read_text())
    assert summary["settings"]["hidden_nodes"] == 2
    assert summary["settings"]["shape"] == [4, 4]
    assert set(summary["playback_atlas"]["patterns"]) == set(completion.PATTERNS)
    for name in completion.PATTERNS:
        assert len(summary["hidden_traces"][name]) == 3
        assert all(len(vector) == 2 for vector in summary["hidden_traces"][name])


def test_patch_box_stays_centered_and_clips_at_world_borders():
    settings = {"shape": [16, 16], "patch_gaze": [.5, .5], "foveal_radius": 3}
    left, top, right, bottom = completion._patch_box(settings, pad=2)
    assert (right - left) == (bottom - top) == 2 * (3 + 2) + 1
    assert left >= 0 and top >= 0 and right <= 16 and bottom <= 16
    clipped = completion._patch_box({"shape": [16, 16], "patch_gaze": [0., 0.], "foveal_radius": 3}, pad=2)
    assert clipped[0] == 0 and clipped[1] == 0
