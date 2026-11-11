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


def test_evaluate_scan_visits_every_fixation_once_per_pass_with_learning_on():
    shape = (4, 4)
    graph = completion.plain_graph(shape, hidden=3, inference_rate=.3, learning_rate=.05, tau=1 / 3)
    fovea = completion.WorldAnchoredFovea(shape, radius=1)
    fixations = completion.raster_fixations(shape, fovea.radius)
    rng = np.random.default_rng(0)
    before = graph.sensory_weights.copy()
    results, fovea_trace = completion.evaluate_scan(
        graph, fovea, shape, dt=1 / 30, hold_frames=2, passes=2,
        rng=rng, sigma=0., clamp_strength=.5,
    )
    expected_frames = 2 * len(fixations) * 2  # passes * fixations * hold_frames
    assert len(fovea_trace) == expected_frames
    for name, case in results.items():
        assert len(case["hidden_trace"]) == expected_frames
        assert len(case["prediction_trace"]) == expected_frames
        for vector in case["hidden_trace"]:
            assert vector.shape == (3,)
    # Learning stayed enabled through the recorded scan: weights kept changing.
    assert np.linalg.norm(graph.sensory_weights - before) > 0.
    # The recorded gaze schedule is the raster path, repeated once per pass.
    centers = [tuple(entry["center"]) for entry in fovea_trace]
    held = centers[::2]  # hold_frames=2, so every other entry starts a new fixation
    first_pass = held[:len(fixations)]
    second_pass = held[len(fixations):]
    assert first_pass == second_pass


def test_evaluate_scan_is_deterministic_given_a_fresh_seeded_rng():
    shape = (4, 4)
    fovea = completion.WorldAnchoredFovea(shape, radius=1)

    def run_once():
        graph = completion.plain_graph(shape, hidden=2, inference_rate=.3, learning_rate=.05, tau=1 / 3)
        rng = np.random.default_rng(0)
        results, _ = completion.evaluate_scan(
            graph, fovea, shape, dt=1 / 30, hold_frames=1, passes=1,
            rng=rng, sigma=0., clamp_strength=.6,
        )
        return results

    first, second = run_once(), run_once()
    for name in completion.PATTERNS:
        np.testing.assert_array_equal(first[name]["prediction_trace"][-1], second[name]["prediction_trace"][-1])


def test_error_trace_matches_the_absolute_prediction_minus_target_residual():
    shape = (4, 4)
    graph = completion.plain_graph(shape, hidden=2, inference_rate=.3, learning_rate=.05, tau=1 / 3)
    fovea = completion.WorldAnchoredFovea(shape, radius=1)
    rng = np.random.default_rng(0)
    results, _ = completion.evaluate_scan(
        graph, fovea, shape, dt=1 / 30, hold_frames=1, passes=1,
        rng=rng, sigma=0., clamp_strength=.5,
    )
    for name, case in results.items():
        image = completion.pattern_image(name, shape)
        for prediction, error in zip(case["prediction_trace"], case["error_trace"]):
            np.testing.assert_allclose(error, np.abs(prediction - image))
        assert len(case["inside_errors"]) == len(case["prediction_trace"])
        assert len(case["outside_errors"]) == len(case["prediction_trace"])
        assert case["rmse_inside_mean"] == pytest.approx(np.mean(case["inside_errors"]))
        assert case["rmse_outside_mean"] == pytest.approx(np.mean(case["outside_errors"]))


def test_playback_and_error_atlas_tile_every_frame_for_every_pattern(tmp_path):
    shape = (4, 4)
    completion.run(
        tmp_path, shape=shape, radius=1, hidden=2, epochs=1, hold_frames=1,
        eval_passes=1, learning_rate=.05, sigma=0., tau=1 / 3, clamp_strength=.5, label="t",
    )
    import json
    from PIL import Image

    summary = json.loads((tmp_path / "t-summary.json").read_text())
    columns = summary["playback_atlas"]["columns"]
    assert summary["playback_atlas"]["rows"] == len(completion.PATTERNS)
    assert summary["error_atlas"]["columns"] == columns
    assert summary["error_atlas"]["scale"] >= 0.
    with Image.open(tmp_path / "t-playback.png") as atlas:
        assert atlas.size == (columns * shape[1], len(completion.PATTERNS) * shape[0])
    with Image.open(tmp_path / "t-errors.png") as atlas:
        assert atlas.size == (columns * shape[1], len(completion.PATTERNS) * shape[0])
        assert atlas.mode == "RGBA"
    for name in completion.PATTERNS:
        assert (tmp_path / f"pattern-{name}.png").exists()
    assert len(summary["fovea_trace"]) == columns


def test_run_grid_covers_every_curated_clamp_and_tau_combination(tmp_path):
    index = completion.run_grid(
        tmp_path, shape=(4, 4), radius=1, hidden=2, epochs=1, hold_frames=1,
        eval_passes=1, learning_rate=.05, sigma=0.,
    )
    labels = {config["label"] for config in index["configs"]}
    assert labels == {
        f"completion-{clamp}-{tau}"
        for clamp in completion.GRID for tau in completion.GRID_TAU
    }
    for config in index["configs"]:
        assert (tmp_path / f"{config['label']}-summary.json").exists()
        assert (tmp_path / f"{config['label']}-playback.png").exists()
        assert (tmp_path / f"{config['label']}-errors.png").exists()


def test_summary_json_matches_the_frontend_contract(tmp_path):
    """The explorer/highlight components read these exact keys; a rename here breaks them silently."""
    import json

    completion.run(
        tmp_path, shape=(4, 4), radius=1, hidden=2, epochs=1, hold_frames=1,
        eval_passes=1, learning_rate=.05, sigma=0., tau=1 / 3, clamp_strength=.5, label="t",
    )
    summary = json.loads((tmp_path / "t-summary.json").read_text())
    assert summary["settings"]["hidden_nodes"] == 2
    assert summary["settings"]["shape"] == [4, 4]
    assert summary["settings"]["foveal_radius"] == 1
    assert set(summary["playback_atlas"]["patterns"]) == set(completion.PATTERNS)
    for name in completion.PATTERNS:
        assert len(summary["hidden_traces"][name]) == len(summary["fovea_trace"])
        assert all(len(vector) == 2 for vector in summary["hidden_traces"][name])
        assert len(summary["error_traces"][name]["inside"]) == len(summary["fovea_trace"])
        assert len(summary["error_traces"][name]["outside"]) == len(summary["fovea_trace"])
    for entry in summary["fovea_trace"]:
        assert len(entry["center"]) == 2
        assert entry["radius"] == 1
