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
