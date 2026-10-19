from dataclasses import replace

import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.readouts import GaussianSpatialPrecision, PredictiveCodingConfig
from audioviz.sources import (
    CameraFrameSourceConfig, CircularDotCapture, CircularDotConfig, CircularDotSource,
    RippleSourceOrchestrator, RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)
from audioviz.sources.gaze import GazePolicy
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig


def test_greedy_gaze_uses_xy_coordinates_preserves_maxima_and_holds_flat_scores():
    policy = GazePolicy()
    current = (0.75, 0.25)
    for value in (0.0, 1e-10, 3.0):
        assert policy.select(np.full((3, 5), value), current) == current
    scores = np.zeros((3, 5))
    scores[1, 3] = 2
    assert policy.select(scores, current) == (0.75, 0.5)
    scores[0, 0] = 2
    assert policy.select(scores, (0.75, 0.5)) == (0.75, 0.5)
    assert policy.select(scores, current) == (0.0, 0.0)
    scores[0, 0] += 1e-6
    assert policy.select(scores, (0.75, 0.5)) == (0.75, 0.5)
    with pytest.raises(ValueError, match="finite"):
        policy.select(np.array([[0, np.nan], [0, 0]]), current)
    with pytest.raises(ValueError, match="two rows"):
        policy.select(np.zeros((1, 5)), current)


def test_fixed_and_random_policies_use_independent_reproducible_state():
    scores = np.arange(117, dtype=np.float32).reshape(9, 13)
    current = (0.5, 0.5)
    assert GazePolicy("fixed").select(scores, current) == current
    first, second = GazePolicy("random", seed=7), GazePolicy("random", seed=7)
    positions = [first.select(scores, current) for _ in range(8)]
    np.random.default_rng(500).normal(size=100)
    assert positions == [second.select(scores, current) for _ in range(8)]
    assert len(set(positions)) > 1
    assert all(
        x * 12 == pytest.approx(round(x * 12)) and y * 8 == pytest.approx(round(y * 8))
        for x, y in positions
    )
    with pytest.raises(ValueError, match="gaze policy must"):
        GazePolicy("truth")


def _orchestrator(seed, *, sample=True):
    engine = RippleEngine(resolution=(12, 16), plane_size_m=(1, 1))
    stimulus = CircularDotSource(CircularDotConfig(
        resolution=engine.resolution, radius=4, dot_radius=2,
        seed=seed, brownian_scale=8, frame_rate=10, duration=0.4,
    ))
    capture = CircularDotCapture(stimulus)
    orchestrator = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False),
            camera_frame=CameraFrameSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(
                dtype="float32", inference_steps=2,
                spatial_precision=GaussianSpatialPrecision(sample=sample),
            ),
        ),
        processor=None, engine=engine, resolution=engine.resolution,
        n_sources=1, camera_capture=capture,
    )
    orchestrator.enable_visual_uncertainty(dt=0.1)
    orchestrator.gaze_policy = GazePolicy()
    return orchestrator, stimulus, capture


def test_policy_runs_before_current_camera_frame_and_freezes_when_observations_pause(monkeypatch):
    orchestrator, stimulus, capture = _orchestrator(7)

    def ordinary_pc(*args, **kwargs):
        pytest.fail("temporal Gaussian frames must not run ordinary PC prediction or inference")

    monkeypatch.setattr(orchestrator.visual_readout, "predict", ordinary_pc)
    monkeypatch.setattr(orchestrator.visual_readout, "infer", ordinary_pc)
    scores = np.zeros((3, 5), dtype=np.float32)
    scores[1, 3] = 2
    calls = []

    def score(precision, *, prior, dt):
        assert stimulus.latest is None
        assert not orchestrator.visual_uncertainty._pending
        assert dt == 0.1 and prior is orchestrator.engine.get_prior()
        calls.append(precision)
        return scores

    monkeypatch.setattr(orchestrator.visual_uncertainty, "expected_information_gain", score)
    try:
        orchestrator.engine.propagate()
        orchestrator.predict_visual()
        assert orchestrator.visual_readout.config.spatial_precision.gaze == (0.75, 0.5)
        assert stimulus.latest is None
        orchestrator.correct_from_observations()
        assert stimulus.latest.time == 0
        orchestrator.update_pathway_config(spatial_precision=replace(calls[0], gaze=(0.75, 0.5), sample=False))
        previous = orchestrator.visual_uncertainty.variance.copy()
        orchestrator.engine.propagate()
        orchestrator.predict_visual()
        orchestrator.correct_from_observations()
        assert len(calls) == 1 and orchestrator.gaze_scores is None
        assert orchestrator.visual_readout.config.spatial_precision.gaze == (0.75, 0.5)
        model = orchestrator.visual_uncertainty
        factor = np.exp(-.2/model.memory_time_constant)
        np.testing.assert_allclose(
            model.variance, factor*previous+model.process_variance*model.memory_time_constant/2*(1-factor),
            rtol=2e-6,
        )
    finally:
        capture.release()


def test_withheld_frames_do_not_leak_stimulus_truth_into_next_policy_choice():
    runs = [_orchestrator(seed, sample=False) for seed in (7, 89)]
    try:
        for orchestrator, _, _ in runs:
            for _ in range(2):
                orchestrator.engine.propagate()
                orchestrator.predict_visual()
                orchestrator.correct_from_observations()
            precision = orchestrator.visual_readout.config.spatial_precision
            orchestrator.update_pathway_config(spatial_precision=replace(precision, sample=True))
            orchestrator.engine.propagate()
            orchestrator.predict_visual()
        left, right = runs[0][0], runs[1][0]
        assert not np.array_equal(runs[0][1].latest.frame, runs[1][1].latest.frame)
        np.testing.assert_array_equal(left.gaze_scores, right.gaze_scores)
        assert left.visual_readout.config.spatial_precision.gaze == right.visual_readout.config.spatial_precision.gaze
        for orchestrator, _, _ in runs:
            orchestrator.correct_from_observations()
    finally:
        for _, _, capture in runs:
            capture.release()
