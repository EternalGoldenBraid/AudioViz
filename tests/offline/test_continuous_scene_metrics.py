import numpy as np
import pytest

from audioviz.offline.continuous_scene_metrics import (
    ContinuousSceneWorldMetrics,
    measure_continuous_scene_step,
    posterior_visual_prediction,
)
from audioviz.readouts.continuous_scene_predictive_coding import (
    ContinuousScenePredictiveCodingGraph,
)
from audioviz.readouts.predictive_coding import PredictiveCodingConfig


def test_measure_continuous_scene_step_and_posterior_prediction():
    rows, cols = 8, 10
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(rows, cols),
        audio_nodes=4,
        hidden_nodes=6,
        config=PredictiveCodingConfig(
            hidden_channels=6,
            inference_steps=6,
            inference_rate=0.1,
            learning_enabled=True,
            learning_rate=0.02,
        ),
    )
    canvas = np.zeros((rows, cols, 3), dtype=np.float64)
    target = np.full((rows, cols, 3), 0.8, dtype=np.float64)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[2:6, 3:7] = True
    dt = 1.0 / 30.0

    with pytest.raises(RuntimeError, match="posterior"):
        graph.forecast(canvas, dt)
        measure_continuous_scene_step(graph, step=1, dt=dt, gaze=(0.5, 0.5))

    graph.condition(target, visual_mask=mask)
    metrics = measure_continuous_scene_step(graph, step=1, dt=dt, gaze=(0.5, 0.5))
    assert isinstance(metrics, ContinuousSceneWorldMetrics)
    assert metrics.step == 1
    assert metrics.time == pytest.approx(dt)
    assert metrics.observed_rgb == int(mask.sum()) * 3
    assert metrics.forecast_mse is not None and metrics.settled_mse is not None
    assert metrics.settled_mse < metrics.forecast_mse
    assert metrics.energy_final < metrics.energy_initial
    assert metrics.energy_final == pytest.approx(
        metrics.hidden_energy + metrics.sensory_energy, abs=1e-14,
    )
    assert metrics.decoded_hidden_norm > 0.0
    assert metrics.visual_potential_norm > 0.0
    assert metrics.neutral_visual_mae == pytest.approx(0.0)
    assert metrics.unobserved_neutral_visual_mae == pytest.approx(0.0)
    assert metrics.learning_updates == 1

    post_pred = posterior_visual_prediction(graph)
    assert post_pred.shape == (rows, cols, 3)
    assert np.mean(post_pred[mask]) > 0.5

    # Next step: prior has non-zero distance from neutral 0.5; full-mask observation sets unobserved MAE to None
    graph.forecast(canvas, dt)
    graph.condition(target, visual_mask=np.ones((rows, cols), dtype=bool))
    full_metrics = measure_continuous_scene_step(graph, step=2, dt=dt, gaze=(0.5, 0.5))
    assert full_metrics.neutral_visual_mae > 0.0
    assert full_metrics.unobserved_neutral_visual_mae is None

    # Observe-off step sets forecast_mse and settled_mse to None while reporting neutral MAEs
    graph.forecast(canvas, dt)
    graph.condition(None, visual_mask=np.zeros((rows, cols), dtype=bool))
    unobs_metrics = measure_continuous_scene_step(graph, step=3, dt=dt, gaze=(0.5, 0.5))
    assert unobs_metrics.observed_rgb == 0
    assert unobs_metrics.forecast_mse is None and unobs_metrics.settled_mse is None
    assert unobs_metrics.unobserved_neutral_visual_mae == pytest.approx(unobs_metrics.neutral_visual_mae)


def test_measure_continuous_scene_step_validates_inputs():
    graph = ContinuousScenePredictiveCodingGraph(visual_shape=(4, 4))
    graph.forecast(np.zeros((4, 4, 3)), 0.05)
    graph.condition(np.zeros((4, 4, 3)))

    with pytest.raises(ValueError, match="ContinuousScenePredictiveCodingGraph"):
        measure_continuous_scene_step(object(), step=0, dt=0.05, gaze=(0.5, 0.5))
    with pytest.raises(ValueError, match="step"):
        measure_continuous_scene_step(graph, step=-1, dt=0.05, gaze=(0.5, 0.5))
    with pytest.raises(ValueError, match="step"):
        measure_continuous_scene_step(graph, step=True, dt=0.05, gaze=(0.5, 0.5))
    with pytest.raises(ValueError, match="dt"):
        measure_continuous_scene_step(graph, step=0, dt=0.0, gaze=(0.5, 0.5))
    with pytest.raises(ValueError, match="gaze"):
        measure_continuous_scene_step(graph, step=0, dt=0.05, gaze=(1.5, 0.5))
