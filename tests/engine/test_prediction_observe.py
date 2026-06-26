import numpy as np

from audioviz.engine import RippleEngine
from audioviz.transforms.prediction_error import (
    PredictionErrorTransform,
    PredictionErrorTransformConfig,
)


def _build_engine() -> RippleEngine:
    return RippleEngine(
        resolution=(2, 2),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )


def test_ripple_engine_observe_updates_symmetric_prediction_edges():
    engine = _build_engine()
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            output_mode="squared_error",
            activation_function="linear_clipped",
            sigma=1.0,
            learning_enabled=True,
            learning_rate=0.5,
            learning_weight_decay=0.0,
            learning_weight_clip=100.0,
            learning_gradient_clip=100.0,
        ),
        resolution=(2, 2),
    )
    engine.Z[:] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    observation = engine.Z + np.ones_like(engine.Z)

    engine.observe(
        source_key="camera_frame",
        observation=observation,
        transform=transform,
    )

    assert engine.prediction_horizontal_edge_weights is not None
    assert engine.prediction_vertical_edge_weights is not None
    np.testing.assert_allclose(
        engine.prediction_horizontal_edge_weights,
        0.5 * np.array([[3.0], [7.0]], dtype=np.float32) / np.log(2.0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        engine.prediction_vertical_edge_weights,
        0.5 * np.array([[4.0, 6.0]], dtype=np.float32) / np.log(2.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_ripple_engine_observe_learning_toggle_takes_effect_on_next_observation():
    engine = _build_engine()
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            sigma=1.0,
            learning_enabled=False,
            learning_rate=0.5,
            learning_weight_decay=0.0,
            learning_weight_clip=100.0,
            learning_gradient_clip=100.0,
        ),
        resolution=(2, 2),
    )
    engine.Z[:] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    observation = engine.Z + 1.0

    engine.observe(
        source_key="camera_frame",
        observation=observation,
        transform=transform,
    )
    assert engine.prediction_horizontal_edge_weights is None
    assert engine.prediction_vertical_edge_weights is None

    transform.update_config(learning_enabled=True)
    engine.observe(
        source_key="camera_frame",
        observation=observation,
        transform=transform,
    )

    assert engine.prediction_horizontal_edge_weights is not None
    assert engine.prediction_vertical_edge_weights is not None
    assert np.any(engine.prediction_horizontal_edge_weights != 0.0)
    assert np.any(engine.prediction_vertical_edge_weights != 0.0)


def test_ripple_engine_predict_observation_uses_learned_symmetric_edges():
    engine = _build_engine()
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            sigma=1.0,
            learning_enabled=True,
            learning_rate=0.0,
            learning_weight_decay=0.0,
            learning_weight_clip=100.0,
            learning_gradient_clip=100.0,
        ),
        resolution=(2, 2),
    )
    engine.Z[:] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    engine.prediction_horizontal_edge_weights = np.array(
        [[0.5], [1.0]],
        dtype=np.float32,
    )
    engine.prediction_vertical_edge_weights = np.array(
        [[0.25, 0.75]],
        dtype=np.float32,
    )

    prediction = engine.predict_observation(transform=transform)

    expected = np.array(
        [
            [1.0 + 0.5 * 2.0 + 0.25 * 3.0, 2.0 + 0.5 * 1.0 + 0.75 * 4.0],
            [3.0 + 1.0 * 4.0 + 0.25 * 1.0, 4.0 + 1.0 * 3.0 + 0.75 * 2.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-6)
