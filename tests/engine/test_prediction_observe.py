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


def _rgb_field(values: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(values, dtype=np.float32)[..., None], 3, axis=2)


def test_ripple_engine_observe_updates_symmetric_prediction_edges():
    engine = _build_engine()
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            activation_function="linear_clipped",
            sigma=1.0,
            learning_enabled=True,
            learning_rate=0.05,
            learning_weight_decay=0.0,
            learning_weight_clip=100.0,
            learning_gradient_clip=100.0,
        ),
        resolution=engine.canvas_shape,
    )
    engine.Z[:] = _rgb_field([[1.0, 2.0], [3.0, 4.0]])
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
        _rgb_field(
            1.0 + 0.05 * np.array([[3.0], [7.0]], dtype=np.float32) / np.log(2.0)
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        engine.prediction_vertical_edge_weights,
        _rgb_field(
            1.0 + 0.05 * np.array([[4.0, 6.0]], dtype=np.float32) / np.log(2.0)
        ),
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
        resolution=engine.canvas_shape,
    )
    engine.Z[:] = _rgb_field([[1.0, 2.0], [3.0, 4.0]])
    observation = engine.Z + 1.0

    engine.observe(
        source_key="camera_frame",
        observation=observation,
        transform=transform,
    )
    np.testing.assert_array_equal(
        engine.prediction_horizontal_edge_weights,
        np.ones((2, 1, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        engine.prediction_vertical_edge_weights,
        np.ones((1, 2, 3), dtype=np.float32),
    )

    transform.update_config(learning_enabled=True)
    engine.observe(
        source_key="camera_frame",
        observation=observation,
        transform=transform,
    )

    assert engine.prediction_horizontal_edge_weights is not None
    assert engine.prediction_vertical_edge_weights is not None
    assert np.any(engine.prediction_horizontal_edge_weights != 1.0)
    assert np.any(engine.prediction_vertical_edge_weights != 1.0)


def test_ripple_engine_predict_observation_reads_canvas_without_graph_aggregation():
    engine = _build_engine()
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            sigma=1.0,
            learning_enabled=False,
            learning_rate=0.0,
            learning_weight_decay=0.0,
            learning_weight_clip=100.0,
            learning_gradient_clip=100.0,
        ),
        resolution=engine.canvas_shape,
    )
    engine.Z[:] = _rgb_field([[1.0, 2.0], [3.0, 4.0]])
    engine.prediction_horizontal_edge_weights = np.array(
        [[1.5], [2.0]],
        dtype=np.float32,
    )
    engine.prediction_vertical_edge_weights = np.array(
        [[1.25, 1.75]],
        dtype=np.float32,
    )

    prediction = engine.predict_observation(transform=transform)

    np.testing.assert_allclose(prediction, engine.Z, rtol=1e-6, atol=1e-6)


def test_ripple_engine_learns_prediction_edges_per_canvas_channel():
    engine = _build_engine()
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            activation_function="linear_clipped",
            sigma=1.0,
            learning_enabled=True,
            learning_rate=0.05,
            learning_weight_decay=0.0,
            learning_weight_clip=100.0,
            learning_gradient_clip=100.0,
        ),
        resolution=engine.canvas_shape,
    )
    engine.Z[..., 0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    observation = engine.Z.copy()
    observation[..., 0] += 1.0

    engine.observe(
        source_key="camera_frame",
        observation=observation,
        transform=transform,
    )

    assert np.any(engine.prediction_horizontal_edge_weights[..., 0] != 1.0)
    assert np.any(engine.prediction_vertical_edge_weights[..., 0] != 1.0)
    np.testing.assert_array_equal(
        engine.prediction_horizontal_edge_weights[..., 1:],
        1.0,
    )
    np.testing.assert_array_equal(
        engine.prediction_vertical_edge_weights[..., 1:],
        1.0,
    )


def test_ripple_engine_normalizes_prediction_conductances_to_stable_degree_budget():
    engine = RippleEngine(
        resolution=(3, 3),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )
    engine.prediction_horizontal_edge_weights = np.full((3, 2), 4.0, dtype=np.float32)
    engine.prediction_vertical_edge_weights = np.full((2, 3), 4.0, dtype=np.float32)

    horizontal = engine.prediction_horizontal_edge_weights
    vertical = engine.prediction_vertical_edge_weights
    degree = np.zeros((3, 3, 3), dtype=np.float32)
    degree[:, :-1] += horizontal
    degree[:, 1:] += horizontal
    degree[:-1, :] += vertical
    degree[1:, :] += vertical

    assert np.all(horizontal >= 0.0)
    assert np.all(vertical >= 0.0)
    assert np.max(degree) <= 4.0 + 1e-6


def test_ripple_engine_aggregates_patch_edge_strengths_from_symmetric_operator():
    engine = RippleEngine(
        resolution=(4, 4),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )
    engine.prediction_vertical_edge_weights = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.8, 0.8, 0.8, 0.8],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    engine.prediction_horizontal_edge_weights = np.array(
        [
            [1.0, 1.05, 1.0],
            [1.0, 1.05, 1.0],
            [1.0, 1.025, 1.0],
            [1.0, 1.025, 1.0],
        ],
        dtype=np.float32,
    )

    segments, strengths = engine.get_prediction_coupling_overlay_edges(
        stride=2,
        threshold=0.0,
    )

    np.testing.assert_allclose(
        segments,
        np.array(
            [
                [[1.0, 1.0], [3.0, 1.0]],
                [[1.0, 3.0], [3.0, 3.0]],
                [[1.0, 1.0], [1.0, 3.0]],
                [[3.0, 1.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        strengths,
        np.array([1.05, 1.025, 0.8, 0.8], dtype=np.float32),
        atol=1e-6,
    )
