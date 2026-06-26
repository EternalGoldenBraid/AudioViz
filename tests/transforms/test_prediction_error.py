import numpy as np
import pytest

from audioviz.transforms.prediction_error import (
    PredictionErrorTransform,
    PredictionErrorTransformConfig,
)


def test_prediction_error_transform_outputs_gaussian_bits():
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            sigma=0.1,
            output_mode="bits",
            activation_function="linear_clipped",
        ),
        resolution=(2, 2),
    )
    ripple_state = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    observation = np.full((2, 2), 0.2, dtype=np.float32)

    transformed = transform.apply(
        source_key="camera_frame",
        observation=observation,
        ripple_state=ripple_state,
    )

    expected_error = observation - ripple_state
    expected = (expected_error * expected_error) / (2.0 * 0.1 * 0.1 * np.log(2.0))
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)


def test_prediction_error_transform_applies_softsign_activation():
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            output_mode="squared_error",
            activation_function="softsign",
            activation_scale=2.0,
            max_output=5.0,
        ),
        resolution=(1, 3),
    )
    observation = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    ripple_state = np.zeros((1, 3), dtype=np.float32)

    transformed = transform.apply(
        source_key="camera_frame",
        observation=observation,
        ripple_state=ripple_state,
    )

    squared_error = observation * observation
    expected = 5.0 * squared_error / (2.0 + squared_error)
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)


def test_prediction_error_learning_updates_edge_weights():
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
    ripple_state = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    observation = ripple_state + np.ones_like(ripple_state)

    transform.apply(
        source_key="camera_frame",
        observation=observation,
        ripple_state=ripple_state,
    )

    assert transform.horizontal_edge_weights is not None
    assert transform.vertical_edge_weights is not None
    np.testing.assert_allclose(
        transform.horizontal_edge_weights,
        0.5 * np.array([[3.0], [7.0]], dtype=np.float32) / np.log(2.0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transform.vertical_edge_weights,
        0.5 * np.array([[4.0, 6.0]], dtype=np.float32) / np.log(2.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_prediction_error_transform_clips_runaway_values():
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
            sigma=0.1,
            output_mode="bits",
            prediction_clip=5.0,
            max_output=3.0,
        ),
        resolution=(2, 2),
    )
    ripple_state = np.array(
        [[1e30, np.inf], [np.nan, -np.inf]],
        dtype=np.float32,
    )
    observation = np.array(
        [[0.0, np.inf], [np.nan, -np.inf]],
        dtype=np.float32,
    )

    with np.errstate(all="raise"):
        transformed = transform.apply(
            source_key="camera_frame",
            observation=observation,
            ripple_state=ripple_state,
        )

    assert np.all(np.isfinite(transformed))
    assert np.min(transformed) >= 0.0
    assert np.max(transformed) <= 3.0


def test_prediction_error_transform_rejects_invalid_config():
    with pytest.raises(ValueError, match="prediction_error_sigma"):
        PredictionErrorTransform(
            config=PredictionErrorTransformConfig(
                enabled=True,
                sigma=0.0,
            ),
            resolution=(2, 2),
        )


def test_prediction_error_transform_learning_toggle_takes_effect_on_next_apply():
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
    ripple_state = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    observation = ripple_state + 1.0

    transform.apply(
        source_key="camera_frame",
        observation=observation,
        ripple_state=ripple_state,
    )
    assert transform.horizontal_edge_weights is None
    assert transform.vertical_edge_weights is None

    transform.update_config(learning_enabled=True)
    transform.apply(
        source_key="camera_frame",
        observation=observation,
        ripple_state=ripple_state,
    )

    assert transform.horizontal_edge_weights is not None
    assert transform.vertical_edge_weights is not None
    assert np.any(transform.horizontal_edge_weights != 0.0)
    assert np.any(transform.vertical_edge_weights != 0.0)
