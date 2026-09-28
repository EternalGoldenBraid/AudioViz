import numpy as np
import pytest

from audioviz.transforms.prediction_error import (
    PredictionErrorTransform,
    PredictionErrorTransformConfig,
)


def test_prediction_error_transform_outputs_signed_raw_error():
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
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

    np.testing.assert_allclose(
        transformed,
        observation - ripple_state,
        rtol=1e-6,
        atol=1e-6,
    )


def test_prediction_error_transform_applies_softsign_activation():
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
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

    expected = 5.0 * observation / (2.0 + np.abs(observation))
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-6)


def test_prediction_error_transform_clips_runaway_values():
    transform = PredictionErrorTransform(
        config=PredictionErrorTransformConfig(
            enabled=True,
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
    assert np.min(transformed) >= -3.0
    assert np.max(transformed) <= 3.0


def test_prediction_neighbor_fields_preserves_rgb_channel_axis():
    field = np.arange(18, dtype=np.float32).reshape(2, 3, 3)

    neighbors = PredictionErrorTransform.neighbor_fields(field)

    assert neighbors.shape == (4, 2, 3, 3)
    np.testing.assert_array_equal(neighbors[0, 1, 2], field[0, 2])
    np.testing.assert_array_equal(neighbors[1, 0, 0], field[1, 0])


def test_prediction_error_transform_rejects_invalid_config():
    with pytest.raises(ValueError, match="prediction_error_sigma"):
        PredictionErrorTransform(
            config=PredictionErrorTransformConfig(
                enabled=True,
                sigma=0.0,
            ),
            resolution=(2, 2),
        )
