from dataclasses import replace

import numpy as np
import pytest

from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout


def _readout(**changes):
    return PredictiveCodingRGBReadout(
        canvas_shape=(1, 1, 3),
        config=PredictiveCodingConfig(hidden_channels=2, **changes),
    )


def _numeric_gradient(values, energy):
    gradient = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + 1e-5
        upper = energy()
        values[index] = original - 1e-5
        lower = energy()
        values[index] = original
        gradient[index] = (upper - lower) / 2e-5
    return gradient


def test_nonlinear_state_updates_match_negative_energy_gradient():
    readout = _readout(inference_steps=1, inference_rate=0.001, canvas_rate=0.001)
    canvas = np.array([[[0.7, -0.4, 0.2]]])
    observation = np.array([[[0.1, 0.8, -0.2]]])
    readout.hidden[:] = [[[0.5, -0.3]]]
    canvas_before = canvas.copy()
    hidden_before = readout.hidden.copy()
    hidden_gradient = _numeric_gradient(
        readout.hidden, lambda: readout.energy(canvas, observation)
    )
    canvas_gradient = _numeric_gradient(
        canvas, lambda: readout.energy(canvas, observation)
    )

    correction = readout.infer(canvas, observation)

    np.testing.assert_allclose(correction / 0.001, -canvas_gradient, rtol=1e-6)
    np.testing.assert_allclose(
        (readout.hidden - hidden_before) / 0.001, -hidden_gradient, rtol=1e-6
    )
    np.testing.assert_array_equal(canvas, canvas_before)


def test_nonlinear_weight_updates_match_negative_energy_gradient():
    readout = _readout(
        inference_steps=1,
        inference_rate=0.0,
        canvas_rate=0.0,
        learning_enabled=True,
        learning_rate=0.001,
        gradient_clip=100.0,
    )
    canvas = np.array([[[0.7, -0.4, 0.2]]])
    observation = np.array([[[0.1, 0.8, -0.2]]])
    readout.hidden[:] = [[[0.5, -0.3]]]
    weights_before = [readout.canvas_weights.copy(), readout.visual_weights.copy()]
    gradients = [
        _numeric_gradient(weights, lambda: readout.energy(canvas, observation))
        for weights in (readout.canvas_weights, readout.visual_weights)
    ]

    readout.infer(canvas, observation)

    for weights, before, gradient in zip(
        (readout.canvas_weights, readout.visual_weights), weights_before, gradients
    ):
        np.testing.assert_allclose((weights - before) / 0.001, -gradient, rtol=1e-6)


def test_inference_reduces_energy_with_fixed_weights():
    readout = _readout(inference_steps=30)
    canvas = np.zeros((1, 1, 3))
    observation = np.array([[[0.7, 0.4, 0.2]]])
    readout.predict(canvas)
    energy_before = readout.energy(canvas, observation)
    weights_before = readout.canvas_weights.copy(), readout.visual_weights.copy()

    correction = readout.infer(canvas, observation)

    assert readout.energy(canvas + correction, observation) < energy_before
    assert np.any(readout.hidden != 0.0)
    assert np.any(correction != 0.0)
    np.testing.assert_array_equal(readout.canvas_weights, weights_before[0])
    np.testing.assert_array_equal(readout.visual_weights, weights_before[1])


def test_predict_advances_memory_without_observing_camera():
    readout = _readout()
    canvas = np.ones((1, 1, 3))
    readout.predict(canvas)
    first_hidden = readout.hidden.copy()

    prediction = readout.predict(np.zeros_like(canvas))

    np.testing.assert_allclose(
        readout.hidden, (1.0 - readout.config.inference_rate) * first_hidden
    )
    assert np.any(prediction != 0.0)


def test_reset_clears_states_and_errors_but_preserves_weights():
    readout = _readout(learning_enabled=True)
    readout.predict(np.ones((1, 1, 3)))
    readout.infer(np.ones((1, 1, 3)), np.zeros((1, 1, 3)))
    weights_before = readout.canvas_weights.copy(), readout.visual_weights.copy()

    readout.reset()

    np.testing.assert_array_equal(readout.hidden, 0.0)
    np.testing.assert_array_equal(readout.hidden_error, 0.0)
    np.testing.assert_array_equal(readout.visual_error, 0.0)
    np.testing.assert_array_equal(readout.canvas_weights, weights_before[0])
    np.testing.assert_array_equal(readout.visual_weights, weights_before[1])


def test_learning_toggle_takes_effect_without_resetting_memory():
    readout = _readout()
    canvas = np.full((1, 1, 3), 0.2)
    observation = np.full((1, 1, 3), 0.7)
    readout.predict(canvas)
    readout.infer(canvas, observation)
    weights_before = readout.visual_weights.copy()
    hidden_before = readout.hidden.copy()

    readout.update_config(learning_enabled=True, learning_rate=0.1)
    np.testing.assert_array_equal(readout.hidden, hidden_before)
    readout.infer(canvas, observation)

    assert np.any(readout.visual_weights != weights_before)


def test_shared_weight_update_is_independent_of_image_size():
    config = PredictiveCodingConfig(
        learning_enabled=True, inference_rate=0.0, canvas_rate=0.0
    )
    small = PredictiveCodingRGBReadout(canvas_shape=(1, 1, 3), config=config)
    large = PredictiveCodingRGBReadout(canvas_shape=(3, 4, 3), config=config)
    for readout in (small, large):
        readout.hidden.fill(0.3)
        readout.infer(
            np.full(readout.canvas_shape, 0.2),
            np.full(readout.canvas_shape, 0.7),
        )

    np.testing.assert_allclose(small.canvas_weights, large.canvas_weights)
    np.testing.assert_allclose(small.visual_weights, large.visual_weights)


def test_local_learning_improves_pre_observation_prediction_over_frozen_weights():
    errors = []
    for learning_enabled in (False, True):
        readout = PredictiveCodingRGBReadout(
            canvas_shape=(2, 2, 3),
            config=PredictiveCodingConfig(
                learning_enabled=learning_enabled, learning_rate=0.05
            ),
        )
        canvas = np.zeros((2, 2, 3), dtype=np.float32)
        observation = np.broadcast_to([0.8, 0.3, 0.5], canvas.shape)
        for _ in range(80):
            prediction = readout.predict(canvas)
            canvas += readout.infer(canvas, observation)
        errors.append(np.mean((observation - prediction) ** 2))

    assert errors[1] < 1e-4
    assert errors[1] < 0.1 * errors[0]


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_channels": 0},
        {"inference_steps": 1.5},
        {"inference_steps": True},
        {"inference_rate": -0.1},
        {"canvas_rate": np.nan},
        {"learning_rate": np.inf},
        {"weight_decay": -1.0},
        {"gradient_clip": 0.0},
        {"weight_clip": np.inf},
        {"cross_modal_enabled": 1},
        {"stream_state_enabled": 1},
    ],
)
def test_config_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        replace(PredictiveCodingConfig(), **changes)


def test_invalid_config_update_does_not_replace_previous_config():
    readout = _readout()
    config_before = readout.config
    with pytest.raises(ValueError):
        readout.update_config(inference_steps=0)
    with pytest.raises(ValueError, match="rebuilding"):
        readout.update_config(hidden_channels=4)
    assert readout.config is config_before


def test_readout_rejects_wrong_shape_and_nonfinite_observations():
    readout = _readout()
    with pytest.raises(ValueError, match="shape"):
        readout.predict(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="finite"):
        readout.infer(np.zeros((1, 1, 3)), np.full((1, 1, 3), np.nan))
