from dataclasses import replace

import numpy as np
import pytest

from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.spatial_precision import GaussianSpatialPrecision
from audioviz.utils.array_backend import to_numpy


def test_gaussian_center_width_boundaries_and_normalized_likelihood():
    precision = GaussianSpatialPrecision(gaze=(0.5, 0.5), sigma=0.2, baseline=0.1, peak=2)
    field = precision.field((5, 9), dtype=np.float64)
    assert field[2, 4] == pytest.approx(2.1)
    assert field[2, 5] == pytest.approx(0.1 + 2 * np.exp(-0.5))
    np.testing.assert_array_equal(field, field[::-1, ::-1])
    assert (field > 0).all()
    edge = replace(precision, gaze=(0, 1)).field((5, 9))
    assert edge.argmax() == 4 * 9

    observation, prediction = np.ones((5, 9, 3)), np.zeros((5, 9, 3))
    expected = 0.5 * 3 * np.sum(field + np.log(2 * np.pi) - np.log(field))
    assert precision.negative_log_likelihood(observation, prediction) == pytest.approx(expected)
    delta = 1e-5
    prediction[2, 4, 0] = delta
    upper = precision.negative_log_likelihood(observation, prediction)
    prediction[2, 4, 0] = -delta
    lower = precision.negative_log_likelihood(observation, prediction)
    assert (upper - lower) / (2 * delta) == pytest.approx(-field[2, 4])
    skipped = replace(precision, sample=False)
    assert not skipped.field((5, 9)).any()
    assert skipped.negative_log_likelihood(observation, prediction) == 0
    assert GaussianSpatialPrecision(baseline=0, peak=0).negative_log_likelihood(observation, prediction) == 0


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_precision_weights_inference_learning_and_energy_once_and_skip_blocks_evidence(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    precision = GaussianSpatialPrecision(gaze=(0, 0.5), sigma=0.2, baseline=0.01, peak=1)
    config = PredictiveCodingConfig(
        hidden_channels=3, inference_steps=1, inference_rate=0.1,
        learning_enabled=True, learning_rate=0.01, spatial_precision=precision,
    )
    readout = PredictiveCodingRGBReadout(canvas_shape=(5, 9, 3), config=config, backend=xp)
    readout.canvas_weights[:] = xp.eye(3)
    readout.visual_weights[:] = xp.eye(3)
    readout.set_diagnostics_enabled(True)
    prior, observation = xp.zeros((5, 9, 3)), xp.ones((5, 9, 3))
    readout.infer(prior, observation)
    field = precision.field((5, 9), dtype=np.float64)[..., None]
    hidden = np.broadcast_to(0.1 * field, (5, 9, 3))
    np.testing.assert_allclose(to_numpy(readout.hidden), hidden)
    residual = 1 - np.tanh(hidden)
    expected_gradient = (field * residual).reshape(-1, 3).T @ np.tanh(hidden).reshape(-1, 3) / 45
    np.testing.assert_allclose(to_numpy(readout.visual_weights), np.eye(3) + 0.01 * expected_gradient)
    assert readout.diagnostics.inference_energy[0] == pytest.approx(0.5 * 3 * field.mean())
    assert readout.diagnostics.visual.mean == pytest.approx((0.5 * field * residual**2).mean())
    assert readout.diagnostics.visual.variance == pytest.approx((0.5 * field * residual**2).var())

    readout.reset()
    moved = replace(precision, gaze=(1, 0.5))
    readout.update_config(spatial_precision=moved, learning_enabled=False)
    weights = to_numpy(readout.visual_weights).copy()
    readout.infer(prior, observation)
    expected_hidden = 0.1 * moved.field((5, 9), dtype=np.float64)[..., None] * (np.ones((5, 9, 3)) @ weights)
    np.testing.assert_allclose(to_numpy(readout.hidden), expected_hidden)

    readout.reset()
    readout.update_config(spatial_precision=replace(precision, sample=False), learning_enabled=True)
    readout.infer(prior, observation)
    np.testing.assert_array_equal(to_numpy(readout.hidden), 0)
    np.testing.assert_array_equal(to_numpy(readout.visual_weights), weights)
    assert readout.diagnostics.inference_energy == (0, 0)


def test_uniform_unit_precision_preserves_original_inference():
    config = PredictiveCodingConfig(inference_steps=3, learning_enabled=True)
    plain = PredictiveCodingRGBReadout(canvas_shape=(5, 9, 3), config=config)
    explicit = PredictiveCodingRGBReadout(
        canvas_shape=(5, 9, 3),
        config=replace(config, spatial_precision=GaussianSpatialPrecision(baseline=1, peak=0)),
    )
    prior = np.full((5, 9, 3), 0.2)
    observation = np.full_like(prior, 0.7)
    np.testing.assert_array_equal(plain.infer(prior, observation), explicit.infer(prior, observation))
    np.testing.assert_array_equal(plain.visual_weights, explicit.visual_weights)
    assert plain.energy(prior, observation) == explicit.energy(prior, observation)


@pytest.mark.parametrize("changes", [
    {"gaze": (-0.1, 0.5)}, {"gaze": (0.5, np.nan)},
    {"sigma": 0}, {"sigma": np.inf}, {"baseline": -1}, {"peak": np.nan}, {"sample": 1},
])
def test_invalid_precision_is_rejected(changes):
    with pytest.raises(ValueError):
        GaussianSpatialPrecision(**changes)
