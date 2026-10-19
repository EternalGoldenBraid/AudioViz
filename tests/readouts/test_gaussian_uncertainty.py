from dataclasses import replace

import numpy as np
import pytest

from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.spatial_precision import GaussianSpatialPrecision
from audioviz.utils.array_backend import to_numpy


def identity_readout(shape=(3, 5), *, backend=np, rate=0):
    readout = PredictiveCodingRGBReadout(
        canvas_shape=(*shape, 3), backend=backend,
        config=PredictiveCodingConfig(hidden_channels=3, inference_rate=rate, dtype="float32"),
    )
    readout.visual_weights[:] = backend.eye(3)
    readout.canvas_weights[:] = backend.eye(3)
    return readout


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_gaussian_gain_equals_entropy_reduction_and_skip_preserves_uncertainty(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    readout = identity_readout(backend=xp)
    prior = xp.zeros(readout.canvas_shape, dtype=xp.float32)
    model = GaussianSensoryUncertainty(readout, initial_variance=4, process_variance=0, backend=xp)
    measurement = GaussianSpatialPrecision(baseline=1, peak=0)
    scores = model.expected_information_gain(measurement, prior=prior, dt=0, grid_shape=(3, 5))
    np.testing.assert_allclose(scores, 1.5*15*np.log(5), rtol=2e-6)
    readout.predict(prior)
    model.predict(dt=0)
    model.condition(measurement, observed=True)
    posterior = to_numpy(model.variance).copy()
    np.testing.assert_allclose(posterior, 1/(1/4+1), rtol=2e-6)
    assert scores[0, 0] == pytest.approx(float(0.5*np.log(4/posterior).sum()), rel=2e-6)
    assert model.expected_information_gain(measurement, prior=prior, dt=0).max() < scores.min()
    skipped = replace(measurement, sample=False)
    for observed in (True, False):
        model.predict(dt=1)
        model.condition(skipped if observed else measurement, observed=observed)
        np.testing.assert_array_equal(to_numpy(model.variance), posterior)
    np.testing.assert_array_equal(model.expected_information_gain(skipped, prior=prior, dt=0), 0)


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_model_transition_and_frozen_decoder_match_dense_gaussian_reference(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    readout = identity_readout((1, 1), backend=xp, rate=0.2)
    weights = np.array([[1, .4, -.2], [.3, 1, .5], [.1, -.3, .7]], dtype=np.float32)
    readout.visual_weights[:] = xp.asarray(weights)
    readout.hidden[:] = xp.asarray([.2, -.3, .5])
    prior = xp.asarray([[[.5, -.2, .1]]], dtype=xp.float32)
    model = GaussianSensoryUncertainty(readout, initial_variance=2, process_variance=.04, backend=xp)
    readout.predict(prior)
    frozen = to_numpy(readout.hidden)[0, 0].copy()
    model.predict(dt=.25)
    variance = 2*.8**2+.04*.25
    np.testing.assert_allclose(to_numpy(model.variance), variance, rtol=2e-6)
    j = weights*(1-np.tanh(frozen)**2)
    dense_prior = variance*np.eye(3)
    dense_posterior = dense_prior-dense_prior@j.T@np.linalg.inv(np.eye(3)+j@dense_prior@j.T)@j@dense_prior
    readout.hidden += .6
    model.condition(None, observed=True)
    np.testing.assert_allclose(to_numpy(model.variance)[0, 0], dense_posterior.diagonal(), rtol=2e-6)

    forecast = .8*to_numpy(readout.hidden)[0, 0]+.2*np.tanh(to_numpy(prior)[0, 0])
    j_next = weights*(1-np.tanh(forecast)**2)
    p_next = .8**2*dense_posterior.diagonal()+.04*.25
    sensory_covariance = (j_next*p_next)@j_next.T
    assert abs(sensory_covariance[0, 1]) > .01
    _, logdet = np.linalg.slogdet(np.eye(3)+sensory_covariance)
    measurement = GaussianSpatialPrecision(baseline=1, peak=0)
    scores = model.expected_information_gain(measurement, prior=prior, dt=.25)
    np.testing.assert_allclose(scores, .5*logdet, rtol=3e-6)
    before = to_numpy(model.variance).copy()
    readout.visual_weights.fill(0)
    np.testing.assert_array_equal(model.expected_information_gain(measurement, prior=prior, dt=.25), 0)
    np.testing.assert_array_equal(to_numpy(model.variance), before)


def test_gaze_reduces_local_uncertainty_and_unobserved_prediction_follows_process_model():
    readout = identity_readout((19, 23))
    prior = np.zeros(readout.canvas_shape, dtype=np.float32)
    model = GaussianSensoryUncertainty(readout, process_variance=.01)
    measurement = GaussianSpatialPrecision(gaze=(.2, .5), sigma=.15, baseline=0, peak=4)
    for _ in range(40):
        model.predict(dt=.1)
        model.condition(measurement, observed=True)
    scores = model.expected_information_gain(measurement, prior=prior, dt=.1, grid_shape=(5, 7))
    assert model.variance[9, 4].sum() < model.variance[9, 18].sum()
    assert scores[2, 1] < scores[2, 5]
    before = model.variance.copy()
    model.predict(dt=2)
    model.condition(measurement, observed=False)
    np.testing.assert_allclose(model.variance, before+.02)
    model.reset()
    np.testing.assert_array_equal(model.variance, 1)


@pytest.mark.parametrize("changes", [
    {"initial_variance": 0}, {"initial_variance": np.inf}, {"initial_variance": 1e-50},
    {"process_variance": -1}, {"process_variance": np.nan},
])
def test_invalid_uncertainty_assumptions_are_rejected(changes):
    with pytest.raises(ValueError):
        GaussianSensoryUncertainty(identity_readout(), **changes)


@pytest.mark.parametrize("changes", [{"learning_enabled": True}, {"cross_modal_enabled": True}])
def test_unrepresented_model_uncertainty_is_rejected(changes):
    readout = identity_readout()
    readout.update_config(**changes)
    with pytest.raises(ValueError, match="uncoupled, fixed-weight"):
        GaussianSensoryUncertainty(readout)


def test_bounded_quadrature_preserves_total_gain_for_constant_precision():
    readout = identity_readout((1080, 1920))
    model = GaussianSensoryUncertainty(readout, initial_variance=.25)
    scores = model.expected_information_gain(
        GaussianSpatialPrecision(baseline=1, peak=0),
        prior=np.zeros(readout.canvas_shape, dtype=np.float32), dt=0, max_side=16,
    )
    np.testing.assert_allclose(scores, 1.5*1080*1920*np.log1p(.25), rtol=2e-6)


def test_forecast_conditioning_boundary_and_invalid_planning_inputs():
    readout = identity_readout()
    model = GaussianSensoryUncertainty(readout)
    prior = np.zeros(readout.canvas_shape, dtype=np.float32)
    measurement = GaussianSpatialPrecision()
    with pytest.raises(RuntimeError, match="pre-evidence"):
        model.condition(measurement, observed=True)
    for dt in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="timestep"):
            model.predict(dt=dt)
    with pytest.raises(ValueError, match="candidate grid"):
        model.expected_information_gain(measurement, prior=prior, dt=0, grid_shape=(1, 3))
    model.predict(dt=.1)
    with pytest.raises(RuntimeError, match="before predicting again"):
        model.predict(dt=.1)
    with pytest.raises(RuntimeError, match="before scoring"):
        model.expected_information_gain(measurement, prior=prior, dt=.1)
    model.condition(measurement, observed=False)
    assert np.isfinite(model.expected_information_gain(measurement, prior=prior, dt=.1)).all()
