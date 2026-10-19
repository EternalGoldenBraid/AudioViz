from dataclasses import replace

import numpy as np
import pytest

from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.spatial_precision import GaussianSpatialPrecision
from audioviz.sources.gaze import information_gaze
from audioviz.utils.array_backend import to_numpy


def identity_readout(shape=(3, 5), *, backend=np, rate=0, dtype="float32"):
    readout = PredictiveCodingRGBReadout(
        canvas_shape=(*shape, 3), backend=backend,
        config=PredictiveCodingConfig(hidden_channels=3, inference_rate=rate, dtype=dtype),
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
    model.predict(prior, dt=0)
    observation = xp.broadcast_to(xp.asarray([.2, -.1, .4]), prior.shape)
    model.condition(observation, measurement)
    posterior = to_numpy(model.variance).copy()
    np.testing.assert_allclose(posterior, 1/(1/4+1), rtol=2e-6)
    np.testing.assert_allclose(to_numpy(readout.hidden), .8*to_numpy(observation), rtol=2e-6)
    assert scores[0, 0] == pytest.approx(float(0.5*np.log(4/posterior).sum()), rel=2e-6)
    assert model.expected_information_gain(measurement, prior=prior, dt=0).max() < scores.min()
    skipped = replace(measurement, sample=False)
    for observed in (True, False):
        model.predict(prior, dt=0)
        model.condition(prior if observed else None, skipped if observed else measurement)
        np.testing.assert_array_equal(to_numpy(model.variance), posterior)
    np.testing.assert_array_equal(model.expected_information_gain(skipped, prior=prior, dt=0), 0)


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("diffusion", (0, .04))
def test_model_transition_and_frozen_decoder_match_dense_gaussian_reference(
    request, backend_name, dtype, diffusion,
):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    readout = identity_readout((1, 1), backend=xp, rate=0.2, dtype=dtype)
    weights = np.array([[1, .4, -.2], [.3, 1, .5], [.1, -.3, .7]], dtype=dtype)
    canvas_weights = np.array([[.8, -.1, .2], [.3, .9, -.4], [-.2, .1, 1]], dtype=dtype)
    readout.visual_weights[:] = xp.asarray(weights)
    readout.canvas_weights[:] = xp.asarray(canvas_weights)
    readout.hidden[:] = xp.asarray([.2, -.3, .5])
    prior = xp.asarray([[[.5, -.2, .1]]], dtype=xp.float32)
    model = GaussianSensoryUncertainty(readout, initial_variance=2, process_variance=diffusion, backend=xp)
    initial_variance = np.array([2, .6, 0])
    model.variance[:] = xp.asarray(initial_variance)
    model.predict(prior, dt=.25)
    np.testing.assert_array_equal(to_numpy(model._frozen_hidden), to_numpy(readout.hidden))
    frozen = to_numpy(readout.hidden)[0, 0].copy()
    retention = np.exp(-.25/model.memory_time_constant)
    drive = canvas_weights@np.tanh(to_numpy(prior)[0, 0])
    np.testing.assert_allclose(frozen, retention*np.array([.2, -.3, .5])+(1-retention)*drive, atol=2e-7)
    increment = diffusion*model.memory_time_constant/2*(1-retention**2)
    variance = initial_variance*retention**2+increment
    np.testing.assert_allclose(to_numpy(model.variance)[0, 0], variance, rtol=2e-6)
    j = weights*(1-np.tanh(frozen)**2)
    dense_prior = np.diag(variance)
    dense_posterior = dense_prior-dense_prior@j.T@np.linalg.inv(np.eye(3)+j@dense_prior@j.T)@j@dense_prior
    observation = xp.asarray([[[.4, .1, -.2]]], dtype=xp.float32)
    delta = dense_prior@j.T@np.linalg.solve(
        np.eye(3)+j@dense_prior@j.T,
        to_numpy(observation)[0, 0]-weights@np.tanh(frozen),
    )
    readout.hidden += .6
    correction = model.condition(observation)
    np.testing.assert_allclose(to_numpy(readout.hidden)[0, 0], frozen+delta, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        to_numpy(correction)[0, 0],
        readout.config.canvas_rate*(1-np.tanh(to_numpy(prior)[0, 0])**2)*(canvas_weights.T@delta),
        rtol=2e-6, atol=2e-7,
    )
    np.testing.assert_allclose(to_numpy(model.variance)[0, 0], dense_posterior.diagonal(), rtol=2e-6)

    forecast = retention*to_numpy(readout.hidden)[0, 0]+(1-retention)*drive
    j_next = weights*(1-np.tanh(forecast)**2)
    p_next = retention**2*dense_posterior.diagonal()+increment
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
        model.predict(prior, dt=.1)
        model.condition(prior, measurement)
    scores = model.expected_information_gain(measurement, prior=prior, dt=.1, grid_shape=(5, 7))
    assert model.variance[9, 4].sum() < model.variance[9, 18].sum()
    assert scores[2, 1] < scores[2, 5]
    before = model.variance.copy()
    model.predict(prior, dt=2)
    model.condition(None, measurement)
    factor = np.exp(-4/model.memory_time_constant)
    np.testing.assert_allclose(model.variance, factor*before+.01*model.memory_time_constant/2*(1-factor))
    model.reset()
    np.testing.assert_array_equal(model.variance, 1)


@pytest.mark.parametrize("changes", [
    {"initial_variance": 0}, {"initial_variance": np.inf}, {"initial_variance": 1e-50},
    {"process_variance": -1}, {"process_variance": np.nan},
    {"memory_time_constant": 0}, {"memory_time_constant": -1},
    {"memory_time_constant": np.inf}, {"memory_time_constant": 1e-50},
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


@pytest.mark.parametrize("shape,max_side", [
    ((1080, 1920), 16), ((1, 31), 1), ((31, 1), 2), ((3, 11), 4), ((3, 11), 32),
])
def test_bounded_quadrature_preserves_total_gain_for_constant_precision(shape, max_side):
    readout = identity_readout(shape)
    model = GaussianSensoryUncertainty(readout, initial_variance=.25)
    scores = model.expected_information_gain(
        GaussianSpatialPrecision(baseline=1, peak=0),
        prior=np.zeros(readout.canvas_shape, dtype=np.float32), dt=0, max_side=max_side,
    )
    np.testing.assert_allclose(scores, 1.5*np.prod(shape)*np.log1p(.25), rtol=2e-6)


def test_forecast_conditioning_boundary_and_invalid_planning_inputs():
    readout = identity_readout()
    model = GaussianSensoryUncertainty(readout)
    prior = np.zeros(readout.canvas_shape, dtype=np.float32)
    measurement = GaussianSpatialPrecision()
    with pytest.raises(RuntimeError, match="pre-evidence"):
        model.condition(prior, measurement)
    for dt in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="timestep"):
            model.predict(prior, dt=dt)
    with pytest.raises(ValueError, match="candidate grid"):
        model.expected_information_gain(measurement, prior=prior, dt=0, grid_shape=(1, 3))
    model.predict(prior, dt=.1)
    with pytest.raises(RuntimeError, match="before predicting again"):
        model.predict(prior, dt=.1)
    with pytest.raises(RuntimeError, match="before scoring"):
        model.expected_information_gain(measurement, prior=prior, dt=.1)
    model.condition(None, measurement)
    assert np.isfinite(model.expected_information_gain(measurement, prior=prior, dt=.1)).all()


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_bounded_gain_preserves_full_grid_action_ranking_after_repeated_center_observations(
    request, backend_name,
):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    shape, dt, rate = (120, 160), 1/30, .1
    readout = identity_readout(shape, backend=xp, rate=rate)
    model = GaussianSensoryUncertainty(readout, backend=xp)
    precision = GaussianSpatialPrecision()
    pi = precision.field(shape, dtype=np.float64)
    factor = np.exp(-2*dt/model.memory_time_constant)
    increment = model.process_variance*model.memory_time_constant/2*(1-factor)
    # Positive Riccati fixed point for repeatedly observing this footprint.
    b = 1-factor+pi*increment
    posterior = 2*increment/(b+np.sqrt(b*b+4*pi*factor*increment))
    model.variance[:] = xp.asarray(posterior[..., None], dtype=xp.float32)
    forecast = factor*posterior+increment
    expected = np.array([
        [1.5*np.log1p(replace(precision, gaze=(float(x), float(y))).field(
            shape, dtype=np.float64,
        )*forecast).sum() for x in np.linspace(0, 1, 13)]
        for y in np.linspace(0, 1, 9)
    ])
    prior = xp.zeros(readout.canvas_shape, dtype=xp.float32)
    bounded = model.expected_information_gain(precision, prior=prior, dt=dt)
    full = model.expected_information_gain(precision, prior=prior, dt=dt, max_side=160)
    np.testing.assert_allclose(full, expected, rtol=2e-6)
    np.testing.assert_allclose(bounded, expected, rtol=1e-3)
    margin = expected[4, 5]-expected[4, 6]
    assert abs((bounded[4, 5]-bounded[4, 6])-margin) < .1*margin
    assert bounded[4, 5] > bounded[4, 6]
    assert information_gaze(bounded, precision.gaze) == information_gaze(full, precision.gaze)
    assert information_gaze(bounded, precision.gaze) == (5/12, .5)


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_temporal_posterior_and_scores_match_scalar_affine_oracle_independently_of_solver(
    request, backend_name,
):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    memory, diffusion, initial_variance = .8, .12, .7
    canvas = np.array([.12, -.08, .2])
    initial_mean = np.array([.1, -.2, .3])
    cases = (
        (.03, [.25, .1, -.2], GaussianSpatialPrecision(baseline=2, peak=0)),
        (.07, None, GaussianSpatialPrecision(baseline=2, peak=0)),
        (.05, [.9, .9, .9], GaussianSpatialPrecision(sample=False)),
        (0, [.2, .2, .2], GaussianSpatialPrecision(baseline=0, peak=0)),
        (.2, [-.1, .3, .6], GaussianSpatialPrecision(baseline=.4, peak=0)),
    )
    trajectories = []
    for rate, settling in ((0, 1), (.8, 37)):
        readout = identity_readout((1, 1), backend=xp, rate=rate)
        readout.update_config(inference_steps=settling)
        readout.set_diagnostics_enabled(True)
        readout.set_spatial_preview_enabled(True)
        readout.hidden[:] = xp.asarray(initial_mean)
        model = GaussianSensoryUncertainty(
            readout, backend=xp, initial_variance=initial_variance,
            process_variance=diffusion, memory_time_constant=memory,
        )
        expected_mean, expected_variance = initial_mean.copy(), np.full(3, initial_variance)
        trajectory = []
        prior = xp.asarray(canvas[None, None, :], dtype=xp.float32)
        for dt, values, precision in cases:
            retention = np.exp(-dt/memory)
            forecast_mean = retention*expected_mean+(1-retention)*np.tanh(canvas)
            forecast_variance = retention**2*expected_variance+diffusion*memory/2*(1-retention**2)
            prediction = model.predict(prior, dt=dt)
            np.testing.assert_allclose(to_numpy(prediction)[0, 0], np.tanh(forecast_mean), atol=2e-7)
            lam = precision.baseline if values is not None and precision.sample else 0
            j = 1-np.tanh(forecast_mean)**2
            residual = np.array(values)-np.tanh(forecast_mean) if values is not None else np.zeros(3)
            delta = lam*forecast_variance*j/(1+lam*forecast_variance*j*j)*residual
            expected_mean = forecast_mean+delta
            expected_variance = forecast_variance/(1+lam*forecast_variance*j*j)
            observation = None if values is None else xp.asarray([[values]], dtype=xp.float32)
            correction = model.condition(observation, precision)
            np.testing.assert_allclose(to_numpy(readout.hidden)[0, 0], expected_mean, atol=2e-7)
            np.testing.assert_allclose(to_numpy(model.variance)[0, 0], expected_variance, rtol=2e-6)
            np.testing.assert_allclose(
                to_numpy(correction)[0, 0], readout.config.canvas_rate*(1-np.tanh(canvas)**2)*delta,
                atol=2e-7,
            )
            next_mean = np.exp(-.03/memory)*expected_mean+(1-np.exp(-.03/memory))*np.tanh(canvas)
            next_variance = np.exp(-.06/memory)*expected_variance+diffusion*memory/2*(1-np.exp(-.06/memory))
            next_lam = precision.baseline if precision.sample else 0
            expected_gain = .5*np.log1p(next_lam*next_variance*(1-np.tanh(next_mean)**2)**2).sum()
            scores = model.expected_information_gain(precision, prior=prior, dt=.03)
            np.testing.assert_allclose(scores, expected_gain, rtol=2e-6, atol=1e-8)
            diagnostics = readout.diagnostics
            assert diagnostics.stream_state is None and diagnostics.weight_update_norm == 0
            if diagnostics.inference_energy:
                assert diagnostics.inference_energy[-1] <= diagnostics.inference_energy[0]+1e-7
            trajectory.extend(to_numpy(value).copy() for value in (
                readout.hidden, model.variance, correction, scores,
            ))
        trajectories.append(trajectory)
    for actual, expected in zip(*trajectories):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_posterior_and_diagnostics_share_configured_or_explicit_likelihood(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    precision = GaussianSpatialPrecision(baseline=.1, peak=2, sigma=.2, gaze=(.7, .3))
    for explicit in (False, True):
        readout = identity_readout(backend=xp)
        readout.set_diagnostics_enabled(True)
        if not explicit:
            readout.update_config(spatial_precision=precision)
        model = GaussianSensoryUncertainty(readout, backend=xp, initial_variance=.7)
        prior = xp.zeros((3, 5, 3), dtype=xp.float32)
        observation = xp.broadcast_to(xp.asarray([.2, .4, .6], dtype=xp.float32), prior.shape)
        model.predict(prior, dt=0)
        readout.update_config(learning_enabled=True)
        with pytest.raises(ValueError, match="fixed-weight"):
            model.condition(observation, precision)
        readout.update_config(learning_enabled=False)
        model.condition(observation, precision if explicit else None)
        assert readout.config.spatial_precision == precision
        pi = precision.field((3, 5))[..., None]
        expected_variance = 1/(1/.7+pi)
        delta = expected_variance*pi*to_numpy(observation)
        np.testing.assert_allclose(to_numpy(readout.hidden), delta, atol=2e-7)
        np.testing.assert_allclose(
            to_numpy(model.variance), np.broadcast_to(expected_variance, prior.shape), atol=2e-7,
        )
        assert readout.diagnostics.hidden.mean == pytest.approx(.5*np.mean(delta**2/.7), abs=1e-7)
        assert readout.diagnostics.visual.mean == pytest.approx(
            .5*np.mean(pi*(to_numpy(observation)-delta)**2), abs=1e-7,
        )
