import numpy as np
import pytest

from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.utils.array_backend import to_numpy


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_known_point_canvas_contracts_uncertainty_to_process_floor_without_observations(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    readout = PredictiveCodingRGBReadout(
        canvas_shape=(1, 1, 3), backend=xp, config=PredictiveCodingConfig(dtype="float32"),
    )
    model = GaussianSensoryUncertainty(readout, backend=xp)
    prior = xp.zeros(readout.canvas_shape, dtype=xp.float32)
    dt, steps = 1/30, 120
    for _ in range(steps):
        readout.predict(prior)
        model.predict(dt=dt)
        model.condition(None, observed=False)
    factor = (1-readout.config.inference_rate)**2
    floor = model.process_variance*dt/(1-factor)
    expected = factor**steps*model.initial_variance + floor*(1-factor**steps)
    variance = to_numpy(model.variance)
    assert np.isfinite(variance).all()
    assert floor == pytest.approx(0.0017543859649122812)
    np.testing.assert_allclose(variance, expected, rtol=2e-6)
    assert variance.max() < model.initial_variance/500
