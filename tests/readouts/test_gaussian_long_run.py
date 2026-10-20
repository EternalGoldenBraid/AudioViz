import numpy as np
import pytest

from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.utils.array_backend import to_numpy


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_temporal_mean_and_covariance_match_exact_ou_independently_of_step_partition(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    initial = np.array([.3, .1, -.2])
    canvas = np.array([[[.2, -.1, .3]]])
    total_time, memory, diffusion = 4, .7, .12
    retention = np.exp(-total_time/memory)
    expected_mean = retention*initial+(1-retention)*np.tanh(canvas)
    expected_variance = retention**2+diffusion*memory/2*(1-retention**2)
    for rate, settling, partitions in ((.05, 1, 120), (.2, 32, 1)):
        readout = PredictiveCodingRGBReadout(
            canvas_shape=(1, 1, 3), backend=xp,
            config=PredictiveCodingConfig(
                hidden_channels=3, dtype="float32", inference_rate=rate, inference_steps=settling,
            ),
        )
        readout.canvas_weights[:] = xp.eye(3)
        readout.hidden[:] = xp.asarray(initial)
        model = GaussianSensoryUncertainty(
            readout, backend=xp, memory_time_constant=memory, process_variance=diffusion,
        )
        prior = xp.asarray(canvas, dtype=xp.float32)
        for _ in range(partitions):
            model.predict(prior, dt=total_time/partitions)
            np.testing.assert_array_equal(to_numpy(model.condition(None)), 0)
        np.testing.assert_allclose(to_numpy(readout.hidden), expected_mean, rtol=2e-6)
        np.testing.assert_allclose(to_numpy(model.variance), expected_variance, rtol=2e-6)
        model.reset()
        np.testing.assert_array_equal(to_numpy(readout.hidden), 0)
        np.testing.assert_array_equal(to_numpy(model.variance), 1)
