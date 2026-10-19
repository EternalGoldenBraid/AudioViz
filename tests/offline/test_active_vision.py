import json
import sys

import numpy as np
import pytest

from audioviz.offline.active_vision import _precision_mass, main, run_gaze_comparison
from audioviz.readouts import GaussianSpatialPrecision
from audioviz.sources import CircularDotConfig
from audioviz.utils.opengl_compute import GLArray


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_gaze_comparison_is_reproducible_and_reports_separate_metrics(qapp, request, monkeypatch, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native OpenGL evaluation requires a window-system context")
    downloads = []
    original = GLArray.download

    def download(array):
        downloads.append(array.shape)
        return original(array)

    monkeypatch.setattr(GLArray, "download", download)
    stimulus = CircularDotConfig(
        resolution=(12, 16), radius=4, dot_radius=2,
        frame_rate=10, duration=0.4, brownian_scale=2, seed=7,
    )
    options = dict(stimulus=stimulus, backend=backend, inference_steps=2, policy_seed=29)
    first = run_gaze_comparison(**options)
    second = run_gaze_comparison(**options)
    assert first == second
    assert [run.policy for run in first] == ["fixed", "random", "autonomous"]
    assert all([frame.time for frame in run.frames] == [0, .1, .2, .3] for run in first)
    assert all(frame.gaze == (.5, .5) for frame in first[0].frames)
    assert len({frame.gaze for frame in first[1].frames}) > 1
    assert len({frame.gaze for frame in first[2].frames}) > 1
    assert first[0].frames[0].forecast_mse == first[2].frames[0].forecast_mse
    for run in first:
        assert run.cumulative_expected_gain_nats == sum(frame.expected_gain_nats for frame in run.frames)
        assert run.final_hidden_variance_mean == run.frames[-1].posterior_hidden_variance_mean
        assert run.mean_forecast_mse == np.mean([frame.forecast_mse for frame in run.frames])
        assert run.cumulative_precision_mass == sum(frame.precision_mass for frame in run.frames)
        for frame in run.frames:
            assert np.isfinite((
                frame.expected_gain_nats, frame.posterior_hidden_variance_mean,
                frame.forecast_mse, frame.precision_mass,
            )).all()
            assert 0 <= frame.posterior_hidden_variance_mean <= 1
            assert frame.expected_gain_nats >= 0 and frame.forecast_mse >= 0
    if backend == "opengl":
        assert downloads and max(np.prod(shape) for shape in downloads) == 117
        assert (1, 2) in downloads
    if backend != "numpy":
        reference = run_gaze_comparison(**{**options, "backend": "numpy"})
        for actual, expected in zip(first, reference):
            np.testing.assert_allclose(
                [frame.gaze for frame in actual.frames], [frame.gaze for frame in expected.frames],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                [[frame.expected_gain_nats, frame.posterior_hidden_variance_mean, frame.forecast_mse]
                 for frame in actual.frames],
                [[frame.expected_gain_nats, frame.posterior_hidden_variance_mean, frame.forecast_mse]
                 for frame in expected.frames], rtol=3e-5, atol=2e-6,
            )


def test_precision_budget_accounts_for_clipping_and_cli_preserves_assumptions(monkeypatch, capsys):
    for gaze in ((0.5, 0.5), (0, 0)):
        precision = GaussianSpatialPrecision(gaze=gaze)
        assert _precision_mass(precision, (12, 16)) == pytest.approx(
            precision.field((12, 16), dtype=np.float64).sum(),
        )
    monkeypatch.setattr(sys, "argv", [
        "active_vision", "--resolution", "12", "16", "--fps", "10", "--duration", ".1",
        "--inference-steps", "2", "--seed", "7", "--policy-seed", "29",
    ])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stimulus"]["seed"] == 7 and report["policy_seed"] == 29
    assert "not sampled or calibrated" in report["measurement_model"]
    assert "clipping" in report["budget"]
    assert len(report["runs"]) == 3
    assert all(len(run["frames"]) == 1 for run in report["runs"])
