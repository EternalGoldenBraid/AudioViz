import numpy as np
import pytest

from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.readouts.continuous_state_covariance import ContinuousStateCovariance
from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.utils.array_backend import to_numpy


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_schur_information_matches_dense_joint_entropy_reduction_with_correlated_observations(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    rng = np.random.default_rng(18)
    a, b = rng.normal(size=(3, 3)), rng.normal(size=(7, 3))
    ph, q, observation = rng.uniform(.2, 3, 3), rng.uniform(.1, 2, 7), rng.uniform(0, 1, 7)
    covariance = ContinuousStateCovariance(
        hidden_jacobian=xp.asarray(a), sensory_jacobian=xp.asarray(b),
        observation_curvature=xp.asarray(observation), hidden_prior_precision=.7,
        sensory_prior_precision=.3, hidden_residual_precision=xp.asarray(ph),
        coupling_precision=xp.asarray(q), backend=xp, stage="forecast",
    )
    precision = np.block([
        [np.diag(.3 + q + observation), -q[:, None] * b],
        [-b.T * q, .7*np.eye(3) + a.T @ (ph[:, None]*a) + b.T @ (q[:, None]*b)],
    ])
    prior = np.linalg.inv(precision)
    indices, added = np.array([0, 2, 5]), np.array([.5, 2., 8.])
    updated = precision.copy()
    updated[indices, indices] += added
    posterior = np.linalg.inv(updated)
    expected = .5 * (np.linalg.slogdet(prior)[1] - np.linalg.slogdet(posterior)[1])
    actual = covariance.sensory_information_gain(xp.asarray(indices), xp.asarray(added))
    assert actual == pytest.approx(expected, abs=1e-12)
    independent_sum = .5 * np.log1p(added*np.diag(prior)[indices]).sum()
    assert abs(actual - independent_sum) > 1e-4
    assert 0 < covariance.sensory_information_gain(indices, added*.01) < actual
    assert covariance.sensory_information_gain(indices, np.zeros(3)) == pytest.approx(0, abs=1e-14)
    assert covariance.sensory_information_gain(np.array([], dtype=int), np.array([])) == 0
    np.testing.assert_allclose(
        to_numpy(covariance.covariance_product(np.ones(10))), prior @ np.ones(10), atol=1e-10,
    )
    for invalid_indices, invalid_added in (
        ([0, 0], [1, 1]), ([-1], [1]), ([7], [1]), ([.5], [1]),
        ([0], [-1]), ([0], [np.nan]), ([0], [np.inf]), ([0], [1, 2]),
    ):
        with pytest.raises(ValueError, match="distinct sensory indices"):
            covariance.sensory_information_gain(invalid_indices, invalid_added)


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_gaze_scores_use_forecast_joint_covariance_noise_and_foveal_support_without_mutation(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(3, 4), audio_nodes=2, hidden_nodes=3, backend=xp,
        config=PredictiveCodingConfig(hidden_channels=3, learning_enabled=True),
    )
    graph.sensory_potential[:] = xp.linspace(-1, .8, graph._sensory_nodes)
    graph.mean[graph.hidden_slice] = xp.asarray([.3, -.2, .1])
    for name in graph._PRECISIONS:
        getattr(graph, name)[:] *= xp.linspace(.6, 1.8, getattr(graph, name).size)
    graph.forecast(np.full((3, 4, 3), .2), 1/30)
    before = graph.belief_snapshot()
    uncertainty = graph.state_covariance()
    precision = np.column_stack([
        to_numpy(uncertainty.precision_product(column)) for column in np.eye(graph.total_nodes)
    ])
    gazes = np.array([[[0., 0.], [.5, .5]], [[1., 1.], [1., 0.]]])
    scores = graph.candidate_information_gains(gazes, foveal_radius=1)
    assert scores.shape == (2, 2) and scores.dtype == np.float64
    rows, cols = np.indices(graph.visual_shape)
    decoded = to_numpy(graph._decode_sensory(graph.sensory_potential))[:graph.visual_nodes]
    added = (
        graph._INTEGRATION_GAIN * before["inference_rate"]
        * before["observation_precision"][:graph.visual_nodes] * (decoded*(1-decoded))**2
    )
    for coordinate in np.ndindex(scores.shape):
        center = np.rint(gazes[coordinate][::-1] * (np.array(graph.visual_shape)-1))
        mask = (rows-center[0])**2 + (cols-center[1])**2 <= 1
        rgb_mask = np.repeat(mask.reshape(-1), 3)
        update = np.zeros(graph.total_nodes)
        update[:graph.visual_nodes] = np.where(rgb_mask, added, 0)
        expected = .5 * (
            np.linalg.slogdet(precision + np.diag(update))[1] - np.linalg.slogdet(precision)[1]
        )
        assert scores[coordinate] == pytest.approx(expected, abs=2e-12)
    assert graph.candidate_information_gain([.5, .5], foveal_radius=1) == pytest.approx(scores[0, 1])
    after = graph.belief_snapshot()
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[name], value)
        else:
            assert after[name] == value
    assert graph.updates == 0 and graph.stage == "forecast"
    for gazes_invalid in ([], [.5, .5], [[np.nan, .5]], [[-.1, .5]], np.zeros((118, 2))):
        with pytest.raises(ValueError, match="gazes"):
            graph.candidate_information_gains(gazes_invalid, foveal_radius=1)
    with pytest.raises(ValueError, match="foveal_radius"):
        graph.candidate_information_gains([[.5, .5]], foveal_radius=37)
    graph.condition(np.full((3, 4, 3), .5))
    with pytest.raises(RuntimeError, match="before condition"):
        graph.candidate_information_gains(gazes, foveal_radius=1)


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_neutral_zero_error_is_informative_and_zero_inference_gain_has_no_observation_information(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(5, 5), audio_nodes=0, hidden_nodes=2, backend=xp,
        config=PredictiveCodingConfig(hidden_channels=2, learning_enabled=False),
    )
    # Zero decoder residual does not remove the information in a reliable observation.
    graph.forecast(np.zeros((5, 5, 3)), 1/30)
    score = graph.candidate_information_gain([.5, .5], foveal_radius=1)
    assert score > 0
    graph.condition(np.full((5, 5, 3), .5), visual_mask=np.eye(5, dtype=bool))
    graph.observation_precision *= .01
    graph.forecast(np.zeros((5, 5, 3)), 1/30)
    assert 0 < graph.candidate_information_gain([.5, .5], foveal_radius=1) < score
    graph.condition()
    graph.update_config(inference_rate=0)
    graph.forecast(np.zeros((5, 5, 3)), 1/30)
    assert graph.candidate_information_gain([.5, .5], foveal_radius=1) == 0
    graph.condition()
