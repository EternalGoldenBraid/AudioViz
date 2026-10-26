import numpy as np
import pytest

from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.readouts.continuous_state_covariance import ContinuousStateCovariance
from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.utils.array_backend import to_numpy


def _two_state_covariance(observed):
    # a ~ N(0,1), b|a ~ N(a,1), and (when observed) y|b ~ N(b,1); state order is [b,a].
    return ContinuousStateCovariance(
        hidden_jacobian=np.ones((1, 1)), sensory_jacobian=np.ones((1, 1)),
        observation_curvature=np.array([float(observed)]),
        hidden_prior_precision=1, sensory_prior_precision=0,
        hidden_residual_precision=0, coupling_precision=1, backend=np, stage="posterior",
    )


def test_two_state_covariance_propagates_evidence_and_solves_variational_covariance_gradient():
    prior = _two_state_covariance(False)
    posterior = _two_state_covariance(True)
    np.testing.assert_allclose(prior.marginal_covariance([0, 1]), [[2, 1], [1, 1]], atol=1e-12)
    covariance = posterior.marginal_covariance([0, 1])
    np.testing.assert_allclose(covariance, np.array([[2, 1], [1, 2]]) / 3, atol=1e-12)
    precision = np.array([[2., -1.], [-1., 2.]])
    np.testing.assert_allclose(posterior.precision_diagonal, np.diag(precision))
    assert covariance[0, 0] != pytest.approx(1 / precision[0, 0])

    def free_energy(candidate):
        return .5 * np.trace(precision @ candidate) - .5 * np.linalg.slogdet(candidate)[1]

    candidate = np.array([[.8, .15], [.15, .6]])
    gradient = .5 * (precision - np.linalg.inv(candidate))
    for direction in (np.diag([1., 0.]), np.diag([0., 1.]), np.array([[0., 1.], [1., 0.]])):
        epsilon = 1e-6
        numerical = (free_energy(candidate + epsilon*direction) - free_energy(candidate - epsilon*direction)) / (2*epsilon)
        assert numerical == pytest.approx(np.sum(gradient * direction), abs=1e-9)
    np.testing.assert_allclose(.5 * (precision - np.linalg.inv(covariance)), 0, atol=1e-12)


def _dense_curvature_from_residual_differences(graph):
    snapshot = graph.belief_snapshot()
    sensory_nodes = graph.hidden_slice.start
    weights = {name: snapshot[name] for name in graph._PARAMETERS}
    gain = graph._INTEGRATION_GAIN * snapshot["inference_rate"]
    observed = snapshot["clamped"][:sensory_nodes]
    center = np.concatenate((snapshot["sensory_potential"], snapshot["mean"][graph.hidden_slice]))
    root = np.where(snapshot["aperture_mask"], np.tanh(snapshot["canvas"]), 0)

    def residuals(state):
        z, h = state[:sensory_nodes], state[sensory_nodes:]
        activity = np.tanh(h)
        prediction = z.copy()
        prediction[graph.visual_slice] = .5 * (1 + np.tanh(.5*z[graph.visual_slice]))
        return np.concatenate((
            np.sqrt(graph._LEAK_ENERGY_WEIGHT) * h,
            np.sqrt(graph._LEAK_ENERGY_WEIGHT * graph._gamma_z) * z,
            np.sqrt(gain * snapshot["hidden_residual_precision"])
            * (h - weights["canvas_weights"] @ root - weights["recurrent_weights"] @ activity),
            np.sqrt(gain * snapshot["coupling_precision"])
            * (z - weights["sensory_weights"] @ activity),
            np.sqrt(gain * snapshot["observation_precision"][observed])
            * (snapshot["evidence"][:sensory_nodes][observed] - prediction[observed]),
        ))

    epsilon = 1e-6
    jacobian = np.column_stack([
        (residuals(center + epsilon*direction) - residuals(center - epsilon*direction)) / (2*epsilon)
        for direction in np.eye(graph.total_nodes)
    ])
    return jacobian.T @ jacobian


@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_neural_covariance_matches_dense_gauss_newton_without_mutating_captured_state(request, backend):
    xp = np if backend == "numpy" else request.getfixturevalue("cupy_backend")
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(2, 2), audio_nodes=2, hidden_nodes=3, backend=xp,
        config=PredictiveCodingConfig(hidden_channels=3, inference_rate=.17, learning_enabled=True),
    )
    graph.mean[graph.hidden_slice] = xp.asarray([.2, -.3, .4])
    graph.sensory_potential[:] = xp.linspace(-.4, .5, graph.hidden_slice.start)
    for name in graph._PRECISIONS:
        precision = getattr(graph, name)
        precision *= xp.linspace(.5, 2., precision.size)
    graph.forecast(np.full((2, 2, 3), .2), 1/30)
    graph.condition(np.full((2, 2, 3), .8), np.array([.3, -.2]), visual_mask=np.eye(2, dtype=bool))
    assert graph.updates == 1
    assert not np.array_equal(to_numpy(graph.sensory_weights), to_numpy(graph._snapshot_weights["sensory_weights"]))
    assert not np.array_equal(
        to_numpy(graph.coupling_precision), to_numpy(graph._snapshot_precisions["coupling_precision"]),
    )
    before = graph.belief_snapshot()
    live_weights = {name: to_numpy(getattr(graph, name)).copy() for name in graph._PARAMETERS}
    live_precisions = {name: to_numpy(getattr(graph, name)).copy() for name in graph._PRECISIONS}
    covariance = graph.state_covariance()
    assert covariance.stage == "posterior"
    expected = _dense_curvature_from_residual_differences(graph)
    basis = np.eye(graph.total_nodes)
    actual = np.column_stack([to_numpy(covariance.precision_product(column)) for column in basis])
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(to_numpy(covariance.precision_diagonal), np.diag(expected), rtol=1e-8)
    rhs = np.linspace(-.3, .8, graph.total_nodes)
    solution = covariance.covariance_product(rhs)
    assert isinstance(solution, xp.ndarray)
    np.testing.assert_allclose(to_numpy(solution), np.linalg.solve(expected, rhs), rtol=1e-8, atol=1e-9)
    indices = [0, 4, graph.audio_slice.start, graph.hidden_slice.start, graph.total_nodes-1]
    block = to_numpy(covariance.marginal_covariance(indices))
    np.testing.assert_allclose(block, np.linalg.inv(expected)[np.ix_(indices, indices)], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(block, block.T, atol=1e-10)
    assert np.linalg.eigvalsh(block).min() > 0
    for name in ("mean", "sensory_potential", "clamped", "evidence"):
        np.testing.assert_array_equal(to_numpy(getattr(graph, name)), before[name])
    for name, values in live_weights.items():
        np.testing.assert_array_equal(to_numpy(getattr(graph, name)), values)
    for name, values in live_precisions.items():
        np.testing.assert_array_equal(to_numpy(getattr(graph, name)), values)

    # A retained operator represents its captured frame, including the pre-learning weights and gain.
    graph.update_config(inference_rate=0)
    np.testing.assert_array_equal(
        to_numpy(graph.state_covariance().precision_product(rhs)),
        to_numpy(covariance.precision_product(rhs)),
    )
    graph.forecast(np.zeros((2, 2, 3)), 1/30)
    forecast = graph.state_covariance()
    assert forecast.stage == "forecast"
    graph.condition()
    np.testing.assert_allclose(
        to_numpy(forecast.covariance_product(rhs)),
        rhs / np.concatenate((np.full(graph.hidden_slice.start, graph._gamma_z), np.ones(graph.hidden_nodes))),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        to_numpy(covariance.precision_product(rhs)), actual @ rhs, atol=1e-10,
    )
    assert not graph.has_gaussian_uncertainty


def test_observation_curvature_is_foveal_and_independent_of_error_value_at_fixed_state():
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(3, 4), audio_nodes=0, hidden_nodes=2,
        config=PredictiveCodingConfig(hidden_channels=2, learning_enabled=False),
    )
    graph.forecast(np.zeros((3, 4, 3)), 1/30)
    prior = graph.state_covariance()
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    # Zero residual keeps the linearization fixed, isolating the precision contributed by observing.
    graph.condition(np.full((3, 4, 3), .5), visual_mask=mask)
    posterior = graph.state_covariance()
    added = posterior.precision_diagonal - prior.precision_diagonal
    expected = np.zeros(graph.total_nodes)
    expected[:graph.visual_nodes].reshape(3, 4, 3)[mask] = (
        graph._INTEGRATION_GAIN * graph.config.inference_rate
        * graph._OBSERVATION_GAIN * graph._gamma_z / 16
    )
    np.testing.assert_allclose(added, expected, atol=1e-12)
    graph.evidence[graph.visual_slice].reshape(3, 4, 3)[~mask] = 100
    np.testing.assert_array_equal(graph.state_covariance().precision_diagonal, posterior.precision_diagonal)
    assert graph.state_covariance().marginal_covariance([18])[0, 0] < prior.marginal_covariance([18])[0, 0]


def test_covariance_queries_are_bounded_and_solver_failures_are_explicit():
    covariance = _two_state_covariance(True)
    for indices in ([], [0, 0], [-1], [2], [0.5], np.arange(65)):
        with pytest.raises(ValueError, match="indices"):
            covariance.marginal_covariance(indices)
    for vector in ([1], [np.nan, 0], [[1, 0]]):
        with pytest.raises(ValueError, match="state vector"):
            covariance.covariance_product(vector)
    for tolerance in (0, -1, np.nan, True):
        with pytest.raises(ValueError, match="rtol"):
            covariance.covariance_product([1, 0], rtol=tolerance)
    np.testing.assert_array_equal(covariance.covariance_product([0, 0]), 0)
    with pytest.raises(RuntimeError, match="did not converge"):
        covariance.covariance_product([1, 0], max_iterations=1)
    with pytest.raises(ValueError, match="max_iterations"):
        covariance.covariance_product([1, 0], max_iterations=0)
    singular = ContinuousStateCovariance(
        hidden_jacobian=np.ones((1, 1)), sensory_jacobian=np.ones((1, 1)),
        observation_curvature=np.zeros(1), hidden_prior_precision=0, sensory_prior_precision=0,
        hidden_residual_precision=0, coupling_precision=1, backend=np, stage="posterior",
    )
    with pytest.raises(RuntimeError, match="non-positive curvature"):
        singular.covariance_product([1, 1])
