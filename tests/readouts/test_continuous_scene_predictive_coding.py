import numpy as np
import pytest

from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.readouts.predictive_coding import PredictiveCodingConfig


def make_graph(**changes):
    return ContinuousScenePredictiveCodingGraph(
        visual_shape=(1, 2), audio_nodes=1, hidden_nodes=2,
        config=PredictiveCodingConfig(hidden_channels=2, **changes),
    )


def test_state_canvas_and_weight_gradients_differentiate_the_same_energy():
    graph = make_graph(weight_decay=.07)
    graph.canvas[:] = np.linspace(-.3, .4, graph.visual_nodes)
    graph.aperture_mask[1] = False
    graph.mean[graph.hidden_slice] = [.2, -.4]
    graph.sensory_potential[:] = np.linspace(-.5, .6, graph._sensory_nodes)
    graph.clamped[:graph._sensory_nodes] = True
    graph.clamped[2] = False
    graph.evidence[:graph._sensory_nodes] = np.linspace(.1, .9, graph._sensory_nodes)
    graph.evidence[2] = np.nan
    h = graph.mean[graph.hidden_slice]
    z = graph.sensory_potential
    terms = graph._terms(h, z)
    gradients = (
        (h, terms[3]), (z, terms[4]), (graph.canvas, terms[5]),
        *(
            (getattr(graph, name), gradient)
            for name, gradient in zip(graph._PARAMETERS, graph._weight_gradients(h, z))
        ),
    )
    for values, analytic in gradients:
        numeric = np.zeros_like(values)
        for index in np.ndindex(values.shape):
            if values is graph.recurrent_weights and index[0] == index[1]:
                continue
            original = values[index]
            values[index] = original + 1e-6
            plus = graph._terms(h, z)[0]
            values[index] = original - 1e-6
            minus = graph._terms(h, z)[0]
            values[index] = original
            numeric[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-6)
    graph.update_config(inference_rate=20.)
    assert graph._terms(h, z)[0] == pytest.approx(terms[0])


def test_unclipped_state_and_learning_steps_match_captured_gradients():
    graph = make_graph(
        inference_steps=1, inference_rate=.7, learning_enabled=True,
        learning_rate=.03, weight_decay=.05, gradient_clip=1e-10, weight_clip=1e-10,
    )
    graph.mean[graph.hidden_slice] = [.3, -.2]
    graph.sensory_potential[:] = .1
    initial_h = graph.mean[graph.hidden_slice].copy()
    initial_z = graph.sensory_potential.copy()
    canvas = np.full((1, 2, 3), .2)
    graph.forecast(canvas, .02)
    graph.clamped[:graph._sensory_nodes] = True
    graph.evidence[:graph.visual_nodes] = .9
    graph.evidence[graph.audio_slice] = -.3
    terms = graph._terms(initial_h, initial_z)
    weights = {name: getattr(graph, name).copy() for name in graph._PARAMETERS}
    direction = graph.condition(np.full_like(canvas, .9), np.array([-.3]))
    np.testing.assert_allclose(graph.mean[graph.hidden_slice], initial_h - .014 * terms[3])
    np.testing.assert_allclose(graph.sensory_potential, initial_z - .014 * terms[4])
    snapshot = graph.belief_snapshot()
    np.testing.assert_array_equal(direction.ravel(), -snapshot["canvas_gradient"])
    for name in graph._PARAMETERS:
        np.testing.assert_array_equal(snapshot[name], weights[name])
        np.testing.assert_allclose(
            getattr(graph, name), weights[name] - .03 * snapshot["weight_gradients"][name],
        )
    for name in graph._PRECISIONS:
        np.testing.assert_array_equal(getattr(graph, name), 1.)
    np.testing.assert_array_equal(np.diag(graph.recurrent_weights), 0.)


def test_black_gray_white_evidence_pushes_decoded_predictions_in_the_right_direction():
    predictions = []
    for level in (0., .5, 1.):
        graph = make_graph(learning_enabled=True, learning_rate=.01)
        canvas = np.zeros((1, 2, 3))
        graph.forecast(canvas, 1/30)
        graph.clamped[:graph.visual_nodes] = True
        graph.evidence[:graph.visual_nodes] = level
        gradient = graph._terms(
            graph.mean[graph.hidden_slice], graph.sensory_potential,
        )[4][:graph.visual_nodes]
        np.testing.assert_allclose(gradient, (.5 - level) * .25)
        for _ in range(300):
            graph.condition(np.full_like(canvas, level))
            graph.forecast(canvas, 1/30)
        graph.condition(np.full_like(canvas, level))
        predictions.append(float(graph.visual_prediction.mean()))
        assert np.all(np.diff(graph.inference_energy) <= 1e-12)
    assert 0. < predictions[0] < .5
    assert predictions[1] == pytest.approx(.5)
    assert .5 < predictions[2] < 1.
    assert predictions[0] + predictions[2] == pytest.approx(1., abs=1e-5)


def test_unobserved_interval_is_not_replayed_and_zero_rate_freezes_decay():
    graph = make_graph()
    canvas = np.zeros((1, 2, 3))
    graph.forecast(canvas, .02)
    graph.condition(np.ones_like(canvas))
    graph.forecast(canvas, .02)
    prior = graph.sensory_potential.copy()
    graph.condition()
    np.testing.assert_array_equal(graph.sensory_potential, prior)
    graph.update_config(inference_rate=0)
    graph.forecast(canvas, .02)
    graph.condition(np.zeros_like(canvas))
    np.testing.assert_array_equal(graph.sensory_potential, prior)
    graph.forecast(canvas, .02)
    with pytest.raises(RuntimeError, match="condition the pending"):
        graph.forecast(canvas, .02)
    with pytest.raises(ValueError, match="finite"):
        graph.condition(np.full_like(canvas, np.nan))
    graph.condition()


def test_covariance_uses_captured_decay_and_matches_dense_precision():
    graph = make_graph()
    graph.forecast(np.zeros((1, 2, 3)), .02)
    graph.condition(np.ones((1, 2, 3)))
    covariance = graph.state_covariance()
    graph.set_memory_time_constant(2.)
    after = graph.state_covariance()
    np.testing.assert_array_equal(covariance.precision_diagonal, after.precision_diagonal)
    basis = np.eye(graph.total_nodes)
    precision = np.column_stack([covariance.precision_product(column) for column in basis])
    np.testing.assert_allclose(
        covariance.marginal_covariance(np.arange(graph.total_nodes)),
        np.linalg.inv(precision), atol=1e-9,
    )


def test_information_gain_uses_unit_precision_independent_of_update_rate():
    scores = []
    for rate in (0., 2.):
        graph = make_graph(inference_rate=rate)
        graph.forecast(np.zeros((1, 2, 3)), .02)
        scores.append(graph.candidate_information_gains([[0., 0.], [1., 0.]], foveal_radius=1))
        assert np.isfinite(scores[-1]).all()
        assert (scores[-1] > 0).all()
        graph.condition()
        with pytest.raises(RuntimeError, match="after forecast and before condition"):
            graph.candidate_information_gain([0., 0.], foveal_radius=1)
    np.testing.assert_array_equal(*scores)


def test_nonfinite_learning_is_reported_without_partially_updating_weights():
    graph = make_graph(learning_enabled=True)
    graph.forecast(np.full((1, 2, 3), .2), .02)
    original = {name: getattr(graph, name).copy() for name in graph._PARAMETERS}
    capture = graph._capture

    def invalid_gradient(terms=None):
        prediction = capture(terms)
        graph.weight_gradients["sensory_weights"].fill(np.inf)
        return prediction

    graph._capture = invalid_gradient
    with pytest.raises(FloatingPointError, match="connection update became non-finite"):
        graph.condition(np.ones((1, 2, 3)))
    for name, values in original.items():
        np.testing.assert_array_equal(getattr(graph, name), values)
