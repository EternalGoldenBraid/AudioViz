from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.readouts.shared_predictive_coding import (
    SharedPredictiveCodingGraph, SharedPredictiveCodingSnapshot,
)


PARAMETERS = (
    "canvas_weights", "recurrent_weights", "sensory_weights", "hidden_bias", "sensory_bias",
)


def make_graph(**kwargs):
    if kwargs.get("config") is not None:
        kwargs["config"] = replace(kwargs["config"], hidden_channels=3)
    return SharedPredictiveCodingGraph(
        visual_shape=(1, 2), audio_nodes=2, hidden_nodes=3, seed=13, **kwargs,
    )


def oracle(snapshot, hidden=None, canvas=None):
    h = snapshot["mean"][slice(*snapshot["hidden_slice"])] if hidden is None else hidden
    c = snapshot["canvas"] if canvas is None else canvas
    t = np.tanh(h)
    eh = (
        h - snapshot["canvas_weights"] @ np.tanh(c)
        - snapshot["recurrent_weights"] @ t - snapshot["hidden_bias"]
    )
    logits = snapshot["sensory_weights"] @ t + snapshot["sensory_bias"]
    prediction = .5 * (1 + np.tanh(.5 * logits))
    observed = snapshot["clamped"][:prediction.size]
    es = np.zeros_like(prediction)
    es[observed] = snapshot["evidence"][:prediction.size][observed] - prediction[observed]
    a = np.eye(h.size) - snapshot["recurrent_weights"] * (1 - t**2)[None, :]
    j = prediction[:, None] * (1 - prediction[:, None]) * snapshot["sensory_weights"]
    j *= (1 - t**2)[None, :]
    energy = .5 * (eh @ eh + es @ es)
    return energy, eh, es, prediction, a, j


def finite_difference(function, values, epsilon=1e-6):
    result = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return result


def test_exact_hidden_root_and_all_local_parameter_gradients():
    graph = make_graph(config=PredictiveCodingConfig(inference_rate=0))
    graph.forecast(np.linspace(-.7, .8, 6), dt=.1)
    graph.condition(
        np.linspace(.1, .8, 6), np.array([.3, .9]),
        visual_mask=np.array([[[True, False, True], [False, True, False]]]),
    )
    graph.mean[graph.hidden_slice] = [.2, -.3, .4]
    snapshot = graph.belief_snapshot()
    hidden = graph.mean[graph.hidden_slice].copy()
    energy, gh, gc, eh, es, prediction, a, j = graph._terms(hidden)
    assert energy == pytest.approx(oracle(snapshot)[0], abs=1e-14)
    np.testing.assert_allclose(
        gh, finite_difference(lambda h: oracle(snapshot, hidden=h)[0], hidden), atol=2e-10,
    )
    np.testing.assert_allclose(
        gc, finite_difference(lambda c: oracle(snapshot, canvas=c)[0], graph.canvas), atol=2e-10,
    )
    for name, gradient in zip(PARAMETERS, graph._weight_gradients(hidden)):
        def energy_with_parameter(value):
            return oracle({**snapshot, name: value})[0]

        expected = finite_difference(energy_with_parameter, snapshot[name])
        if name == "recurrent_weights":
            np.fill_diagonal(expected, 0)
            np.testing.assert_array_equal(np.diag(gradient), 0)
        np.testing.assert_allclose(gradient, expected, atol=2e-10, err_msg=name)
    np.testing.assert_allclose(gh, a.T @ eh - j.T @ es, atol=1e-15)
    assert np.all((prediction >= 0) & (prediction <= 1))


def test_forecast_signed_root_copies_pending_guards_and_pre_evidence_predictions():
    graph = make_graph()
    assert not graph.learning_enabled
    assert graph.learning_rate == .01
    assert graph.stage == "posterior"
    with pytest.raises(RuntimeError):
        graph.condition()
    root = np.linspace(-.8, .4, 6).reshape(1, 2, 3)
    saved = root.copy()
    visual, audio = graph.forecast(root, dt=.1)
    np.testing.assert_array_equal(graph.canvas, saved.reshape(-1))
    root.fill(99)
    np.testing.assert_array_equal(graph.canvas, saved.reshape(-1))
    visual.fill(0)
    audio.fill(0)
    assert np.all(graph.visual_prediction > 0)
    assert np.all(graph.audio_prediction > 0)
    before = graph.visual_prediction.copy()
    with pytest.raises(RuntimeError):
        graph.forecast(root, dt=.1)
    graph.condition(np.ones((1, 2, 3)), np.zeros(2))
    np.testing.assert_array_equal(graph.visual_prediction, before)
    graph.reset()
    changed, _ = graph.predict(-saved, dt=.1)
    assert np.linalg.norm(before - changed) > .001
    assert changed.dtype == np.float32
    assert graph.mean.dtype == np.float64


@pytest.mark.parametrize("modality", ["visual", "audio"])
def test_evidence_reaches_root_only_through_inferred_hidden_residual(modality):
    config = PredictiveCodingConfig(inference_steps=80, inference_rate=.5)
    graph = make_graph(config=config)
    graph.canvas_weights[:] = .2
    graph.sensory_weights[:] = .8
    root = np.zeros((1, 2, 3))
    graph.forecast(root, dt=.2)
    evidence = {"visual": np.ones((1, 2, 3))} if modality == "visual" else {"audio": np.ones(2)}
    direction = graph.condition(**evidence)
    snapshot = graph.belief_snapshot()
    assert snapshot["model"] == "predictive-coding"
    assert np.linalg.norm(snapshot["hidden_error"]) > .01
    assert np.all(direction > 0)
    derivative = finite_difference(lambda c: oracle(snapshot, canvas=c)[0], graph.canvas)
    np.testing.assert_allclose(-direction.reshape(-1), derivative, atol=1e-9)
    np.testing.assert_array_equal(snapshot["canvas_gradient"], -direction.reshape(-1))
    h = graph.mean[graph.hidden_slice]
    expected = (graph.canvas_weights.T @ snapshot["hidden_error"]) * (1 - np.tanh(graph.canvas)**2)
    np.testing.assert_allclose(direction.reshape(-1), expected, atol=1e-15)
    assert np.linalg.norm(h) > .01

    stationary = make_graph(config=replace(config, inference_rate=0))
    stationary.canvas_weights[:] = graph.canvas_weights
    stationary.sensory_weights[:] = graph.sensory_weights
    stationary.forecast(root, dt=.2)
    np.testing.assert_array_equal(stationary.condition(**evidence), 0)
    assert np.linalg.norm(stationary.sensory_error) > .1


def test_both_inference_phases_descend_the_same_energy_with_backtracking():
    graph = make_graph(config=PredictiveCodingConfig(
        inference_steps=30, inference_rate=100, gradient_clip=.04,
    ))
    graph.forecast(np.linspace(-2, 2, 6), dt=.01)
    forecast_energy = graph.inference_energy
    assert len(forecast_energy) > 1
    assert np.all(np.diff(forecast_energy) <= 0)
    assert graph.energy == pytest.approx(oracle(graph.belief_snapshot())[0])
    graph.condition(np.zeros((1, 2, 3)), np.ones(2))
    assert len(graph.inference_energy) > 1
    assert np.all(np.diff(graph.inference_energy) <= 0)
    assert graph.inference_energy[-1] < graph.inference_energy[0]
    assert graph.energy == pytest.approx(oracle(graph.belief_snapshot())[0])
    assert np.isfinite(graph.mean).all()


def test_hard_fresh_zero_masking_free_reconstruction_and_no_pseudo_learning():
    graph = make_graph(config=PredictiveCodingConfig(learning_enabled=True))
    root = np.linspace(-.3, .4, 6)
    graph.forecast(root, dt=.1)
    graph.condition(np.array([0, 0, 0, np.nan, np.inf, np.nan]).reshape(1, 2, 3),
                    visual_mask=np.array([[True, False]]))
    np.testing.assert_array_equal(graph.clamped[:3], True)
    np.testing.assert_array_equal(graph.mean[:3], 0)
    np.testing.assert_array_equal(graph.evidence[:3], 0)
    assert np.isnan(graph.evidence[3:]).all()
    snapshot = graph.belief_snapshot()
    _, _, _, prediction, _, _ = oracle(snapshot)
    np.testing.assert_allclose(graph.mean[3:8], prediction[3:])
    np.testing.assert_array_equal(graph.sensory_error[3:], 0)
    np.testing.assert_array_equal(graph.covariance[:3], 0)
    np.testing.assert_array_equal(graph.covariance[:, :3], 0)
    assert graph.updates == 1
    parameters = {name: getattr(graph, name).copy() for name in PARAMETERS}
    for evidence in (
        {}, {"visual": np.zeros((1, 2, 3)), "visual_mask": np.zeros((1, 2), dtype=bool)},
    ):
        graph.forecast(root, dt=.1)
        graph.condition(**evidence)
        assert not np.any(graph.clamped)
        assert np.isnan(graph.evidence).all()
        np.testing.assert_array_equal(graph.sensory_error, 0)
        assert np.all(np.diag(graph.covariance)[:8] > 0)
        for name in PARAMETERS:
            np.testing.assert_array_equal(getattr(graph, name), parameters[name])
    assert graph.updates == 1


def test_learning_is_exact_clipped_local_gradient_decay_and_constraint_projection():
    config = PredictiveCodingConfig(
        learning_enabled=True, learning_rate=.4, inference_steps=3,
        gradient_clip=.02, weight_decay=.03, weight_clip=.15,
    )
    graph = make_graph(config=config)
    graph.forecast(np.linspace(-.5, .5, 6), dt=.1)
    before = {name: getattr(graph, name).copy() for name in PARAMETERS}
    direction = graph.condition(np.ones((1, 2, 3)), np.zeros(2),
                                visual_mask=np.array([[True, False]]))
    snapshot = graph.belief_snapshot()
    assert graph.updates == 1
    hidden = snapshot["mean"][graph.hidden_slice]
    for name in PARAMETERS:
        expected_gradient = finite_difference(
            lambda value: oracle({**snapshot, name: value}, hidden=hidden)[0], before[name],
        )
        if name == "recurrent_weights":
            np.fill_diagonal(expected_gradient, 0)
        expected = np.clip(
            before[name] - config.learning_rate * np.clip(
                expected_gradient + config.weight_decay * before[name],
                -config.gradient_clip, config.gradient_clip,
            ), -config.weight_clip, config.weight_clip,
        )
        if name == "recurrent_weights":
            np.fill_diagonal(expected, 0)
            expected *= min(1, .5 / max(np.linalg.norm(expected), 1e-30))
        np.testing.assert_allclose(getattr(graph, name), expected, atol=1e-10, err_msg=name)
        np.testing.assert_array_equal(snapshot[name], before[name])
    derivative = finite_difference(lambda c: oracle(snapshot, canvas=c)[0], snapshot["canvas"])
    np.testing.assert_allclose(-direction.reshape(-1), derivative, atol=1e-9)
    np.testing.assert_array_equal(graph.sensory_weights[3:6], (
        before["sensory_weights"][3:6] * (1 - config.learning_rate * config.weight_decay)
    ))
    assert np.linalg.norm(graph.recurrent_weights) <= .5 + 1e-15
    np.testing.assert_array_equal(np.diag(graph.recurrent_weights), 0)


def test_recurrent_projection_remains_dense_and_gn_precision_spd():
    graph = make_graph(config=PredictiveCodingConfig(
        learning_enabled=True, learning_rate=10, weight_clip=2,
    ))
    graph.recurrent_weights[:] = 2
    graph._project_recurrent()
    assert np.linalg.norm(graph.recurrent_weights) == pytest.approx(.5)
    for _ in range(3):
        graph.forecast(np.full(6, -.4), dt=.1)
        graph.condition(np.full(6, .9), np.full(2, .2))
        r = graph.recurrent_weights
        assert np.linalg.norm(r) <= .5 + 1e-15
        np.testing.assert_array_equal(np.diag(r), 0)
        assert np.count_nonzero(r) == graph.hidden_nodes * (graph.hidden_nodes - 1)
        _, _, _, _, a, _ = oracle(graph.belief_snapshot())
        assert np.linalg.eigvalsh(a.T @ a).min() >= .25 - 1e-14


def test_freeze_rate_zero_and_reset_preserve_all_weights_biases_and_updates():
    graph = make_graph(config=PredictiveCodingConfig(learning_enabled=True))
    root = np.ones(6)
    graph.forecast(root, dt=.1)
    graph.condition(np.ones(6), np.ones(2))
    assert graph.updates == 1
    parameters = {name: getattr(graph, name).copy() for name in PARAMETERS}
    for enabled, rate in ((False, .1), (True, 0)):
        graph.set_learning(enabled=enabled, rate=rate)
        graph.forecast(root, dt=.1)
        graph.condition(np.zeros(6), np.zeros(2))
        for name in PARAMETERS:
            np.testing.assert_array_equal(getattr(graph, name), parameters[name])
    graph.forecast(root, dt=.1)
    graph.reset()
    assert graph.stage == "posterior"
    assert graph.updates == 1
    np.testing.assert_array_equal(graph.mean[graph.hidden_slice], 0)
    np.testing.assert_array_equal(graph.canvas, 0)
    assert not graph.clamped.any()
    assert np.isnan(graph.evidence).all()
    for name in PARAMETERS:
        np.testing.assert_array_equal(getattr(graph, name), parameters[name])
    graph.forecast(root, dt=.1)
    graph.condition()


def test_local_gn_covariance_equals_dense_free_state_jacobian_inverse():
    graph = make_graph()
    graph.forecast(np.linspace(-.6, .2, 6), dt=.1)
    forecast = graph.belief_snapshot()
    graph.condition(np.linspace(.1, .9, 6), visual_mask=np.array([[True, False]]))
    snapshot: SharedPredictiveCodingSnapshot = graph.belief_snapshot()
    for snap in (forecast, snapshot):
        _, _, _, _, a, j = oracle(snap)
        observed = snap["clamped"][:8]
        free = np.flatnonzero(~observed)
        n = free.size + 3
        residual_jacobian = np.zeros((3 + 8, n))
        residual_jacobian[:3, free.size:] = a
        residual_jacobian[3:, free.size:] = -j
        residual_jacobian[3 + free, np.arange(free.size)] = 1
        dense = np.linalg.inv(residual_jacobian.T @ residual_jacobian)
        indices = np.r_[free, np.arange(8, 11)]
        expected = np.zeros((11, 11))
        expected[np.ix_(indices, indices)] = dense
        np.testing.assert_allclose(snap["covariance"], expected, atol=5e-15)
        np.testing.assert_allclose(snap["variance"], np.diag(expected), atol=5e-15)
        np.testing.assert_array_equal(snap["covariance"][snap["clamped"]], 0)
    np.testing.assert_array_equal(snapshot["prior_covariance"], forecast["covariance"])
    np.testing.assert_array_equal(snapshot["prior_mean"], forecast["mean"])
    snapshot["mean"].fill(999)
    snapshot["canvas_weights"].fill(999)
    assert np.max(graph.mean) < 999
    assert np.max(graph.canvas_weights) < 999
    assert not any(key.startswith("learned_") for key in snapshot)


def test_information_gain_logdet_oracle_uses_same_prior_counterfactual_after_condition():
    graph = make_graph(config=PredictiveCodingConfig(learning_enabled=True))
    graph.forecast(np.linspace(-.8, .7, 6), dt=.1)
    snapshot = graph.belief_snapshot()
    _, _, _, _, a, j = oracle(snapshot)
    selected = np.array([0, 2, 6, 7])
    q = a.T @ a
    expected = .5 * (
        np.linalg.slogdet(q + j[selected].T @ j[selected])[1] - np.linalg.slogdet(q)[1]
    )
    assert graph.information_gain(selected) == pytest.approx(expected, abs=1e-14)
    assert graph.information_gain([0, 0, 2, 6, 7]) == pytest.approx(expected)
    assert graph.information_gain([]) == 0
    assert np.isfinite(graph.information_gain(np.arange(8)))
    graph.condition(np.zeros(6), np.ones(2))
    assert graph.information_gain(selected) == pytest.approx(expected, abs=1e-14)
    assert graph.information_gain(selected, dt=.1) == pytest.approx(expected, abs=1e-14)
    mask = np.zeros(11, dtype=bool)
    mask[selected] = True
    assert graph.information_gain(mask) == pytest.approx(expected, abs=1e-14)
    for bad in ([8], [-1], [1.2], np.ones(4, dtype=bool)):
        with pytest.raises(ValueError):
            graph.information_gain(bad)


@pytest.mark.parametrize("kwargs", [
    {"visual_shape": (5, 1)}, {"visual_shape": (0, 1)}, {"hidden_nodes": 0},
    {"audio_nodes": -1}, {"hidden_nodes": 80}, {"memory_time_constant": 0},
    {"config": PredictiveCodingConfig(dtype="float32")},
    {"config": PredictiveCodingConfig(cross_modal_enabled=True)},
])
def test_invalid_construction(kwargs):
    with pytest.raises(ValueError):
        SharedPredictiveCodingGraph(**kwargs)


def test_validated_config_and_evidence_failures_leave_pending_retryable():
    graph = make_graph()
    original = graph.config
    for changes in (
        {"hidden_channels": 7}, {"dtype": "float32"}, {"inference_rate": -1},
        {"learning_enabled": 1}, {"learning_rate": float("nan")},
    ):
        with pytest.raises(ValueError):
            graph.update_config(**changes)
        assert graph.config == original
    graph.update_config(inference_steps=4, inference_rate=.2, canvas_rate=.7)
    assert graph.config.inference_steps == 4
    for dt in (0, -1, np.nan):
        with pytest.raises(ValueError):
            graph.forecast(np.zeros(6), dt=dt)
    with pytest.raises(ValueError):
        graph.forecast(np.full(6, np.inf), dt=.1)
    graph.forecast(np.zeros(6), dt=.1)
    with pytest.raises(RuntimeError):
        graph.update_config(inference_rate=.4)
    for evidence in (
        {"visual": np.ones(5)}, {"audio": np.ones(3)}, {"audio": np.array([0, np.nan])},
        {"visual": np.full(6, np.inf)}, {"visual_mask": np.ones((1, 2))},
        {"visual_mask": np.ones(6, dtype=bool)},
    ):
        with pytest.raises(ValueError):
            graph.condition(**evidence)
        assert graph.stage == "forecast"
        assert not graph.clamped.any()
    graph.condition()
    no_audio = SharedPredictiveCodingGraph(visual_shape=(1, 1), audio_nodes=0)
    no_audio.forecast(np.zeros(3), dt=.1)
    with pytest.raises(ValueError):
        no_audio.condition(audio=np.empty(0))
    no_audio.condition()
    assert no_audio.audio_prediction.shape == (0,)


def test_hidden_count_is_a_single_structural_config_invariant():
    default = SharedPredictiveCodingGraph()
    assert default.hidden_nodes == default.config.hidden_channels == 8
    assert default.total_nodes == 72
    config = PredictiveCodingConfig()
    configured = SharedPredictiveCodingGraph(hidden_nodes=config.hidden_channels, config=config)
    assert configured.hidden_nodes == configured.config.hidden_channels == 6
    assert configured.total_nodes == 70
    with pytest.raises(ValueError, match="hidden_channels must match"):
        SharedPredictiveCodingGraph(hidden_nodes=8, config=config)
    with pytest.raises(ValueError, match="rebuilding"):
        configured.update_config(hidden_channels=8)


def test_numpy_cupy_parity_for_inference_learning_gradients_curvature_and_snapshots(cupy_backend):
    cp = cupy_backend
    config = PredictiveCodingConfig(
        inference_steps=12, inference_rate=.4, learning_enabled=True,
        learning_rate=.1, weight_decay=.01,
    )
    cpu, gpu = (make_graph(config=config, backend=xp) for xp in (np, cp))
    for frame in range(3):
        root = np.linspace(-.8, .7, 6).reshape(1, 2, 3) + frame * .1
        visual = np.linspace(.1, .9, 6).reshape(1, 2, 3)
        mask = np.array([[True, frame % 2 == 0]])
        scores, directions = [], []
        for graph, xp in ((cpu, np), (gpu, cp)):
            predictions = graph.forecast(xp.asarray(root), dt=.1)
            assert predictions[0].dtype == xp.float32
            scores.append(graph.information_gain([0, 2, 6]))
            directions.append(graph.condition(
                xp.asarray(visual), xp.asarray([.3, .7]) if frame != 1 else None,
                visual_mask=xp.asarray(mask),
            ))
            assert graph.canvas.dtype == xp.float64
        np.testing.assert_allclose(directions[0], cp.asnumpy(directions[1]), atol=1e-12)
        assert scores[0] == pytest.approx(scores[1], abs=1e-13)
        cpu_snap, gpu_snap = cpu.belief_snapshot(), gpu.belief_snapshot()
        for key, expected in cpu_snap.items():
            if isinstance(expected, np.ndarray):
                assert isinstance(gpu_snap[key], np.ndarray)
                np.testing.assert_allclose(gpu_snap[key], expected, atol=1e-12, equal_nan=True)
            elif key in ("energy", "inference_energy"):
                np.testing.assert_allclose(gpu_snap[key], expected, atol=1e-13)
            else:
                assert gpu_snap[key] == expected
        for name in PARAMETERS:
            np.testing.assert_allclose(cp.asnumpy(getattr(gpu, name)), getattr(cpu, name), atol=1e-12)
        np.testing.assert_allclose(
            cp.asnumpy(gpu._terms(gpu.mean[gpu.hidden_slice])[1]),
            cpu._terms(cpu.mean[cpu.hidden_slice])[1], atol=1e-12,
        )
