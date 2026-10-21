"""Unit tests for the continuous RC-spiking predictive coding generative graph."""
from __future__ import annotations

import numpy as np
import pytest

from audioviz.offline.static_world_metrics import posterior_visual_prediction
from audioviz.readouts import (
    PredictiveCodingConfig,
    RCSpikingPredictiveCodingGraph,
    SharedPredictiveCodingGraph,
)
from audioviz.readouts.shared_sensory_io import circular_foveal_offsets
from audioviz.utils.array_backend import to_numpy


def _make_rc_graph(
    *,
    radius: int = 3,
    audio_nodes: int = 8,
    hidden_nodes: int = 6,
    inference_steps: int = 8,
    inference_rate: float = 0.1,
    learning_enabled: bool = False,
    learning_rate: float = 0.05,
    memory_time_constant: float = 0.25,
    spike_threshold: float = 0.02,
    backend=np,
    seed: int = 0,
) -> RCSpikingPredictiveCodingGraph:
    offsets = circular_foveal_offsets(radius)
    config = PredictiveCodingConfig(
        dtype="float64",
        hidden_channels=hidden_nodes,
        inference_steps=inference_steps,
        inference_rate=inference_rate,
        learning_enabled=learning_enabled,
        learning_rate=learning_rate,
    )
    return RCSpikingPredictiveCodingGraph(
        visual_shape=(len(offsets), 1),
        audio_nodes=audio_nodes,
        hidden_nodes=hidden_nodes,
        config=config,
        memory_time_constant=memory_time_constant,
        spike_threshold=spike_threshold,
        backend=backend,
        seed=seed,
    )


def test_seed_initialization_matches_shared_pc_and_rejects_invalid_arguments():
    rc_graph = _make_rc_graph(radius=2, audio_nodes=4, hidden_nodes=5, seed=11)
    pc_graph = SharedPredictiveCodingGraph(
        visual_shape=rc_graph.visual_shape,
        audio_nodes=4,
        hidden_nodes=5,
        config=rc_graph.config,
        memory_time_constant=rc_graph.memory_time_constant,
        seed=11,
    )
    for name in rc_graph._PARAMETERS:
        np.testing.assert_allclose(getattr(rc_graph, name), getattr(pc_graph, name), atol=1e-15)

    assert not rc_graph.has_gaussian_uncertainty
    for bad in (True, 0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="spike_threshold"):
            _make_rc_graph(spike_threshold=bad)
        with pytest.raises(ValueError, match="memory_time_constant"):
            _make_rc_graph(memory_time_constant=bad)

    canvas = np.zeros(rc_graph.visual_shape + (3,))
    with pytest.raises(RuntimeError, match="Gaussian candidate information gain is unavailable"):
        rc_graph.candidate_information_gain(canvas, dt=1 / 30)
    with pytest.raises(RuntimeError, match="Gaussian candidate information gain is unavailable"):
        rc_graph.candidate_information_gains(canvas[None, ...], dt=1 / 30)


def test_subtractive_reset_transfers_charge_from_voltage_to_decoded_belief():
    graph = _make_rc_graph(
        radius=2,
        hidden_nodes=4,
        inference_steps=1,
        inference_rate=0.2,
        memory_time_constant=0.5,
        spike_threshold=0.02,
        seed=5,
    )
    canvas = np.zeros(graph.visual_shape + (3,))
    # Set a strong positive visual target so conditioning drives error & hidden spikes
    target = np.full(graph.visual_shape + (3,), 0.95)
    graph.forecast(canvas, dt=0.05)
    assert graph.stage == "forecast"
    assert graph.forecast_hidden_spikes == 0
    np.testing.assert_allclose(graph.mean[graph.hidden_slice], 0.0, atol=1e-15)

    graph.condition(target)
    snapshot = graph.belief_snapshot()
    assert snapshot["model"] == "rc-spiking"
    assert snapshot["stage"] == "posterior"
    assert snapshot["condition_error_spikes"] > 0
    assert snapshot["condition_hidden_spikes"] > 0

    # Emitted spikes are strictly ternary in {-1, 0, +1}
    for spikes in (snapshot["last_hidden_spikes"], snapshot["last_error_spikes"]):
        assert set(np.unique(spikes)).issubset({-1.0, 0.0, 1.0})

    # With 1 conditioning micro-step from h=0, decoded hidden state equals theta * s_h
    expected_decoded = graph.spike_threshold * snapshot["last_hidden_spikes"]
    np.testing.assert_allclose(
        snapshot["mean"][slice(*snapshot["hidden_slice"])],
        expected_decoded,
        atol=1e-15,
    )
    # Membrane potential was subtractively reset by theta * s_h rather than erasing decoded h
    assert np.linalg.norm(snapshot["mean"][slice(*snapshot["hidden_slice"])]) > 0.0


def test_subthreshold_constant_current_is_invariant_across_microsteps():
    tau = 0.35
    cap = 0.02
    dt = 1 / 30
    rate = 0.1
    # High spike threshold keeps the membrane strictly in the linear subthreshold regime
    high_threshold = 10.0

    ref_graph = _make_rc_graph(
        radius=2,
        hidden_nodes=5,
        inference_steps=1,
        inference_rate=rate,
        memory_time_constant=tau,
        spike_threshold=high_threshold,
        seed=14,
    )
    rng = np.random.default_rng(21)
    canvas = rng.uniform(-0.4, 0.4, size=ref_graph.visual_shape + (3,))
    target = rng.uniform(0.2, 0.8, size=ref_graph.visual_shape + (3,))
    v_h0 = rng.uniform(-0.05, 0.05, size=(ref_graph.hidden_nodes,))
    v_e0 = rng.uniform(-0.05, 0.05, size=(ref_graph._sensory_nodes,))

    root_act = np.tanh(canvas.reshape(-1))
    ext_current = ref_graph.canvas_weights @ root_act + ref_graph.hidden_bias
    expected_alpha = np.exp(-dt / tau)
    expected_beta = (tau * (1.0 - expected_alpha) / cap) * rate
    expected_vh_forecast = expected_alpha * v_h0 + expected_beta * ext_current

    # At h = 0 (since no spikes fire), sensory error e_s is constant across microsteps
    pred_at_zero = 0.5 * (1.0 + np.tanh(0.5 * ref_graph.sensory_bias))
    es_const = np.zeros(ref_graph._sensory_nodes, dtype=np.float64)
    es_const[ref_graph.visual_slice] = target.reshape(-1) - pred_at_zero[ref_graph.visual_slice]
    expected_ve_condition = np.zeros_like(v_e0)
    expected_ve_condition[ref_graph.visual_slice] = (
        expected_alpha * v_e0[ref_graph.visual_slice]
        + expected_beta * es_const[ref_graph.visual_slice]
    )

    for steps in (1, 2, 5, 12):
        graph = _make_rc_graph(
            radius=2,
            hidden_nodes=5,
            inference_steps=steps,
            inference_rate=rate,
            memory_time_constant=tau,
            spike_threshold=high_threshold,
            seed=14,
        )
        graph.hidden_potential[:] = v_h0
        graph.error_potential[:] = v_e0
        graph.forecast(canvas, dt=dt)
        assert graph.forecast_hidden_spikes == 0
        np.testing.assert_allclose(graph.hidden_potential, expected_vh_forecast, atol=1e-14)
        np.testing.assert_allclose(
            graph.error_potential, expected_alpha * v_e0, atol=1e-14,
        )

        # Replay the same [t, t + dt] interval with observed visual target
        graph.condition(target)
        assert graph.condition_error_spikes == 0
        assert graph.condition_hidden_spikes == 0
        # Because s_e == 0, hidden_potential after condition() equals the single-dt forecast!
        np.testing.assert_allclose(graph.hidden_potential, expected_vh_forecast, atol=1e-14)
        np.testing.assert_allclose(graph.error_potential, expected_ve_condition, atol=1e-14)


def test_single_clock_dt_progression_with_and_without_observation():
    tau = 0.25
    dt = 0.05
    # Verify that conditioning with subthreshold evidence advances the RC leak by
    # exactly ONE dt (matching Observe off), never two dt intervals.
    obs_graph = _make_rc_graph(
        radius=2,
        hidden_nodes=4,
        inference_steps=6,
        inference_rate=0.1,
        memory_time_constant=tau,
        spike_threshold=5.0,
        seed=19,
    )
    off_graph = _make_rc_graph(
        radius=2,
        hidden_nodes=4,
        inference_steps=6,
        inference_rate=0.1,
        memory_time_constant=tau,
        spike_threshold=5.0,
        seed=19,
    )
    h_init = np.array([0.4, -0.3, 0.2, -0.5], dtype=np.float64)
    vh_init = np.array([0.1, -0.2, 0.05, 0.15], dtype=np.float64)
    for g in (obs_graph, off_graph):
        g.mean[g.hidden_slice] = h_init
        g.hidden_potential[:] = vh_init

    zero_canvas = np.zeros(obs_graph.visual_shape + (3,))
    target = np.full(obs_graph.visual_shape + (3,), 0.6)

    obs_graph.forecast(zero_canvas, dt=dt)
    obs_graph.condition(target)

    off_graph.forecast(zero_canvas, dt=dt)
    off_graph.condition(None, None)

    expected_h = np.exp(-dt / tau) * h_init
    expected_vh = np.exp(-dt / tau) * vh_init
    np.testing.assert_allclose(obs_graph.mean[obs_graph.hidden_slice], expected_h, atol=1e-14)
    np.testing.assert_allclose(off_graph.mean[off_graph.hidden_slice], expected_h, atol=1e-14)
    np.testing.assert_allclose(obs_graph.hidden_potential, expected_vh, atol=1e-14)
    np.testing.assert_allclose(off_graph.hidden_potential, expected_vh, atol=1e-14)


def test_unforced_exact_exponential_rc_decay_toward_neutral_half():
    tau = 0.2
    dt = 0.04
    # Verify exact exp(-dt / tau) decay across different inference_steps (sub-stepping invariance)
    for steps in (1, 4, 10):
        graph = _make_rc_graph(
            radius=2,
            hidden_nodes=6,
            inference_steps=steps,
            inference_rate=0.12,
            memory_time_constant=tau,
            spike_threshold=0.02,
            seed=7,
        )
        zero_canvas = np.zeros(graph.visual_shape + (3,))
        target = np.full(graph.visual_shape + (3,), 0.9)

        # Charge the decoded RC belief state with 2 steps of sensory evidence
        for _ in range(2):
            graph.forecast(zero_canvas, dt=dt)
            graph.condition(target)

        # Clear sub-threshold residual voltage to test pure unforced RC capacitor decay
        # (r = 0, b_h = 0, b_s = 0, no sensory evidence, v_h = 0 -> zero spikes)
        graph.hidden_potential.fill(0.0)
        graph.error_potential.fill(0.0)
        graph.last_hidden_spikes.fill(0.0)
        h_initial = graph.mean[graph.hidden_slice].copy()
        assert np.linalg.norm(h_initial) > 0.02

        prev_mae = float(np.mean(np.abs(posterior_visual_prediction(graph) - 0.5)))
        assert prev_mae > 1e-3

        for k in range(1, 6):
            visual_pred, _ = graph.forecast(zero_canvas, dt=dt)
            graph.condition(None, None)
            assert graph.forecast_hidden_spikes == 0
            assert graph.condition_hidden_spikes == 0
            assert graph.condition_error_spikes == 0
            expected_h = np.exp(-k * dt / tau) * h_initial
            np.testing.assert_allclose(
                graph.mean[graph.hidden_slice],
                expected_h,
                rtol=1e-13,
                atol=1e-15,
            )
            mae = float(np.mean(np.abs(visual_pred - 0.5)))
            assert mae < prev_mae
            prev_mae = mae


def test_tonic_forcing_sustains_predictions_when_observe_is_off():
    dt = 1 / 30
    # 1. Held nonzero root canvas sustains tonic hidden spikes and non-neutral predictions
    root_forced = _make_rc_graph(
        radius=2,
        hidden_nodes=6,
        inference_steps=8,
        inference_rate=0.15,
        memory_time_constant=0.15,
        spike_threshold=0.015,
        seed=3,
    )
    nonzero_canvas = np.full(root_forced.visual_shape + (3,), 0.85)
    total_tonic_spikes = 0
    for _ in range(8):
        root_forced.forecast(nonzero_canvas, dt=dt)
        root_forced.condition(None, None)
        total_tonic_spikes += root_forced.forecast_hidden_spikes
        assert root_forced.condition_error_spikes == 0
        assert root_forced.condition_hidden_spikes == 0

    assert total_tonic_spikes > 0
    assert np.linalg.norm(root_forced.mean[root_forced.hidden_slice]) > 0.01
    assert float(np.mean(np.abs(root_forced.visual_prediction - 0.5))) > 1e-3

    # 2. Learned/nonzero sensory_bias sustains non-neutral prediction even when h -> 0
    bias_forced = _make_rc_graph(
        radius=2,
        hidden_nodes=6,
        inference_steps=6,
        inference_rate=0.1,
        memory_time_constant=0.05,
        spike_threshold=0.02,
        seed=3,
    )
    bias_forced.sensory_bias[: bias_forced.visual_nodes] = 0.8
    zero_canvas = np.zeros(bias_forced.visual_shape + (3,))
    for _ in range(6):
        bias_forced.forecast(zero_canvas, dt=dt)
        bias_forced.condition(None, None)
    np.testing.assert_allclose(bias_forced.mean[bias_forced.hidden_slice], 0.0, atol=1e-15)
    expected_val = 1.0 / (1.0 + np.exp(-0.8))
    np.testing.assert_allclose(bias_forced.visual_prediction, expected_val, atol=1e-6)


def test_learned_biases_remain_tonic_drives_without_observation():
    graph = _make_rc_graph(
        radius=2, audio_nodes=0, learning_enabled=True, learning_rate=.03,
        memory_time_constant=1/3,
    )
    canvas = np.zeros(graph.visual_shape + (3,))
    target = np.full_like(canvas, .9)
    for _ in range(160):
        graph.forecast(canvas, dt=1/30)
        graph.condition(target)
    assert np.linalg.norm(graph.hidden_bias) > .1
    assert np.linalg.norm(graph.sensory_bias) > .1
    learned = {name: getattr(graph, name).copy() for name in graph._PARAMETERS}
    graph.set_learning(False)

    # Disabling learning stops parameter updates, not currents from learned biases.
    for _ in range(90):
        graph.forecast(canvas, dt=1/30)
        graph.condition()
        assert graph.condition_error_spikes == 0
    assert graph.committed_hidden_spikes > 0
    assert np.linalg.norm(graph.mean[graph.hidden_slice]) > .1
    assert np.mean(np.abs(graph.visual_prediction - .5)) > .1
    assert graph.updates == 160
    for name, parameter in learned.items():
        np.testing.assert_array_equal(getattr(graph, name), parameter)


def test_reset_config_updates_and_pre_learning_snapshot_consistency():
    graph = _make_rc_graph(
        radius=2,
        hidden_nodes=5,
        inference_steps=6,
        inference_rate=0.15,
        learning_enabled=True,
        learning_rate=0.2,
        memory_time_constant=0.25,
        spike_threshold=0.02,
        seed=9,
    )
    canvas = np.zeros(graph.visual_shape + (3,))
    target = np.full(graph.visual_shape + (3,), 0.85)
    initial_sensory_weights = graph.sensory_weights.copy()

    graph.forecast(canvas, dt=1 / 30)
    with pytest.raises(RuntimeError, match="pending forecast"):
        graph.set_spike_threshold(0.05)
    with pytest.raises(RuntimeError, match="pending forecast"):
        graph.set_memory_time_constant(0.4)
    with pytest.raises(RuntimeError, match="pending forecast"):
        graph.update_config(inference_rate=0.05)

    graph.condition(target)
    assert graph.update_count == 1
    assert not np.allclose(graph.sensory_weights, initial_sensory_weights)

    # Snapshot captures pre-learning weights from before the step's weight update
    snap = graph.belief_snapshot()
    np.testing.assert_allclose(snap["sensory_weights"], initial_sensory_weights, atol=1e-15)
    post_pred = posterior_visual_prediction(graph).reshape(-1)
    np.testing.assert_allclose(
        target.reshape(-1) - post_pred,
        snap["sensory_error"][slice(*snap["visual_slice"])],
        atol=1e-14,
    )

    graph.update_config(spike_threshold=0.04, memory_time_constant=0.5, inference_steps=4)
    assert graph.spike_threshold == pytest.approx(0.04)
    assert graph.memory_time_constant == pytest.approx(0.5)
    assert graph.config.inference_steps == 4

    # reset() clears dynamic RC state and spike counters while retaining learned weights
    learned_weights = graph.sensory_weights.copy()
    graph.reset()
    assert graph.stage == "posterior"
    assert graph.forecast_hidden_spikes == 0
    assert graph.condition_hidden_spikes == 0
    assert graph.condition_error_spikes == 0
    assert graph.inference_energy == ()
    np.testing.assert_allclose(graph.mean[graph.hidden_slice], 0.0, atol=1e-15)
    np.testing.assert_allclose(graph.hidden_potential, 0.0, atol=1e-15)
    np.testing.assert_allclose(graph.error_potential, 0.0, atol=1e-15)
    np.testing.assert_allclose(graph.sensory_weights, learned_weights, atol=1e-15)


def test_cupy_backend_parity():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(19)
    np_graph = _make_rc_graph(
        radius=3,
        hidden_nodes=6,
        inference_steps=6,
        inference_rate=0.12,
        learning_enabled=True,
        learning_rate=0.08,
        spike_threshold=0.02,
        backend=np,
        seed=13,
    )
    cp_graph = _make_rc_graph(
        radius=3,
        hidden_nodes=6,
        inference_steps=6,
        inference_rate=0.12,
        learning_enabled=True,
        learning_rate=0.08,
        spike_threshold=0.02,
        backend=cp,
        seed=13,
    )
    canvas = rng.uniform(-0.6, 0.6, size=np_graph.visual_shape + (3,))
    target = rng.uniform(0.1, 0.9, size=np_graph.visual_shape + (3,))
    audio = rng.uniform(-0.5, 0.5, size=(np_graph.audio_nodes,))

    for _ in range(3):
        np_graph.forecast(canvas, dt=1 / 30)
        dir_np = np_graph.condition(target, audio)
        cp_graph.forecast(cp.asarray(canvas), dt=1 / 30)
        dir_cp = cp_graph.condition(cp.asarray(target), cp.asarray(audio))
        np.testing.assert_allclose(to_numpy(dir_cp), dir_np, atol=1e-14)

    snap_np = np_graph.belief_snapshot()
    snap_cp = cp_graph.belief_snapshot()
    assert snap_cp["forecast_hidden_spikes"] == snap_np["forecast_hidden_spikes"]
    assert snap_cp["condition_hidden_spikes"] == snap_np["condition_hidden_spikes"]
    assert snap_cp["condition_error_spikes"] == snap_np["condition_error_spikes"]
    assert snap_cp["hidden_spike_trajectory"] == snap_np["hidden_spike_trajectory"]
    assert snap_cp["error_spike_trajectory"] == snap_np["error_spike_trajectory"]
    for key in (
        "mean", "prior_mean", "hidden_potential", "error_potential",
        "sensory_error", "hidden_error", "canvas_gradient",
    ):
        np.testing.assert_allclose(snap_cp[key], snap_np[key], atol=1e-14)
