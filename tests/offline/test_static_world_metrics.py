from __future__ import annotations

import numpy as np
import pytest

from audioviz.offline.static_world_metrics import (
    StaticWorldMetrics,
    measure_static_world_step,
    posterior_visual_prediction,
)
from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.readouts.shared_predictive_coding import SharedPredictiveCodingGraph
from audioviz.readouts.shared_sensory_io import circular_foveal_offsets


def _manual_sigmoid(logits: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * logits))


def _make_foveal_graph(
    *,
    radius: int = 4,
    hidden_nodes: int = 6,
    inference_steps: int = 8,
    inference_rate: float = 0.15,
    learning_enabled: bool = False,
    learning_rate: float = 0.08,
    weight_decay: float = 0.0,
    seed: int = 11,
) -> SharedPredictiveCodingGraph:
    offsets = circular_foveal_offsets(radius)
    config = PredictiveCodingConfig(
        dtype="float64",
        hidden_channels=hidden_nodes,
        inference_steps=inference_steps,
        inference_rate=inference_rate,
        learning_enabled=learning_enabled,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return SharedPredictiveCodingGraph(
        visual_shape=(len(offsets), 1),
        audio_nodes=16,
        hidden_nodes=hidden_nodes,
        config=config,
        seed=seed,
    )


def test_stage_and_argument_validation_guards_against_misuse():
    graph = _make_foveal_graph(radius=2)
    canvas = np.zeros(graph.visual_shape + (3,), dtype=np.float64)
    target = np.full(graph.visual_shape + (3,), 0.75, dtype=np.float64)

    # Freshly initialized or reset graph has stage='posterior' but empty inference_energy
    with pytest.raises(RuntimeError, match="after condition"):
        measure_static_world_step(graph, step=0, dt=1 / 30, gaze=(0.5, 0.5))
    with pytest.raises(RuntimeError, match="after condition"):
        posterior_visual_prediction(graph)

    # After forecast(), stage is 'forecast'
    graph.forecast(canvas, dt=1 / 30)
    with pytest.raises(RuntimeError, match="after condition"):
        measure_static_world_step(graph, step=0, dt=1 / 30, gaze=(0.5, 0.5))
    with pytest.raises(RuntimeError, match="after condition"):
        posterior_visual_prediction(graph)

    graph.condition(target)
    metrics = measure_static_world_step(graph, step=1, dt=0.05, gaze=(0.25, 0.75))
    assert isinstance(metrics, StaticWorldMetrics)
    assert metrics.step == 1
    assert metrics.time == pytest.approx(0.05)
    assert metrics.gaze == (0.25, 0.75)
    assert metrics.canvas_rate == pytest.approx(float(graph.config.canvas_rate))
    assert metrics.memory_time_constant == pytest.approx(float(graph.memory_time_constant))

    for bad_step in (-1, True, 1.5):
        with pytest.raises(ValueError, match="step"):
            measure_static_world_step(graph, step=bad_step, dt=0.05, gaze=(0.5, 0.5))
    for bad_dt in (0.0, -0.1, np.nan, np.inf, False):
        with pytest.raises(ValueError, match="dt"):
            measure_static_world_step(graph, step=0, dt=bad_dt, gaze=(0.5, 0.5))
    for bad_gaze in ((-0.1, 0.5), (0.5, 1.1), (0.5,), (0.5, np.nan), (True, 0.5), "0.5,0.5"):
        with pytest.raises(ValueError, match="gaze"):
            measure_static_world_step(graph, step=0, dt=0.05, gaze=bad_gaze)
    with pytest.raises(ValueError, match="SharedPredictiveCodingGraph"):
        measure_static_world_step(object(), step=0, dt=0.05, gaze=(0.5, 0.5))


def test_manual_residual_mse_agreement_and_pre_learning_snapshot_consistency():
    rng = np.random.default_rng(23)
    graph = _make_foveal_graph(
        radius=3,
        hidden_nodes=6,
        inference_steps=10,
        inference_rate=0.2,
        learning_enabled=True,
        learning_rate=0.15,
        weight_decay=0.01,
        seed=5,
    )
    k_slots = graph.visual_pixels
    canvas = rng.uniform(-0.8, 0.8, size=(k_slots, 1, 3))
    aperture = np.ones((k_slots, 1), dtype=bool)
    aperture[-5:] = False
    visual_target = rng.uniform(0.05, 0.95, size=(k_slots, 1, 3))
    audio_target = rng.uniform(0.0, 1.0, size=16)

    pre_weights = {
        name: getattr(graph, name).copy()
        for name in ("canvas_weights", "recurrent_weights", "sensory_weights", "hidden_bias", "sensory_bias")
    }

    graph.forecast(canvas, dt=0.04, aperture_mask=aperture)
    prior_hidden = graph.prior_mean[graph.hidden_slice].copy()
    v0, v1 = graph.visual_slice.start, graph.visual_slice.stop
    a0, a1 = graph.audio_slice.start, graph.audio_slice.stop
    h0, h1 = graph.hidden_slice.start, graph.hidden_slice.stop

    expected_prior_vis = _manual_sigmoid(
        pre_weights["sensory_weights"][v0:v1] @ np.tanh(prior_hidden)
        + pre_weights["sensory_bias"][v0:v1]
    )
    np.testing.assert_allclose(graph.prior_mean[v0:v1], expected_prior_vis, atol=1e-15)

    graph.condition(visual_target, audio_target, visual_mask=aperture)
    metrics = measure_static_world_step(graph, step=3, dt=0.04, gaze=(0.4, 0.6))
    snapshot = graph.belief_snapshot()

    # Clamped graph.mean[v0:v1] equals evidence on observed coordinates, NOT posterior prediction
    mask_rgb = np.repeat(aperture.reshape(-1), 3)
    np.testing.assert_allclose(
        graph.mean[v0:v1][mask_rgb],
        visual_target.reshape(-1)[mask_rgb],
        atol=1e-15,
    )

    # Snapshot weights must match pre-learning weights, while graph public weights moved
    for name, before in pre_weights.items():
        np.testing.assert_allclose(snapshot[name], before, atol=1e-15)
    assert not np.allclose(graph.sensory_weights, pre_weights["sensory_weights"])
    assert metrics.learning_updates == 1

    # Posterior prediction decoded with PRE-learning weights at settled hidden state
    post_hidden = snapshot["mean"][h0:h1]
    expected_post_sensory = _manual_sigmoid(
        pre_weights["sensory_weights"] @ np.tanh(post_hidden) + pre_weights["sensory_bias"]
    )
    expected_post_vis = expected_post_sensory[v0:v1].reshape(k_slots, 1, 3)
    post_vis = posterior_visual_prediction(graph)
    assert post_vis.shape == (k_slots, 1, 3)
    np.testing.assert_allclose(post_vis, expected_post_vis, atol=1e-15)
    np.testing.assert_allclose(posterior_visual_prediction(snapshot), expected_post_vis, atol=1e-15)

    # Disagrees with post-update weights because learning updated parameters after capture
    post_update_vis = _manual_sigmoid(
        graph.sensory_weights[v0:v1] @ np.tanh(post_hidden) + graph.sensory_bias[v0:v1]
    ).reshape(k_slots, 1, 3)
    assert not np.allclose(post_vis, post_update_vis)

    # Manual MSEs on observed RGB coordinates only
    obs_target = visual_target.reshape(-1)[mask_rgb]
    expected_forecast_mse = float(np.mean((obs_target - expected_prior_vis[mask_rgb]) ** 2))
    expected_settled_mse = float(
        np.mean((obs_target - expected_post_vis.reshape(-1)[mask_rgb]) ** 2)
    )
    assert metrics.observed_rgb == int(np.count_nonzero(mask_rgb))
    assert metrics.forecast_mse == pytest.approx(expected_forecast_mse, abs=1e-15)
    assert metrics.settled_mse == pytest.approx(expected_settled_mse, abs=1e-15)
    assert metrics.settled_mse < metrics.forecast_mse

    # Manual energies and root gradient norm at pre-learning weights
    root_activity = np.where(mask_rgb, np.tanh(canvas.reshape(-1)), 0.0)
    expected_eh = (
        post_hidden
        - pre_weights["canvas_weights"] @ root_activity
        - pre_weights["recurrent_weights"] @ np.tanh(post_hidden)
        - pre_weights["hidden_bias"]
    )
    expected_es = np.zeros(graph.visual_nodes + graph.audio_nodes, dtype=np.float64)
    expected_es[v0:v1][mask_rgb] = obs_target - expected_post_sensory[v0:v1][mask_rgb]
    expected_es[a0:a1] = audio_target - expected_post_sensory[a0:a1]
    expected_hidden_energy = 0.5 * float(expected_eh @ expected_eh)
    expected_sensory_energy = 0.5 * float(expected_es @ expected_es)
    expected_root_grad = np.where(
        mask_rgb,
        -(pre_weights["canvas_weights"].T @ expected_eh) * (1.0 - np.tanh(canvas.reshape(-1)) ** 2),
        0.0,
    )

    assert metrics.hidden_energy == pytest.approx(expected_hidden_energy, abs=1e-14)
    assert metrics.sensory_energy == pytest.approx(expected_sensory_energy, abs=1e-14)
    assert metrics.energy_final == pytest.approx(
        expected_hidden_energy + expected_sensory_energy, abs=1e-14
    )
    assert metrics.energy_initial == pytest.approx(metrics.inference_energy[0], abs=1e-15)
    assert metrics.energy_final == pytest.approx(metrics.inference_energy[-1], abs=1e-15)
    assert np.all(np.diff(metrics.inference_energy) <= 0.0)
    assert metrics.canvas_gradient_norm == pytest.approx(
        float(np.linalg.norm(expected_root_grad)), abs=1e-14
    )


def test_zero_inference_rate_and_no_evidence_controls():
    rng = np.random.default_rng(31)
    canvas = rng.uniform(-0.6, 0.6, size=(29, 1, 3))
    target = rng.uniform(0.1, 0.9, size=(29, 1, 3))

    # inference_rate = 0 control: forecast_mse == settled_mse and single-point trajectory
    zero_rate_graph = _make_foveal_graph(
        radius=3,
        inference_steps=6,
        inference_rate=0.0,
        learning_enabled=False,
        seed=9,
    )
    zero_rate_graph.forecast(canvas, dt=0.05)
    zero_rate_graph.condition(target)
    zero_metrics = measure_static_world_step(zero_rate_graph, step=0, dt=0.05, gaze=(0.5, 0.5))
    assert zero_metrics.inference_rate == 0.0
    assert len(zero_metrics.inference_energy) == 1
    assert zero_metrics.energy_initial == pytest.approx(zero_metrics.energy_final, abs=1e-15)
    assert zero_metrics.forecast_mse is not None
    assert zero_metrics.settled_mse == pytest.approx(zero_metrics.forecast_mse, abs=1e-15)

    # No-evidence control: observed_rgb == 0, MSEs are None, sensory_energy == 0, no learning updates
    # (With finite settling steps, hidden_energy and canvas_gradient_norm may remain positive.)
    no_obs_graph = _make_foveal_graph(
        radius=3,
        inference_steps=1,
        inference_rate=0.05,
        learning_enabled=True,
        learning_rate=0.2,
        seed=9,
    )
    before_weights = no_obs_graph.sensory_weights.copy()
    no_obs_graph.forecast(canvas, dt=0.05)
    no_obs_graph.condition(None, None)
    no_obs_metrics = measure_static_world_step(no_obs_graph, step=0, dt=0.05, gaze=(0.5, 0.5))
    assert no_obs_metrics.observed_rgb == 0
    assert no_obs_metrics.forecast_mse is None
    assert no_obs_metrics.settled_mse is None
    assert no_obs_metrics.sensory_energy == 0.0
    assert no_obs_metrics.energy_final == pytest.approx(no_obs_metrics.hidden_energy, abs=1e-15)
    assert no_obs_metrics.learning_updates == 0
    np.testing.assert_array_equal(no_obs_graph.sensory_weights, before_weights)


def test_deterministic_seeded_trials_and_controlled_clone_comparisons():
    rng = np.random.default_rng(42)
    base_graph = _make_foveal_graph(
        radius=4,
        hidden_nodes=6,
        inference_steps=8,
        inference_rate=0.15,
        learning_enabled=False,
        learning_rate=0.1,
        seed=17,
    )
    canvas = rng.uniform(-0.5, 0.5, size=base_graph.visual_shape + (3,))
    target = rng.uniform(0.15, 0.85, size=base_graph.visual_shape + (3,))

    # Deterministic fresh seeded trials produce identical metric sequences
    def run_trial(*, learning_enabled: bool, steps: int = 12) -> list[StaticWorldMetrics]:
        trial_graph = _make_foveal_graph(
            radius=4,
            hidden_nodes=6,
            inference_steps=8,
            inference_rate=0.15,
            learning_enabled=learning_enabled,
            learning_rate=0.12,
            seed=17,
        )
        history = []
        for step_idx in range(steps):
            trial_graph.forecast(canvas, dt=1 / 30)
            trial_graph.condition(target)
            history.append(
                measure_static_world_step(
                    trial_graph, step=step_idx, dt=1 / 30, gaze=(0.5, 0.5)
                )
            )
        return history

    run_a = run_trial(learning_enabled=True, steps=10)
    run_b = run_trial(learning_enabled=True, steps=10)
    assert run_a == run_b

    fixed_run = run_trial(learning_enabled=False, steps=10)
    for item in (*run_a, *fixed_run):
        assert np.all(np.diff(item.inference_energy) <= 0.0)
        assert item.energy_final <= item.energy_initial

    # On this static patch with held root, learning reduces pre-evidence forecast MSE vs step 0
    assert run_a[-1].forecast_mse < run_a[0].forecast_mse
    assert run_a[-1].learning_updates == 10
    assert fixed_run[-1].learning_updates == 0

    # Fresh clones at the identical pre-condition state (e.g. zero canvas root -> h_prior = 0)
    # isolate single-condition inference_steps sweeps on the exact same initial state/objective
    zero_canvas = np.zeros_like(canvas)
    final_energies = []
    settled_mses = []
    for steps in (1, 4, 12):
        clone = _make_foveal_graph(
            radius=4,
            hidden_nodes=6,
            inference_steps=steps,
            inference_rate=0.15,
            learning_enabled=False,
            seed=17,
        )
        clone.forecast(zero_canvas, dt=1 / 30)
        np.testing.assert_allclose(clone.mean[clone.hidden_slice], 0.0, atol=1e-15)
        clone.condition(target)
        m = measure_static_world_step(clone, step=0, dt=1 / 30, gaze=(0.5, 0.5))
        assert np.all(np.diff(m.inference_energy) <= 0.0)
        final_energies.append(m.energy_final)
        settled_mses.append(m.settled_mse)
    assert final_energies[1] <= final_energies[0]
    assert final_energies[2] <= final_energies[1]
    assert settled_mses[2] < settled_mses[0]


def test_cupy_backend_parity():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(51)
    offsets = circular_foveal_offsets(3)
    cfg = PredictiveCodingConfig(
        dtype="float64",
        hidden_channels=6,
        inference_steps=6,
        inference_rate=0.15,
        learning_enabled=True,
        learning_rate=0.1,
    )
    np_graph = SharedPredictiveCodingGraph(
        visual_shape=(len(offsets), 1), audio_nodes=16, hidden_nodes=6, config=cfg, backend=np, seed=3,
    )
    cp_graph = SharedPredictiveCodingGraph(
        visual_shape=(len(offsets), 1), audio_nodes=16, hidden_nodes=6, config=cfg, backend=cp, seed=3,
    )
    canvas = rng.uniform(-0.7, 0.7, size=(len(offsets), 1, 3))
    target = rng.uniform(0.1, 0.9, size=(len(offsets), 1, 3))
    mask = np.ones((len(offsets), 1), dtype=bool)
    mask[:4] = False

    np_graph.forecast(canvas, dt=0.05, aperture_mask=mask)
    np_graph.condition(target, visual_mask=mask)
    cp_graph.forecast(cp.asarray(canvas), dt=0.05, aperture_mask=cp.asarray(mask))
    cp_graph.condition(cp.asarray(target), visual_mask=cp.asarray(mask))

    m_np = measure_static_world_step(np_graph, step=2, dt=0.05, gaze=(0.3, 0.7))
    m_cp = measure_static_world_step(cp_graph, step=2, dt=0.05, gaze=(0.3, 0.7))
    assert m_cp.step == m_np.step
    assert m_cp.time == pytest.approx(m_np.time)
    assert m_cp.gaze == m_np.gaze
    assert m_cp.observed_rgb == m_np.observed_rgb
    assert m_cp.forecast_mse == pytest.approx(m_np.forecast_mse, abs=1e-14)
    assert m_cp.settled_mse == pytest.approx(m_np.settled_mse, abs=1e-14)
    assert m_cp.energy_initial == pytest.approx(m_np.energy_initial, abs=1e-14)
    assert m_cp.energy_final == pytest.approx(m_np.energy_final, abs=1e-14)
    assert m_cp.hidden_energy == pytest.approx(m_np.hidden_energy, abs=1e-14)
    assert m_cp.sensory_energy == pytest.approx(m_np.sensory_energy, abs=1e-14)
    assert m_cp.canvas_gradient_norm == pytest.approx(m_np.canvas_gradient_norm, abs=1e-14)
    assert m_cp.learning_updates == m_np.learning_updates
    assert m_cp.canvas_rate == m_np.canvas_rate
    assert m_cp.memory_time_constant == m_np.memory_time_constant
    np.testing.assert_allclose(m_cp.inference_energy, m_np.inference_energy, atol=1e-14)
    np.testing.assert_allclose(
        posterior_visual_prediction(cp_graph),
        posterior_visual_prediction(np_graph),
        atol=1e-14,
    )
