import numpy as np
import pytest

from audioviz.readouts.continuous_scene_predictive_coding import (
    ContinuousScenePredictiveCodingGraph,
)
from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.utils.array_backend import to_numpy


def test_continuous_scene_graph_supports_full_80x120_world_and_rejects_oversized_configs():
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(80, 120),
        audio_nodes=16,
        hidden_nodes=6,
        config=PredictiveCodingConfig(hidden_channels=6, learning_enabled=False),
    )
    assert graph.visual_pixels == 9600
    assert graph.visual_nodes == 28800
    assert graph.visual_potential.shape == (28800,)
    assert graph.prior_visual_potential.shape == (28800,)
    assert not hasattr(graph, "covariance")
    assert not hasattr(graph, "prior_covariance")
    assert not hasattr(graph, "spike_threshold")
    assert not hasattr(graph, "membrane_capacitance")

    with pytest.raises(ValueError, match="visual_shape"):
        ContinuousScenePredictiveCodingGraph(visual_shape=(150, 150))
    with pytest.raises(ValueError, match="audio_nodes"):
        ContinuousScenePredictiveCodingGraph(audio_nodes=512)
    with pytest.raises(ValueError, match="hidden_nodes"):
        ContinuousScenePredictiveCodingGraph(
            hidden_nodes=128,
            config=PredictiveCodingConfig(hidden_channels=128),
        )
    with pytest.raises(ValueError, match="memory_time_constant"):
        ContinuousScenePredictiveCodingGraph(memory_time_constant=0.0)
    with pytest.raises(ValueError, match="memory_time_constant"):
        ContinuousScenePredictiveCodingGraph(memory_time_constant=True)
    with pytest.raises(RuntimeError, match="candidate information gain"):
        graph.candidate_information_gains(np.zeros((2, 80, 120, 3)), 1 / 30)
    with pytest.raises(RuntimeError, match="candidate information gain"):
        graph.candidate_information_gain(np.zeros((80, 120, 3)), 1 / 30)


def test_conditioned_energy_trace_starts_at_replayed_interval_start():
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(4, 4), audio_nodes=0, hidden_nodes=6,
        config=PredictiveCodingConfig(hidden_channels=6, learning_enabled=False),
    )
    canvas = np.zeros((4, 4, 3))
    for _ in range(10):
        graph.forecast(canvas, 1 / 30)
        graph.condition(np.ones_like(canvas))

    graph.forecast(canvas, 1 / 30)
    initial_hidden, initial_z = (value.copy() for value in graph._interval_start)
    prior_hidden = graph.prior_mean[graph.hidden_slice].copy()
    prior_z = graph.prior_sensory_potential.copy()
    graph.condition(np.zeros_like(canvas))

    def energy(hidden, z):
        return graph._terms_for(
            hidden, z, graph.canvas, graph.aperture_mask,
            graph.clamped, graph.evidence,
        )[0]

    assert graph.inference_energy[0] == pytest.approx(energy(initial_hidden, initial_z))
    assert graph.inference_energy[0] != pytest.approx(energy(prior_hidden, prior_z))
    assert graph.inference_energy[-1] == graph.energy
    assert len(graph.inference_energy) == graph.config.inference_steps + 1


def test_explicit_energy_matches_analytic_gradients_including_both_h_and_z_leak():
    rng = np.random.default_rng(7)
    for rate in (0.02, 0.1, 0.25):
        graph = ContinuousScenePredictiveCodingGraph(
            visual_shape=(4, 4),
            audio_nodes=4,
            hidden_nodes=6,
            config=PredictiveCodingConfig(
                hidden_channels=6,
                inference_rate=rate,
                learning_enabled=False,
            ),
        )
        h0 = rng.normal(0.0, 0.35, size=graph.hidden_nodes)
        z0 = rng.normal(0.0, 0.4, size=graph._sensory_nodes)
        canvas = rng.normal(0.0, 0.25, size=graph.visual_nodes)
        aperture = np.ones(graph.visual_nodes, dtype=bool)
        aperture[3:9] = False
        clamped = np.zeros(graph.total_nodes, dtype=bool)
        clamped[:24] = True
        clamped[graph.visual_nodes:graph._sensory_nodes] = True
        evidence = np.full(graph.total_nodes, np.nan, dtype=np.float64)
        evidence[:24] = rng.uniform(0.1, 0.9, size=24)
        evidence[graph.visual_nodes:graph._sensory_nodes] = rng.normal(0.0, 0.2, size=4)

        energy0, hid_e0, sens_e0, grad_h_int, grad_z_int, root_grad, _, _, _, _ = graph._terms_for(
            h0, z0, canvas, aperture, clamped, evidence,
        )
        assert energy0 == pytest.approx(hid_e0 + sens_e0, abs=1e-14)
        gain = graph._INTEGRATION_GAIN * rate
        # Full natural-metric gradient (M^{-1} dE/dx) includes the leak derivative and gain*g_int:
        grad_h_total = graph._LEAK_ENERGY_WEIGHT * h0 + gain * grad_h_int
        grad_z_total = graph._LEAK_ENERGY_WEIGHT * z0 + gain * grad_z_int

        eps = 1e-7
        num_grad_h = np.zeros_like(h0)
        for i in range(graph.hidden_nodes):
            hp = h0.copy()
            hm = h0.copy()
            hp[i] += eps
            hm[i] -= eps
            ep = graph._terms_for(hp, z0, canvas, aperture, clamped, evidence)[0]
            em = graph._terms_for(hm, z0, canvas, aperture, clamped, evidence)[0]
            num_grad_h[i] = (ep - em) / (2.0 * eps)

        num_grad_z = np.zeros_like(z0)
        for i in range(graph._sensory_nodes):
            zp = z0.copy()
            zm = z0.copy()
            zp[i] += eps
            zm[i] -= eps
            ep = graph._terms_for(h0, zp, canvas, aperture, clamped, evidence)[0]
            em = graph._terms_for(h0, zm, canvas, aperture, clamped, evidence)[0]
            num_grad_z[i] = (ep - em) / (2.0 * eps) / graph._gamma_z

        num_root_grad = np.zeros_like(canvas)
        for i in range(graph.visual_nodes):
            cp = canvas.copy()
            cm = canvas.copy()
            cp[i] += eps
            cm[i] -= eps
            ep = graph._terms_for(h0, z0, cp, aperture, clamped, evidence)[0]
            em = graph._terms_for(h0, z0, cm, aperture, clamped, evidence)[0]
            num_root_grad[i] = (ep - em) / (2.0 * eps) / gain

        np.testing.assert_allclose(grad_h_total, num_grad_h, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(grad_z_total, num_grad_z, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(root_grad, num_root_grad, atol=1e-7, rtol=1e-6)

        # Verify exact sub-step identity: x_next == x - (1 - alpha) * (M^{-1} dE/dx)
        dt = 1.0 / 30.0
        alpha, beta = graph._substep_factors(dt)
        np.testing.assert_allclose(
            alpha * h0 - beta * grad_h_int,
            h0 - (1.0 - alpha) * grad_h_total,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            alpha * z0 - beta * grad_z_int,
            z0 - (1.0 - alpha) * grad_z_total,
            atol=1e-14,
        )


def test_weight_gradients_match_finite_differences_of_interaction_energy():
    rng = np.random.default_rng(19)
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(2, 3),
        audio_nodes=2,
        hidden_nodes=4,
        config=PredictiveCodingConfig(hidden_channels=4, learning_enabled=True),
    )
    graph.canvas = rng.normal(0.0, 0.3, size=graph.visual_nodes)
    graph.aperture_mask = np.ones(graph.visual_nodes, dtype=bool)
    graph.aperture_mask[3:6] = False
    graph.clamped[:] = True
    graph.evidence = rng.uniform(0.15, 0.85, size=graph.total_nodes)
    h0 = rng.normal(0.0, 0.3, size=graph.hidden_nodes)
    z0 = rng.normal(0.0, 0.35, size=graph._sensory_nodes)

    raw_wc, raw_wh, raw_ws = graph._raw_weight_gradients(h0, z0)
    nat_wc, nat_wh, nat_ws = graph._weight_gradients(h0, z0)
    np.testing.assert_allclose(raw_wc, graph._HIDDEN_RESIDUAL_WEIGHT * nat_wc, atol=1e-14)
    np.testing.assert_allclose(raw_wh, graph._HIDDEN_RESIDUAL_WEIGHT * nat_wh, atol=1e-14)
    np.testing.assert_allclose(
        raw_ws, (graph._COUPLING_WEIGHT * graph._gamma_z) * nat_ws, atol=1e-14,
    )

    eps = 1e-7
    for name, analytic in (
        ("canvas_weights", raw_wc),
        ("recurrent_weights", raw_wh),
        ("sensory_weights", raw_ws),
    ):
        weight = getattr(graph, name)
        numeric = np.zeros_like(weight)
        it = np.ndindex(weight.shape)
        for idx in it:
            if name == "recurrent_weights" and idx[0] == idx[1]:
                continue
            orig = weight[idx]
            weight[idx] = orig + eps
            ep = graph._interaction_energy(h0, z0)
            weight[idx] = orig - eps
            em = graph._interaction_energy(h0, z0)
            weight[idx] = orig
            numeric[idx] = (ep - em) / (2.0 * eps)
        np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("rate", (0.02, 0.05, 0.1, 0.2))
def test_nondefault_inference_rates_monotonically_descend_unified_energy(rate):
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(6, 8),
        audio_nodes=4,
        hidden_nodes=6,
        config=PredictiveCodingConfig(
            hidden_channels=6,
            inference_steps=8,
            inference_rate=rate,
            learning_enabled=False,
        ),
    )
    canvas = np.full((6, 8, 3), 0.15, dtype=np.float64)
    target = np.full((6, 8, 3), 0.85, dtype=np.float64)
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 2:6] = True
    graph.forecast(canvas, 1.0 / 30.0)
    graph.condition(target, visual_mask=mask)
    snap = graph.belief_snapshot()
    assert np.all(np.diff(snap["inference_energy"]) <= 1e-12)
    assert snap["energy"] == pytest.approx(
        snap["hidden_energy"] + snap["sensory_energy"], abs=1e-14,
    )


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_single_clock_replay_and_exact_pure_leak_decay_for_both_h_and_z(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    tau = 0.4
    dt = 0.05
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(4, 4),
        audio_nodes=8,
        hidden_nodes=6,
        memory_time_constant=tau,
        config=PredictiveCodingConfig(
            hidden_channels=6,
            inference_steps=8,
            inference_rate=0.1,
            learning_enabled=False,
        ),
        backend=xp,
    )
    canvas = np.zeros((4, 4, 3), dtype=np.float64)
    target = np.full((4, 4, 3), 0.85, dtype=np.float64)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True

    graph.forecast(canvas, dt)
    graph.condition(target, visual_mask=mask)
    snap1 = graph.belief_snapshot()
    assert snap1["model"] == "continuous-scene"
    assert snap1["stage"] == "posterior"
    assert len(snap1["inference_energy"]) == 9
    assert np.all(np.diff(snap1["inference_energy"]) <= 1e-12)
    assert np.linalg.norm(snap1["visual_potential"]) > 0.0
    assert np.linalg.norm(snap1["mean"][slice(*snap1["hidden_slice"])]) > 0.0

    # Unobserved values outside visual_mask cannot affect any state or energy
    graph_alt = ContinuousScenePredictiveCodingGraph(
        visual_shape=(4, 4),
        audio_nodes=8,
        hidden_nodes=6,
        memory_time_constant=tau,
        config=PredictiveCodingConfig(
            hidden_channels=6,
            inference_steps=8,
            inference_rate=0.1,
            learning_enabled=False,
        ),
        backend=xp,
    )
    noisy_target = target.copy()
    noisy_target[~mask] = -999.0
    graph_alt.forecast(canvas, dt)
    graph_alt.condition(noisy_target, visual_mask=mask)
    snap_alt = graph_alt.belief_snapshot()
    np.testing.assert_allclose(snap1["sensory_potential"], snap_alt["sensory_potential"], atol=1e-14)
    np.testing.assert_allclose(snap1["mean"], snap_alt["mean"], atol=1e-14)

    # Pure passive leak (inference_rate == 0) decays both h and z by exact exp(-dt / tau)
    graph.update_config(inference_rate=0.0)
    h_before = to_numpy(graph.mean[graph.hidden_slice]).copy()
    z_before = to_numpy(graph.sensory_potential).copy()
    vis_pred, _ = graph.forecast(canvas, dt)
    expected_factor = np.exp(-dt / tau)
    np.testing.assert_allclose(
        to_numpy(graph.prior_sensory_potential), z_before * expected_factor, atol=1e-12,
    )
    # Conditioning with or without evidence when inference_rate == 0 preserves single-dt leak
    graph.condition(target, visual_mask=mask)
    np.testing.assert_allclose(
        to_numpy(graph.mean[graph.hidden_slice]), h_before * expected_factor, atol=1e-12,
    )
    np.testing.assert_allclose(
        to_numpy(graph.sensory_potential), z_before * expected_factor, atol=1e-12,
    )
    expected_vis = 1.0 / (1.0 + np.exp(-z_before[:graph.visual_nodes].reshape(4, 4, 3) * expected_factor))
    np.testing.assert_allclose(to_numpy(vis_pred), expected_vis, atol=1e-7)


def test_unobserved_world_pixels_evolve_when_gaze_moves_and_decay_after_learning():
    rows, cols = 20, 30
    yy, xx = np.indices((rows, cols))
    mask_a = ((yy - 5)**2 + (xx - 7)**2) <= 4**2
    mask_b = ((yy - 14)**2 + (xx - 22)**2) <= 4**2
    assert not np.any(mask_a & mask_b)

    scene = np.full((rows, cols, 3), 0.15, dtype=np.float64)
    scene[mask_a] = (0.9, 0.2, 0.3)
    scene[mask_b] = (0.2, 0.85, 0.75)
    canvas = np.zeros((rows, cols, 3), dtype=np.float64)
    dt = 1.0 / 30.0

    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(rows, cols),
        audio_nodes=0,
        hidden_nodes=6,
        config=PredictiveCodingConfig(
            hidden_channels=6,
            inference_steps=8,
            inference_rate=0.1,
            learning_enabled=False,
            learning_rate=0.03,
        ),
    )

    for _ in range(25):
        graph.forecast(canvas, dt)
        graph.condition(scene, visual_mask=mask_a)

    z_after_a = graph.visual_potential.reshape(rows, cols, 3).copy()
    assert np.linalg.norm(z_after_a[mask_a]) > 1.0

    # Move gaze to non-overlapping region B with learning enabled
    graph.set_learning(enabled=True)
    initial_ws = graph.sensory_weights.copy()
    for _ in range(60):
        graph.forecast(canvas, dt)
        graph.condition(scene, visual_mask=mask_b)

    z_after_b = graph.visual_potential.reshape(rows, cols, 3).copy()
    # Former patch A is not a frozen cache; it evolved while B was observed
    assert not np.allclose(z_after_b[mask_a], z_after_a[mask_a])
    assert np.linalg.norm(z_after_b[mask_a]) < np.linalg.norm(z_after_a[mask_a]) * 0.25
    assert np.linalg.norm(z_after_b[mask_b]) > 1.0
    assert not np.array_equal(graph.sensory_weights, initial_ws)
    np.testing.assert_array_equal(graph.hidden_bias, 0.0)
    np.testing.assert_array_equal(graph.sensory_bias, 0.0)

    learned_weights = {name: getattr(graph, name).copy() for name in graph._PARAMETERS}
    learned_z_norm = float(np.linalg.norm(graph.visual_potential))
    learned_h_norm = float(np.linalg.norm(graph.mean[graph.hidden_slice]))

    # Turn observation off without resetting state: both h and z decay toward 0
    empty_mask = np.zeros((rows, cols), dtype=bool)
    for _ in range(120):
        vis_pred, _ = graph.forecast(canvas, dt)
        graph.condition(None, visual_mask=empty_mask)

    assert float(np.linalg.norm(graph.visual_potential)) < learned_z_norm * 1e-3
    assert float(np.linalg.norm(graph.mean[graph.hidden_slice])) < learned_h_norm * 1e-3
    assert float(np.mean(np.abs(vis_pred - 0.5))) < 1e-4
    for name, expected in learned_weights.items():
        np.testing.assert_array_equal(getattr(graph, name), expected)
