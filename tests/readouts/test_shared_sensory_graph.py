from __future__ import annotations

import numpy as np
import pytest

from audioviz.readouts.shared_sensory_graph import SharedGraphSnapshot, SharedSensoryGraph


def test_dense_hard_clamp_conditioning_and_masked_em_match_oracle():
    graph = SharedSensoryGraph(
        visual_shape=(2, 2),
        audio_nodes=4,
        hidden_nodes=3,
        initial_variance=1.2,
        process_variance=0.04,
        memory_time_constant=0.5,
        learning_rate=0.2,
        variance_floor=1e-4,
        seed=11,
    )
    assert graph.total_nodes == 12 + 4 + 3
    assert graph.stage == "posterior"

    # Initial covariance is dense and exchangeable (non-zero cross-modality off-diagonals).
    c0 = graph.learned_covariance.copy()
    m0 = graph.learned_mean.copy()
    vis_idx = np.arange(0, 12)
    aud_idx = np.arange(12, 16)
    hid_idx = np.arange(16, 19)
    assert np.linalg.norm(c0[np.ix_(vis_idx, hid_idx)]) > 0
    assert np.linalg.norm(c0[np.ix_(aud_idx, hid_idx)]) > 0

    vis_pred, aud_pred = graph.forecast(dt=0.1)
    assert graph.stage == "forecast"
    assert vis_pred.shape == (2, 2, 3) and aud_pred.shape == (4,)

    prior_mean = graph.prior_mean.copy()
    prior_cov = graph.prior_covariance.copy()

    # Clamp only 2 of the 4 visual pixels (6 RGB coords) and omit audio completely (audio=None).
    vis_obs = np.arange(12, dtype=np.float64).reshape(2, 2, 3) * 0.1 - 0.3
    vis_mask = np.array([[True, False], [False, True]])
    innovation = graph.condition(visual=vis_obs, audio=None, visual_mask=vis_mask)
    assert graph.stage == "posterior"
    assert graph.updates == 1

    obs_idx = np.array([0, 1, 2, 9, 10, 11], dtype=np.int64)
    unobs_idx = np.setdiff1d(np.arange(graph.total_nodes), obs_idx)
    y_o = vis_obs.reshape(-1)[obs_idx]

    # Dense Gaussian conditional oracle for filter posterior.
    p_oo = prior_cov[np.ix_(obs_idx, obs_idx)]
    p_uo = prior_cov[np.ix_(unobs_idx, obs_idx)]
    p_uu = prior_cov[np.ix_(unobs_idx, unobs_idx)]
    gain = p_uo @ np.linalg.inv(p_oo)
    expected_mean = prior_mean.copy()
    expected_mean[obs_idx] = y_o
    expected_mean[unobs_idx] = prior_mean[unobs_idx] + gain @ (y_o - prior_mean[obs_idx])
    expected_cov = np.zeros_like(prior_cov)
    expected_cov[np.ix_(unobs_idx, unobs_idx)] = p_uu - gain @ p_uo.T

    np.testing.assert_allclose(graph.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(graph.covariance, expected_cov, atol=1e-12)
    np.testing.assert_allclose(innovation, expected_mean - prior_mean, atol=1e-12)
    # Observed rows and columns of posterior covariance are identically zero.
    np.testing.assert_array_equal(graph.covariance[obs_idx, :], 0.0)
    np.testing.assert_array_equal(graph.covariance[:, obs_idx], 0.0)

    # Masked EM oracle uses population N(m0, c0) | (x_O = y_O), NOT filter posterior.
    c_oo = c0[np.ix_(obs_idx, obs_idx)]
    c_uo = c0[np.ix_(unobs_idx, obs_idx)]
    c_uu = c0[np.ix_(unobs_idx, unobs_idx)]
    em_gain = c_uo @ np.linalg.inv(c_oo)
    u = m0.copy()
    u[obs_idx] = y_o
    u[unobs_idx] = m0[unobs_idx] + em_gain @ (y_o - m0[obs_idx])
    v = np.zeros_like(c0)
    v[np.ix_(unobs_idx, unobs_idx)] = c_uu - em_gain @ c_uo.T
    alpha = graph.learning_rate
    d = u - m0
    expected_mu1 = m0 + alpha * d
    expected_c1 = (
        (1.0 - alpha) * c0
        + alpha * v
        + alpha * (1.0 - alpha) * np.outer(d, d)
        + alpha * graph.variance_floor * np.eye(graph.total_nodes)
    )
    np.testing.assert_allclose(graph.learned_mean, expected_mu1, atol=1e-12)
    np.testing.assert_allclose(graph.learned_covariance, expected_c1, atol=1e-12)

    # Unobserved audio was NOT clamped to zero: compare against explicitly clamping silent audio=zeros(4).
    graph_silent = SharedSensoryGraph(
        visual_shape=(2, 2),
        audio_nodes=4,
        hidden_nodes=3,
        initial_variance=1.2,
        process_variance=0.04,
        memory_time_constant=0.5,
        learning_rate=0.2,
        variance_floor=1e-4,
        seed=11,
    )
    graph_silent.forecast(dt=0.1)
    graph_silent.condition(visual=vis_obs, audio=np.zeros(4), visual_mask=vis_mask)
    assert not np.allclose(graph.mean[aud_idx], graph_silent.mean[aud_idx])
    assert np.all(np.diag(graph.covariance)[aud_idx] > 0.0)
    np.testing.assert_array_equal(np.diag(graph_silent.covariance)[aud_idx], 0.0)


def test_no_evidence_ou_partition_invariance_and_learning_freeze():
    g1 = SharedSensoryGraph(
        visual_shape=(2, 1), audio_nodes=3, hidden_nodes=2,
        initial_variance=0.8, process_variance=0.05, memory_time_constant=0.4,
        learning_rate=0.15, seed=5,
    )
    # Seed a non-trivial posterior state and learned mean via one observation, then freeze learning.
    g1.forecast(dt=0.05)
    g1.condition(visual=np.full((2, 1, 3), 0.4), audio=np.array([0.2, -0.1, 0.3]))
    assert g1.updates == 1

    g2 = SharedSensoryGraph(
        visual_shape=(2, 1), audio_nodes=3, hidden_nodes=2,
        initial_variance=0.8, process_variance=0.05, memory_time_constant=0.4,
        learning_rate=0.15, seed=5,
    )
    g2.forecast(dt=0.05)
    g2.condition(visual=np.full((2, 1, 3), 0.4), audio=np.array([0.2, -0.1, 0.3]))

    mu_after_obs = g1.learned_mean.copy()
    c_after_obs = g1.learned_covariance.copy()

    # Step g1 in three unequal no-evidence steps summing to 0.35 s; step g2 in one 0.35 s step.
    for step_dt in (0.07, 0.13, 0.15):
        g1.forecast(dt=step_dt)
        innov = g1.condition(visual=None, audio=None)
        np.testing.assert_array_equal(innov, 0.0)
    g2.forecast(dt=0.35)
    g2.condition(visual=None, audio=None)

    assert g1.updates == 1 and g2.updates == 1
    np.testing.assert_array_equal(g1.learned_mean, mu_after_obs)
    np.testing.assert_array_equal(g1.learned_covariance, c_after_obs)
    np.testing.assert_allclose(g1.mean, g2.mean, atol=1e-12)
    np.testing.assert_allclose(g1.covariance, g2.covariance, atol=1e-12)

    # Freezing learning prevents updates even when evidence is present; reset preserves learned parameters.
    g1.set_learning(enabled=False, rate=0.3)
    g1.forecast(dt=0.1)
    g1.condition(visual=np.ones((2, 1, 3)), audio=np.ones(3))
    assert g1.updates == 1
    np.testing.assert_array_equal(g1.learned_mean, mu_after_obs)
    g1.reset()
    np.testing.assert_array_equal(g1.mean, mu_after_obs)
    np.testing.assert_array_equal(g1.covariance, c_after_obs)


def test_finite_target_hidden_mutual_information_and_cupy_parity():
    graph = SharedSensoryGraph(
        visual_shape=(2, 2), audio_nodes=4, hidden_nodes=3,
        initial_variance=1.0, process_variance=0.02, memory_time_constant=0.3,
        seed=19,
    )
    s_idx = np.array([0, 1, 2, 12, 13], dtype=np.int64)
    h_idx = np.arange(graph.hidden_slice.start, graph.hidden_slice.stop)

    # Posterior-stage scoring with explicit dt matches direct logdet oracle on forecast P^-.
    ig_from_post = graph.information_gain(s_idx, dt=0.1)
    graph.forecast(dt=0.1)
    ig_from_forecast = graph.information_gain(s_idx)
    assert ig_from_post == pytest.approx(ig_from_forecast, rel=1e-12)

    p = graph.prior_covariance
    p_hh = p[np.ix_(h_idx, h_idx)]
    p_ss = p[np.ix_(s_idx, s_idx)]
    p_hs = p[np.ix_(h_idx, s_idx)]
    p_hh_cond = p_hh - p_hs @ np.linalg.solve(p_ss, p_hs.T)
    _, logdet_prior = np.linalg.slogdet(p_hh)
    _, logdet_post = np.linalg.slogdet(p_hh_cond)
    expected_ig = 0.5 * float(logdet_prior - logdet_post)
    assert np.isfinite(ig_from_forecast) and ig_from_forecast > 0.0
    assert ig_from_forecast == pytest.approx(expected_ig, rel=1e-12)
    assert graph.information_gain([], dt=0.1) == 0.0

    snap: SharedGraphSnapshot = graph.belief_snapshot()
    assert snap["stage"] == "forecast"
    assert snap["learned_precision"].shape == (graph.total_nodes, graph.total_nodes)
    np.testing.assert_allclose(
        snap["learned_covariance"] @ snap["learned_precision"],
        np.eye(graph.total_nodes),
        atol=1e-10,
    )

    cp = pytest.importorskip("cupy")
    gpu_graph = SharedSensoryGraph(
        visual_shape=(2, 2), audio_nodes=4, hidden_nodes=3,
        initial_variance=1.0, process_variance=0.02, memory_time_constant=0.3,
        seed=19, backend=cp,
    )
    gpu_graph.forecast(dt=0.1)
    assert gpu_graph.information_gain(s_idx) == pytest.approx(ig_from_forecast, rel=1e-12)
    vis_sample = np.linspace(-0.5, 0.5, 12).reshape(2, 2, 3)
    aud_sample = np.array([0.1, -0.2, 0.3, 0.4])
    graph.condition(vis_sample, aud_sample)
    gpu_graph.condition(cp.asarray(vis_sample), cp.asarray(aud_sample))
    gpu_snap = gpu_graph.belief_snapshot()
    np.testing.assert_allclose(gpu_snap["mean"], graph.mean, atol=1e-11)
    np.testing.assert_allclose(gpu_snap["learned_covariance"], graph.learned_covariance, atol=1e-11)


def test_learned_cross_modal_cooccurrence_improves_withheld_modality_completion():
    rng = np.random.default_rng(42)
    # Construct paired visual (1x2 RGB = 6) and audio (4) observations driven by a 2D latent factor z.
    w_vis = rng.standard_normal((6, 2)) * 0.8
    w_aud = rng.standard_normal((4, 2)) * 0.8
    bias_vis = np.array([0.2, -0.1, 0.05, -0.2, 0.15, 0.3])
    bias_aud = np.array([-0.15, 0.25, -0.05, 0.1])

    learned = SharedSensoryGraph(
        visual_shape=(1, 2), audio_nodes=4, hidden_nodes=3,
        initial_variance=0.25, process_variance=0.01, memory_time_constant=0.2,
        learning_rate=0.12, learning_enabled=True, correlation_scale=0.05, seed=7,
    )
    untrained = SharedSensoryGraph(
        visual_shape=(1, 2), audio_nodes=4, hidden_nodes=3,
        initial_variance=0.25, process_variance=0.01, memory_time_constant=0.2,
        learning_rate=0.12, learning_enabled=False, correlation_scale=0.05, seed=7,
    )

    for _ in range(80):
        z = rng.standard_normal(2)
        v_obs = (bias_vis + w_vis @ z + 0.03 * rng.standard_normal(6)).reshape(1, 2, 3)
        a_obs = bias_aud + w_aud @ z + 0.03 * rng.standard_normal(4)
        learned.forecast(dt=0.05)
        learned.condition(visual=v_obs, audio=a_obs)

    # Evaluate cross-modal completion: clamp only visual, withhold audio (audio=None), measure audio MSE.
    learned_errors = []
    untrained_errors = []
    for _ in range(30):
        z = rng.standard_normal(2)
        v_test = (bias_vis + w_vis @ z).reshape(1, 2, 3)
        a_true = bias_aud + w_aud @ z

        learned.reset()
        learned.set_learning(enabled=False)
        learned.forecast(dt=0.05)
        learned.condition(visual=v_test, audio=None)
        learned_errors.append(np.mean((learned.mean[learned.audio_slice] - a_true) ** 2))

        untrained.reset()
        untrained.forecast(dt=0.05)
        untrained.condition(visual=v_test, audio=None)
        untrained_errors.append(np.mean((untrained.mean[untrained.audio_slice] - a_true) ** 2))

    assert float(np.mean(learned_errors)) < 0.25 * float(np.mean(untrained_errors))


def test_validation_rejects_oversized_or_invalid_parameters():
    with pytest.raises(ValueError, match="visual_shape dimensions"):
        SharedSensoryGraph(visual_shape=(5, 4))
    with pytest.raises(ValueError, match="must not exceed 96"):
        SharedSensoryGraph(visual_shape=(4, 4), audio_nodes=32, hidden_nodes=20)
    graph = SharedSensoryGraph(visual_shape=(2, 2), audio_nodes=4, hidden_nodes=2)
    with pytest.raises(RuntimeError, match="Forecast"):
        graph.condition(visual=np.zeros((2, 2, 3)))
    graph.forecast(dt=0.1)
    with pytest.raises(RuntimeError, match="Condition"):
        graph.forecast(dt=0.1)
    with pytest.raises(ValueError, match="sensory node indices"):
        graph.information_gain([graph.hidden_slice.start])
