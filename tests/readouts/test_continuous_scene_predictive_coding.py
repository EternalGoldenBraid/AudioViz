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
        ContinuousScenePredictiveCodingGraph(visual_shape=(481, 640))
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
    with pytest.raises(RuntimeError, match="after forecast and before condition"):
        graph.candidate_information_gains([[.5, .5]], foveal_radius=4)
    with pytest.raises(RuntimeError, match="after forecast and before condition"):
        graph.candidate_information_gain([.5, .5], foveal_radius=4)


def test_full_480x640_world_conditions_only_fovea_and_evolves_without_evidence():
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(480, 640), hidden_nodes=6, audio_nodes=0,
        config=PredictiveCodingConfig(hidden_channels=6, learning_enabled=False),
    )
    assert graph.visual_pixels == graph.MAX_VISUAL_SLOTS == 307200
    assert graph.sensory_weights.shape == (921600, 6)
    rows, cols = np.ogrid[:480, :640]
    mask = (rows - 240) ** 2 + (cols - 320) ** 2 <= 32 ** 2
    canvas = np.zeros((480, 640, 3), dtype=np.float32)
    prediction, _ = graph.forecast(canvas, 1 / 30)
    assert prediction.shape == canvas.shape
    assert np.isfinite(prediction).all()
    graph.condition(np.ones_like(canvas), visual_mask=mask)
    np.testing.assert_array_equal(graph.clamped[graph.visual_slice].reshape(480, 640, 3).all(axis=-1), mask)
    assert graph.clamped.sum() == 3209 * 3
    assert np.linalg.norm(graph.visual_potential) > 0
    weights = graph.sensory_weights.copy()
    graph.update_config(inference_rate=0)
    initial = graph.visual_potential.copy()
    graph.forecast(canvas, 1 / 30)
    graph.condition()
    assert not graph.clamped.any()
    np.testing.assert_allclose(
        graph.visual_potential, initial * np.exp(-(1 / 30) / graph.memory_time_constant), atol=1e-12,
    )
    np.testing.assert_array_equal(graph.sensory_weights, weights)


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


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_condition_reuses_terms_only_within_unchanged_evidence_and_parameters(
    request, monkeypatch, backend_name,
):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(3, 4), audio_nodes=2, hidden_nodes=3, backend=xp,
        config=PredictiveCodingConfig(hidden_channels=3, learning_enabled=True),
    )
    graph.mean[graph.hidden_slice] = xp.asarray([.2, -.1, .3])
    graph.sensory_potential[:] = xp.linspace(-.4, .5, graph._sensory_nodes)
    evaluate = graph._terms_for
    calls = []

    def record(*args):
        terms = evaluate(*args)
        calls.append(terms)
        return terms

    monkeypatch.setattr(graph, "_terms_for", record)
    mask = np.zeros(graph.visual_shape, dtype=bool)
    mask[1, 1:3] = True
    for observed, rate in ((True, .1), (False, .1), (True, 0.0)):
        graph.update_config(inference_rate=rate)
        graph.forecast(np.full((3, 4, 3), .2), 1/30)
        before = graph.belief_snapshot()
        calls.clear()
        if observed:
            direction = graph.condition(
                np.full((3, 4, 3), .8), np.array([.1, -.2]), visual_mask=mask,
            )
        else:
            direction = graph.condition()

        states = graph.config.inference_steps + 1 if observed and rate else 1
        # Learning evaluates its own derivatives before changing weights or precisions.
        assert len(calls) == states + int(observed)
        assert graph.inference_energy == tuple(terms[0] for terms in calls[:states])
        final = calls[states - 1]
        assert (graph.energy, graph.hidden_energy, graph.sensory_energy) == final[:3]
        for actual, expected in (
            (graph.canvas_gradient, final[5]),
            (graph.hidden_error, final[6]),
            (graph.sensory_potential_error, final[7]),
            (graph.sensory_error, final[8]),
            (direction.reshape(-1), -final[5]),
            (graph.visual_prediction.reshape(-1), final[9][graph.visual_slice].astype(xp.float32)),
        ):
            np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))
        after = graph.belief_snapshot()
        for name in (*graph._PARAMETERS, *graph._PRECISIONS):
            np.testing.assert_array_equal(after[name], before[name])
        if not observed or rate == 0:
            np.testing.assert_array_equal(after["sensory_potential"], before["sensory_potential"])
            np.testing.assert_array_equal(
                after["mean"][graph.hidden_slice], before["mean"][graph.hidden_slice],
            )
        if not observed:
            np.testing.assert_array_equal(after["sensory_error"], 0)
        else:
            expected_error = np.where(
                after["clamped"][:graph._sensory_nodes],
                after["evidence"][:graph._sensory_nodes] - to_numpy(final[9]), 0,
            )
            np.testing.assert_array_equal(after["sensory_error"], expected_error)


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
        for name in graph._PRECISIONS:
            precision = getattr(graph, name)
            precision *= rng.uniform(.5, 2., size=precision.shape)
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

        eps = 1e-5
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
    for name in graph._PRECISIONS:
        precision = getattr(graph, name)
        precision *= rng.uniform(.5, 2., size=precision.shape)
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


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_buckley_gaussian_objective_has_normalization_and_log_precision_gradient(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(1, 2), audio_nodes=1, hidden_nodes=2, backend=xp,
        config=PredictiveCodingConfig(hidden_channels=2, learning_enabled=False),
    )
    errors = xp.asarray([2., -.5, 100.])
    precisions = xp.asarray([.25, 4., 9.])
    observed = xp.asarray([True, True, False])
    expected = .5 * (1 + 1 - np.log(.25) - np.log(4.))
    assert graph._gaussian_residual_energy(errors, precisions, observed=observed) == pytest.approx(expected)
    low = graph._gaussian_residual_energy(xp.asarray([2.]), xp.asarray([1e-100]))
    optimum = graph._gaussian_residual_energy(xp.asarray([2.]), xp.asarray([.25]))
    high = graph._gaussian_residual_energy(xp.asarray([2.]), xp.asarray([1e3]))
    assert low > optimum
    assert high > optimum
    assert graph._gaussian_residual_energy(errors, precisions, observed=xp.zeros(3, dtype=bool)) == 0

    graph.mean[graph.hidden_slice] = xp.asarray([.2, -.1])
    graph.sensory_potential[:] = xp.linspace(-.3, .4, graph._sensory_nodes)
    graph.clamped[:3] = True
    graph.clamped[graph.audio_slice] = True
    graph.evidence[:3] = .8
    graph.evidence[graph.audio_slice] = -.2
    graph._capture()
    snapshot = graph.belief_snapshot()
    observed_host = snapshot["clamped"][:graph._sensory_nodes]
    hidden_factor = .5 * np.sum(
        snapshot["hidden_residual_precision"] * snapshot["hidden_error"]**2
        - np.log(snapshot["hidden_residual_precision"]),
    )
    sensory_factors = .5 * np.sum(
        snapshot["coupling_precision"] * snapshot["sensory_potential_error"]**2
        - np.log(snapshot["coupling_precision"]),
    ) + .5 * np.sum(
        (snapshot["observation_precision"] * snapshot["sensory_error"]**2
         - np.log(snapshot["observation_precision"]))[observed_host],
    )
    gain = graph._INTEGRATION_GAIN * graph.config.inference_rate
    expected_hidden = .5 * np.sum(snapshot["mean"][graph.hidden_slice]**2) + gain * hidden_factor
    expected_sensory = .5 * graph._gamma_z * np.sum(snapshot["sensory_potential"]**2) + gain * sensory_factors
    assert graph.hidden_energy == pytest.approx(expected_hidden, abs=1e-12)
    assert graph.sensory_energy == pytest.approx(expected_sensory, abs=1e-12)
    assert graph.energy == pytest.approx(expected_hidden + expected_sensory, abs=1e-12)
    h, z = graph.mean[graph.hidden_slice], graph.sensory_potential
    for name, gradient in zip(graph._PRECISIONS, graph._precision_gradients()):
        precision = getattr(graph, name)
        original = precision.copy()
        numeric = np.zeros(precision.size)
        for index in range(precision.size):
            precision[index] = original[index] * np.exp(1e-6)
            plus = graph._interaction_energy(h, z)
            precision[index] = original[index] * np.exp(-1e-6)
            minus = graph._interaction_energy(h, z)
            precision[index] = original[index]
            numeric[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(to_numpy(gradient), numeric, atol=2e-9)
    np.testing.assert_array_equal(
        to_numpy(graph._precision_gradients()[2])[~to_numpy(graph.clamped[:graph._sensory_nodes])], 0,
    )


@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_precision_learning_uses_captured_residuals_and_preserves_unobserved_observation_precision(request, backend_name):
    xp = np if backend_name == "numpy" else request.getfixturevalue("cupy_backend")
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(2, 3), audio_nodes=1, hidden_nodes=3, backend=xp,
        config=PredictiveCodingConfig(
            hidden_channels=3, learning_enabled=True, learning_rate=.03, gradient_clip=.7,
        ),
    )
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    graph.forecast(np.zeros((2, 3, 3)), 1/30)
    graph.condition(np.full((2, 3, 3), .9), np.array([10.]), visual_mask=mask)
    snapshot = graph.belief_snapshot()
    observed = snapshot["clamped"][:graph._sensory_nodes]
    before_energy, after_energy = 0., 0.
    for name, error_name in zip(
        graph._PRECISIONS, ("hidden_error", "sensory_potential_error", "sensory_error"),
    ):
        before = snapshot[name]
        gradient = .5 * (before * snapshot[error_name]**2 - 1)
        if name == "observation_precision":
            gradient[~observed] = 0
        expected = before * np.exp(-.03 * np.clip(gradient, -.7, .7))
        actual = to_numpy(getattr(graph, name))
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        assert np.isfinite(actual).all() and (actual > 0).all()
        assert isinstance(getattr(graph, name), xp.ndarray)
        assert not np.shares_memory(before, to_numpy(getattr(graph, name)))
        support = observed if name == "observation_precision" else np.ones(before.size, dtype=bool)
        error = snapshot[error_name]
        before_energy += .5 * np.sum((before * error**2 - np.log(before))[support])
        after_energy += .5 * np.sum((actual * error**2 - np.log(actual))[support])
    assert after_energy < before_energy
    assert graph.observation_precision[graph.audio_slice][0] < snapshot["observation_precision"][graph.audio_slice][0]
    assert graph.observation_precision[3] > snapshot["observation_precision"][3]
    np.testing.assert_array_equal(
        to_numpy(graph.observation_precision)[~observed], snapshot["observation_precision"][~observed],
    )

    learned = {name: to_numpy(getattr(graph, name)).copy() for name in graph._PRECISIONS}
    graph.forecast(np.zeros((2, 3, 3)), 1/30)
    graph.condition()
    assert graph.updates == 1
    graph.set_learning(enabled=False)
    graph.forecast(np.zeros((2, 3, 3)), 1/30)
    graph.condition(np.ones((2, 3, 3)), visual_mask=mask)
    graph.set_learning(enabled=True, rate=0)
    graph.forecast(np.zeros((2, 3, 3)), 1/30)
    graph.condition(np.ones((2, 3, 3)), visual_mask=mask)
    graph.reset()
    for name, expected in learned.items():
        np.testing.assert_array_equal(to_numpy(getattr(graph, name)), expected)


def test_precision_learning_rejects_unrepresentable_update_without_parameter_mutation():
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(1, 1), audio_nodes=0, hidden_nodes=2,
        config=PredictiveCodingConfig(hidden_channels=2, learning_enabled=True, learning_rate=2000),
    )
    before = {name: getattr(graph, name).copy() for name in (*graph._PARAMETERS, *graph._PRECISIONS)}
    graph.forecast(np.zeros((1, 1, 3)), 1/30)
    with np.errstate(over="ignore"):
        with pytest.raises(FloatingPointError, match="precision"):
            graph.condition(np.full((1, 1, 3), .5))
    assert graph.updates == 0
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(graph, name), expected)
