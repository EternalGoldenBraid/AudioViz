from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.predictive_coding import PredictiveCodingHiddenCoupling, infer_joint
from audioviz.engine import RippleEngine
from audioviz.sources import (
    AudioSourceConfig, CameraFrameSourceConfig, RippleSourceOrchestrator,
    RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)


def _pair(**changes):
    config = PredictiveCodingConfig(hidden_channels=2, **changes)
    camera = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3), config=config)
    audio = PredictiveCodingAudioReadout(canvas_shape=camera.canvas_shape, config=config)
    coupling = PredictiveCodingHiddenCoupling(camera, audio)
    return camera, audio, coupling


def _energy(canvas, branches, coupling):
    result = 0.0
    for branch, observation in branches:
        parent = coupling.parent(branch)
        prediction = (np.tanh(canvas) @ branch.canvas_weights.T
                      + np.tanh(parent.hidden) @ coupling.weights[branch].T)
        result += 0.5 * np.sum((branch.hidden - prediction)**2, axis=-1).mean()
        if observation is not None:
            error = observation - branch._predict_observation(np.tanh(branch.hidden))
            result += 0.5 * (np.sum(error**2) if error.ndim == 1
                             else np.sum(error**2, axis=-1).mean())
    return float(result)


def _gradient(values, energy):
    result = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + 1e-6
        upper = energy()
        values[index] = original - 1e-6
        lower = energy()
        values[index] = original
        result[index] = (upper - lower) / 2e-6
    return result


@pytest.mark.parametrize("missing", ("neither", "camera", "both"))
def test_recurrent_states_follow_full_nonlinear_energy_gradient_synchronously(missing):
    camera, audio, coupling = _pair(inference_steps=1, inference_rate=.001, canvas_rate=.001)
    rng = np.random.default_rng(4)
    canvas = rng.normal(scale=.2, size=camera.canvas_shape)
    for branch in coupling.branches:
        branch.hidden[:] = rng.normal(scale=.3, size=branch.hidden.shape)
        coupling.weights[branch][:] = rng.normal(scale=.2, size=coupling.weights[branch].shape)
    branches = ((camera, rng.random(canvas.shape) if missing == "neither" else None),
                (audio, rng.random(16) if missing != "both" else None))
    reverse_camera, reverse_audio, reverse_coupling = deepcopy((camera, audio, coupling))
    states = (canvas, camera.hidden, audio.hidden)
    before = [state.copy() for state in states]
    gradients = [_gradient(state, lambda: _energy(canvas, branches, coupling)) for state in states]
    if missing == "both":
        prior_camera, prior_audio, prior_coupling = deepcopy((camera, audio, coupling))
        prior_coupling.predict(canvas)
        for branch, original, gradient in zip(
            (prior_camera, prior_audio), before[1:], gradients[1:]
        ):
            np.testing.assert_allclose(branch.hidden - original, -.001 * 6 * gradient, rtol=1e-5, atol=1e-10)

    correction = infer_joint(canvas, branches, coupling=coupling)
    reverse = infer_joint(
        canvas, ((reverse_audio, branches[1][1]), (reverse_camera, branches[0][1])),
        coupling=reverse_coupling,
    )

    for change, gradient in zip(
        (correction, camera.hidden - before[1], audio.hidden - before[2]), gradients
    ):
        np.testing.assert_allclose(change, -.001 * 6 * gradient, rtol=1e-5, atol=1e-10)
    np.testing.assert_array_equal(canvas, before[0])
    np.testing.assert_array_equal(correction, reverse)
    np.testing.assert_array_equal(camera.hidden, reverse_camera.hidden)
    np.testing.assert_array_equal(audio.hidden, reverse_audio.hidden)


def test_recurrent_weight_learning_is_local_and_missing_sensory_weights_stay_frozen():
    camera, audio, coupling = _pair(
        inference_steps=1, inference_rate=0, canvas_rate=0,
        learning_enabled=True, learning_rate=.001, gradient_clip=100,
    )
    canvas = np.full(camera.canvas_shape, .2)
    camera.hidden.fill(.3)
    audio.hidden.fill(-.2)
    for branch in coupling.branches:
        coupling.weights[branch].fill(.1)
    branches = ((camera, None), (audio, np.linspace(.1, .7, 16)))
    weights = (camera.canvas_weights, audio.canvas_weights, audio.visual_weights,
               coupling.weights[camera], coupling.weights[audio])
    before = [weight.copy() for weight in weights]
    gradients = [_gradient(weight, lambda: _energy(canvas, branches, coupling)) for weight in weights]
    absent_sensory_weights = camera.visual_weights.copy()

    infer_joint(canvas, branches, coupling=coupling)

    for weight, original, gradient in zip(weights, before, gradients):
        np.testing.assert_allclose(weight - original, -.001 * gradient, rtol=1e-5, atol=1e-10)
    np.testing.assert_array_equal(camera.visual_weights, absent_sensory_weights)
    for branch in coupling.branches:
        branch.set_diagnostics_enabled(True)
    before = [weight.copy() for weight in weights]
    infer_joint(canvas, ((camera, None), (audio, None)), coupling=coupling)
    for weight, original in zip(weights, before):
        np.testing.assert_array_equal(weight, original)
    assert camera.diagnostics.weight_update_norm == 0
    assert not camera.diagnostics.observation_present
    assert np.isnan(camera.diagnostics.visual.mean)
    assert not camera.diagnostics.cross_modal_weights.flags.writeable


def test_zero_coupling_matches_existing_joint_inference_with_both_streams_present():
    camera, audio, coupling = _pair(inference_steps=8, learning_enabled=True)
    baseline_camera, baseline_audio = deepcopy((camera, audio))
    canvas = np.full(camera.canvas_shape, .2)
    image = np.full(canvas.shape, .7)
    spectrum = np.linspace(.1, .8, 16)
    actual = infer_joint(canvas, ((camera, image), (audio, spectrum)), coupling=coupling)
    expected = infer_joint(canvas, ((baseline_camera, image), (baseline_audio, spectrum)))
    np.testing.assert_array_equal(actual, expected)
    for branch, baseline in ((camera, baseline_camera), (audio, baseline_audio)):
        np.testing.assert_array_equal(branch.hidden, baseline.hidden)
        np.testing.assert_array_equal(branch.canvas_weights, baseline.canvas_weights)
        np.testing.assert_array_equal(branch.visual_weights, baseline.visual_weights)


def test_loop_requires_both_readouts_and_rejects_shader_before_changing_config():
    engine = RippleEngine(resolution=(6, 8), plane_size_m=(1, 1), speed=1)
    source = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(audio=AudioSourceConfig(enabled=False)),
        processor=None, engine=engine, resolution=engine.resolution, n_sources=1,
    )
    with pytest.raises(ValueError, match="readouts"):
        source.update_pathway_config(cross_modal_enabled=True)
    assert not source.visual_readout.config.cross_modal_enabled
    source = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(audio=AudioSourceConfig(enabled=False)),
        processor=SimpleNamespace(frame_counter=0), engine=engine, resolution=engine.resolution, n_sources=1,
    )
    engine.use_shader = True
    with pytest.raises(ValueError, match="array-backed"):
        source.update_pathway_config(cross_modal_enabled=True)
    assert not source.visual_readout.config.cross_modal_enabled


def test_paired_learning_predicts_missing_camera_and_changes_an_absent_sensor_impulse_response():
    config = PredictiveCodingConfig(
        inference_steps=20, canvas_rate=0, learning_enabled=True, learning_rate=.03,
    )
    camera = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3), config=config)
    audio = PredictiveCodingAudioReadout(canvas_shape=camera.canvas_shape, config=config)
    coupling = PredictiveCodingHiddenCoupling(camera, audio)
    # A fixed zero canvas isolates the hidden-to-hidden route from the shared surface.
    canvas = np.zeros(camera.canvas_shape)
    images = [np.broadcast_to(color, canvas.shape) for color in ([.8, .1, .2], [.1, .8, .4])]
    spectra = [np.r_[np.full(8, .7), np.full(8, .05)],
               np.r_[np.full(8, .05), np.full(8, .7)]]
    for epoch in range(600):
        index = epoch % 2
        coupling.predict(canvas)
        infer_joint(canvas, ((camera, images[index]), (audio, spectra[index])), coupling=coupling)
    for branch in coupling.branches:
        branch.update_config(learning_enabled=False)
        assert np.linalg.norm(coupling.weights[branch]) > .1
    learned_weights = [coupling.weights[branch].copy() for branch in coupling.branches]
    for index, spectrum in enumerate(spectra):
        camera.reset()
        audio.reset()
        for _ in range(40):
            coupling.predict(canvas)
            infer_joint(canvas, ((camera, None), (audio, spectrum)), coupling=coupling)
        prediction = camera._predict_observation(np.tanh(camera.hidden))
        assert np.mean((prediction - images[index])**2) < 1e-3
        assert np.mean((prediction - images[1 - index])**2) > .1
    for branch, before in zip(coupling.branches, learned_weights):
        np.testing.assert_array_equal(coupling.weights[branch], before)

    systems = []
    for keep_loop_weights in (False, True):
        engine = RippleEngine(
            resolution=(6, 8), plane_size_m=(1, 1), speed=1, damping=.98, amplitude=1,
        )
        source = RippleSourceOrchestrator(
            config=RippleSourceOrchestratorConfig(
                synthetic=SyntheticSourceConfig(enabled=False),
                audio=AudioSourceConfig(enabled=False),
                camera_frame=CameraFrameSourceConfig(enabled=False),
                visual_pathway=PredictiveCodingConfig(
                    cross_modal_enabled=True, learning_enabled=False, weight_decay=.1,
                ),
            ),
            processor=SimpleNamespace(frame_counter=0), engine=engine,
            resolution=engine.resolution, n_sources=1,
        )
        for target, trained in zip(source.hidden_coupling.branches, coupling.branches):
            target.canvas_weights[:] = trained.canvas_weights
            target.visual_weights[:] = trained.visual_weights
            if keep_loop_weights:
                source.hidden_coupling.weights[target][:] = coupling.weights[trained]
        source.reset_readouts()
        for branch in source.hidden_coupling.branches:
            branch.set_diagnostics_enabled(True)
        systems.append((engine, source))
    before_weights = [
        [array.copy() for branch in source.hidden_coupling.branches
         for array in (branch.canvas_weights, branch.visual_weights, source.hidden_coupling.weights[branch])]
        for _, source in systems
    ]
    pulse = np.zeros(systems[0][0].canvas_shape, dtype=np.float32)
    pulse[3, 4, 0] = .05
    differences = []
    for step in range(40):
        for engine, source in systems:
            engine.propagate(pulse if step == 0 else None)
            source.predict_visual()
            assert source.correct_from_observations() is not None
            assert source.visual_observation is None
            assert source.visual_readout.diagnostics.weight_update_norm == 0
            assert not source.visual_readout.diagnostics.observation_present
        differences.append(np.max(np.abs(systems[0][0].Z - systems[1][0].Z)))
    assert max(differences) > 1e-6
    for (_, source), before in zip(systems, before_weights):
        arrays = [array for branch in source.hidden_coupling.branches
                  for array in (branch.canvas_weights, branch.visual_weights, source.hidden_coupling.weights[branch])]
        for array, original in zip(arrays, before):
            np.testing.assert_array_equal(array, original)
    source = systems[1][1]
    source.update_pathway_config(cross_modal_enabled=False)
    source.engine.propagate()
    source.predict_visual()
    assert source.correct_from_observations() is not None
    assert source.visual_readout.diagnostics.stream_state.observed is False
    source.reset_readouts()
    source.update_pathway_config(cross_modal_enabled=True)
    for branch, trained in zip(source.hidden_coupling.branches, coupling.branches):
        np.testing.assert_array_equal(source.hidden_coupling.weights[branch], coupling.weights[trained])
