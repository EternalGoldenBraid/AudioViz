from types import SimpleNamespace

import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.predictive_coding import PredictiveCodingHiddenCoupling, infer_joint
from audioviz.sources import AudioSourceConfig, RippleSourceOrchestrator, RippleSourceOrchestratorConfig, SyntheticSourceConfig
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig


def _gradient(values, energy):
    gradient = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + 1e-6
        upper = energy()
        values[index] = original - 1e-6
        lower = energy()
        values[index] = original
        gradient[index] = (upper - lower) / 2e-6
    return gradient


def _energy(canvas, branches, coupling, states):
    energy = 0.0
    for branch, _ in branches:
        prediction = np.tanh(canvas) @ branch.canvas_weights.T
        if coupling is not None:
            prediction += np.tanh(coupling.parent(branch).hidden) @ coupling.weights[branch].T
        energy += .5 * np.sum((branch.hidden - prediction)**2, axis=-1).mean()
        pooled = np.tanh(branch.hidden).mean(axis=(0, 1))
        logit = float((pooled @ branch.stream_state_weights.T + branch.stream_state_bias)[0])
        probability = 1 / (1 + np.exp(-logit))
        energy += .5 * (float(states[branch]) - probability)**2
    return float(energy)


@pytest.mark.parametrize("coupled", (False, True))
@pytest.mark.parametrize("learning", (False, True))
def test_stream_context_state_and_weight_updates_follow_full_nonlinear_energy_gradients(coupled, learning):
    config = PredictiveCodingConfig(
        hidden_channels=2, inference_steps=1,
        inference_rate=0 if learning else .001, canvas_rate=0 if learning else .001,
        learning_enabled=learning, learning_rate=.001, gradient_clip=100,
    )
    camera = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3), config=config)
    audio = PredictiveCodingAudioReadout(canvas_shape=camera.canvas_shape, config=config)
    coupling = PredictiveCodingHiddenCoupling(camera, audio) if coupled else None
    rng = np.random.default_rng(8)
    canvas = rng.normal(scale=.2, size=camera.canvas_shape)
    branches = ((camera, None), (audio, None))
    states = {camera: False, audio: True}
    for branch, _ in branches:
        branch.hidden[:] = rng.normal(scale=.3, size=branch.hidden.shape)
        branch.stream_state_bias[:] = .2
        branch.set_diagnostics_enabled(True)
        if coupling is not None:
            coupling.weights[branch][:] = rng.normal(scale=.2, size=coupling.weights[branch].shape)
    variables = [canvas, camera.hidden, audio.hidden]
    if learning:
        variables = [array for branch, _ in branches for array in (
            branch.canvas_weights, branch.stream_state_weights, branch.stream_state_bias,
        )]
        if coupling is not None:
            variables.extend(coupling.weights.values())
    before = [array.copy() for array in variables]
    gradients = [_gradient(array, lambda: _energy(canvas, branches, coupling, states)) for array in variables]
    sensory_before = [branch.visual_weights.copy() for branch, _ in branches]
    energy_before = _energy(canvas, branches, coupling, states)

    correction = infer_joint(canvas, branches, coupling=coupling, stream_states=states)

    changes = [array - original for array, original in zip(variables, before)]
    if not learning:
        changes[0] = correction
    for change, gradient in zip(changes, gradients):
        np.testing.assert_allclose(change, -.001 * (1 if learning else 6) * gradient, rtol=1e-5, atol=1e-10)
    for (branch, _), weights in zip(branches, sensory_before):
        np.testing.assert_array_equal(branch.visual_weights, weights)
        assert not branch.diagnostics.observation_present
        assert np.isnan(branch.diagnostics.visual.mean)
        assert branch.diagnostics.stream_state.observed is states[branch]
        assert not branch.diagnostics.stream_state.weights.flags.writeable
        np.testing.assert_allclose(branch.diagnostics.inference_energy[0], energy_before)
    if not learning:
        np.testing.assert_allclose(
            camera.diagnostics.inference_energy[-1],
            _energy(canvas + correction, branches, coupling, states), rtol=1e-6,
        )


def test_off_streams_are_context_evidence_learning_is_explicit_and_waiting_is_not_off():
    engine = RippleEngine(resolution=(6, 8), plane_size_m=(1, 1), speed=1, damping=.98)
    source = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False),
            audio=AudioSourceConfig(enabled=False),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(
                cross_modal_enabled=True, learning_enabled=True,
                learning_rate=.1, weight_decay=.001,
            ),
        ),
        processor=SimpleNamespace(frame_counter=0), engine=engine,
        resolution=engine.resolution, n_sources=1,
    )
    source.set_diagnostics_enabled(True)
    sensory_before = [branch.visual_weights.copy() for branch in source.hidden_coupling.branches]
    for _ in range(200):
        engine.propagate()
        source.predict_visual()
        source.correct_from_observations()
    for branch, weights in zip(source.hidden_coupling.branches, sensory_before):
        context = branch.diagnostics.stream_state
        assert context.observed is False
        assert context.prediction < .35
        assert branch.diagnostics.weight_update_norm > 0
        np.testing.assert_array_equal(branch.visual_weights, weights)
    parameters = [
        array for branch in source.hidden_coupling.branches for array in (
            branch.canvas_weights, branch.visual_weights, branch.stream_state_weights,
            branch.stream_state_bias, source.hidden_coupling.weights[branch],
        )
    ]
    before = [array.copy() for array in parameters]
    source.update_pathway_config(learning_enabled=False)
    source.reset_readouts()
    source.set_diagnostics_enabled(True)
    hidden_before = source.visual_readout.hidden.copy()
    # Enabled capture with no sample is ON context and an absent sensory target.
    source.camera_source.enabled = True
    source.camera_source.excitation = lambda: None
    source.audio_source.enabled = True
    engine.propagate()
    source.predict_visual()
    source.correct_from_observations()
    assert source.visual_observation is None
    for branch in source.hidden_coupling.branches:
        assert branch.diagnostics.stream_state.observed is True
        assert not branch.diagnostics.observation_present
        assert branch.diagnostics.weight_update_norm == 0
    assert np.any(source.visual_readout.hidden != hidden_before)
    for array, original in zip(parameters, before):
        np.testing.assert_array_equal(array, original)
    source.camera_source.set_enabled(False)
    source.audio_source.set_enabled(False)
    loop_before = {branch: source.hidden_coupling.weights[branch].copy() for branch in source.hidden_coupling.branches}
    source.update_pathway_config(learning_enabled=True, cross_modal_enabled=False)
    engine.propagate()
    source.predict_visual()
    source.correct_from_observations()
    for branch, sensory_weights in zip(source.hidden_coupling.branches, sensory_before):
        assert branch.diagnostics.stream_state.observed is False
        assert branch.diagnostics.weight_update_norm > 0
        np.testing.assert_array_equal(branch.visual_weights, sensory_weights)
        np.testing.assert_array_equal(
            source.hidden_coupling.weights[branch],
            loop_before[branch],
        )


def test_context_rejects_unknown_branches_and_non_boolean_observations():
    camera = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3))
    canvas = np.zeros(camera.canvas_shape)
    with pytest.raises(ValueError, match="boolean"):
        camera.infer(canvas, None, stream_state=0)
    other = PredictiveCodingRGBReadout(canvas_shape=canvas.shape)
    with pytest.raises(ValueError, match="inferred branch"):
        infer_joint(canvas, ((camera, None),), stream_states={other: False})


def test_camera_only_context_is_independent_of_preview_collection_and_rejects_shader():
    engine = RippleEngine(resolution=(6, 8), plane_size_m=(1, 1), speed=1)
    source = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            audio=AudioSourceConfig(enabled=False),
        ),
        processor=None, engine=engine, resolution=engine.resolution, n_sources=1,
    )
    engine.propagate()
    source.predict_visual()
    assert source.audio_readout is None
    assert source.visual_readout.stream_state_prediction == .5
    source.visual_readout.set_spatial_preview_enabled(False)
    assert source.visual_readout.stream_state_prediction == .5
    assert source.correct_from_observations() is not None
    assert np.any(source.visual_readout.hidden != 0)
    weights = source.visual_readout.stream_state_weights.copy()
    bias = source.visual_readout.stream_state_bias.copy()
    source.reset_readouts()
    assert source.visual_readout.stream_state_prediction is None
    np.testing.assert_array_equal(source.visual_readout.stream_state_weights, weights)
    np.testing.assert_array_equal(source.visual_readout.stream_state_bias, bias)
    engine.use_shader = True
    with pytest.raises(NotImplementedError, match="array-backed"):
        source.correct_from_observations()
