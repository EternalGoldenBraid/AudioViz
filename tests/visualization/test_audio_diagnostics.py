from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from audioviz.sources import AudioSourceConfig, RippleSourceOrchestratorConfig, SyntheticSourceConfig
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS
from audioviz.readouts import PredictiveCodingConfig
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


def test_audio_catalogue_and_independent_scene_hold_real_samples_clear_stale_evidence_and_reset(
    qapp, monkeypatch, layer_view_stub
):
    import audioviz.sources.audio as source_module
    import audioviz.visualization.prediction_scene_window as scene_module

    clock = [0.0]
    monkeypatch.setattr(source_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(scene_module, "monotonic", lambda: clock[0])
    processor = SimpleNamespace(
        current_top_k_frequencies=[], current_signal_level=0, frame_counter=1,
        spectrogram_buffers=[np.full((32, 2), 0.7)], stft_window=np.ones(16),
        n_mels=None, minimum_signal_level=0.05,
    )
    visualizer = RippleWaveVisualizer(
        processor=processor, resolution=(10, 20), plane_size_m=(1, 1), speed=1,
        source_orchestrator_config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False), audio=AudioSourceConfig(enabled=True)
        ),
    )
    visualizer.timer.stop()
    visualizer.set_prediction_diagnostics_visible(True)
    root = visualizer.prediction_diagnostics
    audio = root.audio_view
    root.tabs.setCurrentIndex(1)
    audio.graph_toggle.setChecked(True)
    audio.open_scene_requested.emit()
    scene = visualizer.prediction_scene
    visualizer.update_visualization()
    audio.refresh()
    sources = {source.mapping.key: source for source in scene.sources}

    assert len(audio.history) == 1
    assert root.latest is None
    assert sources["audio"].mapping.representation == "spectrum"
    assert sources["audio"].observation.shape == (AUDIO_SPECTRAL_BANDS,)
    assert sources["audio"].hidden.shape == (10, 20, 6)
    assert sources["camera"].hidden is not None
    assert set(sources) == {"camera", "audio", "synthetic", "pose"}
    np.testing.assert_array_equal(scene.canvas, visualizer.engine.get_field_preview())
    assert not visualizer.source_orchestrator.audio_readout.spatial_preview_enabled
    assert not visualizer.source_orchestrator.visual_readout.spatial_preview_enabled
    clock[0] = 0.05
    visualizer.update_visualization()
    assert len(audio.history) == 1
    assert scene.layer_view.updates == 1
    clock[0] = 0.6
    visualizer.update_visualization()
    assert audio.latest is None
    assert next(source for source in scene.sources if source.mapping.key == "audio").observation is None
    assert scene.canvas is not None

    clock[0] = 0.71
    processor.frame_counter += 1
    processor.spectrogram_buffers[0].fill(0)
    visualizer.update_visualization()
    audio.refresh()
    assert len(audio._graph_nodes.data) == 3 + 6 + AUDIO_SPECTRAL_BANDS
    assert len(audio._graph_edges) == 3 * 6 + 6 * AUDIO_SPECTRAL_BANDS
    np.testing.assert_array_equal(audio.latest.observation_means, 0)
    np.testing.assert_array_equal(
        next(source for source in scene.sources if source.mapping.key == "audio").observation, 0
    )
    visualizer.source_control_binding.update_control("learning-dynamics", "learning_enabled", True)
    visualizer.source_control_binding.update_control("learning-dynamics", "inference_steps", 3)
    assert visualizer.source_orchestrator.audio_readout.config.learning_enabled
    assert visualizer.source_orchestrator.audio_readout.config.inference_steps == 3
    assert visualizer.source_orchestrator.visual_readout.config.inference_steps == 3
    visualizer.source_control_binding.update_control("audio-source", "observation_gain", 2)
    assert visualizer.audio_source.observation_gain == 2

    root.close()
    assert scene.isVisible()
    assert visualizer.source_orchestrator.audio_readout.diagnostics_enabled
    scene.open_catalogue_requested.emit()
    assert root.isVisible()
    scene.close()
    assert root.isVisible()
    assert visualizer.source_orchestrator.audio_readout.diagnostics_enabled
    root.close()
    assert not visualizer.source_orchestrator.audio_readout.diagnostics_enabled
    assert not visualizer.source_orchestrator.visual_readout.diagnostics_enabled
    assert not visualizer.diagnostics_button.isChecked()
    assert not visualizer.scene_button.isChecked()

    scene.show()
    visualizer.engine.reset()
    visualizer._sync_after_reset()
    assert not audio.history
    assert not root.history
    assert scene.canvas is None
    assert not scene.sources
    assert np.all(visualizer.source_orchestrator.audio_readout.hidden == 0)
    visualizer.close()
    assert not scene.isVisible()


@pytest.mark.parametrize(
    "cross_modal_enabled,stream_state_enabled", ((False, False), (True, False), (False, True), (True, True))
)
def test_multimodal_scene_does_not_change_inference_learning_or_wave_state(
    qapp, monkeypatch, layer_view_stub, cross_modal_enabled, stream_state_enabled
):
    import audioviz.sources.audio as source_module
    import audioviz.visualization.prediction_scene_window as scene_module

    clock = [0.0]
    monkeypatch.setattr(source_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(scene_module, "monotonic", lambda: clock[0])
    processor = SimpleNamespace(
        current_top_k_frequencies=[], current_signal_level=0, frame_counter=1,
        spectrogram_buffers=[np.full((32, 2), 0.7)], stft_window=np.ones(16), n_mels=None,
    )
    config = RippleSourceOrchestratorConfig(
        synthetic=SyntheticSourceConfig(enabled=False), audio=AudioSourceConfig(enabled=True),
        prediction_error=PredictionErrorTransformConfig(enabled=True),
        visual_pathway=PredictiveCodingConfig(
            learning_enabled=True, cross_modal_enabled=cross_modal_enabled, stream_state_enabled=stream_state_enabled,
        ),
    )
    baseline, visible = (
        RippleWaveVisualizer(
            processor=deepcopy(processor), resolution=(6, 8), plane_size_m=(1, 1), speed=1,
            source_orchestrator_config=config,
        )
        for _ in range(2)
    )
    for visualizer in (baseline, visible):
        visualizer.timer.stop()
        visualizer.camera_source.enabled = True
    visible.set_prediction_scene_visible(True)
    for time, generation, camera_level in ((0, 1, 0.6), (0.05, 1, 0.7), (0.12, 2, None), (0.63, 2, 0.9)):
        clock[0] = time
        for visualizer in (baseline, visible):
            visualizer.processor.frame_counter = generation
            image = None if camera_level is None else np.full(visualizer.engine.canvas_shape, camera_level)
            visualizer.camera_source.excitation = lambda image=image: image
            visualizer.update_visualization()
        np.testing.assert_array_equal(visible.engine.Z, baseline.engine.Z)
        np.testing.assert_array_equal(visible.engine.Z_old, baseline.engine.Z_old)
        for name in ("visual_readout", "audio_readout"):
            expected = getattr(baseline.source_orchestrator, name)
            actual = getattr(visible.source_orchestrator, name)
            np.testing.assert_array_equal(actual.hidden, expected.hidden)
            np.testing.assert_array_equal(actual.canvas_weights, expected.canvas_weights)
            np.testing.assert_array_equal(actual.visual_weights, expected.visual_weights)
            np.testing.assert_array_equal(actual.stream_state_weights, expected.stream_state_weights)
            np.testing.assert_array_equal(actual.stream_state_bias, expected.stream_state_bias)
            np.testing.assert_array_equal(
                visible.source_orchestrator.hidden_coupling.weights[actual],
                baseline.source_orchestrator.hidden_coupling.weights[expected],
            )
    assert visible.prediction_diagnostics is None
    baseline.close()
    visible.close()


def test_cross_modal_control_keeps_missing_evidence_diagnostics_and_scene_links(
    qapp, layer_view_stub
):
    processor = SimpleNamespace(frame_counter=0, current_top_k_frequencies=[], current_signal_level=0)
    visualizer = RippleWaveVisualizer(
        processor=processor, resolution=(6, 8), plane_size_m=(1, 1), speed=1,
        source_orchestrator_config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False), audio=AudioSourceConfig(enabled=False),
        ),
    )
    visualizer.timer.stop()
    visualizer.set_prediction_diagnostics_visible(True)
    visualizer.set_prediction_scene_visible(True)
    visualizer.toggle_controls()
    checkbox = visualizer.control_panel.source_control_widgets[("learning-dynamics", "cross_modal_enabled")]
    assert not checkbox.isChecked()
    checkbox.click()
    assert visualizer.source_orchestrator.visual_readout.config.cross_modal_enabled
    assert visualizer.source_orchestrator.audio_readout.config.cross_modal_enabled
    visualizer.update_visualization()
    root = visualizer.prediction_diagnostics
    root.graph_toggle.setChecked(True)
    root.refresh()
    assert not root.latest.observation_present
    assert np.isnan(root.latest.visual.mean)
    assert len(root._graph_edges) == 36 + 36
    assert len(root._graph_nodes.data) == 12 + 6
    assert root.latest.cross_modal_weights.shape == (6, 6)
    assert "missing evidence" in root.status.text()
    sources = {source.mapping.key: source for source in visualizer.prediction_scene.sources}
    assert sources["camera"].recurrent_parent == "audio"
    assert sources["audio"].recurrent_parent == "camera"
    assert sources["camera"].observation is None
    assert sources["camera"].prediction is not None
    checkbox.click()
    assert not visualizer.source_orchestrator.visual_readout.config.cross_modal_enabled
    visualizer.control_panel.close()
    visualizer.close()


@pytest.mark.parametrize("audio_available", (False, True))
def test_stream_context_control_plots_real_off_evidence_without_inventing_sensory_samples(
    qapp, monkeypatch, layer_view_stub, audio_available
):
    import audioviz.visualization.prediction_scene_window as scene_module

    clock = [0.0]
    monkeypatch.setattr(scene_module, "monotonic", lambda: clock[0])
    processor = (
        SimpleNamespace(frame_counter=0, current_top_k_frequencies=[], current_signal_level=0)
        if audio_available else None
    )
    visualizer = RippleWaveVisualizer(
        processor=processor, resolution=(6, 8), plane_size_m=(1, 1), speed=1,
        source_orchestrator_config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False), audio=AudioSourceConfig(enabled=False),
        ),
    )
    visualizer.timer.stop()
    visualizer.set_prediction_diagnostics_visible(True)
    visualizer.set_prediction_scene_visible(True)
    visualizer.toggle_controls()
    context_toggle = visualizer.control_panel.source_control_widgets[("learning-dynamics", "stream_state_enabled")]
    assert not context_toggle.isChecked()
    context_toggle.click()
    visualizer.source_control_binding.update_control("learning-dynamics", "learning_enabled", True)
    visualizer.update_visualization()
    root = visualizer.prediction_diagnostics
    root.graph_toggle.setChecked(True)
    root.refresh()
    assert len(root._graph_nodes.data) == 13
    assert len(root._graph_edges) == 36 + 6
    context = root.latest.stream_state
    assert context.observed is False
    assert root.latest.weight_update_norm > 0
    assert not root.latest.observation_present
    assert np.isnan(root.latest.visual.mean)
    np.testing.assert_allclose(root.stream_mean_curve.getData()[1][-1], context.energy)
    np.testing.assert_array_equal(root.stream_variance_curve.getData()[1], 0)
    assert "Stream state OFF (clamped)" in root.status.text()
    sources = {source.mapping.key: source for source in visualizer.prediction_scene.sources}
    for key in (("camera", "audio") if audio_available else ("camera",)):
        assert sources[key].stream_state is False
        assert sources[key].observation is None
        assert 0 <= sources[key].stream_state_prediction <= 1
    if not audio_available:
        assert sources["audio"].stream_state is None
        assert sources["audio"].hidden is None
    visualizer.source_control_binding.update_control("learning-dynamics", "learning_enabled", False)
    visualizer.camera_source.enabled = True
    visualizer.camera_source.excitation = lambda: None
    visualizer.audio_source.enabled = audio_available
    clock[0] = .2
    visualizer.update_visualization()
    root.refresh()
    assert root.latest.stream_state.observed is True
    assert not root.latest.observation_present
    assert root.latest.weight_update_norm == 0
    assert "Stream state ON (clamped)" in root.status.text()
    sources = {source.mapping.key: source for source in visualizer.prediction_scene.sources}
    assert sources["camera"].stream_state is True
    assert sources["audio"].stream_state is (True if audio_available else None)
    context_toggle.click()
    visualizer.update_visualization()
    assert root.latest is None
    root.refresh()
    assert np.isnan(root.stream_mean_curve.getData()[1][-1])
    visualizer.control_panel.close()
    visualizer.close()
