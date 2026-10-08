from types import SimpleNamespace

import numpy as np

from audioviz.sources import AudioSourceConfig, RippleSourceOrchestratorConfig, SyntheticSourceConfig
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


def test_audio_tab_collects_independently_of_camera_and_clears_reset_hide_and_stale_evidence(
    qapp, monkeypatch, layer_view_stub
):
    import audioviz.sources.audio as source_module
    import audioviz.visualization.prediction_diagnostics_view as diagnostics_module

    clock = [0.0]
    monkeypatch.setattr(source_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(diagnostics_module, "monotonic", lambda: clock[0])
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
    audio.layer_toggle.setChecked(True)
    visualizer.update_visualization()
    audio.refresh()

    assert len(audio.history) == 1
    assert root.latest is None
    assert len(audio.latest.observation_means) == AUDIO_SPECTRAL_BANDS
    assert audio.layer_view.preview[1] is audio.latest
    np.testing.assert_array_equal(
        audio.layer_view.preview[0], visualizer.engine.get_field_preview()
    )
    assert visualizer.source_orchestrator.audio_readout.spatial_preview_enabled
    assert not visualizer.source_orchestrator.visual_readout.spatial_preview_enabled
    clock[0] = 0.05
    visualizer.update_visualization()
    assert len(audio.history) == 1
    assert audio.layer_view.preview is not None
    clock[0] = 0.6
    visualizer.update_visualization()
    assert audio.latest is None
    assert audio.layer_view.preview is None

    clock[0] = 0.7
    processor.frame_counter += 1
    processor.spectrogram_buffers[0].fill(0)
    visualizer.update_visualization()
    audio.layer_toggle.setChecked(False)
    audio.refresh()
    assert len(audio._graph_nodes.data) == 3 + 6 + AUDIO_SPECTRAL_BANDS
    assert len(audio._graph_edges) == 3 * 6 + 6 * AUDIO_SPECTRAL_BANDS
    np.testing.assert_array_equal(audio.latest.observation_means, 0)
    visualizer.source_control_binding.update_control("learning-dynamics", "learning_enabled", True)
    visualizer.source_control_binding.update_control("learning-dynamics", "inference_steps", 3)
    assert visualizer.source_orchestrator.audio_readout.config.learning_enabled
    assert visualizer.source_orchestrator.audio_readout.config.inference_steps == 3
    assert visualizer.source_orchestrator.visual_readout.config.inference_steps == 3
    visualizer.source_control_binding.update_control("audio-source", "observation_gain", 2)
    assert visualizer.audio_source.observation_gain == 2
    keys = {control.key for control in visualizer.audio_source.controls()}
    assert "observation_gain" in keys
    assert "mapping_mode" not in keys
    visualizer.engine.reset()
    visualizer._sync_after_reset()
    assert not audio.history
    assert not root.history
    assert np.all(visualizer.source_orchestrator.audio_readout.hidden == 0)
    root.hide()
    assert not visualizer.source_orchestrator.audio_readout.diagnostics_enabled
    assert not visualizer.source_orchestrator.visual_readout.diagnostics_enabled
    visualizer.close()


def test_switching_tabs_releases_camera_preview_and_requests_fresh_evidence_on_return(
    qapp, layer_view_stub
):
    from audioviz.readouts import PredictiveCodingRGBReadout
    from audioviz.visualization.prediction_diagnostics_view import PredictionDiagnosticsView

    canvas = np.full((10, 20, 3), 0.2, dtype=np.float32)
    model = PredictiveCodingRGBReadout(canvas_shape=canvas.shape)
    model.set_diagnostics_enabled(True)
    view = PredictionDiagnosticsView()
    view.add_audio_tab()
    view.spatial_preview_changed.connect(model.set_spatial_preview_enabled)
    view.graph_toggle.setChecked(True)
    view.layer_toggle.setChecked(True)
    view.show()
    assert view.take_spatial_preview_request()
    model.predict(canvas)
    model.infer(canvas, canvas)
    view.record(0.0, model.diagnostics, canvas_preview=canvas)
    view.refresh()
    assert view.layer_view.preview is not None

    view.tabs.setCurrentIndex(1)
    assert view.layer_view.preview is None
    assert view.latest.spatial is None
    assert not model.spatial_preview_enabled
    assert not view.take_spatial_preview_request()

    view.tabs.setCurrentIndex(0)
    assert model.spatial_preview_enabled
    assert view.take_spatial_preview_request()
    assert view.layer_view.preview is None
    view.close()
