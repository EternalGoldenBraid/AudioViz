from dataclasses import replace
import sys
import traceback

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from audioviz.readouts import GaussianSpatialPrecision, PredictiveCodingConfig
from audioviz.sources import (
    CameraFrameSourceConfig, CircularDotCapture, CircularDotConfig, CircularDotSource,
    RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)
from audioviz.sources.gaze import ActiveVisionConfig, GazePolicy
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.visualization import ripple_wave_visualizer as module


@pytest.fixture(autouse=True)
def fail_on_qt_callback_errors(monkeypatch):
    errors = []
    monkeypatch.setattr(
        sys, "excepthook",
        lambda *error: errors.append("".join(traceback.format_exception(*error))),
    )
    yield
    assert not errors, "\n".join(errors)


def source_config():
    return RippleSourceOrchestratorConfig(
        synthetic=SyntheticSourceConfig(enabled=False),
        camera_frame=CameraFrameSourceConfig(enabled=True),
        prediction_error=PredictionErrorTransformConfig(enabled=True),
        visual_pathway=PredictiveCodingConfig(dtype="float32"),
    )


class FakeAudioProcessor:
    def __init__(self):
        self.frame_counter = 0
        self.n_mels = None
        self.stft_window = np.ones(2, dtype=np.float32)
        self.spectrogram_buffers = [np.zeros((32, 4), dtype=np.float32)]
        self.current_signal_level = 0.
        self.minimum_signal_level = .05
        self.current_top_k_frequencies = []

    def push_spectrum(self, value):
        self.spectrogram_buffers[0][:, -1] = value
        self.current_signal_level = float(np.max(value))
        self.frame_counter += 1


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_live_gaze_uses_elapsed_time_and_keeps_selection_before_capture(qapp, request, monkeypatch, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    now = [10.]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    stimulus = CircularDotSource(CircularDotConfig(resolution=(12, 16), radius=3, duration=2))
    precision = GaussianSpatialPrecision(baseline=.03, peak=1.7)
    selected = []

    def select(policy, values, current):
        selected.append(stimulus.latest)
        return (.75, .25)

    monkeypatch.setattr(GazePolicy, "select", select)
    widget = module.RippleWaveVisualizer(
        resolution=(12, 16), update_interval_ms=25, plane_size_m=(1, 1),
        use_shader=backend == "opengl", use_gpu=backend == "cupy",
        camera_capture=CircularDotCapture(stimulus), source_orchestrator_config=source_config(),
        active_vision=ActiveVisionConfig(
            mode="autonomous", precision=precision, memory_time_constant=.5, process_variance=.04,
        ),
    )
    widget.timer.stop()
    try:
        widget.resize(1000, 800)
        widget.show()
        qapp.processEvents()
        widget.update_visualization()
        assert selected == [None]
        assert widget.gaze_controls.gaze_x.value() == .75
        assert widget.source_orchestrator._uncertainty_dt == .025
        assert not widget.gaze_controls.isVisible()
        widget.controls_button.click()
        qapp.processEvents()
        panel = widget.control_panel
        assert panel.gaze_controls is widget.gaze_controls
        assert widget.gaze_controls.window() is panel and widget.gaze_controls.isVisible()
        assert panel.section_widgets["gaze"].is_expanded()
        assert not panel.section_widgets["wave-physics"].is_expanded()
        assert "display" in panel.section_widgets
        panel.close()
        widget.controls_button.click()
        assert widget.control_panel is panel and widget.gaze_controls.isVisible()
        toggles = {toggle.key: toggle for toggle in widget.source_control_binding.build_toggles()}
        assert not toggles["pose"].available
        with pytest.raises(ValueError, match="pose-medium"):
            widget.source_control_binding.update_toggle("pose", True)
        assert not widget.use_pose_sources
        controls = widget.gaze_controls
        model = widget.source_orchestrator.visual_uncertainty
        controls.observe.setChecked(False)
        assert "prediction only" in widget.gaze_status.text()
        assert widget.renderer.prepare_frame()
        previous = spatial_preview(model.variance).copy()
        now[0] += .2
        widget.update_visualization()
        factor = np.exp(-.4/.5)
        np.testing.assert_allclose(
            spatial_preview(model.variance), factor*previous+.04*.5/2*(1-factor), rtol=2e-6,
        )
        assert len(selected) == 1
        assert widget.source_orchestrator._uncertainty_dt == pytest.approx(.2)
        controls.full_canvas.setChecked(True)
        assert "full canvas" in widget.gaze_status.text()
        assert not controls.gaze_x.isEnabled()
        assert not controls.sigma.isEnabled()
        assert not widget.source_orchestrator.visual_readout.config.spatial_precision.sample
        controls.observe.setChecked(True)
        uniform = widget.source_orchestrator.visual_readout.config.spatial_precision
        assert uniform.baseline == 1 and uniform.peak == 0 and uniform.sample
        controls.follow_mouse.setChecked(True)
        qapp.processEvents()
        restored = widget.source_orchestrator.visual_readout.config.spatial_precision
        assert restored.baseline == .03 and restored.peak == 1.7
        controls.sigma.setValue(.2)
        controls.baseline.setValue(.04)
        controls.peak.setValue(2.)
        configured = widget.source_orchestrator.visual_readout.config.spatial_precision
        assert (configured.sigma, configured.baseline, configured.peak) == (.2, .04, 2.)
        if backend == "opengl":
            surface = widget.renderer.widget
            scale = min(surface.width()/16, surface.height()/12)
            point = QtCore.QPointF(
                (surface.width()-16*scale)/2+(.5+.25*15)*scale,
                (surface.height()-12*scale)/2+(.5+.75*11)*scale,
            )
            event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove, point, QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
            )
            QtWidgets.QApplication.sendEvent(surface, event)
            margin = QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove, QtCore.QPointF(-10, -10),
                QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
            )
            QtWidgets.QApplication.sendEvent(surface, margin)
        else:
            image = widget.renderer.image_item
            widget._move_image_gaze(image.mapToScene(QtCore.QPointF(.5+.25*15, .5+.75*11)))
            widget._move_image_gaze(QtCore.QPointF(-1000, -1000))
        assert widget.source_orchestrator.visual_readout.config.spatial_precision.gaze == (.25, .75)
        assert widget.attention_overlay.precision.gaze == (.25, .75)
        widget.set_model_diagnostics_visible(True)
        now[0] += .11
        widget.belief_diagnostics.refresh()
        widget._belief_visibility_changed(True)
        assert widget.belief_diagnostics.latest is not None
        assert widget.belief_diagnostics.latest.visual["sensory_covariance"].shape == (3, 3)
        assert not widget.source_orchestrator.visual_readout.diagnostics_enabled
        widget.belief_diagnostics.hide()
        assert widget._belief_frame is None
        assert not widget.diagnostics_button.isChecked()
        widget.set_prediction_scene_visible(True)
        view = widget.prediction_scene
        layouts = ("comparison", "prediction") if qapp.platformName() == "offscreen" else ("3d", "comparison", "prediction")
        for layout in layouts:
            view.layout_selector.setCurrentIndex(view.layout_selector.findData(layout))
            qapp.processEvents()
            now[0] += .1
            widget.update_visualization()
            camera = next(source for source in view.sources if source.mapping.key == "camera")
            assert camera.prediction is not None and camera.observation is not None
            assert camera.stream_state is None and camera.stream_state_prediction is None
        controls.observe.setChecked(False)
        panel.reset_button.click()
        assert widget.visual_prediction is None
        now[0] += 100
        widget.update_visualization()
        assert widget.source_orchestrator._uncertainty_dt == .025
        widget.camera_source.capture.read = lambda: (False, None)
        now[0] += .2
        widget.update_visualization()
        assert widget.source_orchestrator.visual_observation is None
        assert widget.camera_source.capture is None
        assert widget.attention_overlay.target is None
        assert "off/on to retry" in widget.gaze_status.text()
        assert "off/on to retry" in panel.source_control_widgets[("camera-source", "camera_status")].text()
        now[0] += .2
        widget.update_visualization()
    finally:
        widget.close()
        if widget.control_panel is not None:
            assert not widget.control_panel.isVisible()
        qapp.processEvents()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize("changes,error", (
    ({"processor": object()}, "camera-only"),
    ({"source_orchestrator_config": replace(source_config(), visual_pathway=PredictiveCodingConfig(learning_enabled=True))}, "fixed-weight"),
    ({"source_orchestrator_config": replace(source_config(), prediction_error=PredictionErrorTransformConfig(enabled=False))}, "camera prediction"),
))
def test_live_gaze_rejects_incompatible_paths_before_startup(qapp, changes, error):
    kwargs = dict(active_vision=ActiveVisionConfig(), source_orchestrator_config=source_config())
    kwargs.update(changes)
    with pytest.raises(ValueError, match=error):
        module.RippleWaveVisualizer(**kwargs)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_panel_keeps_global_sound_and_visual_observation_independent(qapp, request, monkeypatch, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    processor = FakeAudioProcessor()
    now = [10.]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    stimulus = CircularDotSource(CircularDotConfig(resolution=(12, 16), radius=3, duration=2))
    widget = module.RippleWaveVisualizer(
        processor=processor, resolution=(12, 16), plane_size_m=(1, 1),
        use_shader=backend == "opengl", use_gpu=backend == "cupy",
        camera_capture=CircularDotCapture(stimulus), source_orchestrator_config=source_config(),
        active_vision=ActiveVisionConfig(audio_precision=1),
    )
    widget.timer.stop()

    def advance():
        now[0] += .03
        widget.update_visualization()

    try:
        widget.resize(1000, 800)
        widget.show()
        qapp.processEvents()
        widget.controls_button.click()
        qapp.processEvents()
        panel = widget.control_panel
        controls = panel.gaze_controls
        assert controls.observe.text() == "Observe vision"
        controls.observe.setChecked(False)
        processor.push_spectrum(.6)
        advance()
        model = widget.source_orchestrator.audio_uncertainty
        assert model is not None
        assert widget.source_orchestrator.audio_prediction.shape == (16,)
        assert not widget.audio_source.has_new_observation()
        assert not widget.source_orchestrator.visual_readout.config.spatial_precision.sample
        assert "Sound:" in widget.gaze_status.text() and "global spectrum" in widget.gaze_status.text()
        assert "No sound direction" in panel.source_control_widgets[("audio-source", "audio_geometry")].text()

        precision = panel.source_control_widgets[("audio-source", "audio_precision")]
        precision.setValue(0)
        assert model.precision == 0
        processor.push_spectrum(.2)
        advance()
        assert not widget.audio_source.has_new_observation()
        assert "Sound: prediction only" in widget.gaze_status.text()

        audio_toggle = panel.source_toggle_checkboxes["audio"]
        audio_toggle.click()
        processor.push_spectrum(.8)
        advance()
        assert not widget.audio_source.enabled and "Sound: off" in widget.gaze_status.text()
        assert widget.audio_source.recent_observation() is None
        audio_toggle.click()
        precision.setValue(1.5)
        controls.observe.setChecked(True)
        processor.push_spectrum(0.)
        advance()
        assert widget.audio_source.enabled and model.precision == 1.5
        np.testing.assert_array_equal(widget.audio_source.recent_observation(), np.zeros(16))
        assert widget.source_orchestrator.visual_readout.config.spatial_precision.sample
        widget.set_model_diagnostics_visible(True)
        now[0] += .11
        widget.belief_diagnostics.refresh()
        beliefs = widget.belief_diagnostics
        assert beliefs.latest.audio["sensory_covariance"].shape == (16, 16)
        assert beliefs.latest.audio_gain >= 0
        assert beliefs.tabs.isTabEnabled(2)
        assert not widget.source_orchestrator.audio_readout.diagnostics_enabled
        beliefs.hide()

        widget.set_prediction_scene_visible(True)
        view = widget.prediction_scene
        view.layout_selector.setCurrentIndex(view.layout_selector.findData("comparison"))
        qapp.processEvents()
        advance()
        sound = next(source for source in view.sources if source.mapping.key == "audio")
        assert sound.prediction.shape == (16,) and sound.observation.shape == (16,)
        assert sound.stream_state is None and sound.stream_state_prediction is None
        panel.reset_button.click()
        assert widget.visual_prediction is None
        assert widget.source_orchestrator.audio_prediction is None
    finally:
        widget.close()
        qapp.processEvents()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
