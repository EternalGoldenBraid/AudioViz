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
from audioviz.utils.array_backend import to_numpy
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


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_shared_graph_clamps_only_fresh_enabled_inputs_and_keeps_joint_learning(qapp, request, monkeypatch, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    now = [10.]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    processor = FakeAudioProcessor()
    stimulus = CircularDotSource(CircularDotConfig(resolution=(12, 16), radius=3, duration=2))
    config = source_config()
    config = replace(config, visual_pathway=replace(
        config.visual_pathway, learning_enabled=True, learning_rate=.05,
    ))
    widget = module.RippleWaveVisualizer(
        processor=processor, resolution=(12, 16), plane_size_m=(1, 1),
        use_shader=backend == "opengl", use_gpu=backend == "cupy",
        camera_capture=CircularDotCapture(stimulus), source_orchestrator_config=config,
        active_vision=ActiveVisionConfig(mode="full", audio_precision=1, shared_sensory_graph=True),
    )
    widget.timer.stop()
    applied = []
    apply_correction = widget.engine.apply_observation_correction

    def capture_correction(correction):
        applied.append(spatial_preview(correction, max_side=16).copy())
        apply_correction(correction)

    monkeypatch.setattr(widget.engine, "apply_observation_correction", capture_correction)

    def weights():
        graph = widget.source_orchestrator.sensory_graph
        return tuple(to_numpy(getattr(graph, key)).copy() for key in (
            "canvas_weights", "recurrent_weights", "sensory_weights", "hidden_bias", "sensory_bias",
        ))

    def advance():
        now[0] += .05
        widget.update_visualization()

    try:
        widget.resize(1000, 800)
        widget.show()
        qapp.processEvents()
        processor.push_spectrum(.4)
        advance()
        source = widget.source_orchestrator
        graph = source.sensory_graph
        assert graph.total_nodes == 64 + config.visual_pathway.hidden_channels
        assert source.visual_uncertainty is None and source.audio_uncertainty is None
        assert source.hidden_coupling is None
        assert source.visual_readout.hidden.shape == (1, 1, config.visual_pathway.hidden_channels)
        if backend == "opengl":
            assert source.shader_model.hidden._handle is None
        assert widget.renderer.prepare_frame()
        observed = to_numpy(source.graph_io.sample(source.visual_observation)).ravel()
        np.testing.assert_allclose(to_numpy(graph.mean[graph.visual_slice]), observed)
        np.testing.assert_allclose(to_numpy(graph.mean[graph.audio_slice]), widget.audio_source.recent_observation())
        assert np.count_nonzero(to_numpy(graph.clamped)) == 64
        assert graph.updates == 1
        snapshot = graph.belief_snapshot()
        assert snapshot["model"] == "predictive-coding"
        assert np.all(np.diff(snapshot["inference_energy"]) <= 1e-10)
        expected_feedback = np.zeros((12, 16, 3))
        rows, cols = source.graph_io.sample_rows, source.graph_io.sample_cols
        expected_feedback[rows[:, None], cols[None, :]] = (
            -snapshot["canvas_gradient"].reshape(4, 4, 3)*source.visual_readout.config.canvas_rate
        )
        np.testing.assert_allclose(applied[-1], expected_feedback, rtol=1e-6, atol=1e-8)
        assert np.linalg.norm(applied[-1]) > 0
        samples = np.arange(48, dtype=np.float32).reshape(4, 4, 3)/47
        rendered = spatial_preview(source.graph_io.render(source.engine.backend.asarray(samples)), max_side=16)
        np.testing.assert_allclose(rendered[rows[:, None], cols[None, :]], samples, atol=1e-7)
        expected = np.empty((12, 16, 3))
        for channel in range(3):
            horizontal = np.array([np.interp(np.arange(16), cols, values[:, channel]) for values in samples])
            for col in range(16):
                expected[:, col, channel] = np.interp(np.arange(12), rows, horizontal[:, col])
        np.testing.assert_allclose(rendered, expected, atol=2e-7)
        scattered = spatial_preview(source.graph_io.scatter(source.engine.backend.asarray(samples)), max_side=16)
        full = np.linspace(-1, 1, 12*16*3).reshape(12, 16, 3)
        np.testing.assert_allclose(
            np.sum(full*scattered), np.sum(full[rows[:, None], cols[None, :]]*samples), atol=1e-7,
        )
        source.graph_io.render(graph.visual_prediction)

        widget.controls_button.click()
        qapp.processEvents()
        panel, controls = widget.control_panel, widget.gaze_controls
        assert not controls.baseline.isVisible() and not controls.peak.isVisible()
        controls.observe.setChecked(False)
        panel.source_toggle_checkboxes["audio"].click()
        learned = weights()
        advance()
        assert not to_numpy(graph.clamped).any()
        assert graph.updates == 1
        for current, before in zip(weights(), learned):
            np.testing.assert_array_equal(current, before)
        assert np.any(to_numpy(graph.mean[graph.audio_slice]) != 0)

        panel.source_toggle_checkboxes["audio"].click()
        processor.push_spectrum(0.)
        advance()
        np.testing.assert_array_equal(to_numpy(graph.mean[graph.audio_slice]), np.zeros(16))
        assert np.count_nonzero(to_numpy(graph.clamped)) == 16
        assert graph.updates == 2
        advance()
        assert not to_numpy(graph.clamped).any() and graph.updates == 2

        panel.source_control_widgets[("learning-dynamics", "learning_enabled")].click()
        assert not graph.learning_enabled
        panel.source_control_widgets[("learning-dynamics", "learning_rate")].setValue(.1)
        assert graph.learning_rate == .1
        assert source.visual_readout.config.learning_rate == source.audio_readout.config.learning_rate == .1
        panel.source_control_widgets[("learning-dynamics", "inference_steps")].setValue(12)
        assert graph.config.inference_steps == 12
        widget.set_model_diagnostics_visible(True)
        now[0] += .11
        widget.belief_diagnostics.refresh()
        view = widget.belief_diagnostics
        assert view.latest.joint["covariance"].shape == (graph.total_nodes, graph.total_nodes)
        np.testing.assert_array_equal(view.latest.visual["mean"], view.latest.audio["mean"])
        assert view.tabs.isTabEnabled(3) and not view.energy_button.isEnabled()
        assert "Gauss-Newton" in view.definition.text()
        view.tabs.setCurrentIndex(3)
        view.refresh()
        assert view.joint_tables[0].rowCount() == graph.total_nodes
        assert "free" == view.joint_tables[0].item(0, 1).text()
        for index, key in ((2, "canvas_weights"), (3, "recurrent_weights"), (4, "sensory_weights")):
            view.joint_tabs.setCurrentIndex(index)
            view.refresh()
            assert view.joint_tables[index].rowCount() == view.latest.joint[key].shape[0]
        view.joint_tabs.setCurrentIndex(0)

        widget.set_prediction_scene_visible(True)
        scene = widget.prediction_scene
        assert scene.layout_selector.currentData() == "comparison"
        assert not scene.layout_selector.model().item(scene.layout_selector.findData("3d")).isEnabled()
        controls.observe.setChecked(True)
        advance()
        camera = next(item for item in scene.sources if item.mapping.key == "camera")
        assert camera.prediction.shape == (12, 16, 3)
        assert camera.stream_state is None and camera.hidden is None
        learned = weights()
        updates = graph.updates
        panel.reset_button.click()
        assert widget.visual_prediction is None and not to_numpy(graph.clamped).any()
        for current, before in zip(weights(), learned):
            np.testing.assert_array_equal(current, before)
        assert graph.updates == updates
    finally:
        widget.close()
        qapp.processEvents()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
