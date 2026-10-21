import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from audioviz.readouts import GaussianSpatialPrecision
from audioviz.sources import CircularDotConfig
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.visualization.active_vision_experiment import ActiveVisionExperiment


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_experiment_delivers_stimulus_updates_gaze_skips_and_stops(qapp, request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl":
        if qapp.platformName() == "offscreen":
            pytest.skip("QOpenGLWidget requires a window-system context")
        context = QtGui.QOpenGLContext()
        surface_format = QtGui.QSurfaceFormat()
        surface_format.setVersion(4, 3)
        context.setFormat(surface_format)
        if not context.create():
            pytest.skip("OpenGL unavailable")
    widget = ActiveVisionExperiment(
        stimulus=CircularDotConfig(
            resolution=(24, 32), radius=6, frame_rate=10, duration=0.3,
        ),
        precision=GaussianSpatialPrecision(), backend=backend, inference_steps=2, memory_time_constant=.7,
    )
    widget.visualizer.timer.stop()
    widget.preview_timer.stop()
    try:
        widget.show()
        qapp.processEvents()
        widget.resize(640, 1100)
        qapp.processEvents()
        assert widget._preview_columns == 1
        for index, plot in enumerate(widget._preview_plots):
            assert widget.previews.ci.items[plot] == [(index, 0)]
            viewport_bounds = widget.previews.viewport().rect().adjusted(-1, -1, 1, 1)
            assert viewport_bounds.contains(
                widget.previews.mapFromScene(plot.sceneBoundingRect()).boundingRect(),
            )
        widget.resize(1180, 860)
        qapp.processEvents()
        assert widget._preview_columns == 3
        assert all(
            widget.previews.ci.items[plot] == [(0, index)]
            for index, plot in enumerate(widget._preview_plots)
        )
        widget.visualizer.update_visualization()
        widget._after_frame()
        widget.refresh_previews()
        assert widget.stimulus.latest.time == 0
        assert widget.truth_image.image.shape == (24, 32, 3)
        assert widget.prediction_image.image.shape == (24, 32, 3)
        assert np.isfinite(widget.prediction_image.image).all()
        assert widget.visualizer.source_orchestrator.visual_readout.config.learning_enabled is False
        assert widget.uncertainty.memory_time_constant == .7 and "memory 0.7 s" in widget.assumptions.text()
        learning = next(
            section for section in widget.visualizer.source_control_binding.build_sections()
            if section.key == "learning-dynamics"
        )
        assert next(control for control in learning.controls if control.key == "learning_enabled").kind == "text"
        assert all(next(control for control in learning.controls if control.key == key).kind == "text"
                   for key in ("inference_steps", "inference_rate"))
        assert not any(control.key == "cross_modal_enabled" for control in learning.controls)
        assert widget.information_scores.shape == (9, 13)
        assert np.isfinite(widget.information_scores).all() and widget.information_scores.max() > 0
        assert widget.information_image.image.shape == (24, 32)
        assert widget.information_image.isVisible()
        assert widget.precision.gaze == (0.5, 0.5)
        variance = spatial_preview(widget.uncertainty.variance, max_side=32).copy()
        widget._after_frame()
        np.testing.assert_array_equal(
            spatial_preview(widget.uncertainty.variance, max_side=32), variance,
        )
        widget.information_toggle.setChecked(False)
        assert not widget.information_image.isVisible()
        assert not widget.direction_arrow.isVisible()
        widget.information_toggle.setChecked(True)

        qapp.processEvents()
        assert widget.follow_mouse.isChecked()
        for image, gaze in (
            (widget.truth_image, (0.75, 0.25)),
            (widget.precision_image, (0.25, 0.75)),
            (widget.prediction_image, (0, 1)),
        ):
            point = QtCore.QPointF(0.5 + gaze[0] * 31, 0.5 + gaze[1] * 23)
            widget.previews.scene().sigMouseMoved.emit(image.mapToScene(point))
            assert widget.precision.gaze == gaze
            assert widget.visualizer.source_orchestrator.visual_readout.config.spatial_precision.gaze == gaze
        widget.previews.scene().sigMouseMoved.emit(QtCore.QPointF(-1000, -1000))
        assert widget.precision.gaze == (0, 1)
        widget.full_canvas.setChecked(True)
        widget.previews.scene().sigMouseMoved.emit(widget.truth_image.mapToScene(QtCore.QPointF(16, 12)))
        assert widget.precision.gaze == (0, 1)
        assert widget.full_canvas.isChecked() and not widget.follow_mouse.isChecked()
        assert not widget.autonomous_gaze.isChecked()
        np.testing.assert_array_equal(widget.precision_image.image, 1)
        assert widget.precision.baseline == 1 and widget.precision.peak == 0
        assert not widget.gaze_marker.isVisible() and not widget.gaze_x.isEnabled()
        widget.refresh_previews()
        assert "full canvas" in widget.information_status.text()
        assert not widget.direction_arrow.isVisible()
        widget.observe.setChecked(False)
        assert not widget.precision_image.image.any()
        widget.observe.setChecked(True)
        widget.follow_mouse.setChecked(True)
        assert widget.precision.baseline == .01 and widget.precision.peak == 1

        widget.gaze_x.setValue(0.9)
        assert widget.visualizer.source_orchestrator.visual_readout.config.spatial_precision.gaze == (0.9, 1)
        widget.refresh_previews()
        assert widget.direction_arrow.isVisible()
        scores_before_missing_frame = widget.information_scores.copy()
        widget.visualizer.camera_source.set_enabled(False)
        widget.visualizer.update_visualization()
        widget._after_frame()
        widget.refresh_previews()
        assert widget.stimulus.latest.time == 0
        factor = np.exp(-.2/widget.uncertainty.memory_time_constant)
        increment = widget.uncertainty.process_variance*widget.uncertainty.memory_time_constant/2*(1-factor)
        np.testing.assert_allclose(
            spatial_preview(widget.uncertainty.variance, max_side=32),
            factor*variance+increment, rtol=2e-6, atol=1e-7,
        )
        assert not np.array_equal(widget.information_scores, scores_before_missing_frame)
        variance = spatial_preview(widget.uncertainty.variance, max_side=32).copy()
        widget.visualizer.camera_source.set_enabled(True)
        widget.observe.setChecked(False)
        assert not widget.precision_image.image.any()
        assert widget.visualizer.source_orchestrator.visual_readout.config.spatial_precision.sample is False
        for _ in range(2):
            widget.visualizer.update_visualization()
            widget._after_frame()
        assert widget.stimulus.exhausted
        assert widget.progress.text() == "Finished: 3 frames - Restart to continue"
        assert not widget.visualizer.timer.isActive() and not widget.preview_timer.isActive()
        np.testing.assert_allclose(
            spatial_preview(widget.uncertainty.variance, max_side=32),
            factor**2*variance + increment*(1+factor), rtol=2e-6, atol=1e-7,
        )
        assert widget.information_scores.max() > 0
        assert "sampling paused" in widget.information_status.text()
        assert widget.precision.gaze == (0.9, 1)
        with pytest.raises(ValueError, match="fixed-weight"):
            widget.visualizer.source_orchestrator.update_pathway_config(learning_enabled=True)
        assert not widget.visualizer.source_orchestrator.visual_readout.config.learning_enabled
        widget.information_scores[:] = 1
        widget._update_advisory_marker()
        assert "no gaze preference" in widget.information_status.text()
        assert not widget.direction_arrow.isVisible()
        widget.information_scores[:] = 0
        widget._update_advisory_marker()
        assert not widget.recommended_marker.isVisible() and not widget.direction_arrow.isVisible()
        assert "No expected gain" in widget.information_status.text()
        widget.visualizer.source_orchestrator.reset_readouts()
        widget.uncertainty._ensure_initialized()
        np.testing.assert_array_equal(spatial_preview(widget.uncertainty.variance, max_side=32), 1)
    finally:
        widget.close()
        qapp.processEvents()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_finished_run_can_restart_with_autonomous_gaze_and_reproducible_stimulus(
    qapp, request, monkeypatch, backend,
):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("QOpenGLWidget requires a window-system context")
    widget = ActiveVisionExperiment(
        stimulus=CircularDotConfig(
            resolution=(24, 32), radius=6, frame_rate=10, duration=0.2,
            seed=7, brownian_scale=2,
        ),
        precision=GaussianSpatialPrecision(), backend=backend, inference_steps=2,
    )
    widget.visualizer.timer.stop()
    widget.preview_timer.stop()
    if backend != "opengl":
        def ordinary_pc(*args, **kwargs):
            pytest.fail("reset must leave the Gaussian mean untouched until its physical-time forecast")
        monkeypatch.setattr(widget.visualizer.source_orchestrator.visual_readout, "predict", ordinary_pc)
    try:
        widget.show()
        qapp.processEvents()
        for index in range(2):
            widget.visualizer.update_visualization()
            widget._after_frame()
            if index == 0:
                first_frame = widget.stimulus.latest.frame.copy()
        assert widget.stimulus.exhausted and widget.restart_button.isEnabled()
        old_capture = widget.visualizer.camera_source.capture
        point = widget.truth_image.mapToScene(QtCore.QPointF(.5+.75*31, .5+.25*23))
        widget.previews.scene().sigMouseMoved.emit(point)
        assert widget.precision.gaze == (.75, .25)
        widget.autonomous_gaze.click()
        assert not widget.visualizer.timer.isActive()
        assert "click Restart" in widget.information_status.text()
        widget.restart_button.click()
        assert widget.visualizer.timer.isActive() and widget.preview_timer.isActive()
        widget.visualizer.timer.stop()
        widget.preview_timer.stop()
        assert old_capture.read() == (False, None)
        assert widget.stimulus.latest is None and not widget.stimulus.exhausted
        assert widget.visualizer.engine.time == 0
        np.testing.assert_array_equal(widget.visualizer.engine.get_field_numpy(), 0)
        widget.uncertainty._ensure_initialized()
        np.testing.assert_array_equal(spatial_preview(widget.uncertainty.variance, max_side=32), 1)
        widget.visualizer.update_visualization()
        widget._after_frame()
        widget.refresh_previews()
        assert widget.stimulus.latest.time == 0
        np.testing.assert_array_equal(widget.stimulus.latest.frame, first_frame)
        assert widget.autonomous_gaze.isChecked() and widget.precision.sample
        assert widget.precision.gaze != (.75, .25)
        assert widget.restart_button.isEnabled()
    finally:
        widget.close()
        qapp.processEvents()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_autonomous_gaze_acts_on_model_steps_not_preview_or_mouse_events(qapp, request, monkeypatch, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("QOpenGLWidget requires a window-system context")
    widget = ActiveVisionExperiment(
        stimulus=CircularDotConfig(resolution=(24, 32), radius=6, frame_rate=10, duration=0.4),
        precision=GaussianSpatialPrecision(baseline=.02, peak=2),
        backend=backend, inference_steps=2, gaze_mode="autonomous",
    )
    widget.visualizer.timer.stop()
    widget.preview_timer.stop()
    calls = []
    policy = widget.visualizer.source_orchestrator.gaze_policy
    original = policy.select
    scores = np.zeros((9, 13), dtype=np.float32)
    scores[2, 9] = 10
    monkeypatch.setattr(widget.uncertainty, "expected_information_gain", lambda *a, **k: scores.copy())

    def select(values, current):
        calls.append(current)
        return original(values, current)

    monkeypatch.setattr(policy, "select", select)
    try:
        widget.show()
        qapp.processEvents()
        widget.visualizer.update_visualization()
        widget._after_frame()
        assert len(calls) == 1
        assert widget.precision.gaze == (.75, .25)
        assert (widget.gaze_x.value(), widget.gaze_y.value()) == (.75, .25)
        assert not widget.follow_mouse.isChecked() and not widget.full_canvas.isChecked()
        widget.previews.scene().sigMouseMoved.emit(widget.truth_image.mapToScene(QtCore.QPointF(16, 12)))
        for _ in range(3):
            widget.refresh_previews()
        assert len(calls) == 1 and widget.precision.gaze == (.75, .25)
        variance = spatial_preview(widget.uncertainty.variance, max_side=32).copy()
        widget.observe.setChecked(False)
        widget.visualizer.update_visualization()
        widget._after_frame()
        assert len(calls) == 1 and widget.precision.gaze == (.75, .25)
        factor = np.exp(-.2/widget.uncertainty.memory_time_constant)
        increment = widget.uncertainty.process_variance*widget.uncertainty.memory_time_constant/2*(1-factor)
        np.testing.assert_allclose(
            spatial_preview(widget.uncertainty.variance, max_side=32),
            factor*variance+increment, rtol=2e-6, atol=1e-7,
        )
        widget.observe.setChecked(True)
        widget.information_toggle.setChecked(False)
        widget.visualizer.update_visualization()
        widget._after_frame()
        assert len(calls) == 2
        widget.information_toggle.setChecked(True)
        widget.full_canvas.setChecked(True)
        assert widget.visualizer.source_orchestrator.gaze_policy is None
        np.testing.assert_array_equal(widget.precision_image.image, 1)
        widget.follow_mouse.setChecked(True)
        assert widget.precision.baseline == .02 and widget.precision.peak == 2
        assert widget.gaze_x.isEnabled()
        assert "deterministic" in widget.assumptions.text()
    finally:
        widget.close()
        qapp.processEvents()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
