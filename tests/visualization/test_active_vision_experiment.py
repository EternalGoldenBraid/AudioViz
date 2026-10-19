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
        precision=GaussianSpatialPrecision(), backend=backend, inference_steps=2,
    )
    widget.visualizer.timer.stop()
    widget.preview_timer.stop()
    try:
        widget.show()
        qapp.processEvents()
        widget.visualizer.update_visualization()
        widget._after_frame()
        widget.refresh_previews()
        assert widget.stimulus.latest.time == 0
        assert widget.truth_image.image.shape == (24, 32, 3)
        assert widget.prediction_image.image.shape == (24, 32, 3)
        assert np.isfinite(widget.prediction_image.image).all()
        assert widget.visualizer.source_orchestrator.visual_readout.config.learning_enabled is False
        learning = next(
            section for section in widget.visualizer.source_control_binding.build_sections()
            if section.key == "learning-dynamics"
        )
        assert next(control for control in learning.controls if control.key == "learning_enabled").kind == "text"
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
        widget.follow_mouse.setChecked(False)
        widget.previews.scene().sigMouseMoved.emit(widget.truth_image.mapToScene(QtCore.QPointF(16, 12)))
        assert widget.precision.gaze == (0, 1)

        widget.gaze_x.setValue(0.9)
        assert widget.visualizer.source_orchestrator.visual_readout.config.spatial_precision.gaze == (0.9, 1)
        assert widget.direction_arrow.isVisible()
        scores_before_missing_frame = widget.information_scores.copy()
        widget.visualizer.camera_source.set_enabled(False)
        widget.visualizer.update_visualization()
        widget._after_frame()
        widget.refresh_previews()
        assert widget.stimulus.latest.time == 0
        np.testing.assert_allclose(
            spatial_preview(widget.uncertainty.variance, max_side=32), .81*variance+.001, atol=1e-7,
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
        assert widget.progress.text() == "Finished: 3 frames"
        assert not widget.visualizer.timer.isActive() and not widget.preview_timer.isActive()
        np.testing.assert_allclose(
            spatial_preview(widget.uncertainty.variance, max_side=32),
            .81**2*variance + .001*(1+.81), atol=1e-7,
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
