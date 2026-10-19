import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from audioviz.readouts import GaussianSpatialPrecision
from audioviz.sources import CircularDotConfig
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
        widget.refresh_previews()
        assert widget.stimulus.latest.time == 0
        assert widget.truth_image.image.shape == (24, 32, 3)
        assert widget.prediction_image.image.shape == (24, 32, 3)
        assert np.isfinite(widget.prediction_image.image).all()
        assert widget.visualizer.source_orchestrator.visual_readout.config.learning_enabled is False

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
        widget.observe.setChecked(False)
        assert not widget.precision_image.image.any()
        assert widget.visualizer.source_orchestrator.visual_readout.config.spatial_precision.sample is False
        for _ in range(2):
            widget.visualizer.update_visualization()
            widget._after_frame()
        assert widget.stimulus.exhausted
        assert widget.progress.text() == "Finished: 3 frames"
        assert not widget.visualizer.timer.isActive() and not widget.preview_timer.isActive()
    finally:
        widget.close()
        qapp.processEvents()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
