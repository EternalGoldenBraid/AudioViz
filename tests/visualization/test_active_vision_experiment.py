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

        widget.gaze_x.setValue(0.9)
        assert widget.visualizer.source_orchestrator.visual_readout.config.spatial_precision.gaze == (0.9, 0.5)
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
