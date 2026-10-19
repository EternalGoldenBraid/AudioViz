from dataclasses import replace

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


def source_config():
    return RippleSourceOrchestratorConfig(
        synthetic=SyntheticSourceConfig(enabled=False),
        camera_frame=CameraFrameSourceConfig(enabled=True),
        prediction_error=PredictionErrorTransformConfig(enabled=True),
        visual_pathway=PredictiveCodingConfig(dtype="float32"),
    )


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
        toggles = {toggle.key: toggle for toggle in widget.source_control_binding.build_toggles()}
        assert not toggles["pose"].available
        with pytest.raises(ValueError, match="pose-medium"):
            widget.source_control_binding.update_toggle("pose", True)
        assert not widget.use_pose_sources
        controls = widget.gaze_controls
        model = widget.source_orchestrator.visual_uncertainty
        controls.observe.setChecked(False)
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
        assert not controls.gaze_x.isEnabled()
        assert not widget.source_orchestrator.visual_readout.config.spatial_precision.sample
        controls.observe.setChecked(True)
        uniform = widget.source_orchestrator.visual_readout.config.spatial_precision
        assert uniform.baseline == 1 and uniform.peak == 0 and uniform.sample
        controls.follow_mouse.setChecked(True)
        qapp.processEvents()
        restored = widget.source_orchestrator.visual_readout.config.spatial_precision
        assert restored.baseline == .03 and restored.peak == 1.7
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
        widget.renderer.prepare_frame()
        widget.engine.reset()
        widget._sync_after_reset()
        assert widget.visual_prediction is None
        now[0] += 100
        widget.update_visualization()
        assert widget.source_orchestrator._uncertainty_dt == .025
    finally:
        widget.close()
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
