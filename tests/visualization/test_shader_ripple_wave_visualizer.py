from types import SimpleNamespace

import numpy as np
import pytest

from audioviz.readouts import PredictiveCodingConfig
from audioviz.sources import AudioSourceConfig, CameraFrameSourceConfig, RippleSourceOrchestratorConfig
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig


@pytest.mark.parametrize("layout", ("comparison", "prediction", "3d"))
def test_shader_live_flow_previews_diagnostics_reset_and_cleanup(qapp, layer_view_stub, monkeypatch, layout):
    from PyQt5 import QtGui
    from audioviz.utils.opengl_compute import GLArray
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    if qapp.platformName() == "offscreen":
        pytest.skip("QOpenGLWidget needs a window-system context; run with QT_QPA_PLATFORM=xcb")
    context = QtGui.QOpenGLContext()
    surface_format = QtGui.QSurfaceFormat()
    surface_format.setVersion(4, 3)
    context.setFormat(surface_format)
    if not context.create():
        pytest.skip("OpenGL unavailable")

    class Capture:
        released = False
        present = True

        def read(self):
            return self.present, np.full((10, 12, 3), 180, dtype=np.uint8) if self.present else None

        def release(self):
            self.released = True

    capture = Capture()
    processor = SimpleNamespace(
        frame_counter=1, n_mels=None, stft_window=np.ones(16),
        spectrogram_buffers=[np.ones((16, 1), dtype=np.float32)],
    )
    widget = RippleWaveVisualizer(
        processor=processor, use_shader=True, resolution=(48, 72),
        plane_size_m=(1, 1), rgb_canvas_enabled=True,
        source_orchestrator_config=RippleSourceOrchestratorConfig(
            audio=AudioSourceConfig(enabled=True),
            camera_frame=CameraFrameSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(
                dtype="float32", inference_steps=2, learning_enabled=True, cross_modal_enabled=True,
            ),
        ),
        camera_capture=capture,
    )
    widget.timer.stop()
    try:
        widget.resize(800, 600)
        widget.show()
        qapp.processEvents()
        widget.set_prediction_scene_visible(True)
        view = widget.prediction_scene
        view.layout_selector.setCurrentIndex(view.layout_selector.findData(layout))
        # Fail on implicit full-field host transfer; the UI may download bounded snapshots.
        monkeypatch.setattr(widget.engine.propagator, "get_state", lambda: pytest.fail("Full texture readback"))
        widget.update_visualization()
        assert isinstance(widget.visual_prediction, GLArray)
        assert isinstance(widget.source_orchestrator.visual_observation, GLArray)
        assert view.canvas is not None
        assert all(
            value is None or (isinstance(value, np.ndarray) and not value.flags.writeable)
            for source in view.sources for value in (source.observation, source.prediction, source.hidden)
        )
        qapp.processEvents()
        assert not widget.renderer.widget.grabFramebuffer().isNull()
        widget.set_prediction_diagnostics_visible(True)
        processor.frame_counter += 1
        widget.update_visualization()
        widget.prediction_diagnostics.refresh()
        widget.prediction_diagnostics.audio_view.refresh()
        assert widget.prediction_diagnostics.latest is not None
        assert widget.prediction_diagnostics.audio_view.latest is not None
        widget.toggle_controls()
        toggles = {toggle.key: toggle for toggle in widget.source_control_binding.build_toggles()}
        assert toggles["camera"].available and toggles["audio"].available
        widget.source_control_binding.update_toggle("audio", False)
        widget.source_control_binding.update_toggle("audio", True)
        widget.source_control_binding.update_toggle("camera", False)
        widget.source_control_binding.update_toggle("camera", True)
        widget.source_orchestrator.update_pathway_config(cross_modal_enabled=False)
        widget.source_orchestrator.update_pathway_config(cross_modal_enabled=True)
        capture.present = False
        widget.update_visualization()
        assert widget.source_orchestrator.visual_observation is None
        widget.renderer.prepare_frame()
        widget.engine.reset()
        widget._sync_after_reset()
        assert view.canvas is None
        assert np.all(widget.engine.get_field().download() == 0)
    finally:
        widget.close()
    assert capture.released
    assert widget.engine.propagator.compute.closed
    assert not widget.engine.propagator.compute.buffers


def test_shader_camera_only_comparison_downloads_only_bounded_snapshots(qapp, monkeypatch):
    from PyQt5.QtTest import QTest
    from audioviz.utils.opengl_compute import GLArray
    from audioviz.visualization.prediction_scene_window import PredictionSceneWindow
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    if qapp.platformName() == "offscreen":
        pytest.skip("QOpenGLWidget needs a window-system context; run with QT_QPA_PLATFORM=xcb")
    capture = SimpleNamespace(
        read=lambda: (True, np.full((12, 18, 3), 160, dtype=np.uint8)),
        release=lambda: None,
    )
    widget = RippleWaveVisualizer(
        use_shader=True, resolution=(600, 1000), plane_size_m=(1, 1),
        camera_capture=capture,
        source_orchestrator_config=RippleSourceOrchestratorConfig(
            camera_frame=CameraFrameSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(dtype="float32", inference_steps=1),
        ),
    )
    widget.timer.stop()
    transfers = []
    original = GLArray.download

    def download(array):
        transfers.append(array.shape)
        return original(array)

    try:
        widget.resize(800, 600)
        widget.show()
        assert QTest.qWaitForWindowExposed(widget)
        qapp.processEvents()
        widget.prediction_scene = PredictionSceneWindow(widget)
        widget.prediction_scene.visibility_changed.connect(widget._set_diagnostics_collection)
        widget.prediction_scene.layout_selector.setCurrentIndex(1)
        widget.set_prediction_scene_visible(True)
        with monkeypatch.context() as patch:
            patch.setattr(GLArray, "download", download)
            widget.update_visualization()
        assert widget.prediction_scene.canvas.shape == (307, 512, 3)
        assert all(max(shape[:2]) <= 512 for shape in transfers)
        assert widget.source_orchestrator.audio_readout is None
        assert not widget.source_orchestrator.visual_readout.diagnostics_enabled
        assert all(source.hidden is None for source in widget.prediction_scene.sources)
    finally:
        widget.close()
