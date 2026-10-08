import numpy as np
from PyQt5 import QtCore

from audioviz.sources.camera import CameraFrameSource
from audioviz.utils.source_preview import SourceScenePreview
from audioviz.visualization.prediction_scene_window import PredictionSceneWindow


def test_scene_throttles_latest_samples_and_releases_hidden_minimized_reset_and_failed_views(
    qapp, monkeypatch, layer_view_stub
):
    import audioviz.visualization.prediction_scene_window as scene_module

    clock = [0.0]
    monkeypatch.setattr(scene_module, "monotonic", lambda: clock[0])
    view = PredictionSceneWindow()
    assert view.layer_view is None
    assert not view.take_spatial_preview_request()
    view.show()
    assert isinstance(view.layer_view, layer_view_stub)
    assert view.take_spatial_preview_request()
    assert not view.take_spatial_preview_request()
    canvas = np.ones((2, 3, 3), dtype=np.float32)
    source = SourceScenePreview(CameraFrameSource.scene_mapping, True, observation=canvas)
    view.record(canvas, (source,))
    assert view.layer_view.updates == 1
    view.invalidate_evidence("camera")
    assert view.sources[0].observation is None
    assert view.layer_view.preview[1][0].observation is None
    assert "no current evidence" in view.status.text()
    clock[0] = 0.1
    assert view.take_spatial_preview_request()
    view.setWindowState(QtCore.Qt.WindowMinimized)
    assert not view.take_spatial_preview_request()
    assert view.canvas is None
    view.setWindowState(QtCore.Qt.WindowNoState)
    assert view.take_spatial_preview_request()
    view.record(canvas, (source,))
    view.hide()
    assert view.canvas is None
    assert view.layer_view.preview is None
    view.show()
    assert view.take_spatial_preview_request()
    view.clear()
    assert view.take_spatial_preview_request()
    view.layer_view.rendering_failed.emit("No compatible OpenGL context")
    assert view.error.isVisible()
    assert "No compatible OpenGL context" in view.error.text()
    assert not view.take_spatial_preview_request()
    assert view.canvas is None
    view.close()
