import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from audioviz.readouts import PredictiveCodingRGBReadout
from audioviz.sources.camera import CameraFrameSource
from audioviz.sources.audio import AudioRippleSource
from audioviz.utils.source_preview import SourceScenePreview
from audioviz.visualization.prediction_layer_view import (
    PredictionLayerView,
    hidden_texture,
    rgb_texture,
    spectrum_mesh,
)


def test_textures_preserve_image_orientation_and_distinguish_signed_hidden_states():
    field = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]])
    texture = rgb_texture(field)
    np.testing.assert_array_equal(texture[0, -1], [255, 0, 0, 255])
    np.testing.assert_array_equal(texture[-1, 0], [255, 255, 255, 255])
    assert texture.dtype == np.uint8
    assert np.all(texture[:, :, 3] == 255)
    signed = hidden_texture(np.array([[-1.0, 0.0, 1.0]]))
    assert signed[0, 0, 2] > signed[0, 0, 0]
    np.testing.assert_array_equal(signed[1, 0], [25, 25, 25, 255])
    assert signed[2, 0, 0] > signed[2, 0, 2]


def test_layer_planes_reuse_items_preserve_camera_and_clear_pixels(qapp):
    field = np.full((2, 3, 3), 0.5, dtype=np.float32)
    readout = PredictiveCodingRGBReadout(canvas_shape=field.shape)
    readout.set_diagnostics_enabled(True)
    readout.set_spatial_preview_enabled(True)
    readout.predict(field)
    readout.infer(field, np.ones_like(field))
    view = PredictionLayerView()
    snapshot = readout.diagnostics
    camera_source = SourceScenePreview(
        CameraFrameSource.scene_mapping, True,
        observation=snapshot.spatial.observation, prediction=snapshot.spatial.prediction,
        hidden=snapshot.spatial.hidden, hidden_energies=snapshot.hidden.channel_means,
    )
    audio_source = SourceScenePreview(
        AudioRippleSource.scene_mapping, True, observation=np.linspace(0, 1, 16),
        prediction=np.linspace(-0.1, 0.1, 16), hidden=snapshot.spatial.hidden,
    )
    view.set_scene(field, (camera_source, audio_source))
    planes = dict(view.planes)

    assert len(planes) == 3 + 2 * readout.config.hidden_channels
    assert set(view.histograms) == {"audio/prediction", "audio/observation"}
    np.testing.assert_array_equal(planes["canvas"].data, rgb_texture(field))
    assert "pre-evidence" in view.labels["camera/prediction"].text
    assert view.labels["camera/hidden/0"].text.startswith("C H0:")
    view.orbit(15, 10)
    camera = view.cameraParams()
    histograms = dict(view.histograms)
    view.set_scene(field, (camera_source, audio_source))
    assert view.planes == planes
    assert view.histograms == histograms
    assert view.cameraParams() == camera
    view.clear_evidence("audio")
    assert not view.histograms["audio/observation"].visible()
    assert view.histograms["audio/prediction"].visible()
    view.clear_preview()
    assert all(not plane.visible() for plane in planes.values())
    assert all(plane.data.shape == (1, 1, 4) for plane in planes.values())
    view.reset_view()
    assert view.cameraParams()["azimuth"] == -90
    view.close()


def test_spectral_histograms_preserve_signed_level_and_use_a_fixed_display_scale():
    vertices, faces, colors = spectrum_mesh(np.array([-0.5, 0, 0.25, 2]), (0, 0, 0), 4)
    bars = vertices.reshape(4, 8, 3)
    np.testing.assert_allclose(bars[0, :, 1].min(), -0.8)
    np.testing.assert_allclose(bars[2, :, 1].max(), 0.4)
    np.testing.assert_allclose(bars[3, :, 1].max(), 1.6)
    np.testing.assert_array_equal(bars[1, :, 1:], 0)
    assert faces.shape == (48, 3)
    assert faces.max() == 31
    assert colors[0, 0] > colors[0, 2]
    assert colors[-1, 2] > colors[-1, 0]


def test_layer_view_drag_orbits_and_wheel_zooms(qapp):
    view = PredictionLayerView()
    camera = view.cameraParams()
    QtWidgets.QApplication.sendEvent(
        view,
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, QtCore.QPointF(40, 40),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
        ),
    )
    QtWidgets.QApplication.sendEvent(
        view,
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(60, 55),
            QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
        ),
    )
    assert view.cameraParams()["azimuth"] == camera["azimuth"] - 20
    assert view.cameraParams()["elevation"] == camera["elevation"] + 15
    QtWidgets.QApplication.sendEvent(
        view,
        QtGui.QWheelEvent(
            QtCore.QPointF(60, 55), QtCore.QPointF(60, 55),
            QtCore.QPoint(), QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False,
        ),
    )
    assert view.cameraParams()["distance"] < camera["distance"]
    view.close()


def test_native_context_mismatch_never_paints_a_success_shaped_blank_view(qapp, monkeypatch):
    import audioviz.visualization.prediction_layer_view as layer_view

    view = PredictionLayerView()
    failures = []
    view.rendering_failed.connect(failures.append)
    monkeypatch.setattr(layer_view.platform, "GetCurrentContext", lambda: None)
    monkeypatch.setattr(view, "isVisible", lambda: True)
    monkeypatch.setattr(view, "isValid", lambda: True)
    view.initializeGL()
    view.paintGL()
    view._check_context()
    assert len(failures) == 1
    assert "incompatible contexts" in failures[0]
    view._check_context()
    assert len(failures) == 2
    view.close()
