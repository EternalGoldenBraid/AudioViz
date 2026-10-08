import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from audioviz.readouts import PredictiveCodingRGBReadout
from audioviz.visualization.prediction_layer_view import (
    PredictionLayerView,
    hidden_texture,
    rgb_texture,
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
    view.set_preview(field, readout.diagnostics)
    planes = dict(view.planes)

    assert len(planes) == 3 + readout.config.hidden_channels
    np.testing.assert_array_equal(planes["Canvas (corrected)"].data, rgb_texture(field))
    assert "before evidence" in view.labels["Prediction (before evidence)"].text
    assert view.labels["H0"].text.startswith("H0:")
    view.orbit(15, 10)
    camera = view.cameraParams()
    view.set_preview(field, readout.diagnostics)
    assert view.planes == planes
    assert view.cameraParams() == camera
    view.clear_preview()
    assert all(not plane.visible() for plane in planes.values())
    assert all(plane.data.shape == (1, 1, 4) for plane in planes.values())
    view.reset_view()
    assert view.cameraParams()["azimuth"] == -90
    view.close()


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
