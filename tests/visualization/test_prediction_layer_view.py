import numpy as np

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
    assert "E=" in view.labels["H0"].text
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
