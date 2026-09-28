import numpy as np

from audioviz.visualization.ripple_renderers import (
    _canvas_to_rgb,
    _percentile_abs_limit,
    _resolve_auto_level_limit,
)


def test_canvas_to_rgb_maps_state_channels_isomorphically():
    canvas = np.array(
        [[[1.0, 0.5, 0.0], [0.0, 1.0, -1.0]]],
        dtype=np.float32,
    )

    rgb = _canvas_to_rgb(canvas, limit=1.0)

    np.testing.assert_array_equal(
        rgb,
        np.array([[[255, 128, 0], [0, 255, 0]]], dtype=np.uint8),
    )


def test_numpy_renderer_rgb_toggle_renders_canvas_channels(qapp):
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer

    class FieldSource:
        field = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]]],
            dtype=np.float32,
        )

        def get_field_numpy(self):
            return self.field

    source = FieldSource()
    renderer = NumpyImageRenderer(
        rgb_canvas_enabled=True,
        auto_percentile_levels=False,
    )

    renderer.render(source)
    np.testing.assert_array_equal(
        renderer.image_item.image,
        np.array([[[255, 0, 0], [0, 128, 255]]], dtype=np.uint8),
    )


def test_percentile_abs_limit_returns_raw_active_percentile():
    field = np.array([[1e-4, -2e-4], [0.0, 0.0]], dtype=np.float32)

    limit = _percentile_abs_limit(field, percentile=98.0)

    assert limit is not None
    assert 1e-4 <= limit <= 2e-4


def test_resolve_auto_level_limit_uses_reference_below_threshold():
    limit = _resolve_auto_level_limit(
        0.05,
        activation_threshold=0.1,
        floor=0.25,
    )

    assert limit == 0.25


def test_resolve_auto_level_limit_uses_active_limit_above_threshold():
    limit = _resolve_auto_level_limit(
        0.3,
        activation_threshold=0.1,
        floor=0.25,
    )

    assert limit == 0.3
