import numpy as np
import pytest

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

    prediction = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]]],
        dtype=np.float32,
    )
    renderer = NumpyImageRenderer(
        rgb_canvas_enabled=True,
        auto_percentile_levels=False,
    )

    renderer.render(prediction)
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


@pytest.mark.parametrize("rgb", (False, True))
@pytest.mark.parametrize("auto", (False, True))
def test_gpu_presentation_matches_numpy_without_downloading_float_rgb(
    cupy_backend, qapp, monkeypatch, rgb, auto,
):
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer

    cp = cupy_backend
    field = np.random.default_rng(10).normal(size=(11, 17, 3)).astype(np.float32)
    reference = NumpyImageRenderer(rgb_canvas_enabled=rgb, auto_percentile_levels=auto)
    actual = NumpyImageRenderer(rgb_canvas_enabled=rgb, auto_percentile_levels=auto)
    transfers = []
    asnumpy = cp.asnumpy

    def download(values, *args, **kwargs):
        transfers.append((values.shape, values.dtype))
        return asnumpy(values, *args, **kwargs)

    try:
        reference.render(field)
        with monkeypatch.context() as patch:
            patch.setattr(cp, "asnumpy", download)
            actual.render(cp.asarray(field))
        np.testing.assert_allclose(actual._display_limit, reference._display_limit, rtol=1e-6)
        np.testing.assert_allclose(
            actual.image_item.image, reference.image_item.image,
            rtol=1e-6, atol=1 if rgb else 1e-7,
        )
        if rgb:
            np.testing.assert_array_equal(actual.image_item.levels, (0, 255))
        assert transfers == (
            [((11, 17, 3), cp.dtype("uint8"))] if rgb else [((11, 17), cp.dtype("float32"))]
        )
    finally:
        reference.widget.close()
        actual.widget.close()


def test_gpu_rgb_conversion_preserves_nonfinite_and_clipping_behavior(cupy_backend):
    cp = cupy_backend
    values = np.array([[[np.nan, np.inf, -np.inf], [-1, .5, 2]]], dtype=np.float32)
    np.testing.assert_array_equal(_canvas_to_rgb(cp.asarray(values), limit=1), _canvas_to_rgb(values, limit=1))
