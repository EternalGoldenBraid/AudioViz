import numpy as np

from audioviz.visualization.ripple_renderers import (
    _percentile_abs_limit,
    _resolve_auto_level_limit,
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
