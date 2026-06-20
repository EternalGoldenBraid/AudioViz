import numpy as np

from audioviz.visualization.ripple_renderers import _percentile_abs_limit


def test_percentile_abs_limit_respects_floor():
    field = np.array([[1e-4, -2e-4], [0.0, 0.0]], dtype=np.float32)

    limit = _percentile_abs_limit(field, percentile=98.0, floor=0.1)

    assert limit == 0.1
