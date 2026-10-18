from __future__ import annotations

from types import ModuleType
from functools import lru_cache

import numpy as np


def array_module(values) -> ModuleType:
    if not hasattr(values, "__cuda_array_interface__"):
        return np
    import cupy as cp

    return cp.get_array_module(values)


def project_channels(activity, weights):
    """Use a local GPU projection for small shared channel graphs."""
    if isinstance(activity, np.ndarray) or activity.ndim < 3:
        return activity @ weights
    if activity.dtype == weights.dtype and activity.shape[-1] <= 32 and weights.shape[-1] <= 32:
        xp = array_module(activity)
        result = xp.empty((*activity.shape[:-1], weights.shape[-1]), dtype=activity.dtype)
        _channel_projection_kernel()(
            xp.ascontiguousarray(activity), xp.ascontiguousarray(weights),
            activity.shape[-1], weights.shape[-1],
            result,
        )
        return result
    projected = activity.reshape(-1, activity.shape[-1]) @ weights
    return projected.reshape(*activity.shape[:-1], weights.shape[-1])


@lru_cache(maxsize=1)
def _channel_projection_kernel():
    import cupy as cp

    return cp.ElementwiseKernel(
        "raw T activity, raw T weights, int32 inputs, int32 outputs",
        "T projected",
        """
        const long long pixel = i / outputs;
        const int channel = i % outputs;
        T value = 0;
        for (int source = 0; source < inputs; ++source) {
            value += activity[pixel * inputs + source]
                * weights[source * outputs + channel];
        }
        projected = value;
        """,
        "audioviz_channel_projection",
    )


def _hidden_direction(rate, error, activity, feedback):
    return rate * (-error + (1.0 - activity**2) * feedback)


def _canvas_direction(rate, activity, projected_error):
    return rate * ((1.0 - activity**2) * projected_error)


@lru_cache(maxsize=1)
def _fused_directions():
    import cupy as cp

    return cp.fuse(_hidden_direction), cp.fuse(_canvas_direction)


def hidden_direction(rate, error, activity, feedback):
    if isinstance(activity, np.ndarray):
        return _hidden_direction(rate, error, activity, feedback)
    return _fused_directions()[0](activity.dtype.type(rate), error, activity, feedback)


def canvas_direction(rate, activity, projected_error):
    if isinstance(activity, np.ndarray):
        return _canvas_direction(rate, activity, projected_error)
    return _fused_directions()[1](activity.dtype.type(rate), activity, projected_error)


def spatial_mean(values):
    if isinstance(values, np.ndarray):
        return values.mean(axis=(0, 1))
    # Give each image row its own reduction instead of only one per channel.
    return values.sum(axis=1).sum(axis=0) / (values.shape[0] * values.shape[1])


def to_numpy(values) -> np.ndarray:
    """Transfer device arrays only at an explicit host-facing boundary."""
    if hasattr(values, "__cuda_array_interface__"):
        import cupy as cp

        return cp.asnumpy(values)
    return np.asarray(values)
