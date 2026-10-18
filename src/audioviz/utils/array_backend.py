from __future__ import annotations

from types import ModuleType

import numpy as np


def array_module(values) -> ModuleType:
    if isinstance(values, np.ndarray):
        return np
    import cupy as cp

    return cp.get_array_module(values)


def to_numpy(values) -> np.ndarray:
    """Transfer device arrays only at an explicit host-facing boundary."""
    if hasattr(values, "__cuda_array_interface__"):
        import cupy as cp

        return cp.asnumpy(values)
    return np.asarray(values)
