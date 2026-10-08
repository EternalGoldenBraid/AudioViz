from __future__ import annotations

import numpy as np


PREVIEW_MAX_SIDE = 64


def preview_indices(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving nearest-neighbor samples, including both image edges."""
    rows, columns = shape[:2]
    scale = max(1.0, max(rows, columns) / PREVIEW_MAX_SIDE)
    return (
        np.linspace(0, rows - 1, max(1, round(rows / scale)), dtype=np.intp),
        np.linspace(0, columns - 1, max(1, round(columns / scale)), dtype=np.intp),
    )


def spatial_preview(field: np.ndarray) -> np.ndarray:
    rows, columns = preview_indices(field.shape)
    sampled = field[rows[:, None], columns[None, :]]
    snapshot = np.asarray(sampled, dtype=np.float32)
    snapshot.setflags(write=False)
    return snapshot
