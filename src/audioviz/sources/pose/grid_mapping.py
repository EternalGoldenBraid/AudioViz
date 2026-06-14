from __future__ import annotations

from typing import Tuple

import numpy as np


FieldRect = Tuple[float, float, float, float]


def centered_field_rect(
    resolution: tuple[int, int],
    *,
    width_fraction: float = 1.0,
    height_fraction: float = 1.0,
) -> FieldRect:
    """Return a centered pixel rectangle as (x, y, width, height)."""
    rows, cols = _validate_resolution(resolution)
    if width_fraction <= 0.0 or height_fraction <= 0.0:
        raise ValueError("width_fraction and height_fraction must be positive")
    if width_fraction > 1.0 or height_fraction > 1.0:
        raise ValueError("width_fraction and height_fraction must be <= 1")

    width = max(1.0, cols * float(width_fraction))
    height = max(1.0, rows * float(height_fraction))
    x = (cols - width) * 0.5
    y = (rows - height) * 0.5
    return (x, y, width, height)


def normalized_pose_coords_to_source_positions(
    coords: np.ndarray,
    resolution: tuple[int, int],
    *,
    field_rect: FieldRect | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Map normalized pose coordinates to ripple source positions.

    MediaPipe pose landmarks are normalized as (x, y) camera coordinates. The
    ripple engine also consumes source positions as (x, y) field pixels, so this
    is the only conversion needed while the pose graph is camera-frame anchored.
    """
    rows, cols = _validate_resolution(resolution)
    positions = np.asarray(coords, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("coords must have shape (num_nodes, 2)")

    if field_rect is None:
        field_rect = (0.0, 0.0, float(cols), float(rows))
    x, y, width, height = _validate_field_rect(field_rect, rows=rows, cols=cols)

    mapped = np.empty_like(positions, dtype=np.float32)
    mapped[:, 0] = x + positions[:, 0] * max(width - 1.0, 0.0)
    mapped[:, 1] = y + positions[:, 1] * max(height - 1.0, 0.0)

    if clip:
        mapped[:, 0] = np.clip(mapped[:, 0], x, x + max(width - 1.0, 0.0))
        mapped[:, 1] = np.clip(mapped[:, 1], y, y + max(height - 1.0, 0.0))
    return mapped


def _validate_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
    if len(resolution) != 2:
        raise ValueError("resolution must contain (rows, cols)")
    rows, cols = int(resolution[0]), int(resolution[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution values must be positive")
    return rows, cols


def _validate_field_rect(
    field_rect: FieldRect,
    *,
    rows: int,
    cols: int,
) -> FieldRect:
    if len(field_rect) != 4:
        raise ValueError("field_rect must contain (x, y, width, height)")
    x, y, width, height = (float(value) for value in field_rect)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("field_rect width and height must be positive")
    if x < 0.0 or y < 0.0 or x + width > cols or y + height > rows:
        raise ValueError("field_rect must fit inside resolution")
    return x, y, width, height
