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
    ripple engine also consumes source positions as (x, y) field pixels, but the
    rendered ripple field should mirror the camera feed left/right so motion
    feels intuitive on screen. Coordinates outside the image support are not
    mapped into the ripple field.
    """
    rows, cols = _validate_resolution(resolution)
    positions = np.asarray(coords, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("coords must have shape (num_nodes, 2)")
    positions = positions[pose_coords_in_image_support(positions)]

    if field_rect is None:
        field_rect = (0.0, 0.0, float(cols), float(rows))
    x, y, width, height = _validate_field_rect(field_rect, rows=rows, cols=cols)

    mapped = np.empty_like(positions, dtype=np.float32)
    mapped[:, 0] = x + (1.0 - positions[:, 0]) * max(width - 1.0, 0.0)
    mapped[:, 1] = y + positions[:, 1] * max(height - 1.0, 0.0)

    if clip:
        mapped[:, 0] = np.clip(mapped[:, 0], x, x + max(width - 1.0, 0.0))
        mapped[:, 1] = np.clip(mapped[:, 1], y, y + max(height - 1.0, 0.0))
    return mapped


def pose_coords_in_image_support(coords: np.ndarray) -> np.ndarray:
    positions = np.asarray(coords, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("coords must have shape (num_nodes, 2)")

    return (
        np.isfinite(positions).all(axis=1)
        & (positions[:, 0] >= 0.0)
        & (positions[:, 0] <= 1.0)
        & (positions[:, 1] >= 0.0)
        & (positions[:, 1] <= 1.0)
    )


def pose_graph_state_to_ripple_sources(
    state,
    resolution: tuple[int, int],
    *,
    field_rect: FieldRect | None = None,
    acceleration_scale: float = 1.0,
    max_excitation: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map pose graph nodes to ripple positions and acceleration excitations."""
    if acceleration_scale < 0.0:
        raise ValueError("acceleration_scale must be non-negative")
    if max_excitation is not None and max_excitation < 0.0:
        raise ValueError("max_excitation must be non-negative")

    valid = pose_coords_in_image_support(state.get_positions())
    source_positions = normalized_pose_coords_to_source_positions(
        state.get_positions()[valid],
        resolution,
        field_rect=field_rect,
    )
    accelerations = np.asarray(state.get_accelerations(), dtype=np.float32)[valid]
    excitations = np.linalg.norm(accelerations, axis=1).astype(np.float32)
    excitations *= np.float32(acceleration_scale)
    if max_excitation is not None:
        excitations = np.clip(excitations, 0.0, max_excitation).astype(np.float32)
    return source_positions, excitations


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
