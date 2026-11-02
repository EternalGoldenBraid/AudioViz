from __future__ import annotations

import numpy as np

from audioviz.utils.spatial_preview import spatial_preview


class CanvasTracerWake:
    """Display-only tracers in a signed canvas field extruded through depth."""

    MAX_PARTICLES = 768
    TRAIL_POINTS = 8
    EMISSION_COUNT = 24
    LIFETIME = 2.0
    VELOCITY_SCALE = 3.0
    CANVAS_ORIGIN = (-7.5, -1.0)
    CANVAS_WIDTH = 15.0

    def __init__(self) -> None:
        self.positions = np.zeros((self.MAX_PARTICLES, 3))
        self.trails = np.zeros((self.MAX_PARTICLES, self.TRAIL_POINTS, 3))
        self.born = np.zeros(self.MAX_PARTICLES)
        self.strength = np.zeros(self.MAX_PARTICLES)
        self.active = np.zeros(self.MAX_PARTICLES, dtype=bool)
        self.field: np.ndarray | None = None
        self._cursor = 0
        self._time: float | None = None

    def set_field(self, field: np.ndarray | None) -> None:
        if field is None:
            self.field = None
            return
        if field.ndim != 3 or field.shape[2] != 3 or min(field.shape[:2]) == 0:
            raise ValueError("The tracer field requires a nonempty three-channel canvas.")
        if not np.all(np.isfinite(field)):
            raise ValueError("The tracer field must contain finite canvas states.")
        self.field = spatial_preview(field)

    def velocity(self, positions: np.ndarray) -> np.ndarray:
        velocity = np.zeros_like(positions, dtype=float)
        if self.field is None:
            return velocity
        rows, columns = self.field.shape[:2]
        origin_x, origin_y = self.CANVAS_ORIGIN
        width = self.CANVAS_WIDTH
        height = width * rows / columns
        x = (positions[:, 0] - origin_x) / width * (columns - 1)
        # Match the backdrop: image row zero is the far (+y) edge.
        y = (1 - (positions[:, 1] - origin_y) / height) * (rows - 1)
        inside = (
            (positions[:, 0] >= origin_x) & (positions[:, 0] <= origin_x + width)
            & (positions[:, 1] >= origin_y) & (positions[:, 1] <= origin_y + height)
        )
        x, y = x[inside], y[inside]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, columns - 1), np.minimum(y0 + 1, rows - 1)
        dx, dy = (x - x0)[:, None], (y - y0)[:, None]
        sampled = (
            (1 - dy) * ((1 - dx) * self.field[y0, x0] + dx * self.field[y0, x1])
            + dy * ((1 - dx) * self.field[y1, x0] + dx * self.field[y1, x1])
        )
        velocity[inside] = self.VELOCITY_SCALE * np.tanh(sampled)
        return velocity

    def advance(self, timestamp: float) -> None:
        if not np.isfinite(timestamp) or (self._time is not None and timestamp < self._time):
            raise ValueError("Tracer time must be finite and monotonic.")
        elapsed = 0 if self._time is None else timestamp - self._time
        self._time = timestamp
        self.active &= timestamp - self.born < self.LIFETIME
        if elapsed == 0:
            return
        indices = np.flatnonzero(self.active)
        # Expire against real time; bound motion after a hidden/stalled GUI.
        self.positions[indices] += min(elapsed, .2) * self.velocity(self.positions[indices])
        self.trails[indices, 1:] = self.trails[indices, :-1]
        self.trails[indices, 0] = self.positions[indices]

    def emit(self, positions: np.ndarray, strength: float) -> None:
        if self._time is None:
            raise RuntimeError("Advance tracer time before emitting a source sample.")
        if positions.shape != (self.EMISSION_COUNT, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("Each tracer emission requires 24 finite 3D source positions.")
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Tracer strength must be in [0, 1].")
        indices = (self._cursor + np.arange(self.EMISSION_COUNT)) % self.MAX_PARTICLES
        self._cursor = (self._cursor + self.EMISSION_COUNT) % self.MAX_PARTICLES
        self.positions[indices] = positions
        self.trails[indices] = positions[:, None, :]
        self.born[indices] = self._time
        self.strength[indices] = strength
        self.active[indices] = True

    def clear(self) -> None:
        self.active.fill(False)
        self._cursor = 0
        self._time = None
