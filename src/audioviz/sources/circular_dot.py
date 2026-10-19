from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class CircularDotConfig:
    resolution: tuple[int, int] = (120, 160)
    center: tuple[float, float] | None = None
    radius: float = 30.0
    angular_velocity: float = np.pi / 2
    phase: float = 0.0
    dot_radius: float = 3.0
    frame_rate: float = 30.0
    duration: float = 10.0
    brownian_scale: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if len(self.resolution) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.resolution
        ):
            raise ValueError("resolution must contain two positive integers")
        if self.center is not None and (
            len(self.center) != 2 or not np.all(np.isfinite(self.center))
        ):
            raise ValueError("center must contain two finite pixel coordinates")
        for name in ("radius", "brownian_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("dot_radius", "frame_rate", "duration"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.all(np.isfinite((self.angular_velocity, self.phase))):
            raise ValueError("angular_velocity and phase must be finite")
        if not np.isfinite(self.frame_rate * self.duration):
            raise ValueError("frame count must be finite")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class CircularDotSample:
    """Evaluator data; only the BGR frame crosses the capture interface."""

    time: float
    position: tuple[float, float]
    offset: tuple[float, float]
    frame: np.ndarray


class CircularDotSource:
    """Finite stimulus with unclipped trajectories and clipped rasterization."""

    def __init__(self, config: CircularDotConfig):
        self.config = config
        self.frame_count = ceil(config.duration * config.frame_rate)
        self.latest: CircularDotSample | None = None
        self._index = 0
        self._offset = np.zeros(2, dtype=np.float64)
        self._rng = np.random.default_rng(config.seed)

    @property
    def exhausted(self):
        return self._index >= self.frame_count

    def next_sample(self) -> CircularDotSample | None:
        if self.exhausted:
            return None
        config = self.config
        rows, cols = config.resolution
        center = config.center if config.center is not None else ((cols - 1) / 2, (rows - 1) / 2)
        time = self._index / config.frame_rate
        if self._index and config.brownian_scale:
            self._offset += self._rng.normal(
                scale=config.brownian_scale / np.sqrt(config.frame_rate), size=2,
            )
        angle = config.phase + config.angular_velocity * time
        x = center[0] + config.radius * np.cos(angle) + self._offset[0]
        y = center[1] + config.radius * np.sin(angle) + self._offset[1]
        frame = np.zeros((rows, cols, 3), dtype=np.uint8)
        radius = config.dot_radius
        left, right = max(0, int(np.ceil(x - radius))), min(cols, int(np.floor(x + radius)) + 1)
        top, bottom = max(0, int(np.ceil(y - radius))), min(rows, int(np.floor(y + radius)) + 1)
        if left < right and top < bottom:
            v, u = np.ogrid[top:bottom, left:right]
            frame[top:bottom, left:right][(u - x)**2 + (v - y)**2 <= radius**2] = 255
        frame.setflags(write=False)
        self.latest = CircularDotSample(
            time=time, position=(float(x), float(y)),
            offset=tuple(float(value) for value in self._offset), frame=frame,
        )
        self._index += 1
        return self.latest


class CircularDotCapture:
    """Camera-compatible adapter exposing frames without evaluator metadata."""

    def __init__(self, source: CircularDotSource):
        self._source = source
        self._closed = False

    def read(self):
        if self._closed:
            return False, None
        sample = self._source.next_sample()
        return (False, None) if sample is None else (True, sample.frame)

    def release(self):
        self._closed = True
