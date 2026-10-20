from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audioviz.readouts.spatial_precision import GaussianSpatialPrecision
from audioviz.readouts.shared_sensory_io import circular_foveal_offsets


@dataclass(frozen=True)
class ActiveVisionConfig:
    mode: str = "mouse"
    precision: GaussianSpatialPrecision = field(default_factory=GaussianSpatialPrecision)
    initial_variance: float = 1.0
    process_variance: float = .01
    memory_time_constant: float = 1/3
    audio_precision: float | None = None
    shared_sensory_graph: bool = False
    foveal_radius: int = 8

    def __post_init__(self):
        if self.mode not in ("mouse", "autonomous", "full"):
            raise ValueError("gaze mode must be mouse, autonomous or full")
        if not isinstance(self.shared_sensory_graph, bool):
            raise ValueError("shared_sensory_graph must be a boolean")
        circular_foveal_offsets(self.foveal_radius)
        if self.shared_sensory_graph and self.mode == "full":
            raise ValueError("Shared foveal PC supports mouse or autonomous gaze, not full-image observation")
        if self.shared_sensory_graph and self.audio_precision not in (None, 1.0):
            raise ValueError("Shared graph uses hard clamps; control audio availability with the Audio source toggle")
        if self.audio_precision is not None:
            if (
                isinstance(self.audio_precision, bool)
                or not np.isfinite(self.audio_precision)
                or self.audio_precision < 0
                or self.audio_precision > np.finfo(np.float32).max
            ):
                raise ValueError("audio_precision must be finite, non-negative and representable as float32")
            object.__setattr__(self, "audio_precision", float(self.audio_precision))


def has_gaze_preference(scores: np.ndarray) -> bool:
    return not np.allclose(scores, scores.flat[0], rtol=1e-6, atol=0)


def information_gaze(scores: np.ndarray, current: tuple[float, float]) -> tuple[float, float]:
    """Choose from a finite non-negative score grid, preserving float32 ties."""
    if not has_gaze_preference(scores):
        return current
    rows, cols = scores.shape
    maxima = np.isclose(scores, scores.max(), rtol=1e-6, atol=0)
    cy, cx = round(current[1] * (rows - 1)), round(current[0] * (cols - 1))
    if (
        np.allclose(current, (cx / (cols - 1), cy / (rows - 1)), rtol=0, atol=1e-6)
        and maxima[cy, cx]
    ):
        return current
    iy, ix = np.unravel_index(np.flatnonzero(maxima)[0], scores.shape)
    return float(ix / (cols - 1)), float(iy / (rows - 1))


class GazePolicy:
    """Select a gaze from model scores without access to observations or evaluator state."""

    def __init__(self, mode: str = "autonomous", *, seed: int = 0):
        if mode not in ("fixed", "random", "autonomous"):
            raise ValueError("gaze policy must be fixed, random or autonomous")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("gaze policy seed must be a non-negative integer")
        self.mode = mode
        self._rng = np.random.default_rng(seed)

    def select(self, scores: np.ndarray, current: tuple[float, float]) -> tuple[float, float]:
        if scores.ndim != 2 or min(scores.shape) < 2:
            raise ValueError("gaze scores must be a grid with at least two rows and columns")
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("gaze scores must be finite and non-negative")
        if len(current) != 2 or any(not np.isfinite(v) or not 0 <= v <= 1 for v in current):
            raise ValueError("current gaze must contain two finite coordinates in [0, 1]")
        if self.mode == "fixed":
            return current
        if self.mode == "autonomous":
            return information_gaze(scores, current)
        rows, cols = scores.shape
        iy, ix = int(self._rng.integers(rows)), int(self._rng.integers(cols))
        return float(ix / (cols - 1)), float(iy / (rows - 1))
