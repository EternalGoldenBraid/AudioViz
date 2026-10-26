"""Configuration shared by sensory inference models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audioviz.readouts.spatial_precision import GaussianSpatialPrecision

MAX_FOVEAL_PIXELS = 4096


@dataclass(frozen=True)
class PredictiveCodingConfig:
    hidden_channels: int = 6
    inference_steps: int = 8
    inference_rate: float = 0.1
    canvas_rate: float = 0.01
    learning_enabled: bool = False
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    weight_clip: float = 10.0
    gradient_clip: float = 1.0
    cross_modal_enabled: bool = False
    dtype: str = "float64"
    spatial_precision: GaussianSpatialPrecision | None = None

    def __post_init__(self) -> None:
        if self.spatial_precision is not None and not isinstance(self.spatial_precision, GaussianSpatialPrecision):
            raise ValueError("spatial_precision must be a GaussianSpatialPrecision or None")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.cross_modal_enabled, bool):
            raise ValueError("cross_modal_enabled must be a boolean")
        for name in ("hidden_channels", "inference_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("inference_rate", "canvas_rate", "learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("weight_clip", "gradient_clip"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
