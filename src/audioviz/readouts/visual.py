from __future__ import annotations

from typing import Protocol

import numpy as np


class VisualReadout(Protocol):
    def predict(self, prior: np.ndarray) -> np.ndarray:
        """Map a canvas prior to an RGB prediction."""

    def backproject(self, visual_correction: np.ndarray) -> np.ndarray:
        """Map a visual-space correction back into the canvas."""


class IdentityRGBReadout:
    def __init__(self, *, canvas_shape: tuple[int, ...]) -> None:
        if len(canvas_shape) != 3 or canvas_shape[2] != 3:
            raise ValueError("identity RGB readout requires a three-channel canvas")
        self.canvas_shape = tuple(canvas_shape)

    def predict(self, prior: np.ndarray) -> np.ndarray:
        return self._coerce(prior, name="canvas prior").copy()

    def backproject(self, visual_correction: np.ndarray) -> np.ndarray:
        return self._coerce(
            visual_correction,
            name="visual correction",
        ).copy()

    def _coerce(self, values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != self.canvas_shape:
            raise ValueError(f"{name} must match the readout canvas shape")
        return array
