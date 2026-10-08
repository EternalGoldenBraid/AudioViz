from __future__ import annotations

from typing import Protocol

import numpy as np

from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.readouts.diagnostics import PredictiveCodingDiagnostics


class VisualReadout(Protocol):
    config: PredictiveCodingConfig
    diagnostics: PredictiveCodingDiagnostics | None
    hidden: np.ndarray
    stream_state_prediction: float | None

    def predict(self, prior: np.ndarray) -> np.ndarray:
        """Advance hidden belief from a canvas prior and predict RGB evidence."""

    def infer(
        self, prior: np.ndarray, observation: np.ndarray | None,
        *, stream_state: bool | None = None,
    ) -> np.ndarray:
        """Infer from available sensory/context evidence; return the canvas correction."""

    def update_config(self, **changes) -> None:
        """Update inference and learning controls."""

    def reset(self) -> None:
        """Clear hidden belief while retaining learned weights."""

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        """Control optional energy and learning telemetry."""

    def set_spatial_preview_enabled(self, enabled: bool) -> None:
        """Capture bounded sensory previews for the next prediction/inference frame."""


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
