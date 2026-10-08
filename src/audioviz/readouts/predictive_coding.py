from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from audioviz.readouts.diagnostics import (
    EnergyStatistics,
    PredictiveCodingDiagnostics,
    SensoryPreview,
    copy_weights,
    mean_energy,
)
from audioviz.utils.spatial_preview import spatial_preview

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

    def __post_init__(self) -> None:
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


class PredictiveCodingRGBReadout:
    """Local PC inference with persistent per-pixel states and shared weights."""

    def __init__(
        self,
        *,
        canvas_shape: tuple[int, ...],
        config: PredictiveCodingConfig | None = None,
    ) -> None:
        if (
            len(canvas_shape) != 3
            or canvas_shape[2] != 3
            or any(size <= 0 for size in canvas_shape)
        ):
            raise ValueError("predictive RGB readout requires a nonempty RGB canvas")
        self.canvas_shape = tuple(canvas_shape)
        self.config = PredictiveCodingConfig() if config is None else config
        self.hidden = np.zeros(
            (*self.canvas_shape[:2], self.config.hidden_channels),
            dtype=np.float64,
        )
        rng = np.random.default_rng(0)
        self.canvas_weights = rng.normal(
            scale=0.1, size=(self.config.hidden_channels, 3)
        )
        self.visual_weights = rng.normal(
            scale=0.1, size=(3, self.config.hidden_channels)
        )
        for channel in range(min(3, self.config.hidden_channels)):
            self.canvas_weights[channel, channel] = 1.0
            self.visual_weights[channel, channel] = 1.0
        self.hidden_error = np.zeros_like(self.hidden)
        self.visual_error = np.zeros(self.canvas_shape, dtype=np.float64)
        self.diagnostics_enabled = False
        self.diagnostics: PredictiveCodingDiagnostics | None = None
        self.spatial_preview_enabled = False
        self._preview_prediction: np.ndarray | None = None

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self.diagnostics_enabled = bool(enabled)
        self.diagnostics = None
        if not enabled:
            self.set_spatial_preview_enabled(False)

    def set_spatial_preview_enabled(self, enabled: bool) -> None:
        self.spatial_preview_enabled = bool(enabled) and self.diagnostics_enabled
        if not self.spatial_preview_enabled:
            self._preview_prediction = None
            if self.diagnostics is not None and self.diagnostics.spatial is not None:
                self.diagnostics = replace(self.diagnostics, spatial=None)

    def update_config(self, **changes) -> None:
        next_config = replace(self.config, **changes)
        if next_config.hidden_channels != self.config.hidden_channels:
            raise ValueError("hidden_channels cannot change without rebuilding the readout")
        self.config = next_config

    def reset(self) -> None:
        """Clear belief states and errors, retaining learned weights."""
        self.hidden.fill(0.0)
        self.hidden_error.fill(0.0)
        self.visual_error.fill(0.0)
        self.diagnostics = None
        self._preview_prediction = None

    def predict(self, prior: np.ndarray) -> np.ndarray:
        self.diagnostics = None
        self._preview_prediction = None
        canvas = self._coerce(prior, name="canvas prior")
        predicted_hidden = np.tanh(canvas) @ self.canvas_weights.T
        # Advance memory from the wave prior before any new camera evidence.
        self.hidden += self.config.inference_rate * (predicted_hidden - self.hidden)
        prediction = np.tanh(self.hidden) @ self.visual_weights.T
        self._require_finite(self.hidden, prediction)
        if self.spatial_preview_enabled:
            self._preview_prediction = spatial_preview(prediction)
        return prediction.astype(np.float32)

    def infer(self, prior: np.ndarray, observation: np.ndarray) -> np.ndarray:
        canvas = self._coerce(prior, name="canvas prior").copy()
        observed = self._coerce(observation, name="camera observation")
        inference_energy: list[float] = []
        for _ in range(self.config.inference_steps):
            canvas_activity = np.tanh(canvas)
            hidden_activity = np.tanh(self.hidden)
            hidden_error = self.hidden - canvas_activity @ self.canvas_weights.T
            visual_error = observed - hidden_activity @ self.visual_weights.T
            if self.diagnostics_enabled:
                inference_energy.append(mean_energy(hidden_error, visual_error))
            canvas_direction = (1.0 - canvas_activity**2) * (
                hidden_error @ self.canvas_weights
            )
            hidden_direction = -hidden_error + (1.0 - hidden_activity**2) * (
                visual_error @ self.visual_weights
            )
            # Both state directions use the same pre-update errors and weights.
            canvas += self.config.canvas_rate * canvas_direction
            self.hidden += self.config.inference_rate * hidden_direction
            self._require_finite(canvas, self.hidden)

        self.hidden_error, self.visual_error = self._errors(canvas, observed)
        weights_before = None
        if self.diagnostics_enabled:
            inference_energy.append(mean_energy(self.hidden_error, self.visual_error))
            weights_before = self.canvas_weights.copy(), self.visual_weights.copy()
        if self.config.learning_enabled and self.config.learning_rate:
            self._learn(canvas)
        if weights_before is not None:
            self.diagnostics = PredictiveCodingDiagnostics(
                hidden=EnergyStatistics.from_error(self.hidden_error),
                visual=EnergyStatistics.from_error(self.visual_error),
                inference_energy=tuple(inference_energy),
                canvas_means=tuple(float(value) for value in canvas.mean(axis=(0, 1))),
                hidden_means=tuple(float(value) for value in self.hidden.mean(axis=(0, 1))),
                observation_means=tuple(float(value) for value in observed.mean(axis=(0, 1))),
                canvas_weights=copy_weights(self.canvas_weights),
                visual_weights=copy_weights(self.visual_weights),
                learning_enabled=self.config.learning_enabled,
                learning_rate=self.config.learning_rate,
                weight_update_norm=float(
                    np.sqrt(
                        np.sum((self.canvas_weights - weights_before[0]) ** 2)
                        + np.sum((self.visual_weights - weights_before[1]) ** 2)
                    )
                ),
                spatial=(
                    SensoryPreview(
                        hidden=spatial_preview(self.hidden),
                        prediction=self._preview_prediction,
                        observation=spatial_preview(observed),
                    )
                    if self.spatial_preview_enabled and self._preview_prediction is not None
                    else None
                ),
            )
        self._preview_prediction = None
        correction = canvas - self._coerce(prior, name="canvas prior")
        self._require_finite(correction)
        return correction.astype(np.float32)

    def energy(self, canvas: np.ndarray, observation: np.ndarray) -> float:
        hidden_error, visual_error = self._errors(
            self._coerce(canvas, name="canvas state"),
            self._coerce(observation, name="camera observation"),
        )
        return mean_energy(hidden_error, visual_error)

    def _errors(
        self, canvas: np.ndarray, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.hidden - np.tanh(canvas) @ self.canvas_weights.T,
            observation - np.tanh(self.hidden) @ self.visual_weights.T,
        )

    def _learn(self, canvas: np.ndarray) -> None:
        pixel_count = self.canvas_shape[0] * self.canvas_shape[1]
        canvas_gradient = (
            self.hidden_error.reshape(-1, self.config.hidden_channels).T
            @ np.tanh(canvas).reshape(-1, 3)
        ) / pixel_count
        visual_gradient = (
            self.visual_error.reshape(-1, 3).T
            @ np.tanh(self.hidden).reshape(-1, self.config.hidden_channels)
        ) / pixel_count
        for weights, gradient in (
            (self.canvas_weights, canvas_gradient),
            (self.visual_weights, visual_gradient),
        ):
            gradient = gradient - self.config.weight_decay * weights
            weights += self.config.learning_rate * np.clip(
                gradient, -self.config.gradient_clip, self.config.gradient_clip
            )
            np.clip(
                weights, -self.config.weight_clip, self.config.weight_clip, out=weights
            )
            self._require_finite(weights)

    def _coerce(self, values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.canvas_shape:
            raise ValueError(f"{name} must match the readout canvas shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _require_finite(*values: np.ndarray) -> None:
        if any(not np.all(np.isfinite(value)) for value in values):
            raise FloatingPointError(
                "Predictive coding diverged; reduce inference or learning rates."
            )
