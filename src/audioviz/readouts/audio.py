from __future__ import annotations

import numpy as np

from audioviz.readouts.diagnostics import copy_weights
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS


class PredictiveCodingAudioReadout(PredictiveCodingRGBReadout):
    """Canvas -> audio hidden field -> spatially pooled spectral prediction."""

    def __init__(
        self, *, canvas_shape: tuple[int, ...], config: PredictiveCodingConfig | None = None
    ) -> None:
        super().__init__(canvas_shape=canvas_shape, config=config)
        rng = np.random.default_rng(1)
        self.canvas_weights = rng.normal(scale=0.1, size=self.canvas_weights.shape)
        self.visual_weights = rng.normal(
            scale=0.1, size=(AUDIO_SPECTRAL_BANDS, self.config.hidden_channels)
        )
        self.visual_error = np.zeros(AUDIO_SPECTRAL_BANDS, dtype=np.float64)

    def _predict_observation(self, hidden_activity: np.ndarray) -> np.ndarray:
        return hidden_activity.mean(axis=(0, 1)) @ self.visual_weights.T

    def _observation_gradient(self) -> np.ndarray:
        return np.outer(self.visual_error, np.tanh(self.hidden).mean(axis=(0, 1)))

    def _coerce_observation(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (AUDIO_SPECTRAL_BANDS,):
            raise ValueError("Audio observation must match the spectral band count.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Audio observation must contain only finite values.")
        return values

    def _preview_observation(self, observation: np.ndarray) -> np.ndarray:
        return copy_weights(np.asarray(observation, dtype=np.float32))
