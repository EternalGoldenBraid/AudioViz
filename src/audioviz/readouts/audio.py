from __future__ import annotations

import numpy as np

from audioviz.readouts.diagnostics import copy_weights
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS


class PredictiveCodingAudioReadout(PredictiveCodingRGBReadout):
    """Canvas -> audio hidden field -> spatially pooled audio-state prediction."""

    def __init__(
        self, *, canvas_shape: tuple[int, ...], config: PredictiveCodingConfig | None = None,
        observation_size: int = AUDIO_SPECTRAL_BANDS,
    ) -> None:
        if isinstance(observation_size, bool) or not isinstance(observation_size, int) or observation_size <= 0:
            raise ValueError("observation_size must be a positive integer")
        self.observation_size = observation_size
        super().__init__(canvas_shape=canvas_shape, config=config)
        rng = np.random.default_rng(1)
        self.canvas_weights = rng.normal(scale=0.1, size=self.canvas_weights.shape)
        self.visual_weights = rng.normal(
            scale=0.1, size=(observation_size, self.config.hidden_channels)
        )
        self.visual_error = np.zeros(observation_size, dtype=np.float64)
        self.stream_state_weights = rng.normal(scale=.1, size=self.stream_state_weights.shape)

    def _predict_observation(self, hidden_activity: np.ndarray) -> np.ndarray:
        return hidden_activity.mean(axis=(0, 1)) @ self.visual_weights.T

    def _observation_gradient(self) -> np.ndarray:
        return np.outer(self.visual_error, np.tanh(self.hidden).mean(axis=(0, 1)))

    def _coerce_observation(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (self.observation_size,):
            raise ValueError("Audio observation must match the configured audio-state count.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Audio observation must contain only finite values.")
        return values

    def _preview_observation(self, observation: np.ndarray) -> np.ndarray:
        return copy_weights(np.asarray(observation, dtype=np.float32))
