from __future__ import annotations

from types import ModuleType

import numpy as np

from audioviz.readouts.diagnostics import copy_weights
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS
from audioviz.utils.array_backend import spatial_mean


class PredictiveCodingAudioReadout(PredictiveCodingRGBReadout):
    """Canvas -> audio hidden field -> spatially pooled spectral prediction."""

    def __init__(
        self, *, canvas_shape: tuple[int, ...], config: PredictiveCodingConfig | None = None,
        backend: ModuleType = np,
    ) -> None:
        super().__init__(canvas_shape=canvas_shape, config=config, backend=backend)
        xp = self.backend
        rng = np.random.default_rng(1)
        dtype = self.hidden.dtype
        self.canvas_weights = xp.asarray(rng.normal(scale=0.1, size=self.canvas_weights.shape), dtype=dtype)
        self.visual_weights = xp.asarray(rng.normal(
            scale=0.1, size=(AUDIO_SPECTRAL_BANDS, self.config.hidden_channels)
        ), dtype=dtype)
        self.visual_error = xp.zeros(AUDIO_SPECTRAL_BANDS, dtype=dtype)
        self.stream_state_weights = xp.asarray(rng.normal(scale=.1, size=self.stream_state_weights.shape), dtype=dtype)

    def _predict_observation(self, hidden_activity: np.ndarray) -> np.ndarray:
        return spatial_mean(hidden_activity) @ self.visual_weights.T

    def _observation_gradient(self) -> np.ndarray:
        xp = self.backend
        return xp.outer(self.visual_error, spatial_mean(xp.tanh(self.hidden)))

    def _coerce_observation(self, observation: np.ndarray) -> np.ndarray:
        xp = self.backend
        values = xp.asarray(observation, dtype=self.hidden.dtype)
        if values.shape != (AUDIO_SPECTRAL_BANDS,):
            raise ValueError("Audio observation must match the spectral band count.")
        if not bool(xp.all(xp.isfinite(values))):
            raise ValueError("Audio observation must contain only finite values.")
        return values

    def _preview_observation(self, observation: np.ndarray) -> np.ndarray:
        return copy_weights(observation.astype(self.backend.float32))
