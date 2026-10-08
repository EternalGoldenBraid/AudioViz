from __future__ import annotations

import numpy as np


AUDIO_SPECTRAL_BANDS = 16


def spectral_observation(
    spectra: list[np.ndarray], *, window_sum: float, mel_power: bool, gain: float
) -> np.ndarray:
    """Fixed-scale magnitude bands; channel averaging cannot cancel opposing phases."""
    if not spectra or not np.isfinite(window_sum) or window_sum <= 0:
        raise ValueError("Audio evidence requires spectra and a positive window sum.")
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("Audio evidence gain must be finite and non-negative.")
    latest = []
    for spectrum in spectra:
        values = np.asarray(spectrum)
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError("Audio spectrograms must have nonempty frequency and time axes.")
        frame = values[:, -1]
        if not np.all(np.isfinite(frame)) or np.any(frame < 0):
            raise ValueError("Audio spectral magnitudes must be finite and non-negative.")
        latest.append(np.sqrt(frame) if mel_power else frame)
    magnitudes = np.stack(latest).mean(axis=0) * (2.0 * gain / window_sum)
    if magnitudes.size >= AUDIO_SPECTRAL_BANDS:
        bands = np.array([group.mean() for group in np.array_split(magnitudes, AUDIO_SPECTRAL_BANDS)])
    else:
        bands = np.interp(
            np.linspace(0, magnitudes.size - 1, AUDIO_SPECTRAL_BANDS),
            np.arange(magnitudes.size), magnitudes,
        )
    if not np.all(np.isfinite(bands)):
        raise FloatingPointError("Audio spectral normalization produced nonfinite evidence.")
    return bands.astype(np.float32)
