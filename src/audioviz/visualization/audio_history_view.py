"""Numerical display conventions shared by Qt plots and terminal thumbnails."""

import numpy as np
from matplotlib import colormaps

from audioviz.audio_processing.history import AudioHistory, history_grid

COCHLEA_DB_RANGE = (-80.0, 0.0)
COCHLEA_COLORMAP = "magma"
SPECTRUM_DB_RANGE = (-80.0, 0.0)
SPECTRUM_COLORMAP = "viridis"


def waveform_envelope(history: AudioHistory, columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak-preserving display bins on the same clock as both heatmaps."""
    size = len(history.waveform)
    count = min(columns, size)
    if count < 1:
        raise ValueError("waveform envelope requires samples and positive columns")
    starts = np.arange(count) * size // count
    ends = np.r_[starts[1:], size]
    first = max(0, size - history.sample_end)
    valid = ends > first
    low, high = np.full(count, np.nan), np.full(count, np.nan)
    if valid.any():
        indices = np.maximum(starts[valid], first)
        low[valid] = np.minimum.reduceat(history.waveform, indices)
        high[valid] = np.maximum.reduceat(history.waveform, indices)
    times = ((starts + ends) / 2 - size) / history.sample_rate
    return times, low, high


def spectrum_db(history: AudioHistory, columns: int) -> np.ndarray:
    values = history_grid(
        history.spectrum, sample_end=history.sample_end,
        sample_span=len(history.waveform), columns=columns,
    )
    factor = 10 if history.mel_power else 20
    return factor * np.log10(np.maximum(values, 1e-10))


def cochlea_db(history: AudioHistory, columns: int) -> np.ndarray | None:
    if history.cochlea is None:
        return None
    amplitude = history_grid(
        history.cochlea, sample_end=history.sample_end,
        sample_span=len(history.waveform), columns=columns,
    )
    return 20 * np.log10(np.maximum(amplitude, 1e-8))


def level_rgb(values: np.ndarray, limits: tuple[float, float], colormap: str) -> np.ndarray:
    low, high = limits
    return colormaps[colormap]((values - low) / (high - low), bytes=True)[..., :3]
