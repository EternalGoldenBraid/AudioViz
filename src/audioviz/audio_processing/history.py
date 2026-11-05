"""Sample-clock snapshots shared by full-resolution and thumbnail renderers."""

from dataclasses import dataclass

import numpy as np


def readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class TimedHistory:
    values: np.ndarray
    samples: np.ndarray
    window_samples: int = 0


@dataclass(frozen=True)
class AudioHistory:
    sample_rate: float
    sample_end: int
    waveform: np.ndarray
    spectrum: TimedHistory
    frequencies: np.ndarray
    mel_power: bool
    cochlea: TimedHistory | None

    @property
    def duration(self) -> float:
        return len(self.waveform) / self.sample_rate


def history_grid(history: TimedHistory, *, sample_end: int, sample_span: int, columns: int) -> np.ndarray:
    """Sample-and-hold on a common time grid; unseen history remains NaN."""
    if columns < 1 or sample_span < 1:
        raise ValueError("history grid requires positive span and column count")
    result = np.full((history.values.shape[0], columns), np.nan)
    if not history.samples.size:
        return result
    positions = sample_end - sample_span + (np.arange(columns) + 0.5) * sample_span / columns
    indices = np.searchsorted(history.samples - history.window_samples, positions, side="right") - 1
    valid = indices >= 0
    if history.window_samples:
        valid &= positions < history.samples[-1]
    result[:, valid] = history.values[:, indices[valid]]
    return result
