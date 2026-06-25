from __future__ import annotations

import numpy as np

from audioviz.source_controls import SourceControl


class SyntheticRippleSource:
    def __init__(
        self,
        *,
        frequency: float | list[float] | tuple[float, ...] | np.ndarray,
        n_sources: int,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.frequencies_matrix = self.coerce_frequencies(
            frequency,
            n_sources=n_sources,
        )

    @property
    def frequency(self) -> float:
        return float(self.frequencies_matrix[0, 0])

    def frequencies(self) -> np.ndarray | None:
        if not self.enabled:
            return None
        return self.frequencies_matrix.copy()

    def set_frequency(self, index: int, frequency_hz: float) -> None:
        self.frequencies_matrix[int(index), 0] = float(frequency_hz)

    def controls_for_index(self, index: int) -> tuple[SourceControl, ...]:
        return (
            SourceControl(
                key="frequency_hz",
                label="Frequency",
                default=float(self.frequencies_matrix[int(index), 0]),
                minimum=0.1,
                maximum=20_000.0,
                step=0.1,
                unit="Hz",
            ),
        )

    @staticmethod
    def coerce_frequencies(
        frequency: float | list[float] | tuple[float, ...] | np.ndarray,
        *,
        n_sources: int,
    ) -> np.ndarray:
        values = np.asarray(frequency, dtype=np.float32)
        if values.ndim == 0:
            return np.full((n_sources, 1), float(values), dtype=np.float32)
        if values.ndim == 1 and values.shape[0] == n_sources:
            return values.reshape(n_sources, 1).astype(np.float32, copy=False)
        if values.ndim == 2 and values.shape == (n_sources, 1):
            return values.astype(np.float32, copy=False)
        raise ValueError(
            "frequency must be a scalar, a length-n_sources vector, "
            "or an (n_sources, 1) matrix"
        )
