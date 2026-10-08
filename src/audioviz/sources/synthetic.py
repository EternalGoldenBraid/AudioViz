from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audioviz.source_controls import SourceControl
from audioviz.sources.ripple_grid import frequency_excitation_grid
from audioviz.utils.source_preview import SourceSceneMapping


@dataclass(frozen=True)
class SyntheticSourceConfig:
    enabled: bool = True
    frequency: float | list[float] | tuple[float, ...] = 440.0


class SyntheticRippleSource:
    scene_mapping = SourceSceneMapping("synthetic", "Synthetic", "signed", "drive")

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

    def excitation_grid(
        self,
        *,
        t: float,
        source_positions: np.ndarray,
        resolution: tuple[int, int],
        grid_spacing: float,
        speed: float,
        max_frequency: float,
        decay_alpha: float,
    ) -> np.ndarray | None:
        frequencies = self.frequencies()
        if frequencies is None:
            return None
        return frequency_excitation_grid(
            t=t,
            frequencies=frequencies,
            source_positions=source_positions,
            resolution=resolution,
            grid_spacing=grid_spacing,
            speed=speed,
            max_frequency=max_frequency,
            decay_alpha=decay_alpha,
        )

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
