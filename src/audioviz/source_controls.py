from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np


ControlKind = Literal["number", "toggle", "choice", "text"]
ControlValue = bool | float | int | str


@dataclass(frozen=True)
class SourceControl:
    key: str
    label: str
    default: ControlValue
    kind: ControlKind = "number"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str | None = None
    choices: tuple[ControlValue, ...] = ()


class SourceControls(Protocol):
    def get_controls(self) -> Sequence[SourceControl]:
        """Return metadata for source-specific controls."""
        ...


class SourceControlProvider:
    def get_controls(self) -> Sequence[SourceControl]:
        return ()


@dataclass
class SyntheticFrequencySource(SourceControlProvider):
    frequency_hz: float = 1.0
    n_sources: int = 1

    def get_controls(self) -> Sequence[SourceControl]:
        return (
            SourceControl(
                key="frequency_hz",
                label="Frequency",
                default=self.frequency_hz,
                minimum=0.1,
                maximum=20_000.0,
                step=0.1,
                unit="Hz",
            ),
        )

    def frequencies(self) -> np.ndarray:
        return np.full((self.n_sources, 1), self.frequency_hz, dtype=np.float32)


@dataclass
class AudioSourceControls(SourceControlProvider):
    signal_gate_threshold: float = 0.05
    drive_amplitude: float = 1.0
    minimum_peak_magnitude: float = 0.1
    peak_prominence_ratio: float = 5.0
    top_k_count: int = 3
    mapping_mode: str = "legacy"
    mapping_alpha: float = 50.0
    mapping_f0: float = 50.0
    mapping_fc: float = 2000.0
    linear_scale: float = 0.05
    linear_offset: float = 0.0

    def get_controls(self) -> Sequence[SourceControl]:
        return (
            SourceControl(
                key="signal_level",
                label="Signal Level",
                default="0.00",
                kind="text",
            ),
            SourceControl(
                key="gate_open",
                label="Gate",
                default="closed",
                kind="text",
            ),
            SourceControl(
                key="detected_frequencies",
                label="Detected Audio Hz",
                default="—",
                kind="text",
            ),
            SourceControl(
                key="mapped_frequencies",
                label="Mapped Ripple Hz",
                default="—",
                kind="text",
            ),
            SourceControl(
                key="signal_gate_threshold",
                label="Signal Gate Threshold",
                default=self.signal_gate_threshold,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
            ),
            SourceControl(
                key="drive_amplitude",
                label="Audio Drive Amplitude",
                default=self.drive_amplitude,
                minimum=0.0,
                maximum=10.0,
                step=0.05,
            ),
            SourceControl(
                key="minimum_peak_magnitude",
                label="Min Peak Magnitude",
                default=self.minimum_peak_magnitude,
                minimum=0.0,
                maximum=10.0,
                step=0.01,
            ),
            SourceControl(
                key="peak_prominence_ratio",
                label="Peak Prominence Ratio",
                default=self.peak_prominence_ratio,
                minimum=1.0,
                maximum=50.0,
                step=0.1,
            ),
            SourceControl(
                key="top_k_count",
                label="Top-K Peaks",
                default=int(self.top_k_count),
                minimum=1,
                maximum=8,
                step=1,
            ),
            SourceControl(
                key="mapping_mode",
                label="Mapping Mode",
                default=self.mapping_mode,
                kind="choice",
                choices=("legacy", "linear"),
            ),
            SourceControl(
                key="mapping_alpha",
                label="Mapping Alpha",
                default=self.mapping_alpha,
                minimum=0.0,
                maximum=200.0,
                step=1.0,
            ),
            SourceControl(
                key="mapping_f0",
                label="Mapping f0",
                default=self.mapping_f0,
                minimum=1.0,
                maximum=2000.0,
                step=1.0,
                unit="Hz",
            ),
            SourceControl(
                key="mapping_fc",
                label="Mapping fc",
                default=self.mapping_fc,
                minimum=1.0,
                maximum=20000.0,
                step=10.0,
                unit="Hz",
            ),
            SourceControl(
                key="linear_scale",
                label="Linear Scale",
                default=self.linear_scale,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
            ),
            SourceControl(
                key="linear_offset",
                label="Linear Offset",
                default=self.linear_offset,
                minimum=0.0,
                maximum=200.0,
                step=0.1,
            ),
        )
