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
