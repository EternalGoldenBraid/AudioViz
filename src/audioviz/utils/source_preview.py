from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from audioviz.utils.spatial_preview import PREVIEW_MAX_SIDE, spatial_preview


@dataclass(frozen=True)
class SourceSceneMapping:
    key: str
    label: str
    representation: Literal["rgb", "spectrum", "signed", "graph"]
    role: Literal["sensory", "drive", "medium"]

    def sample(self, values: np.ndarray) -> np.ndarray:
        if self.representation in ("rgb", "signed"):
            return spatial_preview(values)
        if self.representation == "spectrum":
            if values.ndim != 1 or not 0 < len(values) <= PREVIEW_MAX_SIDE:
                raise ValueError("A spectral scene source must provide 1-64 bands.")
            sampled = np.array(values, dtype=np.float32, copy=True)
        elif self.representation == "graph":
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("A graph scene source must provide 2D node positions.")
            sampled = np.array(values[:PREVIEW_MAX_SIDE], dtype=np.float32, copy=True)
        else:
            raise ValueError(f"Unsupported scene representation: {self.representation}")
        sampled.setflags(write=False)
        return sampled


@dataclass(frozen=True)
class SourceScenePreview:
    mapping: SourceSceneMapping
    enabled: bool
    observation: np.ndarray | None = None
    prediction: np.ndarray | None = None
    hidden: np.ndarray | None = None
    hidden_energies: tuple[float, ...] = ()
    edges: np.ndarray | None = None
    recurrent_parent: str | None = None
    stream_state: bool | None = None
    stream_state_prediction: float | None = None
