from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseGraphFrame:
    coords: np.ndarray
    adjacency: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"coords": self.coords, "adjacency": self.adjacency}


class PoseGraphExtractor(ABC):
    @abstractmethod
    def extract(self, frame: np.ndarray) -> PoseGraphFrame:
        """Extract normalized 2D node coordinates and graph adjacency from a frame."""

    def close(self) -> None:
        """Release extractor resources."""

