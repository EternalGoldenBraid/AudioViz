from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PoseGraphFrame:
    """Pose graph frame with reusable backing arrays and active views."""

    def __init__(
        self,
        coords: np.ndarray | None = None,
        adjacency: np.ndarray | None = None,
        segmentation_mask: np.ndarray | None = None,
        *,
        max_nodes: int | None = None,
        array_module=np,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        if coords is None and max_nodes is None:
            raise ValueError("coords or max_nodes is required")

        self._xp = array_module
        self._dtype = dtype
        active_nodes = 0 if coords is None else int(coords.shape[0])
        max_nodes = active_nodes if max_nodes is None else int(max_nodes)
        if max_nodes < active_nodes:
            raise ValueError("max_nodes must be at least the number of coords")
        if max_nodes < 0:
            raise ValueError("max_nodes must be non-negative")

        self._coords = self._xp.zeros((max_nodes, 2), dtype=dtype)
        self._adjacency = self._xp.zeros((max_nodes, max_nodes), dtype=dtype)
        self._num_nodes = 0
        self._segmentation_mask: np.ndarray | None = None

        if coords is not None:
            self.update(coords, adjacency, segmentation_mask=segmentation_mask)
        elif adjacency is not None:
            adjacency_array = self._xp.asarray(adjacency, dtype=dtype)
            if adjacency_array.shape != (max_nodes, max_nodes):
                raise ValueError("adjacency shape must match max_nodes")
            self._adjacency[...] = adjacency_array
            self.set_segmentation_mask(segmentation_mask)
        elif segmentation_mask is not None:
            self.set_segmentation_mask(segmentation_mask)

    @classmethod
    def empty(
        cls,
        max_nodes: int,
        *,
        adjacency: np.ndarray | None = None,
        array_module=np,
        dtype: np.dtype | type = np.float32,
    ) -> "PoseGraphFrame":
        return cls(
            max_nodes=max_nodes,
            adjacency=adjacency,
            array_module=array_module,
            dtype=dtype,
        )

    @property
    def coords(self) -> np.ndarray:
        return self._coords[: self._num_nodes]

    @property
    def adjacency(self) -> np.ndarray:
        return self._adjacency[: self._num_nodes, : self._num_nodes]

    @property
    def max_nodes(self) -> int:
        return int(self._coords.shape[0])

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def coords_buffer(self) -> np.ndarray:
        return self._coords

    @property
    def adjacency_buffer(self) -> np.ndarray:
        return self._adjacency

    @property
    def segmentation_mask(self) -> np.ndarray | None:
        return self._segmentation_mask

    def clear(self) -> None:
        self._num_nodes = 0
        self._segmentation_mask = None

    def update(
        self,
        coords: np.ndarray,
        adjacency: np.ndarray | None = None,
        *,
        segmentation_mask: np.ndarray | None = None,
    ) -> None:
        coords_array = self._xp.asarray(coords, dtype=self._dtype)
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError("coords shape must be (num_nodes, 2)")
        num_nodes = int(coords_array.shape[0])
        if num_nodes > self.max_nodes:
            raise ValueError("coords exceed max_nodes")

        self._coords[:num_nodes] = coords_array
        self._num_nodes = num_nodes

        if adjacency is not None:
            adjacency_array = self._xp.asarray(adjacency, dtype=self._dtype)
            if adjacency_array.shape != (num_nodes, num_nodes):
                raise ValueError("adjacency shape must match coords")
            self._adjacency[:num_nodes, :num_nodes] = adjacency_array
        self.set_segmentation_mask(segmentation_mask)

    def update_xy(self, points: object) -> None:
        num_nodes = len(points)  # type: ignore[arg-type]
        if num_nodes > self.max_nodes:
            raise ValueError("points exceed max_nodes")
        for index, point in enumerate(points):  # type: ignore[operator]
            self._coords[index, 0] = point.x
            self._coords[index, 1] = point.y
        self._num_nodes = num_nodes

    def set_segmentation_mask(self, segmentation_mask: np.ndarray | None) -> None:
        if segmentation_mask is None:
            self._segmentation_mask = None
            return
        mask_array = self._xp.asarray(segmentation_mask, dtype=self._dtype)
        if mask_array.ndim != 2:
            raise ValueError("segmentation_mask must have shape (rows, cols)")
        self._segmentation_mask = np.array(mask_array, copy=True)

    def as_dict(self) -> dict[str, np.ndarray]:
        result = {"coords": self.coords, "adjacency": self.adjacency}
        if self._segmentation_mask is not None:
            result["segmentation_mask"] = self._segmentation_mask
        return result


class PoseGraphExtractor(ABC):
    @abstractmethod
    def extract(self, frame: np.ndarray) -> PoseGraphFrame:
        """Extract normalized 2D node coordinates and graph adjacency from a frame."""

    def close(self) -> None:
        """Release extractor resources."""
