from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

PoseEdge = tuple[int, int]

MEDIAPIPE_POSE_CONNECTIONS: tuple[PoseEdge, ...] = (
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
    (23, 24),
    (11, 23),
    (12, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (15, 21),
    (16, 22),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
)


def adjacency_from_edges(
    num_nodes: int,
    edges: Iterable[PoseEdge],
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=dtype)
    for i, j in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    return adjacency


def mediapipe_pose_adjacency(num_nodes: int = 33) -> np.ndarray:
    return adjacency_from_edges(num_nodes, MEDIAPIPE_POSE_CONNECTIONS)


def iter_adjacency_edges(adjacency: np.ndarray) -> Iterator[PoseEdge]:
    rows, cols = np.nonzero(np.triu(adjacency, k=1))
    return ((int(i), int(j)) for i, j in zip(rows, cols))

