from __future__ import annotations

import numpy as np


class PoseMediumController:
    def __init__(
        self,
        *,
        engine,
        graph_stiffness: float = 0.25,
    ) -> None:
        self.engine = engine
        self.graph_stiffness = float(graph_stiffness)
        self.adjacency: np.ndarray | None = None
        self.degree: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.values_old: np.ndarray | None = None
        self.positions: np.ndarray | None = None
        self.valid: np.ndarray | None = None

    def configure(self, adjacency: np.ndarray) -> None:
        if self.engine.use_gpu or self.engine.use_shader:
            raise NotImplementedError(
                "Pose-medium coupling is currently implemented for the CPU backend only."
            )
        adjacency_array = np.asarray(adjacency, dtype=np.float32)
        if adjacency_array.ndim != 2 or adjacency_array.shape[0] != adjacency_array.shape[1]:
            raise ValueError("pose adjacency must be a square matrix")
        num_nodes = adjacency_array.shape[0]
        if (
            self.adjacency is not None
            and self.adjacency.shape == adjacency_array.shape
            and np.array_equal(self.adjacency, adjacency_array)
        ):
            return
        self.adjacency = adjacency_array
        self.degree = adjacency_array.sum(axis=1).astype(np.float32)
        self.values = np.zeros(num_nodes, dtype=np.float32)
        self.values_old = np.zeros(num_nodes, dtype=np.float32)
        self.positions = np.zeros((num_nodes, 2), dtype=np.float32)
        self.valid = np.zeros(num_nodes, dtype=bool)

    def update(
        self,
        *,
        positions: np.ndarray,
        valid: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> None:
        if adjacency is not None or self.adjacency is None:
            self.configure(
                adjacency
                if adjacency is not None
                else np.zeros((len(positions), len(positions)), dtype=np.float32)
            )
        assert self.positions is not None
        assert self.valid is not None
        positions_array = np.asarray(positions, dtype=np.float32)
        valid_array = np.asarray(valid, dtype=bool)
        num_nodes = len(valid_array)
        if positions_array.shape != (num_nodes, 2):
            raise ValueError("positions must have shape (num_nodes, 2)")
        if self.positions.shape != (num_nodes, 2):
            raise ValueError("pose medium size does not match current pose graph")
        self.positions[:] = positions_array
        self.valid[:] = valid_array

    def step(self, *, grid_excitation: np.ndarray | None = None) -> np.ndarray:
        if self.values is None or self.values_old is None:
            raise RuntimeError("Pose medium has not been configured.")
        assert self.adjacency is not None
        assert self.degree is not None
        self.engine.time += self.engine.dt
        grid = self.engine.Z
        pose = self.values
        driven_grid = grid.copy()
        driven_pose = pose.copy()
        if grid_excitation is not None:
            driven_grid += self.engine._coerce_grid_excitation(grid_excitation) * np.float32(
                self.engine.amplitude
            )
        grid_laplacian = self.engine._grid_laplacian_with_internal_boundaries(driven_grid)
        pose_laplacian = self.adjacency @ driven_pose - self.degree * driven_pose
        pose_laplacian *= self.graph_stiffness
        new_grid = (
            2 * driven_grid
            - self.engine.Z_old
            + self.engine.propagator.c2_dt2 * grid_laplacian
        )
        new_pose = (
            2 * driven_pose
            - self.values_old
            + self.engine.propagator.c2_dt2 * pose_laplacian
        )
        new_grid *= self.engine.damping
        new_pose *= self.engine.damping
        if self.engine.body_boundary_mask is not None:
            new_grid[self.engine.body_boundary_mask] *= np.float32(
                1.0 - self.engine.body_boundary_dissipation
            )
        self.engine.Z_old = grid.copy()
        self.engine.Z = new_grid.astype(np.float32, copy=False)
        self.values_old = pose.copy()
        self.values = new_pose.astype(np.float32, copy=False)
        return self.engine.Z

    def get_state(self) -> np.ndarray:
        if self.values is None:
            raise RuntimeError("Pose medium has not been configured.")
        return self.values.copy()

    def get_positions(self, *, valid_only: bool = False) -> np.ndarray:
        if self.positions is None:
            raise RuntimeError("Pose medium has not been configured.")
        if not valid_only or self.valid is None:
            return self.positions.copy()
        return self.positions[self.valid].copy()

    def reset(self) -> None:
        if self.values is not None:
            self.values[:] = 0
        if self.values_old is not None:
            self.values_old[:] = 0
        if self.valid is not None:
            self.valid[:] = False

