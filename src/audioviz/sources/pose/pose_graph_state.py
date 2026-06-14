from __future__ import annotations

import numpy as np


class PoseGraphState:
    def __init__(
        self,
        num_nodes: int,
        adjacency: np.ndarray | None = None,
        *,
        position_smoothing_alpha: float = 1.0,
        velocity_smoothing_alpha: float = 1.0,
    ) -> None:
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        if not 0.0 <= position_smoothing_alpha <= 1.0:
            raise ValueError("position_smoothing_alpha must be between 0 and 1")
        if not 0.0 <= velocity_smoothing_alpha <= 1.0:
            raise ValueError("velocity_smoothing_alpha must be between 0 and 1")

        self.num_nodes = num_nodes
        self.adjacency = (
            np.zeros((num_nodes, num_nodes), dtype=np.float32)
            if adjacency is None
            else np.asarray(adjacency, dtype=np.float32)
        )
        if self.adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adjacency shape must match (num_nodes, num_nodes)")

        self.position_smoothing_alpha = position_smoothing_alpha
        self.velocity_smoothing_alpha = velocity_smoothing_alpha
        self.state = np.zeros((num_nodes, 7), dtype=np.float32)
        self.prev_positions = np.zeros((num_nodes, 2), dtype=np.float32)
        self.prev_velocities = np.zeros((num_nodes, 2), dtype=np.float32)
        self.smooth_positions = np.zeros((num_nodes, 2), dtype=np.float32)
        self.smooth_velocities = np.zeros((num_nodes, 2), dtype=np.float32)
        self._has_observation = False

    def __call__(self) -> np.ndarray:
        return np.linalg.norm(self.get_velocities(), axis=1)

    def update(self, new_positions: np.ndarray, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")

        positions = np.asarray(new_positions, dtype=np.float32)
        if positions.shape != (self.num_nodes, 2):
            raise ValueError("new_positions shape must match (num_nodes, 2)")

        if not self._has_observation:
            self.smooth_positions = positions.copy()
            self._has_observation = True
        else:
            alpha = self.position_smoothing_alpha
            self.smooth_positions = (
                alpha * positions + (1.0 - alpha) * self.smooth_positions
            )

        velocities = (self.smooth_positions - self.prev_positions) / dt
        alpha = self.velocity_smoothing_alpha
        self.smooth_velocities = (
            alpha * velocities + (1.0 - alpha) * self.smooth_velocities
        )
        accelerations = (self.smooth_velocities - self.prev_velocities) / dt

        self.state[:, 0:2] = self.smooth_positions
        self.state[:, 2:4] = self.smooth_velocities
        self.state[:, 4:6] = accelerations

        self.prev_positions = self.smooth_positions.copy()
        self.prev_velocities = self.smooth_velocities.copy()

    def set_ripple_states(self, values: np.ndarray) -> None:
        ripple = np.asarray(values, dtype=np.float32)
        if ripple.shape != (self.num_nodes,):
            raise ValueError("values shape must match (num_nodes,)")
        self.state[:, 6] = ripple

    def get_state_array(self) -> np.ndarray:
        return self.state.copy()

    def get_positions(self) -> np.ndarray:
        return self.state[:, 0:2].copy()

    def get_velocities(self) -> np.ndarray:
        return self.state[:, 2:4].copy()

    def get_accelerations(self) -> np.ndarray:
        return self.state[:, 4:6].copy()

    def get_ripple_states(self) -> np.ndarray:
        return self.state[:, 6].copy()
