from typing import Tuple

import numpy as np

from audioviz.physics.wave_propagator import (
    BoundaryCondition,
    WavePropagatorCPU,
    WavePropagatorGPU,
    _laplacian_neumann,
    _laplacian_periodic,
    coerce_boundary_condition,
    load_cupy,
)
from audioviz.physics.opengl_wave_propagator import WavePropagatorOpenGL
from audioviz.transforms.prediction_error import PredictionErrorTransform


class RippleEngine:
    def __init__(
        self,
        *,
        resolution: Tuple[int, int],
        plane_size_m: Tuple[float, float],
        n_sources: int = 1,
        speed: float = 340.0,
        damping: float = 0.999,
        amplitude: float = 1.0,
        decay_alpha: float = 0.0,
        use_gpu: bool = False,
        use_shader: bool = False,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
        pose_graph_stiffness: float = 0.25,
        body_boundary_transmission: float = 0.0,
        body_boundary_dissipation: float = 1.0,
        use_external_opengl_context: bool = False,
    ):
        if use_gpu and use_shader:
            raise ValueError("Choose either use_gpu=True or use_shader=True, not both.")

        self.resolution = resolution
        self.plane_size_m = plane_size_m
        self.n_sources = n_sources
        self.speed = speed
        self.damping = damping
        self.amplitude = amplitude
        self.decay_alpha = decay_alpha
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.boundary_condition = coerce_boundary_condition(boundary_condition)
        self.pose_graph_stiffness = pose_graph_stiffness
        self.body_boundary_transmission = self._validate_unit_interval(
            body_boundary_transmission,
            name="body_boundary_transmission",
        )
        self.body_boundary_dissipation = self._validate_unit_interval(
            body_boundary_dissipation,
            name="body_boundary_dissipation",
        )
        self.use_external_opengl_context = use_external_opengl_context

        self.backend = load_cupy() if use_gpu else np
        self.dx = self.plane_size_m[0] / self.resolution[0]
        self.dy = self.plane_size_m[1] / self.resolution[1]
        self.grid_spacing = min(self.dx, self.dy)
        self.dt = self._stable_dt()
        self.time = 0.0
        self.max_frequency = self.speed / (2 * self.grid_spacing)

        self.source_positions = self._make_source_positions()
        propagator_kwargs = {
            "shape": self.resolution,
            "dx": self.grid_spacing,
            "dt": self.dt,
            "speed": self.speed,
            "damping": self.damping,
            "boundary_condition": self.boundary_condition,
        }
        if use_shader:
            self.propagator = WavePropagatorOpenGL(
                **propagator_kwargs,
                use_current_context=self.use_external_opengl_context,
            )
        elif use_gpu:
            self.propagator = WavePropagatorGPU(
                **propagator_kwargs,
                cupy_module=self.backend,
            )
        else:
            self.propagator = WavePropagatorCPU(**propagator_kwargs)

        self.Z = self.backend.zeros(self.resolution, dtype=self.backend.float32)
        self.Z_old = self.backend.zeros(self.resolution, dtype=self.backend.float32)
        self.pose_medium_enabled = False
        self.pose_values = None
        self.pose_values_old = None
        self.pose_adjacency = None
        self.pose_degree = None
        self.pose_positions = None
        self.pose_valid = None
        self.body_boundary_mask = None
    def _stable_dt(self) -> float:
        return (self.grid_spacing / self.speed) * 1 / np.sqrt(2)

    @property
    def prediction_horizontal_edge_weights(self) -> np.ndarray:
        return self._edge_weights_to_numpy(self.propagator.horizontal_edge_weights)

    @prediction_horizontal_edge_weights.setter
    def prediction_horizontal_edge_weights(self, weights: np.ndarray) -> None:
        self.propagator.horizontal_edge_weights[...] = self._coerce_edge_weight_array(
            weights,
            expected_shape=self.propagator.horizontal_edge_weights.shape,
        )

    @property
    def prediction_vertical_edge_weights(self) -> np.ndarray:
        return self._edge_weights_to_numpy(self.propagator.vertical_edge_weights)

    @prediction_vertical_edge_weights.setter
    def prediction_vertical_edge_weights(self, weights: np.ndarray) -> None:
        self.propagator.vertical_edge_weights[...] = self._coerce_edge_weight_array(
            weights,
            expected_shape=self.propagator.vertical_edge_weights.shape,
        )

    def _make_source_positions(self):
        rng = np.random.default_rng(42)
        rows, cols = self.resolution
        positions = np.column_stack(
            (
                rng.integers(0, cols, size=self.n_sources),
                rng.integers(0, rows, size=self.n_sources),
            )
        )
        return positions.astype(np.float32)

    def set_source_positions(self, source_positions: np.ndarray) -> None:
        positions = np.asarray(source_positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("source_positions must have shape (n_sources, 2)")
        rows, cols = self.resolution
        if np.any(positions[:, 0] < 0.0) or np.any(positions[:, 0] > cols - 1):
            raise ValueError("source x positions must fit inside the field width")
        if np.any(positions[:, 1] < 0.0) or np.any(positions[:, 1] > rows - 1):
            raise ValueError("source y positions must fit inside the field height")

        self.source_positions = positions.copy()
        self.n_sources = len(positions)

    def set_speed(self, speed: float) -> None:
        self.speed = speed
        self.dt = self._stable_dt()
        self.propagator.dt = self.dt
        self.propagator.c = self.speed
        self.propagator.c2_dt2 = (self.speed * self.dt / self.grid_spacing) ** 2
        self.max_frequency = self.speed / (2 * self.grid_spacing)

    def set_damping(self, damping: float) -> None:
        self.damping = damping
        self.propagator.damping = damping

    def set_body_boundary_transmission(self, transmission: float) -> None:
        self.body_boundary_transmission = self._validate_unit_interval(
            transmission,
            name="body_boundary_transmission",
        )

    def set_body_boundary_dissipation(self, dissipation: float) -> None:
        self.body_boundary_dissipation = self._validate_unit_interval(
            dissipation,
            name="body_boundary_dissipation",
        )

    def reset(self) -> None:
        self.propagator.reset()
        self.Z[:] = 0
        self.Z_old[:] = 0
        if self.pose_values is not None:
            self.pose_values[:] = 0
        if self.pose_values_old is not None:
            self.pose_values_old[:] = 0
        self.time = 0.0

    def set_body_boundary_mask(self, body_boundary_mask: np.ndarray | None) -> None:
        if body_boundary_mask is None:
            self.body_boundary_mask = None
            return
        mask = np.asarray(body_boundary_mask, dtype=bool)
        if mask.shape != self.resolution:
            raise ValueError("body_boundary_mask must match engine resolution")
        self.body_boundary_mask = mask.copy()

    def step_grid_excitation(
        self,
        excitation_grid: np.ndarray,
    ):
        self.time += self.dt
        self._add_grid_excitation(excitation_grid)
        self.propagator.step()
        if self.use_shader:
            return self.Z
        if hasattr(self.propagator, "Z_old"):
            self.Z_old = np.array(self.propagator.Z_old, copy=True)
        self.Z[:] = self.propagator.get_state()
        return self.Z

    def observe(
        self,
        *,
        source_key: str,
        observation: np.ndarray,
        transform: PredictionErrorTransform,
    ) -> np.ndarray:
        observed = self._coerce_prediction_field(
            observation,
            prediction_clip=transform.config.prediction_clip,
        )
        if not transform.applies_to(source_key):
            return observed

        prediction = self.predict_observation(transform=transform)
        error = observed.astype(np.float64, copy=False) - prediction.astype(
            np.float64,
            copy=False,
        )
        self._update_prediction_edge_weights(error=error, transform=transform)
        return transform.shape_error(error)

    def predict_observation(
        self,
        *,
        transform: PredictionErrorTransform,
    ) -> np.ndarray:
        field = self._coerce_prediction_field(
            self.get_field_numpy(),
            prediction_clip=transform.config.prediction_clip,
        )
        if not transform.config.learning_enabled:
            return field
        prediction = field.astype(np.float64) + self._shared_neighbor_sum(field)
        clip = np.float32(transform.config.prediction_clip)
        return np.clip(prediction, -clip, clip).astype(np.float32, copy=False)

    def step_without_excitation(self):
        self.time += self.dt
        self.propagator.step()
        if self.use_shader:
            return self.Z
        if hasattr(self.propagator, "Z_old"):
            self.Z_old = np.array(self.propagator.Z_old, copy=True)
        self.Z[:] = self.propagator.get_state()
        return self.Z

    def get_field_numpy(self) -> np.ndarray:
        if self.use_shader:
            return self.propagator.get_state()
        if self.use_gpu:
            return self.backend.asnumpy(self.Z)
        return self.Z

    def get_opengl_field_texture_id(self) -> int:
        if not self.use_shader:
            raise RuntimeError(
                "OpenGL field textures are only available with use_shader=True."
            )
        return self.propagator.get_current_texture_id()

    def get_opengl_field_shape(self) -> Tuple[int, int]:
        if not self.use_shader:
            raise RuntimeError(
                "OpenGL field textures are only available with use_shader=True."
            )
        return self.propagator.get_texture_shape()

    def configure_pose_medium(self, adjacency: np.ndarray) -> None:
        if self.use_gpu or self.use_shader:
            raise NotImplementedError(
                "Pose-medium coupling is currently implemented for the CPU backend only."
            )

        adjacency_array = np.asarray(adjacency, dtype=np.float32)
        if adjacency_array.ndim != 2 or adjacency_array.shape[0] != adjacency_array.shape[1]:
            raise ValueError("pose adjacency must be a square matrix")

        num_nodes = adjacency_array.shape[0]
        if (
            self.pose_adjacency is not None
            and self.pose_adjacency.shape == adjacency_array.shape
            and np.array_equal(self.pose_adjacency, adjacency_array)
        ):
            return

        self.pose_medium_enabled = True
        self.pose_adjacency = adjacency_array
        self.pose_degree = adjacency_array.sum(axis=1).astype(np.float32)
        self.pose_values = np.zeros(num_nodes, dtype=np.float32)
        self.pose_values_old = np.zeros(num_nodes, dtype=np.float32)
        self.pose_positions = np.zeros((num_nodes, 2), dtype=np.float32)
        self.pose_valid = np.zeros(num_nodes, dtype=bool)

    def update_pose_medium(
        self,
        *,
        positions: np.ndarray,
        valid: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> None:
        if adjacency is not None or self.pose_adjacency is None:
            self.configure_pose_medium(
                adjacency if adjacency is not None else np.zeros((len(positions), len(positions)), dtype=np.float32)
            )

        assert self.pose_positions is not None
        assert self.pose_valid is not None
        positions_array = np.asarray(positions, dtype=np.float32)
        valid_array = np.asarray(valid, dtype=bool)
        num_nodes = len(valid_array)
        if positions_array.shape != (num_nodes, 2):
            raise ValueError("positions must have shape (num_nodes, 2)")
        if self.pose_positions.shape != (num_nodes, 2):
            raise ValueError("pose medium size does not match current pose graph")

        self.pose_positions[:] = positions_array
        self.pose_valid[:] = valid_array

    def step_pose_medium(
        self,
        grid_excitation: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.pose_medium_enabled:
            raise RuntimeError("Pose medium has not been configured.")
        assert self.pose_values is not None
        assert self.pose_values_old is not None
        assert self.pose_adjacency is not None
        assert self.pose_degree is not None
        assert self.pose_positions is not None
        assert self.pose_valid is not None

        self.time += self.dt
        grid = self.Z
        pose = self.pose_values
        driven_grid = grid.copy()
        driven_pose = pose.copy()
        if grid_excitation is not None:
            driven_grid += self._coerce_grid_excitation(grid_excitation) * np.float32(
                self.amplitude
            )

        grid_laplacian = self._grid_laplacian_with_internal_boundaries(driven_grid)
        pose_laplacian = self.pose_adjacency @ driven_pose - self.pose_degree * driven_pose
        pose_laplacian *= self.pose_graph_stiffness
        new_grid = 2 * driven_grid - self.Z_old + self.propagator.c2_dt2 * grid_laplacian
        new_pose = (
            2 * driven_pose
            - self.pose_values_old
            + self.propagator.c2_dt2 * pose_laplacian
        )
        new_grid *= self.damping
        new_pose *= self.damping
        if self.body_boundary_mask is not None:
            new_grid[self.body_boundary_mask] *= np.float32(
                1.0 - self.body_boundary_dissipation
            )

        self.Z_old = grid.copy()
        self.Z = new_grid.astype(np.float32, copy=False)
        self.pose_values_old = pose.copy()
        self.pose_values = new_pose.astype(np.float32, copy=False)
        return self.Z

    def get_pose_medium_state(self) -> np.ndarray:
        if self.pose_values is None:
            raise RuntimeError("Pose medium has not been configured.")
        return self.pose_values.copy()

    def get_pose_medium_positions(self, *, valid_only: bool = False) -> np.ndarray:
        if self.pose_positions is None:
            raise RuntimeError("Pose medium has not been configured.")
        if not valid_only or self.pose_valid is None:
            return self.pose_positions.copy()
        return self.pose_positions[self.pose_valid].copy()

    def _add_grid_excitation(self, excitation_grid: np.ndarray) -> None:
        self.propagator.add_excitation(
            self._coerce_grid_excitation(excitation_grid) * self.amplitude
        )

    def _coerce_grid_excitation(self, excitation_grid: np.ndarray):
        xp = self.backend
        excitation = xp.asarray(excitation_grid, dtype=xp.float32)
        if excitation.shape != self.resolution:
            raise ValueError("excitation_grid must match engine resolution")
        return excitation

    def _coerce_prediction_field(
        self,
        values: np.ndarray,
        *,
        prediction_clip: float,
    ) -> np.ndarray:
        field = np.asarray(values, dtype=np.float32)
        if field.shape != self.resolution:
            raise ValueError("prediction field must match engine resolution")
        clip = np.float32(prediction_clip)
        field = np.nan_to_num(field, nan=0.0, posinf=clip, neginf=-clip)
        return np.clip(field, -clip, clip).astype(np.float32, copy=False)

    def _shared_neighbor_sum(self, field: np.ndarray) -> np.ndarray:
        field64 = field.astype(np.float64, copy=False)
        prediction = np.zeros(self.resolution, dtype=np.float64)
        horizontal = self.prediction_horizontal_edge_weights.astype(
            np.float64,
            copy=False,
        )
        vertical = self.prediction_vertical_edge_weights.astype(np.float64, copy=False)
        if horizontal.size:
            prediction[:, :-1] += horizontal * field64[:, 1:]
            prediction[:, 1:] += horizontal * field64[:, :-1]
        if vertical.size:
            prediction[:-1, :] += vertical * field64[1:, :]
            prediction[1:, :] += vertical * field64[:-1, :]
        return prediction

    @staticmethod
    def _shared_edge_gradients(
        *,
        error: np.ndarray,
        field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        field64 = field.astype(np.float64, copy=False)
        error64 = error.astype(np.float64, copy=False)
        horizontal_gradient = (
            error64[:, :-1] * field64[:, 1:] + error64[:, 1:] * field64[:, :-1]
        )
        vertical_gradient = (
            error64[:-1, :] * field64[1:, :] + error64[1:, :] * field64[:-1, :]
        )
        return horizontal_gradient, vertical_gradient

    def _update_prediction_edge_weights(
        self,
        *,
        error: np.ndarray,
        transform: PredictionErrorTransform,
    ) -> None:
        config = transform.config
        if not config.learning_enabled or config.learning_rate == 0.0:
            return
        field = self._coerce_prediction_field(
            self.get_field_numpy(),
            prediction_clip=config.prediction_clip,
        )
        denom = (
            np.float64(config.sigma)
            * np.float64(config.sigma)
            * np.float64(np.log(2.0))
        )
        horizontal_gradient, vertical_gradient = self._shared_edge_gradients(
            error=error,
            field=field,
        )
        horizontal_gradient = np.nan_to_num(
            horizontal_gradient / denom,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        vertical_gradient = np.nan_to_num(
            vertical_gradient / denom,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        horizontal_gradient = np.clip(
            horizontal_gradient,
            -config.learning_gradient_clip,
            config.learning_gradient_clip,
        )
        vertical_gradient = np.clip(
            vertical_gradient,
            -config.learning_gradient_clip,
            config.learning_gradient_clip,
        )
        horizontal_weights = self.prediction_horizontal_edge_weights.astype(
            np.float64,
            copy=False,
        )
        vertical_weights = self.prediction_vertical_edge_weights.astype(
            np.float64,
            copy=False,
        )
        if config.learning_weight_decay:
            decay = np.float64(1.0 - config.learning_weight_decay)
            horizontal_weights *= decay
            vertical_weights *= decay
        horizontal_weights += np.float64(config.learning_rate) * horizontal_gradient
        vertical_weights += np.float64(config.learning_rate) * vertical_gradient
        horizontal_weights = np.clip(
            horizontal_weights,
            -config.learning_weight_clip,
            config.learning_weight_clip,
        )
        vertical_weights = np.clip(
            vertical_weights,
            -config.learning_weight_clip,
            config.learning_weight_clip,
        )
        self.prediction_horizontal_edge_weights = horizontal_weights.astype(
            np.float32,
            copy=False,
        )
        self.prediction_vertical_edge_weights = vertical_weights.astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _coerce_edge_weight_array(
        weights: np.ndarray,
        *,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        array = np.asarray(weights, dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError("edge weight array shape must match propagator operator")
        return array

    def get_prediction_coupling_vectors(
        self,
        *,
        stride: int = 8,
        threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stride = max(int(stride), 1)
        threshold = max(float(threshold), 0.0)
        horizontal = self.prediction_horizontal_edge_weights
        vertical = self.prediction_vertical_edge_weights
        rows, cols = self.resolution
        vx_field = np.zeros((rows, cols), dtype=np.float32)
        vy_field = np.zeros((rows, cols), dtype=np.float32)
        if horizontal.size:
            vx_field[:, 0] = horizontal[:, 0]
            vx_field[:, -1] = horizontal[:, -1]
            if cols > 2:
                vx_field[:, 1:-1] = 0.5 * (horizontal[:, :-1] + horizontal[:, 1:])
        if vertical.size:
            vy_field[0, :] = vertical[0, :]
            vy_field[-1, :] = vertical[-1, :]
            if rows > 2:
                vy_field[1:-1, :] = 0.5 * (vertical[:-1, :] + vertical[1:, :])
        magnitude_field = np.hypot(vx_field, vy_field)
        positions: list[tuple[float, float]] = []
        vectors: list[tuple[float, float]] = []
        magnitudes: list[float] = []
        for row_start in range(0, rows, stride):
            row_end = min(row_start + stride, rows)
            for col_start in range(0, cols, stride):
                col_end = min(col_start + stride, cols)
                tile_magnitudes = magnitude_field[row_start:row_end, col_start:col_end]
                if tile_magnitudes.size == 0:
                    continue
                flat_index = int(np.argmax(tile_magnitudes))
                local_row, local_col = np.unravel_index(flat_index, tile_magnitudes.shape)
                row = row_start + local_row
                col = col_start + local_col
                vx = float(vx_field[row, col])
                vy = float(vy_field[row, col])
                magnitude = float(tile_magnitudes[local_row, local_col])
                if magnitude <= threshold:
                    continue
                positions.append((float(col) + 0.5, float(row) + 0.5))
                vectors.append((vx, vy))
                magnitudes.append(magnitude)
        if not positions:
            empty = np.zeros((0, 2), dtype=np.float32)
            return empty, empty, np.zeros((0,), dtype=np.float32)
        return (
            np.asarray(positions, dtype=np.float32),
            np.asarray(vectors, dtype=np.float32),
            np.asarray(magnitudes, dtype=np.float32),
        )

    def _edge_weights_to_numpy(self, weights) -> np.ndarray:
        if self.use_gpu:
            return self.backend.asnumpy(weights).astype(np.float32, copy=False)
        return np.asarray(weights, dtype=np.float32)

    def _grid_laplacian_with_internal_boundaries(self, field: np.ndarray) -> np.ndarray:
        if self.body_boundary_mask is None:
            return (
                _laplacian_periodic(
                    np,
                    field,
                    horizontal_edge_weights=self.prediction_horizontal_edge_weights,
                    vertical_edge_weights=self.prediction_vertical_edge_weights,
                )
                if self.boundary_condition is BoundaryCondition.CYCLIC
                else _laplacian_neumann(
                    np,
                    field,
                    horizontal_edge_weights=self.prediction_horizontal_edge_weights,
                    vertical_edge_weights=self.prediction_vertical_edge_weights,
                )
            )

        mask = self.body_boundary_mask
        laplacian = np.zeros_like(field)
        transmission = np.float32(self.body_boundary_transmission)
        horizontal_conductance = 1.0 + self.prediction_horizontal_edge_weights
        vertical_conductance = 1.0 + self.prediction_vertical_edge_weights

        vertical_open = mask[:-1, :] == mask[1:, :]
        vertical_diff = field[1:, :] - field[:-1, :]
        laplacian[:-1, :] += vertical_open * (vertical_conductance * vertical_diff)
        laplacian[1:, :] -= vertical_open * (vertical_conductance * vertical_diff)
        vertical_closed = ~vertical_open
        laplacian[:-1, :] += vertical_closed * (
            transmission * vertical_conductance * vertical_diff
        )
        laplacian[1:, :] -= vertical_closed * (
            transmission * vertical_conductance * vertical_diff
        )

        horizontal_open = mask[:, :-1] == mask[:, 1:]
        horizontal_diff = field[:, 1:] - field[:, :-1]
        laplacian[:, :-1] += horizontal_open * (
            horizontal_conductance * horizontal_diff
        )
        laplacian[:, 1:] -= horizontal_open * (
            horizontal_conductance * horizontal_diff
        )
        horizontal_closed = ~horizontal_open
        laplacian[:, :-1] += horizontal_closed * (
            transmission * horizontal_conductance * horizontal_diff
        )
        laplacian[:, 1:] -= horizontal_closed * (
            transmission * horizontal_conductance * horizontal_diff
        )

        if self.boundary_condition is BoundaryCondition.CYCLIC:
            vertical_wrap_open = mask[-1, :] == mask[0, :]
            vertical_wrap_diff = field[0, :] - field[-1, :]
            laplacian[-1, :] += vertical_wrap_open * vertical_wrap_diff
            laplacian[0, :] -= vertical_wrap_open * vertical_wrap_diff
            vertical_wrap_closed = ~vertical_wrap_open
            laplacian[-1, :] += vertical_wrap_closed * (
                transmission * vertical_wrap_diff
            )
            laplacian[0, :] -= vertical_wrap_closed * (
                transmission * vertical_wrap_diff
            )

            horizontal_wrap_open = mask[:, -1] == mask[:, 0]
            horizontal_wrap_diff = field[:, 0] - field[:, -1]
            laplacian[:, -1] += horizontal_wrap_open * horizontal_wrap_diff
            laplacian[:, 0] -= horizontal_wrap_open * horizontal_wrap_diff
            horizontal_wrap_closed = ~horizontal_wrap_open
            laplacian[:, -1] += horizontal_wrap_closed * (
                transmission * horizontal_wrap_diff
            )
            laplacian[:, 0] -= horizontal_wrap_closed * (
                transmission * horizontal_wrap_diff
            )

        return laplacian

    @staticmethod
    def _validate_unit_interval(value: float, *, name: str) -> float:
        scalar = float(value)
        if scalar < 0.0 or scalar > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return scalar
