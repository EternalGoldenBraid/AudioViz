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
from audioviz.utils.spatial_preview import preview_indices


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
        canvas_channels: int = 3,
        use_gpu: bool = False,
        use_shader: bool = False,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
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
        self.canvas_channels = int(canvas_channels)
        if self.canvas_channels <= 0:
            raise ValueError("canvas_channels must be positive")
        self.canvas_shape = (*self.resolution, self.canvas_channels)
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.boundary_condition = coerce_boundary_condition(boundary_condition)
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
            "shape": self.canvas_shape,
            "dx": self.grid_spacing,
            "dt": self.dt,
            "speed": self.speed,
            "damping": self.damping,
            "boundary_condition": self.boundary_condition,
        }
        if use_shader:
            if self.canvas_channels != 1:
                raise NotImplementedError(
                    "OpenGL propagation does not yet support multi-channel canvas states."
                )
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

        self.Z = self.backend.zeros(self.canvas_shape, dtype=self.backend.float32)
        self.Z_old = self.backend.zeros(self.canvas_shape, dtype=self.backend.float32)
        self.prior_Z = self.backend.zeros(
            self.canvas_shape,
            dtype=self.backend.float32,
        )
        self._has_prior = False
        self.body_boundary_mask = None

    def _stable_dt(self) -> float:
        return (self.grid_spacing / self.speed) * 1 / np.sqrt(2)

    @property
    def prediction_horizontal_edge_weights(self) -> np.ndarray:
        return self._edge_weights_to_numpy(self.propagator.horizontal_edge_weights)

    @prediction_horizontal_edge_weights.setter
    def prediction_horizontal_edge_weights(self, weights: np.ndarray) -> None:
        self._set_prediction_conductances(
            horizontal=self._coerce_edge_weight_array(
                weights,
                expected_shape=self.propagator.horizontal_edge_weights.shape,
            )
        )

    @property
    def prediction_vertical_edge_weights(self) -> np.ndarray:
        return self._edge_weights_to_numpy(self.propagator.vertical_edge_weights)

    @prediction_vertical_edge_weights.setter
    def prediction_vertical_edge_weights(self, weights: np.ndarray) -> None:
        self._set_prediction_conductances(
            vertical=self._coerce_edge_weight_array(
                weights,
                expected_shape=self.propagator.vertical_edge_weights.shape,
            )
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
        self.prior_Z[:] = 0
        self._has_prior = False
        self.time = 0.0

    def set_body_boundary_mask(self, body_boundary_mask: np.ndarray | None) -> None:
        if body_boundary_mask is None:
            self.body_boundary_mask = None
            return
        mask = np.asarray(body_boundary_mask, dtype=bool)
        if mask.shape != self.resolution:
            raise ValueError("body_boundary_mask must match engine resolution")
        self.body_boundary_mask = mask.copy()

    def propagate(
        self,
        excitation_grid: np.ndarray | None = None,
    ):
        self.time += self.dt
        excitation = (
            None
            if excitation_grid is None
            else self._coerce_grid_excitation(excitation_grid)
            * np.float32(self.amplitude)
        )
        if self._uses_custom_grid_step():
            self._step_cpu_grid(excitation=excitation)
            return self._capture_prior()
        if excitation is not None:
            self.propagator.add_excitation(excitation)
        self.propagator.step()
        if self.use_shader:
            return self._capture_prior()
        if hasattr(self.propagator, "Z_old"):
            self.Z_old = self.propagator.Z_old.copy()
        self.Z[:] = self.propagator.get_state()
        return self._capture_prior()

    def step_grid_excitation(
        self,
        excitation_grid: np.ndarray,
    ):
        return self.propagate(excitation_grid)

    def step_without_excitation(self):
        return self.propagate()

    def apply_observation_correction(self, correction_grid: np.ndarray) -> np.ndarray:
        if self.use_shader:
            raise NotImplementedError(
                "Observation correction does not yet support OpenGL propagation."
            )
        if not self._has_prior:
            raise RuntimeError("propagate must be called before correcting observations")

        correction = self._coerce_grid_excitation(correction_grid)
        self.Z += correction
        self.Z_old += correction
        self.propagator.Z[...] = self.Z
        if hasattr(self.propagator, "Z_old"):
            self.propagator.Z_old[...] = self.Z_old
        return self.Z

    def get_field_numpy(self) -> np.ndarray:
        if self.use_shader:
            return self.propagator.get_state()
        if self.use_gpu:
            return self.backend.asnumpy(self.Z)
        return self.Z

    def get_field_preview(self) -> np.ndarray:
        """Sample the corrected field before transferring GPU data to the host."""
        if self.use_shader:
            raise RuntimeError("Spatial previews require an array-backed field.")
        rows, columns = preview_indices(self.Z.shape)
        sampled = self.Z[rows[:, None], columns[None, :]]
        if self.use_gpu:
            sampled = self.backend.asnumpy(sampled)
        snapshot = np.asarray(sampled, dtype=np.float32)
        snapshot.setflags(write=False)
        return snapshot

    def get_prior_numpy(self) -> np.ndarray:
        if not self._has_prior:
            raise RuntimeError("propagate must be called before reading the prior")
        if self.use_gpu:
            return self.backend.asnumpy(self.prior_Z)
        return self.prior_Z

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

    def _uses_custom_grid_step(self) -> bool:
        return (
            self.body_boundary_mask is not None
            and not self.use_gpu
            and not self.use_shader
        )

    def _step_cpu_grid(self, *, excitation: np.ndarray | None) -> None:
        grid = np.asarray(self.Z, dtype=np.float32)
        driven_grid = grid.copy()
        if excitation is not None:
            driven_grid += np.asarray(excitation, dtype=np.float32)
        grid_laplacian = self._grid_laplacian_with_internal_boundaries(driven_grid)
        new_grid = (
            2 * driven_grid - self.Z_old + self.propagator.c2_dt2 * grid_laplacian
        )
        new_grid *= self.damping
        if self.body_boundary_mask is not None:
            new_grid[self.body_boundary_mask] *= np.float32(
                1.0 - self.body_boundary_dissipation
            )
        old_grid = grid.copy()
        self.Z[:] = new_grid.astype(np.float32, copy=False)
        self.Z_old[:] = old_grid
        self.propagator.Z[:] = self.Z
        if hasattr(self.propagator, "Z_old"):
            self.propagator.Z_old[:] = self.Z_old

    def _coerce_grid_excitation(self, excitation_grid: np.ndarray):
        xp = self.backend
        excitation = xp.asarray(excitation_grid, dtype=xp.float32)
        if excitation.shape == self.resolution:
            excitation = xp.broadcast_to(
                excitation[..., None],
                self.canvas_shape,
            )
        elif excitation.shape != self.canvas_shape:
            raise ValueError(
                "excitation_grid must match engine resolution or canvas shape"
            )
        return excitation

    def _capture_prior(self):
        if self.use_shader:
            prior = self.propagator.get_state()
            self.prior_Z[:] = prior
        else:
            self.prior_Z[...] = self.Z
        self._has_prior = True
        return self.prior_Z

    @staticmethod
    def _coerce_edge_weight_array(
        weights: np.ndarray,
        *,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        array = np.asarray(weights, dtype=np.float32)
        if array.shape == expected_shape[:2] and len(expected_shape) == 3:
            array = np.broadcast_to(array[..., None], expected_shape).copy()
        if array.shape != expected_shape:
            raise ValueError("edge weight array shape must match propagator operator")
        return array

    def get_prediction_coupling_overlay_edges(
        self,
        *,
        stride: int = 8,
        threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        stride = max(int(stride), 1)
        threshold = max(float(threshold), 0.0)
        horizontal = self.prediction_horizontal_edge_weights
        vertical = self.prediction_vertical_edge_weights
        rows, cols = self.resolution
        row_ranges = [
            (row_start, min(row_start + stride, rows))
            for row_start in range(0, rows, stride)
        ]
        col_ranges = [
            (col_start, min(col_start + stride, cols))
            for col_start in range(0, cols, stride)
        ]
        centers_y = [0.5 * (row_start + row_end) for row_start, row_end in row_ranges]
        centers_x = [0.5 * (col_start + col_end) for col_start, col_end in col_ranges]
        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        strengths: list[float] = []
        for row_index, (row_start, row_end) in enumerate(row_ranges):
            for col_index in range(len(col_ranges) - 1):
                _, col_end = col_ranges[col_index]
                boundary_col = col_end - 1
                boundary_patch = horizontal[row_start:row_end, boundary_col : boundary_col + 1]
                if boundary_patch.size == 0:
                    continue
                strength = float(np.mean(boundary_patch, dtype=np.float64))
                if abs(strength) <= threshold:
                    continue
                segments.append(
                    (
                        (centers_x[col_index], centers_y[row_index]),
                        (centers_x[col_index + 1], centers_y[row_index]),
                    )
                )
                strengths.append(strength)
        for row_index in range(len(row_ranges) - 1):
            _, row_end = row_ranges[row_index]
            boundary_row = row_end - 1
            for col_index, (col_start, col_end) in enumerate(col_ranges):
                boundary_patch = vertical[
                    boundary_row : boundary_row + 1,
                    col_start:col_end,
                ]
                if boundary_patch.size == 0:
                    continue
                strength = float(np.mean(boundary_patch, dtype=np.float64))
                if abs(strength) <= threshold:
                    continue
                segments.append(
                    (
                        (centers_x[col_index], centers_y[row_index]),
                        (centers_x[col_index], centers_y[row_index + 1]),
                    )
                )
                strengths.append(strength)
        if not segments:
            return np.zeros((0, 2, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return (
            np.asarray(segments, dtype=np.float32),
            np.asarray(strengths, dtype=np.float32),
        )

    def _edge_weights_to_numpy(self, weights) -> np.ndarray:
        if self.use_gpu:
            return self.backend.asnumpy(weights).astype(np.float32, copy=False)
        return np.asarray(weights, dtype=np.float32)

    def _set_prediction_conductances(
        self,
        *,
        horizontal: np.ndarray | None = None,
        vertical: np.ndarray | None = None,
    ) -> None:
        horizontal_array = (
            self._edge_weights_to_numpy(self.propagator.horizontal_edge_weights)
            if horizontal is None
            else np.asarray(horizontal, dtype=np.float32)
        )
        vertical_array = (
            self._edge_weights_to_numpy(self.propagator.vertical_edge_weights)
            if vertical is None
            else np.asarray(vertical, dtype=np.float32)
        )
        normalized_horizontal, normalized_vertical = self._normalize_prediction_conductances(
            horizontal_array,
            vertical_array,
        )
        self.propagator.horizontal_edge_weights[...] = normalized_horizontal
        self.propagator.vertical_edge_weights[...] = normalized_vertical

    @staticmethod
    def _normalize_prediction_conductances(
        horizontal: np.ndarray,
        vertical: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        horizontal = np.clip(np.asarray(horizontal, dtype=np.float32), 0.0, None)
        vertical = np.clip(np.asarray(vertical, dtype=np.float32), 0.0, None)
        rows = horizontal.shape[0]
        cols = vertical.shape[1]
        feature_shape = horizontal.shape[2:]
        degree = np.zeros((rows, cols, *feature_shape), dtype=np.float32)
        if horizontal.size:
            degree[:, :-1] += horizontal
            degree[:, 1:] += horizontal
        if vertical.size:
            degree[:-1, :] += vertical
            degree[1:, :] += vertical
        factors = np.ones_like(degree)
        overloaded = degree > 4.0
        factors[overloaded] = 4.0 / degree[overloaded]
        if horizontal.size:
            horizontal = horizontal * np.minimum(factors[:, :-1], factors[:, 1:])
        if vertical.size:
            vertical = vertical * np.minimum(factors[:-1, :], factors[1:, :])
        return horizontal.astype(np.float32, copy=False), vertical.astype(
            np.float32,
            copy=False,
        )

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
        horizontal_conductance = self.prediction_horizontal_edge_weights
        vertical_conductance = self.prediction_vertical_edge_weights

        vertical_open = (mask[:-1, :] == mask[1:, :])[..., None]
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

        horizontal_open = (mask[:, :-1] == mask[:, 1:])[..., None]
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
            vertical_wrap_open = (mask[-1, :] == mask[0, :])[..., None]
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

            horizontal_wrap_open = (mask[:, -1] == mask[:, 0])[..., None]
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
