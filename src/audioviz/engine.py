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
        self.dt = self._stable_dt()
        self.time = 0.0
        self.max_frequency = self.speed / (2 * max(self.dx, self.dy))

        self.source_positions = self._make_source_positions()
        self.xs, self.ys = self.backend.meshgrid(
            self.backend.arange(self.resolution[1]),
            self.backend.arange(self.resolution[0]),
        )

        propagator_kwargs = {
            "shape": self.resolution,
            "dx": self.dx,
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
        self.body_boundary_inner_ring_mask = None

    def _stable_dt(self) -> float:
        return (max(self.dx, self.dy) / self.speed) * 1 / np.sqrt(2)

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
        self.propagator.c2_dt2 = (self.speed * self.dt / self.dx) ** 2
        self.max_frequency = self.speed / (2 * max(self.dx, self.dy))

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
        if self.pose_values is not None:
            self.pose_values[:] = 0
        if self.pose_values_old is not None:
            self.pose_values_old[:] = 0
        self.time = 0.0

    def set_body_boundary_mask(self, body_boundary_mask: np.ndarray | None) -> None:
        if body_boundary_mask is None:
            self.body_boundary_mask = None
            self.body_boundary_inner_ring_mask = None
            return
        mask = np.asarray(body_boundary_mask, dtype=bool)
        if mask.shape != self.resolution:
            raise ValueError("body_boundary_mask must match engine resolution")
        self.body_boundary_mask = mask.copy()
        self.body_boundary_inner_ring_mask = self._compute_body_boundary_inner_ring(mask)

    def step(self, frequencies: np.ndarray):
        self.time += self.dt
        self._add_ripple_excitation(self.time, frequencies)
        if self.use_shader:
            return self.Z
        if hasattr(self.propagator, "Z_old"):
            self.Z_old = np.array(self.propagator.Z_old, copy=True)
        self.Z[:] = self.propagator.get_state()
        return self.Z

    def step_source_excitations(self, source_excitations: np.ndarray):
        self.time += self.dt
        self._add_source_excitation(source_excitations)
        self.propagator.step()
        if self.use_shader:
            return self.Z
        if hasattr(self.propagator, "Z_old"):
            self.Z_old = np.array(self.propagator.Z_old, copy=True)
        self.Z[:] = self.propagator.get_state()
        return self.Z

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

    def _add_ripple_excitation(self, t: float, frequencies: np.ndarray) -> None:
        xp = self.backend
        frequencies = xp.asarray(frequencies, dtype=xp.float32)
        n_sources, _ = frequencies.shape

        if n_sources != self.n_sources:
            raise ValueError(
                f"Expected {self.n_sources} source rows, got {n_sources}."
            )

        x0 = xp.array([p[0] for p in self.source_positions]).reshape(n_sources, 1, 1)
        y0 = xp.array([p[1] for p in self.source_positions]).reshape(n_sources, 1, 1)

        xs = self.xs[None, :, :]
        ys = self.ys[None, :, :]

        r_pixels = xp.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
        r_meters = r_pixels * self.dx
        decay = xp.exp(-self.decay_alpha * r_meters)

        frequencies = xp.clip(frequencies, 1e-3, self.max_frequency)
        wavelengths = self.speed / frequencies
        phases = 2 * xp.pi * frequencies * t

        r = r_meters[:, None, :, :]
        decay = decay[:, None, :, :]
        wavelengths = wavelengths[:, :, None, None]
        phases = phases[:, :, None, None]

        ripple = self.amplitude * decay * xp.sin(
            phases - 2 * xp.pi * r / wavelengths
        )

        self.propagator.add_excitation(ripple.sum(axis=(0, 1)))
        self.propagator.step()

    def _add_source_excitation(self, source_excitations: np.ndarray) -> None:
        xp = self.backend
        values = xp.asarray(source_excitations, dtype=xp.float32).reshape(-1)
        if len(values) != self.n_sources:
            raise ValueError(
                f"Expected {self.n_sources} source excitations, got {len(values)}."
            )

        rows, cols = self.resolution
        positions = xp.asarray(self.source_positions, dtype=xp.float32)
        xs = xp.clip(xp.rint(positions[:, 0]).astype(xp.int32), 0, cols - 1)
        ys = xp.clip(xp.rint(positions[:, 1]).astype(xp.int32), 0, rows - 1)

        excitation = xp.zeros(self.resolution, dtype=xp.float32)
        xp.add.at(excitation, (ys, xs), values * self.amplitude)
        self.propagator.add_excitation(excitation)

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

    def step_pose_medium(self, frequencies: np.ndarray | None = None) -> np.ndarray:
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
        if frequencies is not None:
            driven_grid += self._compute_ripple_excitation_field(self.time, frequencies)

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
        if self.body_boundary_inner_ring_mask is not None:
            new_grid[self.body_boundary_inner_ring_mask] *= np.float32(
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

    def _compute_ripple_excitation_field(self, t: float, frequencies: np.ndarray) -> np.ndarray:
        xp = self.backend
        frequencies = xp.asarray(frequencies, dtype=xp.float32)
        n_sources, _ = frequencies.shape

        if n_sources != self.n_sources:
            raise ValueError(
                f"Expected {self.n_sources} source rows, got {n_sources}."
            )

        x0 = xp.array([p[0] for p in self.source_positions]).reshape(n_sources, 1, 1)
        y0 = xp.array([p[1] for p in self.source_positions]).reshape(n_sources, 1, 1)

        xs = self.xs[None, :, :]
        ys = self.ys[None, :, :]

        r_pixels = xp.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
        r_meters = r_pixels * self.dx
        decay = xp.exp(-self.decay_alpha * r_meters)

        frequencies = xp.clip(frequencies, 1e-3, self.max_frequency)
        wavelengths = self.speed / frequencies
        phases = 2 * xp.pi * frequencies * t

        r = r_meters[:, None, :, :]
        decay = decay[:, None, :, :]
        wavelengths = wavelengths[:, :, None, None]
        phases = phases[:, :, None, None]

        ripple = self.amplitude * decay * xp.sin(
            phases - 2 * xp.pi * r / wavelengths
        )
        return ripple.sum(axis=(0, 1))

    def _grid_laplacian_with_internal_boundaries(self, field: np.ndarray) -> np.ndarray:
        if self.body_boundary_mask is None:
            return (
                _laplacian_periodic(np, field)
                if self.boundary_condition is BoundaryCondition.CYCLIC
                else _laplacian_neumann(np, field)
            )

        mask = self.body_boundary_mask
        laplacian = np.zeros_like(field)
        transmission = np.float32(self.body_boundary_transmission)

        vertical_open = mask[:-1, :] == mask[1:, :]
        vertical_diff = field[1:, :] - field[:-1, :]
        laplacian[:-1, :] += vertical_open * vertical_diff
        laplacian[1:, :] -= vertical_open * vertical_diff
        vertical_closed = ~vertical_open
        laplacian[:-1, :] += vertical_closed * (transmission * vertical_diff)
        laplacian[1:, :] -= vertical_closed * (transmission * vertical_diff)

        horizontal_open = mask[:, :-1] == mask[:, 1:]
        horizontal_diff = field[:, 1:] - field[:, :-1]
        laplacian[:, :-1] += horizontal_open * horizontal_diff
        laplacian[:, 1:] -= horizontal_open * horizontal_diff
        horizontal_closed = ~horizontal_open
        laplacian[:, :-1] += horizontal_closed * (transmission * horizontal_diff)
        laplacian[:, 1:] -= horizontal_closed * (transmission * horizontal_diff)

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

    def _compute_body_boundary_inner_ring(self, mask: np.ndarray) -> np.ndarray:
        outside_neighbor = np.zeros_like(mask, dtype=bool)
        outside_neighbor[1:, :] |= ~mask[:-1, :]
        outside_neighbor[:-1, :] |= ~mask[1:, :]
        outside_neighbor[:, 1:] |= ~mask[:, :-1]
        outside_neighbor[:, :-1] |= ~mask[:, 1:]

        if self.boundary_condition is BoundaryCondition.CYCLIC:
            outside_neighbor[0, :] |= ~mask[-1, :]
            outside_neighbor[-1, :] |= ~mask[0, :]
            outside_neighbor[:, 0] |= ~mask[:, -1]
            outside_neighbor[:, -1] |= ~mask[:, 0]

        return mask & outside_neighbor

    @staticmethod
    def _validate_unit_interval(value: float, *, name: str) -> float:
        scalar = float(value)
        if scalar < 0.0 or scalar > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return scalar
