from typing import Tuple

import numpy as np

from audioviz.physics.wave_propagator import (
    BoundaryCondition,
    WavePropagatorCPU,
    WavePropagatorGPU,
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

    def reset(self) -> None:
        self.propagator.reset()
        self.Z[:] = 0
        self.time = 0.0

    def step(self, frequencies: np.ndarray):
        self.time += self.dt
        self._add_ripple_excitation(self.time, frequencies)
        if self.use_shader:
            return self.Z
        self.Z[:] = self.propagator.get_state()
        return self.Z

    def step_source_excitations(self, source_excitations: np.ndarray):
        self.time += self.dt
        self._add_source_excitation(source_excitations)
        self.propagator.step()
        if self.use_shader:
            return self.Z
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
