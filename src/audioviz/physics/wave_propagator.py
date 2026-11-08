from enum import Enum
from typing import Any

import numpy as np


class BoundaryCondition(str, Enum):
    CYCLIC = "cyclic"
    NEUMANN = "neumann"


def coerce_boundary_condition(value: BoundaryCondition | str) -> BoundaryCondition:
    if isinstance(value, BoundaryCondition):
        return value
    try:
        return BoundaryCondition(value)
    except ValueError as exc:
        valid = ", ".join(option.value for option in BoundaryCondition)
        raise ValueError(f"boundary_condition must be one of: {valid}") from exc


def load_cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU ripple rendering requires CuPy. Install CuPy/CUDA or use the CPU backend."
        ) from exc
    return cp


def _weighted_laplacian_periodic(
    xp,
    Z,
    *,
    horizontal_edge_weights=None,
    vertical_edge_weights=None,
):
    laplacian = (
        -4 * Z
        + xp.roll(Z, 1, axis=0)
        + xp.roll(Z, -1, axis=0)
        + xp.roll(Z, 1, axis=1)
        + xp.roll(Z, -1, axis=1)
    )
    if horizontal_edge_weights is not None and horizontal_edge_weights.size:
        horizontal_delta = (horizontal_edge_weights - 1.0) * (Z[:, 1:] - Z[:, :-1])
        laplacian[:, :-1] += horizontal_delta
        laplacian[:, 1:] -= horizontal_delta
    if vertical_edge_weights is not None and vertical_edge_weights.size:
        vertical_delta = (vertical_edge_weights - 1.0) * (Z[1:, :] - Z[:-1, :])
        laplacian[:-1, :] += vertical_delta
        laplacian[1:, :] -= vertical_delta
    return laplacian


def _weighted_laplacian_neumann(
    xp,
    Z,
    *,
    horizontal_edge_weights=None,
    vertical_edge_weights=None,
):
    laplacian = xp.zeros_like(Z)
    vertical_conductance = 1.0
    horizontal_conductance = 1.0
    if vertical_edge_weights is not None:
        vertical_conductance = vertical_edge_weights
    if horizontal_edge_weights is not None:
        horizontal_conductance = horizontal_edge_weights

    vertical_diff = Z[1:, :] - Z[:-1, :]
    laplacian[:-1, :] += vertical_conductance * vertical_diff
    laplacian[1:, :] -= vertical_conductance * vertical_diff

    horizontal_diff = Z[:, 1:] - Z[:, :-1]
    laplacian[:, :-1] += horizontal_conductance * horizontal_diff
    laplacian[:, 1:] -= horizontal_conductance * horizontal_diff
    return laplacian


def _laplacian_periodic(
    xp,
    Z,
    *,
    horizontal_edge_weights=None,
    vertical_edge_weights=None,
):
    return _weighted_laplacian_periodic(
        xp,
        Z,
        horizontal_edge_weights=horizontal_edge_weights,
        vertical_edge_weights=vertical_edge_weights,
    )


def _laplacian_neumann(
    xp,
    Z,
    *,
    horizontal_edge_weights=None,
    vertical_edge_weights=None,
):
    return _weighted_laplacian_neumann(
        xp,
        Z,
        horizontal_edge_weights=horizontal_edge_weights,
        vertical_edge_weights=vertical_edge_weights,
    )


class WavePropagatorCPU:
    def __init__(
        self,
        shape,
        dx,
        dt,
        speed,
        damping,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
    ):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping
        self.boundary_condition = coerce_boundary_condition(boundary_condition)

        self.Z = np.zeros(shape, dtype=np.float32)
        self.Z_old = np.zeros_like(self.Z)
        self.Z_new = np.zeros_like(self.Z)
        rows, cols = shape[:2]
        feature_shape = tuple(shape[2:])
        self.horizontal_edge_weights = np.ones(
            (rows, max(cols - 1, 0), *feature_shape),
            dtype=np.float32,
        )
        self.vertical_edge_weights = np.ones(
            (max(rows - 1, 0), cols, *feature_shape),
            dtype=np.float32,
        )

        self.c2_dt2 = (self.c * self.dt / self.dx) ** 2

    def add_excitation(self, excitation: np.ndarray):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        Z = self.Z
        laplacian = (
            _weighted_laplacian_periodic(
                np,
                Z,
                horizontal_edge_weights=self.horizontal_edge_weights,
                vertical_edge_weights=self.vertical_edge_weights,
            )
            if self.boundary_condition is BoundaryCondition.CYCLIC
            else _weighted_laplacian_neumann(
                np,
                Z,
                horizontal_edge_weights=self.horizontal_edge_weights,
                vertical_edge_weights=self.vertical_edge_weights,
            )
        )
        self.Z_new = 2 * Z - self.Z_old + self.c2_dt2 * laplacian
        self.Z_new *= self.damping
        self.Z_old = Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self):
        return self.Z

    def reset(self):
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0
        self.horizontal_edge_weights[:] = 1
        self.vertical_edge_weights[:] = 1


class ForceDrivenCanvasCPU:
    """Unit-mass waves with velocity damping and externally evaluated gradient forces.

    Force is acceleration evaluated at the current position. Its restoring
    curvature bound is combined with the unweighted wave operator's bound.
    Damping is integrated implicitly; velocity then advances position.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dx: float,
        speed: float,
        damping_rate: float,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
    ):
        if len(shape) < 2 or any(not isinstance(size, (int, np.integer)) or size <= 0 for size in shape):
            raise ValueError("shape must contain at least two positive grid dimensions")
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("dx must be finite and positive")
        for name, value in (("speed", speed), ("damping_rate", damping_rate)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.dx = float(dx)
        self.speed = float(speed)
        self.damping_rate = float(damping_rate)
        self.boundary_condition = coerce_boundary_condition(boundary_condition)
        self.position = np.zeros(shape, dtype=np.float64)
        self.velocity = np.zeros_like(self.position)
        self.time = 0.0

    def max_step(self, force_curvature: float) -> float:
        if not np.isfinite(force_curvature) or force_curvature < 0:
            raise ValueError("force_curvature must be finite and non-negative")
        curvature = 8 * (self.speed / self.dx)**2 + force_curvature
        return 2 / np.sqrt(curvature) if curvature > 0 else np.inf

    def step(self, force: np.ndarray, dt: float, *, force_curvature: float) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if dt > self.max_step(force_curvature):
            raise ValueError("dt exceeds the combined wave and force stability bound")
        force = np.asarray(force, dtype=np.float64)
        if force.shape != self.position.shape:
            raise ValueError("force must match the canvas shape")
        if not np.isfinite(force).all():
            raise FloatingPointError("canvas force became non-finite")
        laplacian = (
            _weighted_laplacian_periodic(np, self.position)
            if self.boundary_condition is BoundaryCondition.CYCLIC
            else _weighted_laplacian_neumann(np, self.position)
        )
        acceleration = (self.speed / self.dx)**2 * laplacian + force
        velocity = (self.velocity + dt * acceleration) / (1 + self.damping_rate * dt)
        position = self.position + dt * velocity
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise FloatingPointError("force-driven canvas update became non-finite")
        self.position[:] = position
        self.velocity[:] = velocity
        self.time += dt
        return self.position


class WavePropagatorGPU:
    def __init__(
        self,
        shape,
        dx,
        dt,
        speed,
        damping,
        cupy_module=None,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
    ):
        self.cp = cupy_module if cupy_module is not None else load_cupy()
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping
        self.boundary_condition = coerce_boundary_condition(boundary_condition)

        cp = self.cp
        self.Z = cp.zeros(shape, dtype=cp.float32)
        self.Z_old = cp.zeros_like(self.Z)
        self.Z_new = cp.zeros_like(self.Z)
        rows, cols = shape[:2]
        feature_shape = tuple(shape[2:])
        self.horizontal_edge_weights = cp.ones(
            (rows, max(cols - 1, 0), *feature_shape),
            dtype=cp.float32,
        )
        self.vertical_edge_weights = cp.ones(
            (max(rows - 1, 0), cols, *feature_shape),
            dtype=cp.float32,
        )

        self.c2_dt2 = (self.c * self.dt / self.dx) ** 2

    def add_excitation(self, excitation):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        cp = self.cp
        Z = self.Z
        laplacian = (
            _weighted_laplacian_periodic(
                cp,
                Z,
                horizontal_edge_weights=self.horizontal_edge_weights,
                vertical_edge_weights=self.vertical_edge_weights,
            )
            if self.boundary_condition is BoundaryCondition.CYCLIC
            else _weighted_laplacian_neumann(
                cp,
                Z,
                horizontal_edge_weights=self.horizontal_edge_weights,
                vertical_edge_weights=self.vertical_edge_weights,
            )
        )
        self.Z_new = 2 * Z - self.Z_old + self.c2_dt2 * laplacian
        self.Z_new *= self.damping
        self.Z_old = Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self):
        return self.Z

    def reset(self):
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0
        self.horizontal_edge_weights[:] = 1
        self.vertical_edge_weights[:] = 1
