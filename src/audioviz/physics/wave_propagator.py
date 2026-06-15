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


def _laplacian_periodic(xp, Z):
    return (
        -4 * Z
        + xp.roll(Z, 1, axis=0)
        + xp.roll(Z, -1, axis=0)
        + xp.roll(Z, 1, axis=1)
        + xp.roll(Z, -1, axis=1)
    )


def _laplacian_neumann(xp, Z):
    laplacian = xp.zeros_like(Z)
    degree = xp.zeros_like(Z)

    laplacian[1:, :] += Z[:-1, :]
    degree[1:, :] += 1
    laplacian[:-1, :] += Z[1:, :]
    degree[:-1, :] += 1
    laplacian[:, 1:] += Z[:, :-1]
    degree[:, 1:] += 1
    laplacian[:, :-1] += Z[:, 1:]
    degree[:, :-1] += 1

    laplacian -= degree * Z
    return laplacian


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

        self.c2_dt2 = (self.c * self.dt / self.dx) ** 2

    def add_excitation(self, excitation: np.ndarray):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        Z = self.Z
        laplacian = (
            _laplacian_periodic(np, Z)
            if self.boundary_condition is BoundaryCondition.CYCLIC
            else _laplacian_neumann(np, Z)
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

        self.c2_dt2 = (self.c * self.dt / self.dx) ** 2

    def add_excitation(self, excitation):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        cp = self.cp
        Z = self.Z
        laplacian = (
            _laplacian_periodic(cp, Z)
            if self.boundary_condition is BoundaryCondition.CYCLIC
            else _laplacian_neumann(cp, Z)
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
