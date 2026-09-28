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
