from typing import Any

import numpy as np


def load_cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU ripple rendering requires CuPy. Install CuPy/CUDA or use the CPU backend."
        ) from exc
    return cp


class WavePropagatorCPU:
    def __init__(self, shape, dx, dt, speed, damping):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping

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
            -4 * Z
            + np.roll(Z, 1, axis=0)
            + np.roll(Z, -1, axis=0)
            + np.roll(Z, 1, axis=1)
            + np.roll(Z, -1, axis=1)
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
    def __init__(self, shape, dx, dt, speed, damping, cupy_module=None):
        self.cp = cupy_module if cupy_module is not None else load_cupy()
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping

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
            -4 * Z
            + cp.roll(Z, 1, axis=0)
            + cp.roll(Z, -1, axis=0)
            + cp.roll(Z, 1, axis=1)
            + cp.roll(Z, -1, axis=1)
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
