"""Geometric test scene, seeded Brownian translation, and BGR snapshots."""

import numpy as np


def geometric_scene(resolution=(80, 120)) -> np.ndarray:
    if len(resolution) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in resolution
    ):
        raise ValueError("resolution must contain two positive integers")
    rows, cols = resolution
    image = np.full((rows, cols, 3), 20, dtype=np.uint8)
    image[rows//4:3*rows//4, 5*cols//12:7*cols//12] = (230, 170, 45)
    y, x = np.ogrid[:rows, :cols]
    image[(y-rows*.45)**2+(x-cols*.23)**2 <= (min(rows, cols)*.16)**2] = (45, 185, 210)
    image[rows//3:2*rows//3, 4*cols//5:9*cols//10] = (185, 65, 170)
    image.setflags(write=False)
    return image


class StaticSceneCapture:
    """Repeat an immutable BGR snapshot until explicitly replaced."""

    def __init__(self, rgb: np.ndarray):
        self._closed = False
        self.read_count = 0
        self.set_frame(rgb)

    def set_frame(self, rgb: np.ndarray):
        if self._closed:
            raise RuntimeError("cannot replace a released scene capture")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError("static scene must be a nonempty HxWx3 uint8 RGB image")
        self._frame = rgb[..., ::-1].copy()
        self._frame.setflags(write=False)

    def read(self):
        if self._closed:
            return False, None
        self.read_count += 1
        return True, self._frame

    def release(self):
        self._closed = True


class BrownianScene:
    """Evaluator-owned scene translation; advancing time does not observe it."""

    def __init__(self, resolution=(80, 120), *, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self._base = geometric_scene(resolution)
        self.reset()

    def reset(self):
        self._rng = np.random.default_rng(self.seed)
        self.offset = np.zeros(2, dtype=np.float64)
        self.pixel_offset = (0, 0)
        self.time = 0.
        self.frame = self._base

    def advance(self, dt: float, *, scale: float = 4.) -> np.ndarray:
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if isinstance(scale, bool) or not np.isfinite(scale) or scale < 0:
            raise ValueError("Brownian scale must be finite and non-negative")
        deviation = scale*np.sqrt(dt)
        if not np.isfinite(deviation) or not np.isfinite(self.time+dt):
            raise ValueError("Brownian step and world time must remain finite")
        offset = self.offset+self._rng.normal(scale=deviation, size=2)
        if not np.isfinite(offset).all():
            raise FloatingPointError("Brownian displacement must remain finite")
        self.offset = offset
        self.time += dt
        self.pixel_offset = tuple(int(np.rint(value)) for value in self.offset)
        dx, dy = self.pixel_offset
        rows, cols = self._base.shape[:2]
        left, right = max(0, -dx), min(cols, cols-dx)
        top, bottom = max(0, -dy), min(rows, rows-dy)
        frame = np.full_like(self._base, 20)
        if left < right and top < bottom:
            frame[top+dy:bottom+dy, left+dx:right+dx] = self._base[top:bottom, left:right]
        frame.setflags(write=False)
        self.frame = frame
        return frame
