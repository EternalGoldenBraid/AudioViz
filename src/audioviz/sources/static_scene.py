"""Deterministic RGB test world and its camera-compatible BGR interface."""

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
    """Expose only an immutable, repeated BGR frame to the camera pathway."""

    def __init__(self, rgb: np.ndarray):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError("static scene must be a nonempty HxWx3 uint8 RGB image")
        self._frame = rgb[..., ::-1].copy()
        self._frame.setflags(write=False)
        self._closed = False
        self.read_count = 0

    def read(self):
        if self._closed:
            return False, None
        self.read_count += 1
        return True, self._frame

    def release(self):
        self._closed = True
