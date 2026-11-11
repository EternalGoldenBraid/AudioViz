"""World-anchored circular evidence routing."""

import numpy as np

MAX_FOVEAL_PIXELS = 4096


def circular_foveal_offsets(radius):
    if (
        isinstance(radius, bool) or not isinstance(radius, (int, np.integer))
        or radius < 1 or int(radius)**2 > MAX_FOVEAL_PIXELS
    ):
        raise ValueError("foveal_radius must be a positive integer within the disk pixel limit")
    radius = int(radius)
    offsets = np.array([
        (y, x) for y in range(-radius, radius+1) for x in range(-radius, radius+1)
        if y*y+x*x <= radius*radius
    ], dtype=np.intp)
    if len(offsets) > MAX_FOVEAL_PIXELS:
        raise ValueError(f"foveal_radius={radius} exceeds {MAX_FOVEAL_PIXELS} disk pixels")
    return offsets


class WorldAnchoredFovea:
    """Move an observation disk without moving or resizing world sensory slots."""

    def __init__(self, shape, *, radius=8):
        if len(shape) != 2 or any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1
            for size in shape
        ):
            raise ValueError("world shape must contain two positive integer dimensions")
        self.shape = tuple(shape)
        self.offsets = circular_foveal_offsets(radius)
        self.radius = int(radius)
        self.positions = np.indices(self.shape).reshape(2, -1).T
        self.gaze = (.5, .5)

    def set_gaze(self, gaze):
        values = np.asarray(gaze, dtype=np.float64)
        if (
            values.shape != (2,) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
        ):
            raise ValueError("gaze must contain two finite normalized coordinates")
        self.gaze = tuple(values)

    @property
    def center(self):
        return np.rint(np.asarray(self.gaze)[::-1]*(np.asarray(self.shape)-1)).astype(np.intp)

    @property
    def observation_mask(self):
        return (
            np.sum((self.positions-self.center)**2, axis=1) <= self.radius**2
        ).reshape(self.shape)
