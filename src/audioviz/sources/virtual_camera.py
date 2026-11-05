"""Synthetic cameras: a cv2.VideoCapture lookalike so camera paths can be tested without hardware.

Virtual devices take indices from VIRTUAL_BASE upward, so they flow through the same device
selection as /dev/video* numbers.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

VIRTUAL_BASE = 100
FRAME_SHAPE = (120, 160)


def _bar(tick: int) -> np.ndarray:
    frame = np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)
    x = (tick * 4) % FRAME_SHAPE[1]
    frame[:, x:x + 12] = 255
    return frame


def _dark(tick: int) -> np.ndarray:
    return np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)


def _bright(tick: int) -> np.ndarray:
    return np.full((*FRAME_SHAPE, 3), 255, dtype=np.uint8)


def _gradient(tick: int) -> np.ndarray:
    ramp = np.linspace(0, 255, FRAME_SHAPE[1], dtype=np.uint8)
    return np.broadcast_to(ramp[None, :, None], (*FRAME_SHAPE, 3)).copy()


SCENES: dict[str, Callable[[int], np.ndarray]] = {
    "bar": _bar, "dark": _dark, "bright": _bright, "gradient": _gradient,
}


def virtual_choices() -> tuple[str, ...]:
    return tuple(f"{VIRTUAL_BASE + i}: virtual {name}" for i, name in enumerate(SCENES))


class VirtualCapture:
    """Returns BGR frames of a named scene; frame `tick` advances on every read."""

    def __init__(self, index: int):
        names = tuple(SCENES)
        self.scene = SCENES[names[index - VIRTUAL_BASE]] if 0 <= index - VIRTUAL_BASE < len(names) else None
        self.tick = 0

    def isOpened(self) -> bool:
        return self.scene is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.scene is None:
            return False, None
        self.tick += 1
        return True, self.scene(self.tick)

    def release(self) -> None:
        self.scene = None
