from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class CameraFrameSourceConfig:
    enabled: bool = False
    camera_index: int | str = 0
    gain: float = 1.0


def load_cv2():
    import cv2

    return cv2


class CameraFrameSource:
    def __init__(
        self,
        *,
        resolution: tuple[int, int],
        camera_index: int | str = 0,
        gain: float = 1.0,
        enabled: bool = False,
        capture=None,
        cv2_loader: Callable = load_cv2,
    ) -> None:
        self.resolution = tuple(int(value) for value in resolution)
        self.camera_index = camera_index
        self.gain = float(gain)
        self.enabled = bool(enabled)
        self.capture = capture
        self.cv2_loader = cv2_loader

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self.ensure_open()
        self.enabled = bool(enabled)

    def set_gain(self, gain: float) -> None:
        self.gain = float(gain)

    def ensure_open(self) -> None:
        if self.capture is not None:
            return
        cv2 = self.cv2_loader()
        logger.info("Opening camera source: {}", self.camera_index)
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            self.capture = None
            raise RuntimeError(f"Failed to open camera source index {self.camera_index}")
        self.capture = capture

    def excitation(self) -> np.ndarray | None:
        if not self.enabled:
            return None
        self.ensure_open()
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        return self.frame_to_excitation_grid(
            frame,
            resolution=self.resolution,
            gain=self.gain,
        )

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @staticmethod
    def frame_to_excitation_grid(
        frame: np.ndarray,
        *,
        resolution: tuple[int, int],
        gain: float = 1.0,
    ) -> np.ndarray:
        values = np.asarray(frame)
        if values.ndim == 3:
            if values.shape[2] < 3:
                raise ValueError("camera frame must provide at least three channels")
            rgb = values[..., :3][..., ::-1].astype(np.float32)
        elif values.ndim == 2:
            gray = values.astype(np.float32)
            rgb = np.repeat(gray[..., None], 3, axis=2)
        else:
            raise ValueError(
                "camera frame must have shape (rows, cols) or (rows, cols, channels)"
            )
        if rgb.size == 0:
            raise ValueError("camera frame must not be empty")
        if values.dtype.kind in {"u", "i"}:
            rgb = rgb / np.float32(255.0)
        else:
            max_value = float(np.nanmax(rgb))
            if max_value > 1.0:
                rgb = rgb / np.float32(255.0)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = np.clip(rgb, 0.0, 1.0)
        mapped = CameraFrameSource.resize_grid(rgb, resolution=resolution)
        return (mapped * np.float32(gain)).astype(np.float32, copy=False)

    @staticmethod
    def resize_grid(
        values: np.ndarray,
        *,
        resolution: tuple[int, int],
    ) -> np.ndarray:
        rows, cols = resolution
        src_rows, src_cols = values.shape[:2]
        y_index = np.rint(np.linspace(0, src_rows - 1, rows)).astype(np.int32)
        x_index = np.rint(np.linspace(0, src_cols - 1, cols)).astype(np.int32)
        return np.ascontiguousarray(values[y_index][:, x_index], dtype=np.float32)
