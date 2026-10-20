from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import numpy as np
from loguru import logger

from audioviz.utils.source_preview import SourceSceneMapping


@dataclass(frozen=True)
class CameraFrameSourceConfig:
    enabled: bool = False
    camera_index: int | str = 0
    gain: float = 1.0


def load_cv2():
    import cv2

    return cv2


class CameraUnavailableError(RuntimeError):
    pass


class CameraFrameSource:
    scene_mapping = SourceSceneMapping("camera", "Camera", "rgb", "sensory")

    def __init__(
        self,
        *,
        resolution: tuple[int, int],
        camera_index: int | str = 0,
        gain: float = 1.0,
        enabled: bool = False,
        capture=None,
        cv2_loader: Callable = load_cv2,
        backend: ModuleType = np,
        frame_mapper: Callable | None = None,
    ) -> None:
        self.resolution = tuple(int(value) for value in resolution)
        self.camera_index = camera_index
        self.gain = float(gain)
        self.enabled = bool(enabled)
        self.capture = capture
        self.cv2_loader = cv2_loader
        self.backend = backend
        self.frame_mapper = frame_mapper
        self.error: str | None = None
        self._has_frame = False

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.error is not None:
            return self.error
        return "receiving" if self._has_frame else "waiting for first frame"

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            if self.error is not None:
                self.close()
        self.enabled = bool(enabled)

    def set_gain(self, gain: float) -> None:
        self.gain = float(gain)

    def ensure_open(self) -> None:
        if self.capture is not None:
            return
        cv2 = self.cv2_loader()
        logger.info("Opening camera source: {}", self.camera_index)
        linux_device = (
            isinstance(self.camera_index, str)
            and self.camera_index.startswith("/dev/video")
            and self.camera_index.removeprefix("/dev/video").isdecimal()
        )
        capture = (
            cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if linux_device else cv2.VideoCapture(self.camera_index)
        )
        if not capture.isOpened():
            capture.release()
            self.capture = None
            self.error = (
                f"Camera {self.camera_index}: cannot open device. "
                "Toggle Camera Frame off/on to retry."
            )
            raise CameraUnavailableError(f"Failed to open camera source index {self.camera_index}")
        self.capture = capture
        self.error = None

    def excitation(self) -> np.ndarray | None:
        if not self.enabled or self.error is not None:
            return None
        try:
            self.ensure_open()
        except CameraUnavailableError:
            logger.warning(self.error)
            return None
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.error = (
                f"Camera {self.camera_index}: no frame. "
                "Toggle Camera Frame off/on to retry."
            )
            self.capture.release()
            self.capture = None
            self._has_frame = False
            logger.warning(self.error)
            return None
        self._has_frame = True
        if self.frame_mapper is not None:
            return self.frame_mapper(frame, gain=self.gain)
        return self.frame_to_excitation_grid(
            frame,
            resolution=self.resolution,
            gain=self.gain,
            backend=self.backend,
        )

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.error = None
        self._has_frame = False

    @staticmethod
    def frame_to_excitation_grid(
        frame: np.ndarray,
        *,
        resolution: tuple[int, int],
        gain: float = 1.0,
        backend: ModuleType = np,
    ) -> np.ndarray:
        xp = backend
        values = xp.asarray(frame)
        if values.ndim == 3:
            if values.shape[2] < 3:
                raise ValueError("camera frame must provide at least three channels")
            rgb = values[..., :3][..., ::-1].astype(xp.float32)
        elif values.ndim == 2:
            gray = values.astype(xp.float32)
            rgb = xp.repeat(gray[..., None], 3, axis=2)
        else:
            raise ValueError(
                "camera frame must have shape (rows, cols) or (rows, cols, channels)"
            )
        if rgb.size == 0:
            raise ValueError("camera frame must not be empty")
        if values.dtype.kind in {"u", "i"}:
            rgb = rgb / xp.float32(255.0)
        else:
            max_value = float(xp.nanmax(rgb))
            if max_value > 1.0:
                rgb = rgb / xp.float32(255.0)
        rgb = xp.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = xp.clip(rgb, 0.0, 1.0)
        mapped = CameraFrameSource.resize_grid(rgb, resolution=resolution, backend=xp)
        return (mapped * xp.float32(gain)).astype(xp.float32, copy=False)

    @staticmethod
    def resize_grid(
        values: np.ndarray,
        *,
        resolution: tuple[int, int],
        backend: ModuleType = np,
    ) -> np.ndarray:
        rows, cols = resolution
        src_rows, src_cols = values.shape[:2]
        y_index = np.rint(np.linspace(0, src_rows - 1, rows)).astype(np.int32)
        x_index = np.rint(np.linspace(0, src_cols - 1, cols)).astype(np.int32)
        xp = backend
        values = xp.asarray(values)
        y_index, x_index = xp.asarray(y_index), xp.asarray(x_index)
        return xp.ascontiguousarray(values[y_index[:, None], x_index[None, :]], dtype=xp.float32)
