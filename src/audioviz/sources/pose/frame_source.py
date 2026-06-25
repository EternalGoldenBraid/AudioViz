from __future__ import annotations

from collections.abc import Callable

import numpy as np

from audioviz.sources.camera import load_cv2
from audioviz.sources.pose.base import PoseGraphExtractor, PoseGraphFrame
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor


class PoseFrameSource:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        camera_index: int = 0,
        enabled: bool = False,
        extractor: PoseGraphExtractor | None = None,
        capture=None,
        cv2_loader: Callable = load_cv2,
    ) -> None:
        self.model_path = model_path
        self.camera_index = int(camera_index)
        self.enabled = bool(enabled)
        self.extractor = extractor
        self.capture = capture
        self.cv2_loader = cv2_loader

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self.ensure_open()
        self.enabled = bool(enabled)

    def ensure_open(self) -> None:
        if self.extractor is None:
            self.extractor = MediaPipePoseExtractor(model_path=self.model_path)
        if self.capture is None:
            cv2 = self.cv2_loader()
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                self.extractor.close()
                self.extractor = None
                raise RuntimeError(f"Failed to open pose camera index {self.camera_index}")

    def read(self) -> tuple[np.ndarray, PoseGraphFrame] | None:
        if not self.enabled:
            return None
        self.ensure_open()
        if self.capture is None or self.extractor is None:
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame, self.extractor.extract(frame)

    def close(self) -> None:
        if self.extractor is not None:
            self.extractor.close()
            self.extractor = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
