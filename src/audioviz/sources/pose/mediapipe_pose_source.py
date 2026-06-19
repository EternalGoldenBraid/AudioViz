from __future__ import annotations

import numpy as np

from audioviz.sources.pose.adjacency import mediapipe_pose_adjacency
from audioviz.sources.pose.base import PoseGraphExtractor, PoseGraphFrame


class MediaPipePoseExtractor(PoseGraphExtractor):
    def __init__(
        self,
        *,
        model_path: str | None = None,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        array_module=np,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipePoseExtractor requires the optional 'mediapipe' dependency. "
                "Install the pose/demo dependency group to use it."
            ) from exc

        self._mp = mp
        self._frame = PoseGraphFrame.empty(
            33,
            adjacency=mediapipe_pose_adjacency(33),
            array_module=array_module,
        )
        self._mode = "legacy" if hasattr(mp, "solutions") else "tasks"
        if self._mode == "legacy":
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=static_image_mode,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                enable_segmentation=True,
            )
        else:
            if model_path is None:
                raise ValueError(
                    "This MediaPipe version requires a pose landmarker .task model. "
                    "Pass model_path or run scripts/pose_graph_demo.py --model-path PATH."
                )
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._pose = vision.PoseLandmarker.create_from_options(options)

    def extract(self, frame: np.ndarray) -> PoseGraphFrame:
        rgb_frame = np.ascontiguousarray(frame[..., ::-1])
        if self._mode == "legacy":
            results = self._pose.process(rgb_frame)
            landmarks = (
                results.pose_landmarks.landmark if results.pose_landmarks else None
            )
            segmentation_mask = (
                np.asarray(results.segmentation_mask, dtype=np.float32)
                if getattr(results, "segmentation_mask", None) is not None
                else None
            )
        else:
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            results = self._pose.detect(image)
            landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
            segmentation_mask = None

        if not landmarks:
            self._frame.clear()
            self._frame.set_segmentation_mask(segmentation_mask)
            return self._frame

        self._frame.update_xy(landmarks)
        self._frame.set_segmentation_mask(segmentation_mask)
        return self._frame

    def close(self) -> None:
        self._pose.close()
