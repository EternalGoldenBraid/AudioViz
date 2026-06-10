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
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipePoseExtractor requires the optional 'mediapipe' dependency. "
                "Install the pose/demo dependency group to use it."
            ) from exc

        self._mp = mp
        self._mode = "legacy" if hasattr(mp, "solutions") else "tasks"
        if self._mode == "legacy":
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=static_image_mode,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
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
        else:
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            results = self._pose.detect(image)
            landmarks = results.pose_landmarks[0] if results.pose_landmarks else None

        if not landmarks:
            return PoseGraphFrame(
                coords=np.zeros((0, 2), dtype=np.float32),
                adjacency=np.zeros((0, 0), dtype=np.float32),
            )

        coords = np.array(
            [(landmark.x, landmark.y) for landmark in landmarks],
            dtype=np.float32,
        )
        return PoseGraphFrame(coords=coords, adjacency=mediapipe_pose_adjacency(len(coords)))

    def close(self) -> None:
        self._pose.close()
