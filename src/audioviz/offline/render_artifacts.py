"""Frame recording and image/video writing for offline ripple renders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps

ALLOWED_VIDEO_SUFFIXES = {".gif", ".mp4", ".mov", ".avi"}


class RecordingFieldRenderer:
    def __init__(self):
        self.fields: list[np.ndarray] = []

    @property
    def widget(self):  # pragma: no cover - only for renderer interface compatibility.
        return None

    def prepare_frame(self) -> bool:
        return True

    def render(self, canvas: np.ndarray) -> None:
        self.fields.append(np.asarray(canvas).copy())



def write_render_artifacts(
    fields: list[np.ndarray],
    *,
    output_dir: Path,
    video_path: Path,
    fps: float,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    limit = max((float(np.max(np.abs(field))) for field in fields), default=1e-9)
    limit = max(limit, 1e-9)
    rgb_frames = [_field_to_rgb(field, limit=limit) for field in fields]

    frame_paths = []
    for index, rgb_frame in enumerate(rgb_frames, start=1):
        frame_path = output_dir / f"ripple_{index:03d}.png"
        _save_rgb_image(rgb_frame, frame_path)
        frame_paths.append(frame_path)

    _write_animation(rgb_frames, video_path=video_path, fps=fps)
    return tuple(frame_paths)


def _field_to_rgb(field: np.ndarray, *, limit: float) -> np.ndarray:
    if field.ndim == 3:
        normalized = np.clip(field / limit, 0.0, 1.0)
        return np.ascontiguousarray(np.rint(normalized * 255.0).astype(np.uint8))
    normalized = np.clip((field / limit + 1.0) * 0.5, 0.0, 1.0)
    rgba = colormaps["inferno"](normalized, bytes=True)
    return np.ascontiguousarray(rgba[..., :3])


def _save_rgb_image(rgb_frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(rgb_frame, mode="RGB").save(path)


def _write_animation(
    rgb_frames: list[np.ndarray],
    *,
    video_path: Path,
    fps: float,
) -> None:
    suffix = video_path.suffix.lower()
    if suffix == ".gif":
        _write_gif(rgb_frames, video_path=video_path, fps=fps)
        return
    _write_opencv_video(rgb_frames, video_path=video_path, fps=fps)


def _write_gif(
    rgb_frames: list[np.ndarray],
    *,
    video_path: Path,
    fps: float,
) -> None:
    from PIL import Image

    duration_ms = max(int(round(1000.0 / fps)), 1)
    frames = [Image.fromarray(frame, mode="RGB") for frame in rgb_frames]
    frames[0].save(
        video_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _write_opencv_video(
    rgb_frames: list[np.ndarray],
    *,
    video_path: Path,
    fps: float,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for non-GIF video recording. "
            "Use a .gif output path or install opencv-python."
        ) from exc

    height, width = rgb_frames[0].shape[:2]
    fourcc_by_suffix = {
        ".mp4": "mp4v",
        ".mov": "mp4v",
        ".avi": "MJPG",
    }
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*fourcc_by_suffix[video_path.suffix.lower()]),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for frame in rgb_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
