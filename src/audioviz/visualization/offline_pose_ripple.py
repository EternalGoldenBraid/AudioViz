from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from audioviz.sources.pose import (
    PoseGraphFrame,
    adjacency_from_edges,
    build_pose_graph_segmentation_mask,
)
from audioviz.visualization.ripple_wave_visualizer import (
    POSE_RENDER_MODE_OVERLAY,
    RippleWaveVisualizer,
    normalize_pose_render_mode,
)

DEFAULT_OUTPUT_DIR = Path("outputs/pose_ripple_validation")
DEFAULT_VIDEO_NAME = "pose_ripple_validation.gif"
_ALLOWED_VIDEO_SUFFIXES = {".gif", ".mp4", ".mov", ".avi"}


@dataclass(frozen=True)
class OfflinePoseRippleResult:
    output_dir: Path
    video_path: Path
    frame_paths: tuple[Path, ...]
    max_abs_by_frame: tuple[float, ...]
    capture_released: bool
    extractor_closed: bool


class SyntheticPoseCapture:
    def __init__(self, frame_count: int, frame_shape: tuple[int, int, int]):
        self._remaining = frame_count
        self._frame = self._build_frame(frame_shape)
        self.released = False

    @staticmethod
    def _build_frame(frame_shape: tuple[int, int, int]) -> np.ndarray:
        height, width, channels = frame_shape
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        frame = np.zeros((height, width, channels), dtype=np.uint8)
        frame[..., 0] = np.rint(xx * 96).astype(np.uint8)
        frame[..., 1] = np.rint(yy * 128).astype(np.uint8)
        frame[..., 2] = np.rint((1.0 - xx) * 160).astype(np.uint8)
        return frame

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


class DummyPoseExtractor:
    def __init__(
        self,
        frame_count: int,
        *,
        frame_shape: tuple[int, int],
        node_count: int,
        edges: list[tuple[int, int]],
        horizontal_motion: float = 0.12,
        vertical_motion: float = 0.08,
    ):
        if node_count <= 0:
            raise ValueError("node_count must be positive")
        self._frame_count = frame_count
        self._index = 0
        self._frame_shape = frame_shape
        self._horizontal_motion = horizontal_motion
        self._vertical_motion = vertical_motion
        self._base_coords = self._make_base_coords(node_count)
        self._node_phase = np.linspace(0.0, np.pi, node_count, dtype=np.float32)
        self._adjacency = adjacency_from_edges(node_count, edges)
        self.closed = False

    @staticmethod
    def _make_base_coords(node_count: int) -> np.ndarray:
        xs = np.linspace(0.2, 0.8, node_count, dtype=np.float32)
        ys = 0.5 + 0.18 * np.sin(np.linspace(0.0, np.pi, node_count, dtype=np.float32))
        return np.column_stack((xs, ys)).astype(np.float32)

    def extract(self, _frame) -> PoseGraphFrame:
        phase = 2.0 * np.pi * self._index / max(self._frame_count - 1, 1)
        self._index += 1
        coords = self._base_coords.copy()
        coords[:, 0] += self._horizontal_motion * np.sin(phase + self._node_phase)
        coords[:, 1] += self._vertical_motion * np.cos(phase * 1.5 + 0.5 * self._node_phase)
        coords = np.clip(coords, 0.05, 0.95)
        segmentation_mask = self._build_segmentation_mask(coords)
        return PoseGraphFrame(
            coords=coords,
            adjacency=self._adjacency,
            segmentation_mask=segmentation_mask,
        )

    def close(self) -> None:
        self.closed = True

    def _build_segmentation_mask(self, coords: np.ndarray) -> np.ndarray:
        return build_pose_graph_segmentation_mask(
            coords,
            self._adjacency,
            self._frame_shape,
        )


class RecordingFieldRenderer:
    def __init__(self):
        self.fields: list[np.ndarray] = []

    @property
    def widget(self):  # pragma: no cover - only for renderer interface compatibility.
        return None

    def prepare_frame(self) -> bool:
        return True

    def render(self, field_source) -> None:
        self.fields.append(field_source.get_field_numpy().copy())


def resolve_pose_edges(
    node_count: int,
    *,
    pose_graph: str,
    pose_edges: str | None,
) -> list[tuple[int, int]]:
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if pose_edges:
        return _parse_pose_edges(pose_edges, node_count)
    if pose_graph == "chain":
        return [(index, index + 1) for index in range(node_count - 1)]
    if pose_graph == "ring":
        if node_count == 1:
            return []
        return [(index, (index + 1) % node_count) for index in range(node_count)]
    if pose_graph == "star":
        return [(0, index) for index in range(1, node_count)]
    raise ValueError(f"Unsupported pose_graph preset: {pose_graph}")


def run_offline_pose_ripple(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    video_path: Path | None = None,
    frame_count: int = 8,
    resolution: tuple[int, int] = (96, 128),
    synthetic_frequencies: tuple[float, ...] = (440.0,),
    pose_nodes: int = 4,
    pose_graph: str = "chain",
    pose_edges: str | None = None,
    pose_render_mode: str = POSE_RENDER_MODE_OVERLAY,
    fps: float = 12.0,
) -> OfflinePoseRippleResult:
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2 for pose-motion validation")
    if fps <= 0:
        raise ValueError("fps must be positive")
    resolved_pose_render_mode = normalize_pose_render_mode(pose_render_mode)

    resolved_output_dir = Path(output_dir)
    resolved_video_path = Path(video_path) if video_path is not None else resolved_output_dir / DEFAULT_VIDEO_NAME
    if resolved_video_path.suffix.lower() not in _ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(
            f"video_path must use one of {sorted(_ALLOWED_VIDEO_SUFFIXES)}"
        )

    edges = resolve_pose_edges(
        pose_nodes,
        pose_graph=pose_graph,
        pose_edges=pose_edges,
    )
    capture = SyntheticPoseCapture(frame_count, (resolution[0], resolution[1], 3))
    extractor = DummyPoseExtractor(
        frame_count,
        frame_shape=resolution,
        node_count=pose_nodes,
        edges=edges,
    )
    renderer = RecordingFieldRenderer()

    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        n_sources=max(len(synthetic_frequencies), 1),
        resolution=resolution,
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=0.995,
        amplitude=1.0,
        frequency=list(synthetic_frequencies) if synthetic_frequencies else 440.0,
        use_synthetic=bool(synthetic_frequencies),
        use_pose_sources=True,
        pose_render_mode=resolved_pose_render_mode,
        pose_capture=capture,
        pose_extractor=extractor,
    )
    visualizer.timer.stop()
    visualizer.renderer = renderer

    try:
        for _ in range(frame_count):
            visualizer.update_visualization()
            app.processEvents()
    finally:
        visualizer.close()
        app.processEvents()

    if len(renderer.fields) != frame_count:
        raise RuntimeError(
            f"Rendered {len(renderer.fields)} frames; expected {frame_count}."
        )

    max_abs_by_frame = tuple(float(np.max(np.abs(field))) for field in renderer.fields)
    if synthetic_frequencies and not any(value > 1e-9 for value in max_abs_by_frame[1:]):
        raise RuntimeError("Pose-driven ripple validation stayed blank after frame 1.")

    frame_paths = _write_render_artifacts(
        renderer.fields,
        output_dir=resolved_output_dir,
        video_path=resolved_video_path,
        fps=fps,
    )
    return OfflinePoseRippleResult(
        output_dir=resolved_output_dir,
        video_path=resolved_video_path,
        frame_paths=frame_paths,
        max_abs_by_frame=max_abs_by_frame,
        capture_released=capture.released,
        extractor_closed=extractor.closed,
    )



def _write_render_artifacts(
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
        frame_path = output_dir / f"pose_ripple_{index:03d}.png"
        _save_rgb_image(rgb_frame, frame_path)
        frame_paths.append(frame_path)

    _write_animation(rgb_frames, video_path=video_path, fps=fps)
    return tuple(frame_paths)


def _field_to_rgb(field: np.ndarray, *, limit: float) -> np.ndarray:
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
            "Use a .gif output path or install the pose-demo dependency group."
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


def _parse_pose_edges(edge_spec: str, node_count: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for token in edge_spec.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(
                f"Invalid pose edge '{part}'. Expected pairs like '0-1,1-2'."
            )
        start_text, end_text = part.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if start == end:
            raise ValueError("Pose edges must connect two distinct nodes")
        if not 0 <= start < node_count or not 0 <= end < node_count:
            raise ValueError(
                f"Pose edge '{part}' references nodes outside 0..{node_count - 1}"
            )
        edge = tuple(sorted((start, end)))
        if edge not in edges:
            edges.append(edge)
    return edges
