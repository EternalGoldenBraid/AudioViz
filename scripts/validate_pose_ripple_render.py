from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audioviz.sources.pose import PoseGraphFrame, adjacency_from_edges
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


class SyntheticPoseCapture:
    def __init__(self, frame_count: int):
        self._remaining = frame_count
        self.released = False

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((64, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class SyntheticPoseExtractor:
    def __init__(self, frame_count: int):
        self._frame_count = frame_count
        self._index = 0
        self._adjacency = adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3)])
        self.closed = False

    def extract(self, _frame) -> PoseGraphFrame:
        phase = self._index / max(self._frame_count - 1, 1)
        self._index += 1
        coords = np.array(
            [
                [0.25, 0.35],
                [0.45 + 0.25 * phase, 0.42],
                [0.55, 0.58 + 0.20 * np.sin(phase * np.pi)],
                [0.75 - 0.20 * phase, 0.70],
            ],
            dtype=np.float32,
        )
        return PoseGraphFrame(coords=coords, adjacency=self._adjacency)

    def close(self) -> None:
        self.closed = True


class SavingFieldRenderer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fields: list[np.ndarray] = []

    @property
    def widget(self):  # pragma: no cover - only needed to match renderer shape.
        return None

    def prepare_frame(self) -> bool:
        return True

    def render(self, field_source) -> None:
        field = field_source.get_field_numpy().copy()
        self.fields.append(field)
        _save_field_png(field, self.output_dir / f"pose_ripple_{len(self.fields):03d}.png")


def _save_field_png(field: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    max_abs = float(np.max(np.abs(field)))
    limit = max(max_abs, 1e-9)
    plt.imsave(path, field, cmap="inferno", vmin=-limit, vmax=limit)


def run(
    *,
    output_dir: Path,
    frame_count: int,
    resolution: tuple[int, int],
    acceleration_scale: float,
    max_excitation: float,
) -> int:
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = SyntheticPoseCapture(frame_count)
    extractor = SyntheticPoseExtractor(frame_count)
    renderer = SavingFieldRenderer(output_dir)
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=resolution,
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=0.995,
        amplitude=1.0,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_acceleration_scale=acceleration_scale,
        pose_max_excitation=max_excitation,
    )
    visualizer.timer.stop()
    visualizer.renderer = renderer

    for _ in range(frame_count):
        visualizer.update_visualization()
        app.processEvents()

    visualizer.close_pose_sources()
    if len(renderer.fields) != frame_count:
        raise RuntimeError(
            f"Rendered {len(renderer.fields)} frames; expected {frame_count}."
        )

    max_abs_values = [float(np.max(np.abs(field))) for field in renderer.fields]
    active_frames = sum(value > 1e-9 for value in max_abs_values[1:])
    if active_frames == 0:
        raise RuntimeError("Pose-driven ripple validation stayed blank after frame 1.")

    print(f"Saved {len(renderer.fields)} frames to {output_dir}")
    print(f"max_abs_by_frame={max_abs_values}")
    print(f"capture_released={capture.released} extractor_closed={extractor.closed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render deterministic synthetic pose motion through the ripple visualizer."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pose_ripple_validation"),
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--acceleration-scale", type=float, default=0.01)
    parser.add_argument("--max-excitation", type=float, default=0.05)
    args = parser.parse_args(argv)

    return run(
        output_dir=args.output_dir,
        frame_count=args.frames,
        resolution=(args.rows, args.cols),
        acceleration_scale=args.acceleration_scale,
        max_excitation=args.max_excitation,
    )


if __name__ == "__main__":
    sys.exit(main())
