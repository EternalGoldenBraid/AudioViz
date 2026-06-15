from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from audioviz.sources.pose import MediaPipePoseExtractor, PoseGraphState, iter_adjacency_edges


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for the live pose graph demo. "
            "Install the pose-demo dependency group first."
        ) from exc
    return cv2


def _to_pixels(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([width, height], dtype=np.float32)
    return np.rint(coords * scale).astype(np.int32)


def _draw_graph(cv2, frame: np.ndarray, state: PoseGraphState) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    positions = state.get_positions()
    velocities = state.get_velocities()
    accelerations = state.get_accelerations()
    coords_px = _to_pixels(positions, width, height)

    for i, j in iter_adjacency_edges(state.adjacency):
        cv2.line(output, tuple(coords_px[i]), tuple(coords_px[j]), (80, 160, 255), 2)

    velocity_norm = np.linalg.norm(velocities, axis=1)
    acceleration_norm = np.linalg.norm(accelerations, axis=1)
    max_velocity = max(float(velocity_norm.max(initial=0.0)), 1e-6)
    max_acceleration = max(float(acceleration_norm.max(initial=0.0)), 1e-6)

    for idx, (x, y) in enumerate(coords_px):
        velocity_ratio = float(np.clip(velocity_norm[idx] / max_velocity, 0.0, 1.0))
        acceleration_ratio = float(np.clip(acceleration_norm[idx] / max_acceleration, 0.0, 1.0))
        color = (int(255 * (1.0 - velocity_ratio)), 64, int(255 * velocity_ratio))
        radius = int(5 + 14 * acceleration_ratio)
        origin = (int(x), int(y))

        cv2.circle(output, origin, radius, color, -1)
        cv2.circle(output, origin, radius, (255, 255, 255), 1)

        velocity_tip = tuple(np.rint(coords_px[idx] + velocities[idx] * 0.08 * [width, height]).astype(np.int32))
        acceleration_tip = tuple(np.rint(coords_px[idx] + accelerations[idx] * 0.008 * [width, height]).astype(np.int32))
        cv2.arrowedLine(output, origin, velocity_tip, (255, 255, 0), 1, tipLength=0.25)
        cv2.arrowedLine(output, origin, acceleration_tip, (0, 255, 255), 1, tipLength=0.25)

    cv2.putText(
        output,
        "Pose graph: edges=orange, velocity=blue/red + cyan arrows, acceleration=size + yellow arrows",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def run(
    camera_index: int,
    width: int | None,
    height: int | None,
    smoothing: float,
    model_path: str | None,
) -> int:
    cv2 = _load_cv2()
    extractor = MediaPipePoseExtractor(model_path=model_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        extractor.close()
        raise SystemExit(f"Failed to open camera index {camera_index}")

    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    state: PoseGraphState | None = None
    last_time = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            now = time.monotonic()
            dt = max(now - last_time, 1e-6)
            last_time = now

            pose = extractor.extract(frame)
            output = frame
            if pose.coords.size:
                if state is None or state.num_nodes != len(pose.coords):
                    state = PoseGraphState(
                        len(pose.coords),
                        pose.adjacency,
                        velocity_smoothing_alpha=smoothing,
                    )
                state.update(pose.coords, dt)
                output = _draw_graph(cv2, frame, state)

            cv2.imshow("AudioViz pose graph demo", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    finally:
        extractor.close()
        capture.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize a live MediaPipe pose graph from a camera.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--smoothing", type=float, default=0.8)
    parser.add_argument(
        "--model-path",
        help=(
            "Path to a MediaPipe pose_landmarker.task model. Required for "
            "MediaPipe versions that use the tasks API instead of mp.solutions."
        ),
    )
    args = parser.parse_args(argv)

    return run(
        args.camera_index,
        args.width,
        args.height,
        args.smoothing,
        args.model_path,
    )


if __name__ == "__main__":
    sys.exit(main())
