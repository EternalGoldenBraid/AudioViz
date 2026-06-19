from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audioviz.visualization.offline_pose_ripple import (
    DEFAULT_OUTPUT_DIR,
    OfflinePoseRippleResult,
    run_offline_pose_ripple,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic offline pose+ripple validation and record an animation. "
            "GIF output works without extra codecs; mp4/mov/avi output requires OpenCV."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered PNG frames.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Animation output path (.gif by default, or .mp4/.mov/.avi with OpenCV).",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--synthetic-frequency",
        action="append",
        type=float,
        dest="synthetic_frequencies",
        help="Synthetic ripple frequency in Hz. Repeat to add multiple deterministic sources.",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Disable the synthetic ripple source and validate the pose-coupled path only.",
    )
    parser.add_argument("--pose-nodes", type=int, default=4)
    parser.add_argument(
        "--pose-graph",
        choices=("chain", "ring", "star"),
        default="chain",
        help="Dummy pose graph preset used when --pose-edges is omitted.",
    )
    parser.add_argument(
        "--pose-edges",
        help="Custom pose graph edges like '0-1,1-2,2-3'. Overrides --pose-graph.",
    )
    args = parser.parse_args(argv)

    if args.no_synthetic and args.synthetic_frequencies:
        parser.error("--no-synthetic cannot be combined with --synthetic-frequency")

    synthetic_frequencies = (
        tuple(args.synthetic_frequencies or (440.0,))
        if not args.no_synthetic
        else ()
    )
    result = run_offline_pose_ripple(
        output_dir=args.output_dir,
        video_path=args.video_path,
        frame_count=args.frames,
        resolution=(args.rows, args.cols),
        synthetic_frequencies=synthetic_frequencies,
        pose_nodes=args.pose_nodes,
        pose_graph=args.pose_graph,
        pose_edges=args.pose_edges,
        fps=args.fps,
    )
    _print_summary(result)
    return 0


def _print_summary(result: OfflinePoseRippleResult) -> None:
    print(f"Saved {len(result.frame_paths)} frames to {result.output_dir}")
    print(f"Recorded animation to {result.video_path}")
    print(f"max_abs_by_frame={list(result.max_abs_by_frame)}")
    print(
        "capture_released="
        f"{result.capture_released} extractor_closed={result.extractor_closed}"
    )


if __name__ == "__main__":
    sys.exit(main())
