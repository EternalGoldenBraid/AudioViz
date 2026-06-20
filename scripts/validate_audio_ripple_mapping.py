from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import librosa as lr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audioviz.utils.signal_processing import SUPPORTED_AUDIO_VISUAL_MAPPING_MODES
from audioviz.visualization.offline_audio_ripple import (
    DEFAULT_OUTPUT_DIR,
    OfflineAudioRippleResult,
    run_offline_audio_ripple,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline audio-driven ripple validation from an audio sample and "
            "record an animation."
        )
    )
    parser.add_argument("audio_path", type=Path, help="Audio file to analyze.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--io-blocksize", type=int, default=2048)
    parser.add_argument("--mapping-mode", choices=SUPPORTED_AUDIO_VISUAL_MAPPING_MODES, default="legacy")
    parser.add_argument("--mapping-alpha", type=float, default=50.0)
    parser.add_argument("--mapping-f0", type=float, default=50.0)
    parser.add_argument("--mapping-fc", type=float, default=2000.0)
    parser.add_argument("--linear-scale", type=float, default=0.05)
    parser.add_argument("--linear-offset", type=float, default=0.0)
    args = parser.parse_args(argv)

    samples, sr = lr.load(args.audio_path, sr=None, mono=True)
    result = run_offline_audio_ripple(
        audio_samples=samples,
        sr=int(sr),
        output_dir=args.output_dir,
        video_path=args.video_path,
        resolution=(args.rows, args.cols),
        fps=args.fps,
        io_blocksize=args.io_blocksize,
        audio_visual_mapping_mode=args.mapping_mode,
        audio_visual_mapping_alpha=args.mapping_alpha,
        audio_visual_mapping_f0=args.mapping_f0,
        audio_visual_mapping_fc=args.mapping_fc,
        audio_visual_linear_scale=args.linear_scale,
        audio_visual_linear_offset=args.linear_offset,
    )
    _print_summary(result)
    return 0


def _print_summary(result: OfflineAudioRippleResult) -> None:
    print(f"Saved {len(result.frame_paths)} frames to {result.output_dir}")
    print(f"Recorded animation to {result.video_path}")
    print(f"max_abs_by_frame={list(result.max_abs_by_frame)}")
    print(
        "mapped_visual_frequencies_by_frame="
        f"{list(result.mapped_visual_frequencies_by_frame)}"
    )


if __name__ == "__main__":
    sys.exit(main())
