import os
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audioviz.visualization.offline_audio_ripple import run_offline_audio_ripple


_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "audio_ripple_validation"
    / "test_offline_audio_ripple"
)


def _clear(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _clear_root_if_empty() -> None:
    if _OUTPUT_ROOT.exists() and not any(_OUTPUT_ROOT.iterdir()):
        _OUTPUT_ROOT.rmdir()


def test_offline_audio_ripple_linear_mapping_pipeline_writes_frames_and_gif():
    output_dir = _OUTPUT_ROOT / "linear_mapping_pipeline"
    video_path = _OUTPUT_ROOT / "linear_mapping_pipeline.gif"
    _clear(output_dir)
    if video_path.exists():
        video_path.unlink()

    sr = 48_000
    duration_s = 0.12
    time = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    tone_hz = 440.0
    audio = 0.05 * np.sin(2.0 * np.pi * tone_hz * time)

    try:
        result = run_offline_audio_ripple(
            audio_samples=audio,
            sr=sr,
            output_dir=output_dir,
            video_path=video_path,
            resolution=(24, 32),
            fps=12.0,
            io_blocksize=2048,
            audio_visual_mapping_mode="linear",
            audio_visual_linear_scale=0.05,
            audio_visual_linear_offset=0.0,
        )

        assert len(result.frame_paths) >= 2
        assert all(path.exists() for path in result.frame_paths)
        assert result.video_path.exists()
        assert result.video_path.stat().st_size > 0
        assert any(value > 1e-9 for value in result.max_abs_by_frame)
        assert any(frame for frame in result.detected_audio_frequencies_by_frame)
        assert any(frame for frame in result.mapped_visual_frequencies_by_frame)

        mapped_pairs = [
            (detected, mapped)
            for detected, mapped in zip(
                result.detected_audio_frequencies_by_frame,
                result.mapped_visual_frequencies_by_frame,
            )
            if detected and mapped
        ]
        assert mapped_pairs
        for detected, mapped in mapped_pairs:
            assert len(detected) == len(mapped)
            np.testing.assert_allclose(
                np.asarray(mapped, dtype=np.float32),
                0.05 * np.asarray(detected, dtype=np.float32),
                rtol=1e-5,
                atol=1e-5,
            )
    finally:
        _clear(output_dir)
        if video_path.exists():
            video_path.unlink()
        _clear_root_if_empty()
