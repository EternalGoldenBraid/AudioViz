import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from audioviz.visualization.offline_pose_ripple import DummyPoseExtractor, run_offline_pose_ripple


_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "pose_ripple_validation"
    / "test_offline_pose_ripple"
)


def _clear(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _clear_root_if_empty() -> None:
    if _OUTPUT_ROOT.exists() and not any(_OUTPUT_ROOT.iterdir()):
        _OUTPUT_ROOT.rmdir()


def test_offline_pose_ripple_writes_frames_and_gif():
    output_dir = _OUTPUT_ROOT / "with_synthetic"
    video_path = _OUTPUT_ROOT / "with_synthetic.gif"
    _clear(output_dir)
    if video_path.exists():
        video_path.unlink()

    try:
        result = run_offline_pose_ripple(
            output_dir=output_dir,
            video_path=video_path,
            frame_count=4,
            resolution=(24, 32),
            synthetic_frequencies=(220.0, 330.0),
            pose_nodes=5,
            pose_graph="ring",
            fps=8.0,
        )

        assert len(result.frame_paths) == 4
        assert all(path.exists() for path in result.frame_paths)
        assert result.video_path.exists()
        assert result.video_path.stat().st_size > 0
        assert any(value > 1e-9 for value in result.max_abs_by_frame[1:])
        assert result.capture_released
        assert result.extractor_closed
    finally:
        _clear(output_dir)
        if video_path.exists():
            video_path.unlink()
        _clear_root_if_empty()


def test_offline_pose_ripple_runs_pose_only_with_custom_edges():
    output_dir = _OUTPUT_ROOT / "pose_only"
    video_path = _OUTPUT_ROOT / "pose_only.gif"
    _clear(output_dir)
    if video_path.exists():
        video_path.unlink()

    try:
        result = run_offline_pose_ripple(
            output_dir=output_dir,
            video_path=video_path,
            frame_count=4,
            resolution=(20, 28),
            synthetic_frequencies=(),
            pose_nodes=4,
            pose_edges="0-1,1-2,1-3",
            fps=6.0,
        )

        assert len(result.frame_paths) == 4
        assert result.video_path.exists()
        assert result.video_path.stat().st_size > 0
        assert all(value <= 1e-9 for value in result.max_abs_by_frame)
        assert result.capture_released
        assert result.extractor_closed
    finally:
        _clear(output_dir)
        if video_path.exists():
            video_path.unlink()
        _clear_root_if_empty()


def test_dummy_pose_extractor_builds_segmentation_mask_from_dummy_graph():
    extractor = DummyPoseExtractor(
        frame_count=2,
        frame_shape=(20, 28),
        node_count=4,
        edges=[(0, 1), (1, 2), (2, 3)],
    )

    frame = extractor.extract(np.zeros((20, 28, 3), dtype=np.uint8))

    assert frame.segmentation_mask is not None
    assert frame.segmentation_mask.shape == (20, 28)
    assert np.count_nonzero(frame.segmentation_mask) > 0
