import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.sources.pose import (
    centered_field_rect,
    normalized_pose_coords_to_source_positions,
)


def test_normalized_pose_coords_map_to_full_ripple_field():
    coords = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    positions = normalized_pose_coords_to_source_positions(
        coords,
        resolution=(100, 200),
    )

    np.testing.assert_allclose(
        positions,
        [
            [0.0, 0.0],
            [99.5, 49.5],
            [199.0, 99.0],
        ],
    )


def test_normalized_pose_coords_can_map_to_centered_field_rect():
    field_rect = centered_field_rect(
        (100, 200),
        width_fraction=0.5,
        height_fraction=0.5,
    )

    positions = normalized_pose_coords_to_source_positions(
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        resolution=(100, 200),
        field_rect=field_rect,
    )

    np.testing.assert_allclose(
        positions,
        [
            [50.0, 25.0],
            [149.0, 74.0],
        ],
    )


def test_normalized_pose_mapping_clips_landmarks_to_field_rect():
    positions = normalized_pose_coords_to_source_positions(
        np.array([[-1.0, 0.5], [2.0, 2.0]], dtype=np.float32),
        resolution=(10, 20),
        field_rect=(5.0, 2.0, 10.0, 5.0),
    )

    np.testing.assert_allclose(
        positions,
        [
            [5.0, 4.0],
            [14.0, 6.0],
        ],
    )


def test_normalized_pose_mapping_validates_input_shape_and_rect():
    with pytest.raises(ValueError, match="coords"):
        normalized_pose_coords_to_source_positions(
            np.zeros((2, 3), dtype=np.float32),
            resolution=(10, 10),
        )

    with pytest.raises(ValueError, match="field_rect"):
        normalized_pose_coords_to_source_positions(
            np.zeros((1, 2), dtype=np.float32),
            resolution=(10, 10),
            field_rect=(9.0, 0.0, 2.0, 2.0),
        )


def test_ripple_engine_accepts_pose_mapped_source_positions():
    engine = RippleEngine(
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        n_sources=1,
        speed=10.0,
        use_gpu=False,
    )
    positions = np.array([[0.0, 0.0], [19.0, 9.0]], dtype=np.float32)

    engine.set_source_positions(positions)

    assert engine.n_sources == 2
    np.testing.assert_allclose(engine.source_positions, positions)
    engine.step(np.full((2, 1), 5.0, dtype=np.float32))


def test_ripple_engine_source_positions_use_xy_field_bounds():
    engine = RippleEngine(
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        n_sources=64,
        speed=10.0,
        use_gpu=False,
    )

    assert np.all(engine.source_positions[:, 0] >= 0.0)
    assert np.all(engine.source_positions[:, 0] <= 19.0)
    assert np.all(engine.source_positions[:, 1] >= 0.0)
    assert np.all(engine.source_positions[:, 1] <= 9.0)

    with pytest.raises(ValueError, match="field width"):
        engine.set_source_positions(np.array([[20.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="field height"):
        engine.set_source_positions(np.array([[0.0, 10.0]], dtype=np.float32))
