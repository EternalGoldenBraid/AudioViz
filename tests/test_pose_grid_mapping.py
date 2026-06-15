import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.sources.pose import (
    PoseGraphState,
    centered_field_rect,
    normalized_pose_coords_to_source_positions,
    pose_coords_in_image_support,
    pose_graph_state_to_ripple_sources,
)


def test_normalized_pose_coords_map_to_full_ripple_field_with_horizontal_mirroring():
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
            [199.0, 0.0],
            [99.5, 49.5],
            [0.0, 99.0],
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
            [149.0, 25.0],
            [50.0, 74.0],
        ],
    )


def test_pose_coords_in_image_support_marks_only_in_frame_landmarks():
    mask = pose_coords_in_image_support(
        np.array(
            [[0.0, 0.0], [1.0, 1.0], [-0.1, 0.5], [0.5, 1.1], [np.nan, 0.2]],
            dtype=np.float32,
        )
    )

    np.testing.assert_array_equal(mask, [True, True, False, False, False])


def test_normalized_pose_mapping_filters_landmarks_outside_field_support():
    positions = normalized_pose_coords_to_source_positions(
        np.array([[-1.0, 0.5], [0.5, 0.5], [2.0, 2.0]], dtype=np.float32),
        resolution=(10, 20),
        field_rect=(5.0, 2.0, 10.0, 5.0),
    )

    np.testing.assert_allclose(
        positions,
        [
            [9.5, 4.0],
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


def test_pose_graph_state_maps_acceleration_norm_to_source_excitations():
    state = PoseGraphState(2, velocity_smoothing_alpha=1.0)
    state.update(np.array([[0.1, 0.2], [0.8, 0.5]], dtype=np.float32), dt=1.0)
    state.update(np.array([[0.3, 0.2], [0.8, 0.8]], dtype=np.float32), dt=1.0)

    positions, excitations = pose_graph_state_to_ripple_sources(
        state,
        resolution=(10, 20),
        acceleration_scale=2.0,
    )

    np.testing.assert_allclose(positions, [[13.3, 1.8], [3.8, 7.2]])
    np.testing.assert_allclose(excitations, [0.4472136, 1.6492423], atol=1e-6)


def test_pose_graph_state_excitations_can_be_clipped():
    state = PoseGraphState(1, velocity_smoothing_alpha=1.0)
    state.update(np.array([[1.0, 0.0]], dtype=np.float32), dt=1.0)

    _, excitations = pose_graph_state_to_ripple_sources(
        state,
        resolution=(10, 10),
        acceleration_scale=10.0,
        max_excitation=3.0,
    )

    np.testing.assert_allclose(excitations, [3.0])


def test_pose_graph_state_filters_out_of_frame_landmarks_from_ripple_sources():
    state = PoseGraphState(3, velocity_smoothing_alpha=1.0)
    state.update(
        np.array(
            [
                [0.25, 0.5],
                [-0.1, 0.2],
                [1.1, 0.3],
            ],
            dtype=np.float32,
        ),
        dt=1.0,
    )

    positions, excitations = pose_graph_state_to_ripple_sources(
        state,
        resolution=(10, 20),
        acceleration_scale=1.0,
    )

    np.testing.assert_allclose(positions, [[14.25, 4.5]])
    np.testing.assert_allclose(excitations, [np.sqrt(0.25**2 + 0.5**2)])


def test_ripple_engine_steps_direct_source_excitations():
    engine = RippleEngine(
        resolution=(5, 6),
        plane_size_m=(1.0, 1.0),
        n_sources=1,
        speed=1.0,
        damping=1.0,
        amplitude=2.0,
        use_gpu=False,
    )
    engine.set_source_positions(np.array([[2.0, 3.0], [2.4, 3.4]], dtype=np.float32))

    engine.step_source_excitations(np.array([1.0, 3.0], dtype=np.float32))

    state = engine.get_field_numpy()
    assert engine.n_sources == 2
    assert state[3, 1] > 0.0
    assert state[3, 3] > 0.0
    np.testing.assert_allclose(engine.propagator.Z_old[3, 2], 8.0)


def test_ripple_engine_validates_direct_source_excitations_shape():
    engine = RippleEngine(
        resolution=(5, 6),
        plane_size_m=(1.0, 1.0),
        n_sources=2,
        speed=1.0,
        use_gpu=False,
    )

    with pytest.raises(ValueError, match="source excitations"):
        engine.step_source_excitations(np.ones(1, dtype=np.float32))


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
