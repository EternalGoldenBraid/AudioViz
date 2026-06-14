import numpy as np
import pytest

from audioviz.sources.pose import (
    MEDIAPIPE_POSE_CONNECTIONS,
    PoseGraphState,
    adjacency_from_edges,
    iter_adjacency_edges,
    mediapipe_pose_adjacency,
)


def test_pose_graph_state_tracks_velocity_acceleration_and_excitation():
    state = PoseGraphState(2, velocity_smoothing_alpha=1.0)

    state.update(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32), dt=0.5)

    np.testing.assert_allclose(state.get_positions(), [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(state.get_velocities(), [[2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(state.get_accelerations(), [[4.0, 0.0], [0.0, 8.0]])
    np.testing.assert_allclose(state(), [2.0, 4.0])

    state.update(np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32), dt=0.5)

    np.testing.assert_allclose(state.get_velocities(), [[2.0, 0.0], [0.0, -2.0]])
    np.testing.assert_allclose(state.get_accelerations(), [[0.0, 0.0], [0.0, -12.0]])


def test_pose_graph_state_smooths_velocity_and_copies_outputs():
    state = PoseGraphState(1, velocity_smoothing_alpha=0.5)

    state.update(np.array([[1.0, 0.0]], dtype=np.float32), dt=1.0)

    np.testing.assert_allclose(state.get_velocities(), [[0.5, 0.0]])
    positions = state.get_positions()
    positions[:] = 99
    np.testing.assert_allclose(state.get_positions(), [[1.0, 0.0]])


def test_pose_graph_state_smooths_positions_before_velocity_updates():
    state = PoseGraphState(
        1,
        position_smoothing_alpha=0.5,
        velocity_smoothing_alpha=1.0,
    )

    state.update(np.array([[1.0, 0.0]], dtype=np.float32), dt=1.0)
    state.update(np.array([[2.0, 0.0]], dtype=np.float32), dt=1.0)

    np.testing.assert_allclose(state.get_positions(), [[1.5, 0.0]])
    np.testing.assert_allclose(state.get_velocities(), [[0.5, 0.0]])
    np.testing.assert_allclose(state.get_accelerations(), [[-0.5, 0.0]])


def test_pose_graph_state_validates_inputs():
    with pytest.raises(ValueError, match="num_nodes"):
        PoseGraphState(-1)
    with pytest.raises(ValueError, match="position_smoothing_alpha"):
        PoseGraphState(1, position_smoothing_alpha=1.5)
    with pytest.raises(ValueError, match="velocity_smoothing_alpha"):
        PoseGraphState(1, velocity_smoothing_alpha=1.5)
    with pytest.raises(ValueError, match="adjacency shape"):
        PoseGraphState(2, np.zeros((1, 1), dtype=np.float32))

    state = PoseGraphState(1)
    with pytest.raises(ValueError, match="dt"):
        state.update(np.zeros((1, 2), dtype=np.float32), dt=0.0)
    with pytest.raises(ValueError, match="new_positions shape"):
        state.update(np.zeros((2, 2), dtype=np.float32), dt=1.0)
    with pytest.raises(ValueError, match="values shape"):
        state.set_ripple_states(np.zeros(2, dtype=np.float32))


def test_static_mediapipe_adjacency_is_symmetric():
    adjacency = mediapipe_pose_adjacency()

    assert adjacency.shape == (33, 33)
    assert adjacency.dtype == np.float32
    np.testing.assert_array_equal(adjacency, adjacency.T)
    assert set(iter_adjacency_edges(adjacency)) == set(MEDIAPIPE_POSE_CONNECTIONS)


def test_static_adjacency_ignores_edges_outside_node_range():
    adjacency = adjacency_from_edges(3, [(0, 1), (1, 4), (-1, 2)])

    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(adjacency, expected)


def test_mediapipe_extractor_constructs_when_dependency_available():
    mp = pytest.importorskip("mediapipe")
    from audioviz.sources.pose import MediaPipePoseExtractor

    if not hasattr(mp, "solutions"):
        with pytest.raises(ValueError, match="pose landmarker .task model"):
            MediaPipePoseExtractor()
        return

    extractor = MediaPipePoseExtractor()
    extractor.close()
