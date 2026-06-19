from types import SimpleNamespace

import numpy as np
import pytest

from audioviz.sources.pose import PoseGraphFrame, adjacency_from_edges
from audioviz.sources.pose.adjacency import mediapipe_pose_adjacency
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor


def test_pose_graph_frame_preserves_constructor_behavior():
    coords = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32)
    adjacency = adjacency_from_edges(2, [(0, 1)])
    segmentation_mask = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    frame = PoseGraphFrame(
        coords=coords,
        adjacency=adjacency,
        segmentation_mask=segmentation_mask,
    )

    np.testing.assert_allclose(frame.coords, coords)
    np.testing.assert_array_equal(frame.adjacency, adjacency)
    np.testing.assert_array_equal(frame.segmentation_mask, segmentation_mask)
    assert frame.as_dict()["coords"].shape == (2, 2)


def test_pose_graph_frame_reuses_backing_arrays_across_updates():
    frame = PoseGraphFrame.empty(4, adjacency=adjacency_from_edges(4, [(0, 1), (2, 3)]))
    coords_buffer = frame.coords_buffer
    adjacency_buffer = frame.adjacency_buffer

    frame.update(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    first_coords_view = frame.coords
    first_adjacency_view = frame.adjacency

    np.testing.assert_allclose(first_coords_view, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(first_adjacency_view, [[0.0, 1.0], [1.0, 0.0]])

    frame.update(
        np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    )

    assert frame.coords_buffer is coords_buffer
    assert frame.adjacency_buffer is adjacency_buffer
    assert np.shares_memory(frame.coords, coords_buffer)
    assert np.shares_memory(frame.adjacency, adjacency_buffer)
    np.testing.assert_allclose(frame.coords, [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])


def test_pose_graph_frame_updates_from_landmark_like_points_without_new_frame():
    frame = PoseGraphFrame.empty(3, adjacency=adjacency_from_edges(3, [(0, 2)]))
    coords_buffer = frame.coords_buffer

    frame.update_xy(
        [
            SimpleNamespace(x=0.1, y=0.2),
            SimpleNamespace(x=0.3, y=0.4),
        ]
    )

    assert frame.coords_buffer is coords_buffer
    np.testing.assert_allclose(frame.coords, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(frame.adjacency, [[0.0, 0.0], [0.0, 0.0]])

    frame.clear()

    assert frame.coords.shape == (0, 2)
    assert frame.adjacency.shape == (0, 0)
    assert frame.coords_buffer is coords_buffer


def test_pose_graph_frame_validates_capacity_and_shapes():
    frame = PoseGraphFrame.empty(1)

    with pytest.raises(ValueError, match="coords exceed max_nodes"):
        frame.update(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="coords shape"):
        frame.update(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="adjacency shape"):
        frame.update(np.zeros((1, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="segmentation_mask"):
        frame.set_segmentation_mask(np.zeros((1, 2, 3), dtype=np.float32))


def test_mediapipe_extractor_reuses_frame_for_repeated_extracts():
    extractor = MediaPipePoseExtractor.__new__(MediaPipePoseExtractor)
    extractor._mode = "legacy"
    extractor._frame = PoseGraphFrame.empty(33, adjacency=mediapipe_pose_adjacency(33))
    extractor._pose = _FakePose(
        [
            (
                [
                    SimpleNamespace(x=0.1, y=0.2),
                    SimpleNamespace(x=0.3, y=0.4),
                ],
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            ),
            (
                [
                    SimpleNamespace(x=0.5, y=0.6),
                    SimpleNamespace(x=0.7, y=0.8),
                ],
                np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            ),
            ([], None),
        ]
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    first = extractor.extract(frame)
    coords_buffer = first.coords_buffer
    adjacency_buffer = first.adjacency_buffer
    second = extractor.extract(frame)
    second_segmentation_mask = np.array(second.segmentation_mask, copy=True)
    empty = extractor.extract(frame)

    assert first is second is empty
    assert second.coords_buffer is coords_buffer
    assert second.adjacency_buffer is adjacency_buffer
    np.testing.assert_allclose(coords_buffer[:2], [[0.5, 0.6], [0.7, 0.8]])
    np.testing.assert_array_equal(second_segmentation_mask, [[1.0, 1.0], [0.0, 0.0]])
    assert empty.coords.shape == (0, 2)
    assert empty.adjacency.shape == (0, 0)
    assert empty.segmentation_mask is None


class _FakePose:
    def __init__(self, landmark_batches):
        self._landmark_batches = iter(landmark_batches)

    def process(self, _frame):
        landmarks, segmentation_mask = next(self._landmark_batches)
        pose_landmarks = None
        if landmarks:
            pose_landmarks = SimpleNamespace(landmark=landmarks)
        return SimpleNamespace(
            pose_landmarks=pose_landmarks,
            segmentation_mask=segmentation_mask,
        )
