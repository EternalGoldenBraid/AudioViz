import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audioviz.sources.pose import PoseGraphFrame, adjacency_from_edges


class _FakeCapture:
    def __init__(self, frame_count: int = 2):
        self.frames = [
            np.zeros((4, 4, 3), dtype=np.uint8)
            for _ in range(frame_count)
        ]
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class _FakeExtractor:
    def __init__(self):
        self.frames = [
            PoseGraphFrame(
                coords=np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
                adjacency=adjacency_from_edges(2, [(0, 1)]),
            ),
            PoseGraphFrame(
                coords=np.array([[0.25, 0.25], [0.95, 0.75]], dtype=np.float32),
                adjacency=adjacency_from_edges(2, [(0, 1)]),
            ),
        ]
        self.closed = False

    def extract(self, _frame):
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class _FakeRenderer:
    def __init__(self):
        self.render_count = 0

    def prepare_frame(self):
        return True

    def render(self, _engine):
        self.render_count += 1


def test_ripple_visualizer_feeds_pose_landmarks_into_source_excitations():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture()
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_acceleration_scale=1.0,
        pose_max_excitation=10.0,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()
    first_source_positions = visualizer.engine.source_positions.copy()
    first_field = visualizer.engine.get_field_numpy().copy()

    visualizer.update_visualization()
    second_source_positions = visualizer.engine.source_positions.copy()
    second_field = visualizer.engine.get_field_numpy().copy()

    assert visualizer.engine.n_sources == 2
    np.testing.assert_allclose(first_source_positions, [[4.75, 2.25], [14.25, 6.75]])
    np.testing.assert_allclose(second_source_positions, [[4.75, 2.25], [18.05, 6.75]])
    assert np.count_nonzero(first_field) == 0
    assert np.count_nonzero(second_field) > 0
    assert visualizer.renderer.render_count == 2

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()
