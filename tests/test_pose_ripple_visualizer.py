import numpy as np

from audioviz.sources.pose import PoseGraphFrame, adjacency_from_edges


class _FakeCapture:
    def __init__(self, frame_count: int = 2):
        self.frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(frame_count)]
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


class _StandingRenderer(_FakeRenderer):
    def __init__(self):
        super().__init__()
        self.rgb_frame = None
        lut = np.stack([np.arange(256), np.arange(256), np.arange(256)], axis=1).astype(np.uint8)
        self.image_item = type("DummyImageItem", (), {"lut": lut})()

    def render_rgb_frame(self, rgb_frame):
        self.rgb_frame = np.asarray(rgb_frame)


def test_ripple_visualizer_pose_medium_overlay_smoke():
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
        use_synthetic=False,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_acceleration_scale=1.0,
        pose_max_excitation=10.0,
        pose_debug_view=False,
        pose_render_mode="overlay",
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()
    first_pose_positions = visualizer.engine.get_pose_medium_positions(valid_only=True)
    first_field = visualizer.engine.get_field_numpy().copy()

    visualizer.update_visualization()
    second_pose_positions = visualizer.engine.get_pose_medium_positions(valid_only=True)
    second_field = visualizer.engine.get_field_numpy().copy()

    np.testing.assert_allclose(first_pose_positions, [[14.25, 2.25], [4.75, 6.75]])
    np.testing.assert_allclose(second_pose_positions, [[14.25, 2.25], [0.95, 6.75]], atol=1e-5)
    assert np.count_nonzero(first_field) == 0
    assert np.count_nonzero(second_field) == 0
    assert visualizer.renderer.render_count == 2

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()


def test_ripple_visualizer_pose_medium_standing_body_smoke():
    from PyQt5 import QtWidgets

    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture()
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_synthetic=False,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_debug_view=False,
        pose_render_mode="standing-body",
    )
    visualizer.timer.stop()
    visualizer.renderer = _StandingRenderer()

    visualizer.update_visualization()

    assert visualizer.renderer.render_count == 1
    assert visualizer.renderer.rgb_frame is not None
    assert visualizer.renderer.rgb_frame.shape == (24, 32, 3)
    assert np.count_nonzero(visualizer.renderer.rgb_frame) > 0

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()
