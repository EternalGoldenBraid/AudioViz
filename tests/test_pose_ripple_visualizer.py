import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audioviz.sources.pose import (
    PoseGraphFrame,
    adjacency_from_edges,
    build_pose_graph_segmentation_mask,
    map_pose_segmentation_to_field_mask,
)


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


class _MissingPoseExtractor:
    def __init__(self):
        self.frames = [
            PoseGraphFrame(
                coords=np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
                adjacency=adjacency_from_edges(2, [(0, 1)]),
            ),
            PoseGraphFrame.empty(2, adjacency=adjacency_from_edges(2, [(0, 1)])),
        ]
        self.closed = False

    def extract(self, _frame):
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class _OutOfFrameExtractor:
    def __init__(self):
        self.frames = [
            PoseGraphFrame(
                coords=np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
                adjacency=adjacency_from_edges(2, [(0, 1)]),
            ),
            PoseGraphFrame(
                coords=np.array([[0.25, 0.25], [3.0, 0.75]], dtype=np.float32),
                adjacency=adjacency_from_edges(2, [(0, 1)]),
            ),
        ]
        self.closed = False

    def extract(self, _frame):
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class _MaskedExtractor:
    def __init__(self):
        self.frames = [
            PoseGraphFrame(
                coords=np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
                adjacency=adjacency_from_edges(2, [(0, 1)]),
                segmentation_mask=np.array(
                    [
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ),
        ]
        self.closed = False

    def extract(self, _frame):
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class _BlankMaskExtractor:
    def __init__(self):
        self.frames = [
            PoseGraphFrame(
                coords=np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
                adjacency=adjacency_from_edges(2, [(0, 1)]),
                segmentation_mask=np.zeros((4, 4), dtype=np.float32),
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


class _FieldSource:
    def __init__(self, field):
        self._field = np.asarray(field, dtype=np.float32)

    def get_field_numpy(self):
        return self._field


class _FakeProcessor:
    def __init__(self, frequencies):
        self.current_top_k_frequencies = list(frequencies)


def test_ripple_visualizer_updates_pose_medium_without_self_excitation():
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
        pose_debug_view=True,
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
    np.testing.assert_allclose(
        second_pose_positions,
        [[14.25, 2.25], [0.95, 6.75]],
        atol=1e-5,
    )
    assert np.count_nonzero(first_field) == 0
    assert np.count_nonzero(second_field) == 0
    assert visualizer.renderer.render_count == 2
    assert visualizer.pose_debug_widget is not None
    assert visualizer.pose_debug_frame_count == 2

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()


def test_ripple_visualizer_keeps_rendering_when_pose_detection_drops_out():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture()
    extractor = _MissingPoseExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=0.99,
        amplitude=1.0,
        use_synthetic=False,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_acceleration_scale=1.0,
        pose_max_excitation=10.0,
        pose_debug_view=True,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()
    field_after_detection = visualizer.engine.get_field_numpy().copy()

    visualizer.update_visualization()
    field_after_dropout = visualizer.engine.get_field_numpy().copy()

    assert visualizer.renderer.render_count == 2
    assert visualizer.pose_debug_frame_count == 2
    assert np.count_nonzero(field_after_detection) == 0
    assert np.count_nonzero(field_after_dropout) == 0

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()


def test_pose_debug_view_mirrors_frame_and_graph_horizontally():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        use_pose_sources=False,
        pose_debug_view=True,
    )
    visualizer.timer.stop()

    frame = np.array(
        [
            [[0, 1, 2], [10, 11, 12], [20, 21, 22]],
            [[30, 31, 32], [40, 41, 42], [50, 51, 52]],
        ],
        dtype=np.uint8,
    )
    pose = PoseGraphFrame(
        coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        adjacency=adjacency_from_edges(2, [(0, 1)]),
    )

    visualizer._update_pose_debug_view(frame, pose)

    np.testing.assert_array_equal(
        visualizer.pose_debug_image.image,
        np.ascontiguousarray(frame[:, ::-1, ::-1]),
    )
    np.testing.assert_allclose(
        [(point.pos().x(), point.pos().y()) for point in visualizer.pose_debug_points.points()],
        [(2.0, 0.0), (0.0, 1.0)],
    )
    edge_xs, edge_ys = visualizer.pose_debug_edges.getData()
    np.testing.assert_allclose(edge_xs[:2], [2.0, 0.0])
    np.testing.assert_allclose(edge_ys[:2], [0.0, 1.0])
    assert np.isnan(edge_xs[2])
    assert np.isnan(edge_ys[2])

    visualizer.close()
    app.processEvents()


def test_pose_debug_view_overlays_segmentation_mask_with_same_mirroring():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        use_pose_sources=False,
        pose_debug_view=True,
    )
    visualizer.timer.stop()

    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    pose = PoseGraphFrame(
        coords=np.array([[0.5, 0.5]], dtype=np.float32),
        adjacency=adjacency_from_edges(1, []),
        segmentation_mask=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    visualizer._update_pose_debug_view(frame, pose)

    expected = np.zeros((2, 4, 3), dtype=np.uint8)
    expected[:, 2:] = [255, 255, 255]
    np.testing.assert_array_equal(visualizer.pose_debug_image.image, expected)

    visualizer.close()
    app.processEvents()


def test_pose_debug_view_falls_back_to_pose_graph_mask_when_segmentation_missing():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        use_pose_sources=False,
        pose_debug_view=True,
    )
    visualizer.timer.stop()

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    pose = PoseGraphFrame(
        coords=np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float32),
        adjacency=adjacency_from_edges(2, [(0, 1)]),
    )

    fallback_mask = visualizer._resolve_pose_segmentation_mask(frame, pose)

    assert fallback_mask is not None
    assert np.count_nonzero(fallback_mask) > 0

    visualizer._update_pose_debug_view(frame, pose, segmentation_mask=fallback_mask)

    assert np.count_nonzero(visualizer.pose_debug_image.image) > 0

    visualizer.close()
    app.processEvents()


def test_pose_debug_view_omits_out_of_frame_landmarks_and_edges():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        use_pose_sources=False,
        pose_debug_view=True,
    )
    visualizer.timer.stop()

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    pose = PoseGraphFrame(
        coords=np.array([[0.0, 0.0], [1.25, 1.0]], dtype=np.float32),
        adjacency=adjacency_from_edges(2, [(0, 1)]),
    )

    visualizer._update_pose_debug_view(frame, pose)

    points = [(point.pos().x(), point.pos().y()) for point in visualizer.pose_debug_points.points()]
    assert points == [(2.0, 0.0)]
    edge_xs, edge_ys = visualizer.pose_debug_edges.getData()
    assert edge_xs is None or len(edge_xs) == 0
    assert edge_ys is None or len(edge_ys) == 0

    visualizer.close()
    app.processEvents()


def test_ripple_visualizer_filters_out_of_frame_pose_landmarks_without_resetting_state():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture()
    extractor = _OutOfFrameExtractor()
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
        pose_debug_view=True,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()
    first_pose_state = visualizer.pose_state
    visualizer.update_visualization()

    assert visualizer.pose_state is first_pose_state
    assert visualizer.pose_state is not None
    assert visualizer.pose_state.num_nodes == 2
    np.testing.assert_allclose(
        visualizer.engine.get_pose_medium_positions(valid_only=True),
        [[14.25, 2.25]],
    )
    assert len(visualizer.pose_debug_points.points()) == 1


def test_ripple_visualizer_maps_segmentation_mask_into_engine_boundary_mask():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture(frame_count=1)
    extractor = _MaskedExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(4, 6),
        plane_size_m=(1.0, 1.0),
        use_synthetic=False,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_debug_view=False,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()

    expected = np.array(
        [
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(visualizer.engine.body_boundary_mask, expected)

    visualizer.close_pose_sources()
    app.processEvents()


def test_ripple_visualizer_falls_back_when_segmentation_mask_is_blank():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture(frame_count=1)
    extractor = _BlankMaskExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        use_synthetic=False,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_debug_view=True,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()

    expected = map_pose_segmentation_to_field_mask(
        build_pose_graph_segmentation_mask(
            np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
            adjacency_from_edges(2, [(0, 1)]),
            (4, 4),
        ),
        resolution=(10, 20),
        field_rect=visualizer.pose_field_rect,
    )
    np.testing.assert_array_equal(visualizer.engine.body_boundary_mask, expected)
    assert np.count_nonzero(visualizer.pose_debug_image.image) > 0

    visualizer.close_pose_sources()
    app.processEvents()


def test_ripple_visualizer_keeps_synthetic_source_when_pose_boundary_is_enabled():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture(frame_count=1)
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        n_sources=1,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        frequency=2.0,
        use_synthetic=True,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
        pose_acceleration_scale=0.0,
        pose_max_excitation=0.0,
        pose_debug_view=False,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()
    visualizer.engine.set_source_positions(np.array([[10.0, 5.0]], dtype=np.float32))

    visualizer.update_visualization()

    np.testing.assert_allclose(visualizer.engine.source_positions, [[10.0, 5.0]])
    assert visualizer.engine.n_sources == 1
    assert np.count_nonzero(visualizer.engine.get_field_numpy()) > 0

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()


def test_ripple_visualizer_updates_per_source_synthetic_frequencies_from_controls():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        n_sources=2,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        frequency=[110.0, 220.0],
        use_synthetic=True,
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.toggle_controls()

    panel = visualizer.control_panel
    assert panel is not None
    panel.source_control_widgets[("synthetic-source-1", "frequency_hz")].setValue(330.0)
    app.processEvents()

    np.testing.assert_allclose(
        visualizer._resolve_ripple_frequencies(),
        [[110.0], [330.0]],
    )

    visualizer.close()
    app.processEvents()


def test_ripple_visualizer_initializes_decay_alpha_from_constructor():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        decay_alpha=3.5,
        use_synthetic=True,
        use_pose_sources=False,
    )
    visualizer.timer.stop()

    assert visualizer.decay_alpha == 3.5
    assert visualizer.engine.decay_alpha == 3.5

    visualizer.close()
    app.processEvents()


def test_ripple_visualizer_combines_synthetic_and_audio_sources_when_enabled():
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=_FakeProcessor([220.0, 330.0]),
        n_sources=2,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        frequency=[110.0, 120.0],
        use_synthetic=True,
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.use_audio_source = True

    np.testing.assert_allclose(
        visualizer._resolve_ripple_frequencies(),
        [[110.0, 220.0, 330.0], [120.0, 220.0, 330.0]],
    )

    visualizer.close()


def test_ripple_visualizer_source_dropdown_toggles_audio_and_synthetic():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=_FakeProcessor([220.0]),
        n_sources=1,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        frequency=110.0,
        use_synthetic=False,
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.toggle_controls()

    panel = visualizer.control_panel
    assert panel is not None
    assert visualizer.use_audio_source is True
    assert visualizer.use_synthetic is False

    panel.source_toggle_actions["audio"].trigger()
    panel.source_toggle_actions["synthetic"].trigger()
    app.processEvents()

    assert visualizer.use_audio_source is False
    assert visualizer.use_synthetic is True
    np.testing.assert_allclose(visualizer._resolve_ripple_frequencies(), [[110.0]])

    visualizer.close()
    app.processEvents()


def test_ripple_visualizer_source_dropdown_can_enable_pose_graph():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture(frame_count=1)
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        n_sources=1,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        frequency=110.0,
        use_synthetic=False,
        use_pose_sources=False,
        pose_capture=capture,
        pose_extractor=extractor,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()
    visualizer.toggle_controls()

    panel = visualizer.control_panel
    assert panel is not None
    assert visualizer.use_pose_sources is False

    panel.source_toggle_actions["pose"].trigger()
    app.processEvents()
    visualizer.update_visualization()

    assert visualizer.use_pose_sources is True
    assert visualizer.pose_state is not None
    assert visualizer.renderer.render_count == 1

    visualizer.close_pose_sources()
    app.processEvents()


def test_ripple_visualizer_source_dropdown_can_disable_pose_graph():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = _FakeCapture(frame_count=2)
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        n_sources=1,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        frequency=110.0,
        use_synthetic=False,
        use_pose_sources=True,
        pose_capture=capture,
        pose_extractor=extractor,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()
    visualizer.update_visualization()
    visualizer.toggle_controls()

    panel = visualizer.control_panel
    assert panel is not None
    assert visualizer.use_pose_sources is True
    assert visualizer.pose_state is not None

    panel.source_toggle_actions["pose"].trigger()
    app.processEvents()
    visualizer.update_visualization()

    assert visualizer.use_pose_sources is False
    assert visualizer.pose_state is None
    assert visualizer.renderer.render_count == 2

    visualizer.close_pose_sources()
    app.processEvents()


def test_ripple_visualizer_steps_without_excitation_when_all_sources_disabled():
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        n_sources=1,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        frequency=110.0,
        use_synthetic=False,
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    initial_time = visualizer.engine.time
    visualizer.update_visualization()

    assert visualizer.renderer.render_count == 1
    assert visualizer.engine.time > initial_time

    visualizer.close()


def test_numpy_ripple_renderer_uses_top_left_image_origin():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    renderer = NumpyImageRenderer()

    assert renderer.plot.getViewBox().state["yInverted"] is True

    renderer.widget.close()
    app.processEvents()


def test_numpy_ripple_renderer_uses_percentile_auto_levels_for_sparse_fields():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    renderer = NumpyImageRenderer(auto_level_percentile=98.0)

    renderer.render(_FieldSource([[0.0, 0.0, 100.0], [0.0, 1.0, 2.0]]))
    low, high = renderer.histogram.getLevels()

    assert low < 0.0
    assert high > 0.0
    assert high < 100.0

    renderer.widget.close()
    app.processEvents()


def test_numpy_ripple_renderer_can_freeze_levels_when_auto_scaling_is_disabled():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    renderer = NumpyImageRenderer(auto_level_percentile=98.0)

    renderer.render(_FieldSource([[0.0, 0.0, 4.0], [0.0, 1.0, 2.0]]))
    first_levels = renderer.histogram.getLevels()
    renderer.set_auto_percentile_levels(False)
    renderer.render(_FieldSource([[0.0, 0.0, 200.0], [0.0, 1.0, 2.0]]))
    second_levels = renderer.histogram.getLevels()

    assert second_levels == first_levels

    renderer.widget.close()
    app.processEvents()


def test_ripple_visualizer_control_panel_toggles_renderer_auto_color_scaling():
    from PyQt5 import QtWidgets
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.renderer = NumpyImageRenderer()

    visualizer.toggle_controls()
    assert visualizer.control_panel is not None
    assert visualizer.renderer.auto_percentile_levels is True

    visualizer.control_panel.auto_color_levels_checkbox.setChecked(False)
    app.processEvents()

    assert visualizer.renderer.auto_percentile_levels is False

    visualizer.control_panel.close()
    visualizer.renderer.widget.close()
    visualizer.close()
    app.processEvents()
