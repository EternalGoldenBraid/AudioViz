import numpy as np

from audioviz.sources.pose import PoseGraphFrame, adjacency_from_edges
from audioviz.utils.signal_processing import map_audio_freq_to_visual_freq


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


class _StandingRendererWithCallableLut(_FakeRenderer):
    def __init__(self):
        super().__init__()
        self.rgb_frame = None
        self._lut = np.stack(
            [np.arange(256), np.arange(256), np.arange(256)],
            axis=1,
        ).astype(np.uint8)
        self.image_item = type(
            "DummyImageItem",
            (),
            {"lut": lambda item: item._lut, "_lut": self._lut},
        )()

    def render_rgb_frame(self, rgb_frame):
        self.rgb_frame = np.asarray(rgb_frame)


class _FakeProcessor:
    def __init__(self, current_top_k_frequencies, current_signal_level=1.0):
        self.current_top_k_frequencies = current_top_k_frequencies
        self.current_signal_level = current_signal_level
        self.minimum_frequency_peak_magnitude = 0.1
        self.minimum_frequency_peak_to_median_ratio = 5.0
        self.minimum_signal_level = 0.05
        self.num_top_frequencies = len(current_top_k_frequencies)

    def set_num_top_frequencies(self, count):
        count = int(count)
        self.num_top_frequencies = count
        values = [value for value in self.current_top_k_frequencies if value is not None][:count]
        values.extend([None] * (count - len(values)))
        self.current_top_k_frequencies = values


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


def test_ripple_visualizer_pose_medium_standing_body_accepts_callable_lut():
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
    visualizer.renderer = _StandingRendererWithCallableLut()

    visualizer.update_visualization()

    assert visualizer.renderer.render_count == 1
    assert visualizer.renderer.rgb_frame is not None
    assert visualizer.renderer.rgb_frame.shape == (24, 32, 3)
    assert np.count_nonzero(visualizer.renderer.rgb_frame) > 0

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()


def test_ripple_visualizer_pose_medium_standing_body_with_numpy_renderer_smoke():
    from PyQt5 import QtWidgets

    from audioviz.visualization.ripple_renderers import NumpyImageRenderer
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
    visualizer.renderer = NumpyImageRenderer()

    visualizer.update_visualization()

    assert visualizer.renderer.image_item.image is not None
    assert visualizer.renderer.image_item.image.shape == (24, 32, 3)

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed
    app.processEvents()


def test_ripple_visualizer_maps_audio_frequencies_before_ripple_drive():
    from PyQt5 import QtWidgets

    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    processor = _FakeProcessor([440.0, 880.0, None])
    visualizer = RippleWaveVisualizer(
        processor=processor,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_synthetic=False,
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    resolved = visualizer._resolve_audio_frequencies()

    expected = map_audio_freq_to_visual_freq(
        np.asarray([440.0, 880.0], dtype=np.float32)
    ).astype(np.float32)
    assert resolved is not None
    np.testing.assert_allclose(resolved, expected.reshape(1, -1))

    app.processEvents()


def test_ripple_visualizer_uses_configured_audio_frequency_mapping():
    from PyQt5 import QtWidgets

    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    processor = _FakeProcessor([440.0, 880.0, None])
    visualizer = RippleWaveVisualizer(
        processor=processor,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_synthetic=False,
        use_pose_sources=False,
        audio_visual_mapping_alpha=10.0,
        audio_visual_mapping_f0=100.0,
        audio_visual_mapping_fc=10_000.0,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    resolved = visualizer._resolve_audio_frequencies()

    expected = map_audio_freq_to_visual_freq(
        np.asarray([440.0, 880.0], dtype=np.float32),
        alpha=10.0,
        f0=100.0,
        fc=10_000.0,
    ).astype(np.float32)
    assert resolved is not None
    np.testing.assert_allclose(resolved, expected.reshape(1, -1))

    app.processEvents()


def test_ripple_visualizer_scales_audio_only_excitation_by_signal_level():
    from PyQt5 import QtWidgets

    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    processor = _FakeProcessor([440.0], current_signal_level=0.25)
    visualizer = RippleWaveVisualizer(
        processor=processor,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=2.0,
        use_synthetic=False,
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    amplitude = visualizer._current_excitation_amplitude(
        np.asarray([[1.0]], dtype=np.float32)
    )

    assert amplitude == 0.5

    app.processEvents()


def test_ripple_visualizer_audio_source_controls_update_processor_and_mapping():
    from PyQt5 import QtWidgets

    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    processor = _FakeProcessor([440.0, 880.0, None], current_signal_level=0.25)
    visualizer = RippleWaveVisualizer(
        processor=processor,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=2.0,
        use_synthetic=False,
        use_pose_sources=False,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer._update_source_control("audio-source", "signal_gate_threshold", 0.2)
    visualizer._update_source_control("audio-source", "drive_amplitude", 1.5)
    visualizer._update_source_control("audio-source", "minimum_peak_magnitude", 0.3)
    visualizer._update_source_control("audio-source", "peak_prominence_ratio", 7.0)
    visualizer._update_source_control("audio-source", "top_k_count", 2)
    visualizer._update_source_control("audio-source", "mapping_mode", "linear")
    visualizer._update_source_control("audio-source", "linear_scale", 0.1)

    assert visualizer.audio_signal_gate_threshold == 0.2
    assert processor.minimum_signal_level == 0.2
    assert visualizer.audio_drive_amplitude == 1.5
    assert processor.minimum_frequency_peak_magnitude == 0.3
    assert processor.minimum_frequency_peak_to_median_ratio == 7.0
    assert processor.num_top_frequencies == 2
    assert visualizer.audio_visual_mapping_mode == "linear"
    assert visualizer.audio_visual_linear_scale == 0.1

    app.processEvents()
