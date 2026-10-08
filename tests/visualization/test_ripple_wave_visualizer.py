import numpy as np

from audioviz.sources import (
    AudioSourceConfig,
    CameraFrameSourceConfig,
    RipplePoseConfig,
    RippleSourceOrchestratorConfig,
    SyntheticSourceConfig,
)
from audioviz.sources.pose import PoseGraphFrame, adjacency_from_edges
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.readouts import PredictiveCodingConfig


class _FakeCapture:
    def __init__(self, frame_count: int = 2, frames: list[np.ndarray] | None = None):
        self.frames = (
            [np.array(frame, copy=True) for frame in frames]
            if frames is not None
            else [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(frame_count)]
        )
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
        self.field = np.asarray(_engine).copy()


class _StandingRenderer(_FakeRenderer):
    def __init__(self):
        super().__init__()
        self.rgb_frame = None
        self._lut = np.stack([np.arange(256), np.arange(256), np.arange(256)], axis=1).astype(np.uint8)
        self.image_item = type(
            "DummyImageItem",
            (),
            {"lut": lambda item: item._lut, "_lut": self._lut},
        )()

    def render_rgb_frame(self, rgb_frame):
        self.rgb_frame = np.asarray(rgb_frame)


def _source_config(
    *,
    use_synthetic: bool = False,
    use_camera_source: bool = False,
    prediction_error: PredictionErrorTransformConfig | None = None,
    camera_source_index: int = 0,
    camera_source_gain: float = 1.0,
) -> RippleSourceOrchestratorConfig:
    return RippleSourceOrchestratorConfig(
        synthetic=SyntheticSourceConfig(enabled=use_synthetic, frequency=440.0),
        audio=AudioSourceConfig(enabled=None),
        camera_frame=CameraFrameSourceConfig(
            enabled=use_camera_source,
            camera_index=camera_source_index,
            gain=camera_source_gain,
        ),
        prediction_error=(
            PredictionErrorTransformConfig()
            if prediction_error is None
            else prediction_error
        ),
    )


def _pose_config(
    *,
    enabled: bool = False,
    render_mode: str = "overlay",
    debug_view: bool = False,
) -> RipplePoseConfig:
    return RipplePoseConfig(
        enabled=enabled,
        render_mode=render_mode,
        debug_view=debug_view,
    )


def test_ripple_visualizer_pose_medium_overlay_smoke(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    capture = _FakeCapture()
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
        pose_config=_pose_config(enabled=True, render_mode="overlay"),
        pose_capture=capture,
        pose_extractor=extractor,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()
    first_pose_positions = visualizer._latest_pose_render_positions[visualizer._latest_pose_valid]
    first_field = visualizer.engine.get_field_numpy().copy()

    visualizer.update_visualization()
    second_pose_positions = visualizer._latest_pose_render_positions[visualizer._latest_pose_valid]
    second_field = visualizer.engine.get_field_numpy().copy()

    np.testing.assert_allclose(first_pose_positions, [[14.25, 2.25], [4.75, 6.75]])
    np.testing.assert_allclose(second_pose_positions, [[14.25, 2.25], [0.95, 6.75]], atol=1e-5)
    assert np.count_nonzero(first_field) == 0
    assert np.count_nonzero(second_field) == 0
    assert visualizer.renderer.render_count == 2

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed


def test_ripple_visualizer_camera_source_updates_field_and_releases_capture(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    frame = np.full((4, 4, 3), 255, dtype=np.uint8)
    capture = _FakeCapture(frames=[frame])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(6, 8),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(
            use_synthetic=False,
            use_camera_source=True,
            prediction_error=PredictionErrorTransformConfig(enabled=True),
        ),
        camera_capture=capture,
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()

    visualizer.update_visualization()

    assert visualizer.renderer.render_count == 1
    assert np.count_nonzero(visualizer.engine.get_field_numpy()) > 0

    visualizer.close_camera_source()
    assert capture.released


def test_ripple_visualizer_renders_canvas_after_camera_correction(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    frame = np.array(
        [
            [[0, 0, 255], [0, 255, 0]],
            [[255, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    capture = _FakeCapture(frames=[frame])
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(6, 8),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(
            use_synthetic=False,
            use_camera_source=True,
            prediction_error=PredictionErrorTransformConfig(
                enabled=True,
                activation_function="linear_clipped",
                gain=1.0,
            ),
        ),
        camera_capture=capture,
    )
    visualizer.timer.stop()
    visualizer._update_rgb_canvas(True)

    visualizer.update_visualization()
    first_prediction = visualizer.visual_prediction.copy()
    first_rendered = visualizer.renderer.image_item.image.copy()
    posterior = visualizer.engine.get_field_numpy().copy()

    assert np.count_nonzero(first_prediction) == 0
    assert np.count_nonzero(first_rendered) > 0
    assert np.any(posterior[..., 0] != posterior[..., 1])
    assert np.any(posterior[..., 1] != posterior[..., 2])

    visualizer.update_visualization()

    prediction = visualizer.visual_prediction
    rendered = visualizer.renderer.image_item.image
    assert prediction.shape == (6, 8, 3)
    assert rendered is not None
    assert rendered.shape == (6, 8, 3)
    assert np.any(prediction[..., 0] != prediction[..., 1])
    assert np.any(prediction[..., 1] != prediction[..., 2])
    assert np.any(rendered[..., 0] != rendered[..., 1])
    assert np.any(rendered[..., 1] != rendered[..., 2])
    visualizer.renderer.render(visualizer.engine.get_field_numpy())
    np.testing.assert_array_equal(rendered, visualizer.renderer.image_item.image)

    visualizer.close_camera_source()
    assert capture.released


def test_ripple_visualizer_render_uses_surface_not_camera_prediction(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        resolution=(6, 8),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        source_orchestrator_config=_source_config(),
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()
    visualizer.visual_prediction.fill(0.9)
    visualizer.engine.Z.fill(0.2)

    visualizer._render_scene()

    np.testing.assert_array_equal(
        visualizer.renderer.field, visualizer.engine.get_field_numpy()
    )
    assert not np.array_equal(visualizer.renderer.field, visualizer.visual_prediction)


def test_ripple_visualizer_reset_clears_hidden_memory_and_renders_zero_surface(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        resolution=(6, 8),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        source_orchestrator_config=RippleSourceOrchestratorConfig(
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(learning_enabled=True),
        ),
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()
    readout = visualizer.source_orchestrator.visual_readout
    readout.predict(np.ones(visualizer.engine.canvas_shape))
    readout.infer(
        np.ones(visualizer.engine.canvas_shape),
        np.full(visualizer.engine.canvas_shape, 0.2),
    )
    weights_before = readout.visual_weights.copy()
    visualizer.engine.Z.fill(0.5)

    visualizer.engine.reset()
    visualizer._sync_after_reset()

    np.testing.assert_array_equal(readout.hidden, 0.0)
    np.testing.assert_array_equal(readout.hidden_error, 0.0)
    np.testing.assert_array_equal(readout.visual_error, 0.0)
    np.testing.assert_array_equal(readout.visual_weights, weights_before)
    np.testing.assert_array_equal(visualizer.visual_prediction, 0.0)
    np.testing.assert_array_equal(visualizer.renderer.field, 0.0)


def test_ripple_visualizer_pose_path_uses_same_visual_inference(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        resolution=(10, 20),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        source_orchestrator_config=_source_config(
            prediction_error=PredictionErrorTransformConfig(enabled=True)
        ),
        pose_config=_pose_config(enabled=True),
        pose_capture=_FakeCapture(),
        pose_extractor=_FakeExtractor(),
    )
    visualizer.timer.stop()
    visualizer.renderer = _FakeRenderer()
    visualizer.camera_source.excitation = lambda: np.full(
        visualizer.engine.canvas_shape, 0.7, dtype=np.float32
    )

    visualizer.update_visualization()

    assert np.any(visualizer.source_orchestrator.visual_readout.hidden != 0.0)
    assert np.any(visualizer.engine.get_field_numpy() != 0.0)
    np.testing.assert_array_equal(
        visualizer.renderer.field, visualizer.engine.get_field_numpy()
    )
    visualizer.close_pose_sources()


def test_ripple_visualizer_pose_medium_standing_body_accepts_callable_lut(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    capture = _FakeCapture()
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
        pose_config=_pose_config(enabled=True, render_mode="standing-body"),
        pose_capture=capture,
        pose_extractor=extractor,
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


def test_ripple_visualizer_pose_medium_standing_body_with_numpy_renderer_smoke(qapp):
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    capture = _FakeCapture()
    extractor = _FakeExtractor()
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
        pose_config=_pose_config(enabled=True, render_mode="standing-body"),
        pose_capture=capture,
        pose_extractor=extractor,
    )
    visualizer.timer.stop()
    visualizer.renderer = NumpyImageRenderer()

    visualizer.update_visualization()

    assert visualizer.renderer.image_item.image is not None
    assert visualizer.renderer.image_item.image.shape == (24, 32, 3)

    visualizer.close_pose_sources()
    assert capture.released
    assert extractor.closed


def test_ripple_visualizer_non_pose_mode_does_not_apply_standing_body_projection(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
        pose_config=_pose_config(enabled=False, render_mode="standing-body"),
    )
    visualizer.timer.stop()
    visualizer.renderer = _StandingRenderer()

    visualizer.update_visualization()

    assert visualizer.renderer.render_count == 1
    assert visualizer.renderer.rgb_frame is None


def test_ripple_visualizer_auto_color_controls_take_effect_on_next_render(qapp):
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        auto_color_floor=0.05,
        source_orchestrator_config=_source_config(use_synthetic=False),
    )
    visualizer.timer.stop()
    renderer = NumpyImageRenderer(
        auto_percentile_levels=True,
        auto_level_floor=0.05,
        auto_level_activation_threshold=0.1,
    )
    visualizer.renderer = renderer
    visualizer.visual_prediction[:] = np.full((24, 32, 3), 0.2, dtype=np.float32)

    renderer.render(visualizer.visual_prediction)
    first_levels = renderer.histogram.getLevels()

    visualizer._update_auto_color_activation_threshold(0.35)
    visualizer._update_auto_color_floor(0.2)
    second_levels_before_render = renderer.histogram.getLevels()
    renderer.render(visualizer.visual_prediction)
    second_levels_after_render = renderer.histogram.getLevels()

    np.testing.assert_allclose(first_levels, (-0.2, 0.2))
    np.testing.assert_allclose(second_levels_before_render, first_levels)
    np.testing.assert_allclose(second_levels_after_render, (-0.2, 0.2))

    visualizer._update_auto_color_floor(0.05)
    renderer.render(visualizer.visual_prediction)
    third_levels = renderer.histogram.getLevels()

    np.testing.assert_allclose(third_levels, (-0.05, 0.05))


def test_ripple_visualizer_renders_learning_overlay_edges(qapp):
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
    )
    visualizer.timer.stop()
    visualizer.renderer = NumpyImageRenderer()
    visualizer.show_learning_overlay = True
    visualizer.learning_overlay_stride = 4
    visualizer.learning_overlay_threshold = 0.0
    visualizer.learning_overlay_scale = 2.0
    visualizer.engine.prediction_horizontal_edge_weights = np.ones((8, 7), dtype=np.float32)
    visualizer.engine.prediction_vertical_edge_weights = np.ones((7, 8), dtype=np.float32)

    visualizer._render_scene()

    overlay = visualizer.renderer.prediction_overlay_image.image
    assert overlay is not None
    assert overlay.shape == (8, 8, 4)
    assert np.count_nonzero(overlay[..., 3]) > 0


def test_ripple_visualizer_updates_learning_overlay_in_normal_render_loop(qapp):
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
    )
    visualizer.timer.stop()
    visualizer.renderer = NumpyImageRenderer()
    visualizer.show_learning_overlay = True
    visualizer.learning_overlay_stride = 4
    visualizer.learning_overlay_threshold = 0.0
    visualizer.learning_overlay_scale = 2.0
    visualizer.engine.prediction_horizontal_edge_weights = np.ones((8, 7), dtype=np.float32)
    visualizer.engine.prediction_vertical_edge_weights = np.ones((7, 8), dtype=np.float32)

    visualizer.update_visualization()

    overlay = visualizer.renderer.prediction_overlay_image.image
    assert overlay is not None
    assert np.count_nonzero(overlay[..., 3]) > 0


def test_ripple_visualizer_updates_learning_overlay_with_standing_body_config_disabled_pose(
    qapp,
):
    from audioviz.visualization.ripple_renderers import NumpyImageRenderer
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
        pose_config=_pose_config(enabled=False, render_mode="standing-body"),
    )
    visualizer.timer.stop()
    visualizer.renderer = NumpyImageRenderer()
    visualizer.show_learning_overlay = True
    visualizer.learning_overlay_stride = 4
    visualizer.learning_overlay_threshold = 0.0
    visualizer.learning_overlay_scale = 2.0
    visualizer.engine.prediction_horizontal_edge_weights = np.ones((8, 7), dtype=np.float32)
    visualizer.engine.prediction_vertical_edge_weights = np.ones((7, 8), dtype=np.float32)

    visualizer.update_visualization()

    assert visualizer.renderer.image_item.image is not None
    assert visualizer.renderer.image_item.image.ndim == 2
    overlay = visualizer.renderer.prediction_overlay_image.image
    assert overlay is not None
    assert np.count_nonzero(overlay[..., 3]) > 0
