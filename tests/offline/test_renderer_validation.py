import numpy as np
from PyQt5 import QtGui

from audioviz.sources import RipplePoseConfig, RippleSourceOrchestratorConfig
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


def _seed_field(visualizer: RippleWaveVisualizer) -> None:
    field = np.zeros(visualizer.engine.canvas_shape, dtype=np.float32)
    field[6:14, 9:17, 0] = 0.75
    field[10:18, 15:23, 2] = 0.35
    visualizer.engine.propagator.Z[:] = field
    if hasattr(visualizer.engine.propagator, "Z_old"):
        visualizer.engine.propagator.Z_old[:] = 0
    visualizer.engine.Z[:] = field
    visualizer.engine.Z_old[:] = 0


def _grab_widget_rgb(widget) -> np.ndarray:
    image = widget.grab().toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
    width = image.width()
    height = image.height()
    bits = image.bits()
    bits.setsize(image.byteCount())
    rgba = np.frombuffer(bits, dtype=np.uint8).reshape(
        height,
        image.bytesPerLine() // 4,
        4,
    )
    return np.ascontiguousarray(rgba[:, :width, :3])


def _count_changed_pixels(before: np.ndarray, after: np.ndarray) -> int:
    difference = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return int(np.count_nonzero(np.any(difference > 0, axis=2)))


def _build_non_pose_visualizer() -> RippleWaveVisualizer:
    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=0.995,
        amplitude=1.0,
        source_orchestrator_config=RippleSourceOrchestratorConfig(),
        pose_config=RipplePoseConfig(
            enabled=False,
            render_mode="standing-body",
        ),
    )
    visualizer.timer.stop()
    visualizer.resize(420, 360)
    _seed_field(visualizer)
    return visualizer


def test_offline_render_validation_keeps_non_pose_view_flat(qapp):
    visualizer = _build_non_pose_visualizer()
    visualizer.show()
    qapp.processEvents()

    try:
        visualizer.update_visualization()
        qapp.processEvents()
        screenshot = _grab_widget_rgb(visualizer.renderer.widget)

        assert visualizer.renderer.image_item.image is not None
        assert visualizer.renderer.image_item.image.shape == (24, 32)
        assert screenshot.ndim == 3
        assert screenshot.shape[2] == 3
        assert np.count_nonzero(screenshot) > 0
    finally:
        visualizer.close()
        qapp.processEvents()


def test_offline_render_validation_rgb_canvas_produces_colored_widget_pixels(qapp):
    visualizer = _build_non_pose_visualizer()
    visualizer._update_rgb_canvas(True)
    visualizer.show()
    qapp.processEvents()

    try:
        visualizer._render_scene()
        qapp.processEvents()
        screenshot = _grab_widget_rgb(visualizer.renderer.widget)
        channel_spread = np.max(screenshot, axis=2) - np.min(screenshot, axis=2)

        assert visualizer.renderer.image_item.image.shape == (24, 32, 3)
        assert np.count_nonzero(channel_spread > 20) > 100
    finally:
        visualizer.close()
        qapp.processEvents()


def test_offline_render_validation_shows_learning_overlay_pixels(qapp):
    visualizer = _build_non_pose_visualizer()
    visualizer.show()
    qapp.processEvents()

    try:
        visualizer.show_learning_overlay = False
        visualizer.update_visualization()
        qapp.processEvents()
        baseline = _grab_widget_rgb(visualizer.renderer.widget)

        visualizer.show_learning_overlay = True
        visualizer.learning_overlay_stride = 4
        visualizer.learning_overlay_threshold = 0.0
        visualizer.learning_overlay_scale = 2.0
        visualizer.engine.prediction_horizontal_edge_weights = np.ones(
            (24, 31),
            dtype=np.float32,
        )
        visualizer.engine.prediction_vertical_edge_weights = np.ones(
            (23, 32),
            dtype=np.float32,
        )
        visualizer.update_visualization()
        qapp.processEvents()
        screenshot = _grab_widget_rgb(visualizer.renderer.widget)
        overlay = visualizer.renderer.prediction_overlay_image.image

        assert overlay is not None
        assert np.count_nonzero(overlay[..., 3]) > 0
        assert _count_changed_pixels(baseline, screenshot) > 20
    finally:
        visualizer.close()
        qapp.processEvents()
