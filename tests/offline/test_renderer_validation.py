import numpy as np
from PyQt5 import QtGui

from audioviz.sources import RipplePoseConfig, RippleSourceOrchestratorConfig
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


def _seed_field(visualizer: RippleWaveVisualizer) -> None:
    field = np.zeros(visualizer.resolution, dtype=np.float32)
    field[6:14, 9:17] = 0.75
    field[10:18, 15:23] -= 0.35
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


def _count_cyan_overlay_pixels(rgb_frame: np.ndarray) -> int:
    red = rgb_frame[..., 0]
    green = rgb_frame[..., 1]
    blue = rgb_frame[..., 2]
    return int(np.count_nonzero((red < 120) & (green > 150) & (blue > 150)))


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


def test_offline_render_validation_shows_learning_overlay_pixels(qapp):
    visualizer = _build_non_pose_visualizer()
    visualizer.show_learning_overlay = True
    visualizer.learning_overlay_stride = 4
    visualizer.learning_overlay_threshold = 0.0
    visualizer.learning_overlay_scale = 2.0
    horizontal = np.zeros((24, 31), dtype=np.float32)
    horizontal[:, ::2] = 1e-4
    horizontal[:, 1::2] = -1e-4
    visualizer.engine.prediction_horizontal_edge_weights = horizontal
    visualizer.engine.prediction_vertical_edge_weights = np.zeros(
        (23, 32),
        dtype=np.float32,
    )
    visualizer.show()
    qapp.processEvents()

    try:
        visualizer.update_visualization()
        qapp.processEvents()
        screenshot = _grab_widget_rgb(visualizer.renderer.widget)
        xs, ys = visualizer.renderer.prediction_overlay_lines.getData()

        assert xs is not None
        assert ys is not None
        assert len(xs) > 0
        assert len(ys) > 0
        assert _count_cyan_overlay_pixels(screenshot) > 20
    finally:
        visualizer.close()
        qapp.processEvents()
