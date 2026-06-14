import time
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtWidgets
from audioviz.engine import RippleEngine
from audioviz.sources.pose import (
    MediaPipePoseExtractor,
    PoseGraphExtractor,
    PoseGraphState,
    centered_field_rect,
    pose_graph_state_to_ripple_sources,
)
from audioviz.visualization.ripple_renderers import (
    NumpyImageRenderer,
    OpenGLFieldRenderer,
)
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.visualization.ripple_control_panel import RippleControlPanel


class RippleWaveVisualizer(VisualizerBase):
    def __init__(self,
                 processor: Optional[AudioProcessor] = None,
                 n_sources: int = 1,
                 plane_size_m: Tuple[float, float] = (0.36, 0.62),
                 resolution: Tuple[int, int] = (400, 400),
                 frequency: float = 440.0,
                 amplitude: float = 1.0,
                 speed: float = 340.0,
                 damping: float = 0.999,
                 use_synthetic: bool = True,
                 apply_gaussian_smoothing: bool = False,
                 use_gpu: bool = False,
                 use_shader: bool = False,
                 use_pose_sources: bool = False,
                 pose_model_path: str | None = None,
                 pose_camera_index: int = 0,
                 pose_acceleration_scale: float = 1.0,
                 pose_max_excitation: float | None = None,
                 pose_field_width_fraction: float = 1.0,
                 pose_field_height_fraction: float = 1.0,
                 pose_extractor: PoseGraphExtractor | None = None,
                 pose_capture=None,
                 **kwargs):

        super().__init__(processor, **kwargs)

        self.processor = processor
        self.use_synthetic = processor is None or use_synthetic
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.use_pose_sources = use_pose_sources

        self.n_sources = n_sources
        self.plane_size_m = plane_size_m
        self.resolution = resolution
        self.frequency = frequency
        self.apply_gaussian_smoothing = apply_gaussian_smoothing
        self.amplitude = amplitude
        self.decay_alpha = 0.0
        self.speed = speed
        self.damping = damping
        self.time = 0.0
        self.control_panel: Optional[RippleControlPanel] = None
        self.pose_acceleration_scale = pose_acceleration_scale
        self.pose_max_excitation = pose_max_excitation
        self.pose_field_rect = centered_field_rect(
            self.resolution,
            width_fraction=pose_field_width_fraction,
            height_fraction=pose_field_height_fraction,
        )
        self.pose_state: PoseGraphState | None = None
        self.pose_last_update_time = time.monotonic()
        self.pose_extractor = pose_extractor
        self.pose_capture = pose_capture

        if self.use_pose_sources:
            self._ensure_pose_source(
                model_path=pose_model_path,
                camera_index=pose_camera_index,
            )

        self.engine = RippleEngine(
            resolution=self.resolution,
            plane_size_m=self.plane_size_m,
            n_sources=self.n_sources,
            speed=self.speed,
            damping=damping,
            amplitude=self.amplitude,
            use_gpu=self.use_gpu,
            use_shader=self.use_shader,
            use_external_opengl_context=self.use_shader,
        )
        self.dt = self.engine.dt

        self.renderer = (
            OpenGLFieldRenderer() if self.use_shader else NumpyImageRenderer()
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.renderer.widget)

        controls_button = QtWidgets.QPushButton("Show Controls")
        controls_button.clicked.connect(self.toggle_controls)
        layout.addWidget(controls_button)

    def _update_speed(self, val: float):
        self.speed = val
        self.dt = self.engine.dt

    def _update_amplitude(self, val: float):
        self.amplitude = val

    def _update_decay_alpha(self, val: float):
        self.decay_alpha = val
    
    def _update_damping(self, val: float):
        self.damping = val

    def toggle_controls(self):
        if self.control_panel is None:
            self.control_panel = RippleControlPanel(
                self.engine,
                on_speed_changed=self._update_speed,
                on_amplitude_changed=self._update_amplitude,
                on_decay_alpha_changed=self._update_decay_alpha,
                on_damping_changed=self._update_damping,
                before_reset=self.renderer.prepare_frame,
                on_reset=self._sync_after_reset,
            )
            self.control_panel.resize(300, 300)
        self.control_panel.show()
        self.control_panel.raise_()
        self.control_panel.activateWindow()

    def _sync_after_reset(self) -> None:
        self.time = self.engine.time

    def update_visualization(self):
        if self.use_pose_sources:
            self._update_pose_visualization()
            return

        if self.use_synthetic or self.processor is None:
            freqs = np.full((self.n_sources, 1), self.frequency, dtype=np.float32)
        else:
            top_k = self.processor.current_top_k_frequencies
            top_k = [f for f in top_k if f is not None and np.isfinite(f)]
            if len(top_k) == 0:
                return
            freqs = np.tile(top_k, (self.n_sources, 1))

        if not self.renderer.prepare_frame():
            return

        self.engine.step(freqs)
        self.time = self.engine.time
        self.renderer.render(self.engine)

    def _ensure_pose_source(
        self,
        *,
        model_path: str | None,
        camera_index: int,
    ) -> None:
        if self.pose_extractor is None:
            self.pose_extractor = MediaPipePoseExtractor(model_path=model_path)
        if self.pose_capture is None:
            cv2 = self._load_cv2()
            self.pose_capture = cv2.VideoCapture(camera_index)
            if not self.pose_capture.isOpened():
                self.pose_extractor.close()
                self.pose_extractor = None
                raise RuntimeError(f"Failed to open pose camera index {camera_index}")

    def _update_pose_visualization(self) -> None:
        if self.pose_capture is None or self.pose_extractor is None:
            return

        ok, frame = self.pose_capture.read()
        if not ok:
            return

        pose = self.pose_extractor.extract(frame)
        if not pose.coords.size:
            return

        now = time.monotonic()
        dt = max(now - self.pose_last_update_time, 1e-6)
        self.pose_last_update_time = now

        initialized_pose_state = False
        if self.pose_state is None or self.pose_state.num_nodes != len(pose.coords):
            self.pose_state = PoseGraphState(
                len(pose.coords),
                pose.adjacency,
                velocity_smoothing_alpha=0.8,
            )
            initialized_pose_state = True
        self.pose_state.update(pose.coords, dt)

        source_positions, source_excitations = pose_graph_state_to_ripple_sources(
            self.pose_state,
            self.resolution,
            field_rect=self.pose_field_rect,
            acceleration_scale=self.pose_acceleration_scale,
            max_excitation=self.pose_max_excitation,
        )
        if initialized_pose_state:
            source_excitations = np.zeros_like(source_excitations)

        if not self.renderer.prepare_frame():
            return

        self.engine.set_source_positions(source_positions)
        self.engine.step_source_excitations(source_excitations)
        self.time = self.engine.time
        self.renderer.render(self.engine)

    def closeEvent(self, event):
        self.close_pose_sources()
        super().closeEvent(event)

    def close_pose_sources(self) -> None:
        if self.pose_extractor is not None:
            self.pose_extractor.close()
            self.pose_extractor = None
        if self.pose_capture is not None:
            self.pose_capture.release()
            self.pose_capture = None

    @staticmethod
    def _load_cv2():
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for pose-driven ripple sources. "
                "Install the pose-demo dependency group first."
            ) from exc
        return cv2


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
