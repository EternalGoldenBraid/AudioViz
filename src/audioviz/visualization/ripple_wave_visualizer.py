from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtWidgets
from audioviz.engine import RippleEngine
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
                 **kwargs):

        super().__init__(processor, **kwargs)

        self.processor = processor
        self.use_synthetic = processor is None or use_synthetic
        self.use_gpu = use_gpu
        self.use_shader = use_shader

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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
