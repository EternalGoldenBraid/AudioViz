from typing import Optional, Tuple

import numpy as np
import matplotlib.cm as cm
from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt
from audioviz.engine import RippleEngine
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor


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
                 **kwargs):

        super().__init__(processor, **kwargs)

        self.processor = processor
        self.use_synthetic = processor is None or use_synthetic
        self.use_gpu = use_gpu

        self.n_sources = n_sources
        self.plane_size_m = plane_size_m
        self.resolution = resolution
        self.frequency = frequency
        self.apply_gaussian_smoothing = apply_gaussian_smoothing
        self.amplitude = amplitude
        self.speed = speed
        self.time = 0.0

        self.engine = RippleEngine(
            resolution=self.resolution,
            plane_size_m=self.plane_size_m,
            n_sources=self.n_sources,
            speed=self.speed,
            damping=damping,
            amplitude=self.amplitude,
            use_gpu=self.use_gpu,
        )
        self.dt = self.engine.dt

        self.image_item = pg.ImageItem()
        # cmap = cm.get_cmap("seismic")
        cmap = cm.get_cmap("inferno")
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.image_item.setLookupTable(lut)

        self.plot = pg.PlotItem()
        self.plot.setTitle("Ripple Simulation")
        self.plot.addItem(self.image_item)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.addItem(self.plot)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        # Decay Alpha Slider
        self.decay_alpha = 0.0
        self.decay_label = QLabel("Decay α: 0.0")
        self.decay_title = QLabel("Excitation Falloff (α)")
        self.decay_title.setToolTip(
            "Controls how quickly the excitation decays away from the source point. " 
            "Higher α = more localized excitation. Lower α = more spread out excitation e.g. larger object causing the ripple)."
        )
        self.decay_slider = QSlider(Qt.Horizontal)
        self.decay_slider.setMinimum(0)
        self.decay_slider.setMaximum(1000)  # maps to 0.0 - 20.0
        self.decay_slider.valueChanged.connect(
            lambda val: self._update_decay_alpha(val / 10.0)
        )
        
        # Damping Slider
        self.damping_label = QLabel(f"Damping: {damping:.3f}")
        self.damping_title = QLabel("Wave Damping")
        self.damping_title.setToolTip("Controls how quickly the wave loses energy as it propagates. 1.0 = no damping.")
        self.damping_slider = QSlider(Qt.Horizontal)
        self.damping_slider.setMinimum(0)
        self.damping_slider.setMaximum(1000) # maps to 0.0 - 1.0
        self.damping_slider.valueChanged.connect(
            lambda val: self._update_damping(val / 1000) # step size of 0.01
        )

        # Speed Slider
        self.speed_label = QLabel(f"Speed: {self.speed:.1f}")
        self.speed_title = QLabel("Wave Speed (m/s)")
        self.speed_title.setToolTip("Controls how fast the wave propagates across the surface.")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(1000)  # Maps to 1–1000 m/s
        self.speed_slider.setValue(int(self.speed))
        self.speed_slider.valueChanged.connect(
            lambda val: self._update_speed(val)
        )

        # Amplitude Slider
        self.amplitude_label = QLabel(f"Amplitude: {self.amplitude:.2f}")
        self.amplitude_title = QLabel("Excitation Amplitude")
        self.amplitude_title.setToolTip("Controls the strength of the input excitation added to the wave field.")
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setMinimum(0)
        self.amplitude_slider.setMaximum(500)  # Maps to 0.00–5.00
        self.amplitude_slider.setValue(int(self.amplitude * 100))
        self.amplitude_slider.valueChanged.connect(
            lambda val: self._update_amplitude(val / 100.0)
        )

        # Add to layout
        layout.addWidget(self.decay_title)
        layout.addWidget(self.decay_label)
        layout.addWidget(self.decay_slider)

        layout.addWidget(self.damping_title)
        layout.addWidget(self.damping_label)
        layout.addWidget(self.damping_slider)

        layout.addWidget(self.speed_title)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_slider)

        layout.addWidget(self.amplitude_title)
        layout.addWidget(self.amplitude_label)
        layout.addWidget(self.amplitude_slider)

    def _update_speed(self, val: float):
        self.speed = val
        self.speed_label.setText(f"Speed: {val:.1f}")
        self.engine.set_speed(val)
        self.dt = self.engine.dt

    def _update_amplitude(self, val: float):
        self.amplitude = val
        self.engine.amplitude = val
        self.amplitude_label.setText(f"Amplitude: {val:.2f}")

    def _update_decay_alpha(self, val: float):
        self.engine.decay_alpha = val
        self.decay_label.setText(f"Decay α: {val:.1f}")
    
    def _update_damping(self, val: float):
        self.engine.set_damping(val)
        self.damping_label.setText(f"Damping: {val:.3f}")

    def update_visualization(self):
        if self.use_synthetic or self.processor is None:
            freqs = np.full((self.n_sources, 1), self.frequency, dtype=np.float32)
        else:
            top_k = self.processor.current_top_k_frequencies
            top_k = [f for f in top_k if f is not None and np.isfinite(f)]
            if len(top_k) == 0:
                return
            freqs = np.tile(top_k, (self.n_sources, 1))

        self.engine.step(freqs)
        self.time = self.engine.time

        Z_vis = self.engine.get_field_numpy()
        max_abs = np.max(np.abs(Z_vis))
        self.image_item.setLevels([-max_abs, max_abs])
        self.image_item.setImage(Z_vis, autoLevels=False)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
