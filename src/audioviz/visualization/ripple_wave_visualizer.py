from typing import Optional, Tuple, Dict, Any

import numpy as np
import cupy as cp
import matplotlib.cm as cm
from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.signal_processing import map_audio_freq_to_visual_freq

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
                 use_gpu: bool = True,
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

        self.backend = cp if use_gpu else np

        self.dx = self.plane_size_m[0] / self.resolution[0]
        self.dy = self.plane_size_m[1] / self.resolution[1]
        self.dt = (max(self.dx, self.dy) / speed) * 1 / np.sqrt(2)

        propagator_kwargs = {
            "shape": self.resolution,
            "dx": self.dx,
            "dt": self.dt,
            "speed": self.speed,
            "damping": damping
        }
        if use_gpu:
            self.propagator = WavePropagatorGPU(**propagator_kwargs)
        else:
            self.propagator = WavePropagatorCPU(**propagator_kwargs)

        self.Z = self.backend.zeros(self.resolution, dtype=self.backend.float32)

        self.max_frequency = self.speed / (2 * max(self.dx, self.dy))

        self.source_positions = []
        np.random.seed(42)
        for _ in range(n_sources):
            x = np.random.randint(0, resolution[0])
            y = np.random.randint(0, resolution[1])
            self.source_positions.append((x, y))

        self.xs, self.ys = self.backend.meshgrid(
            self.backend.arange(self.resolution[1]),
            self.backend.arange(self.resolution[0])
        )

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
        self.damping_label = QLabel("Damping: 0.999")
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

        self.compute_ripple = self.compute_ripple_cupy if use_gpu else self.compute_ripple_numpy

    def _update_speed(self, val: float):
        self.speed = val
        self.speed_label.setText(f"Speed: {val:.1f}")
        self.dt = (max(self.dx, self.dy) / self.speed) * 1 / np.sqrt(2)
        self.propagator.dt = self.dt
        self.propagator.c = self.speed
        self.propagator.c2_dt2 = (self.speed * self.dt / self.dx)**2

    def _update_amplitude(self, val: float):
        self.amplitude = val
        self.amplitude_label.setText(f"Amplitude: {val:.2f}")

    def _update_decay_alpha(self, val: float):
        self.decay_alpha = val
        self.decay_label.setText(f"Decay α: {val:.1f}")
    
    def _update_damping(self, val: float):
        self.propagator.damping = val
        self.damping_label.setText(f"Damping: {val:.3f}")

    def compute_ripple_numpy(self, t: float, frequencies: np.ndarray):
        self._compute_ripple(np, t, frequencies)

    def compute_ripple_cupy(self, t: float, frequencies: np.ndarray):
        frequencies = cp.asarray(frequencies)
        self._compute_ripple(cp, t, frequencies)

    def _compute_ripple(self, xp, t: float, frequencies):
        N, k = frequencies.shape

        x0 = xp.array([p[0] for p in self.source_positions]).reshape(N, 1, 1)
        y0 = xp.array([p[1] for p in self.source_positions]).reshape(N, 1, 1)

        xs = self.xs[None, :, :]
        ys = self.ys[None, :, :]

        r_pixels = xp.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
        r_meters = r_pixels * self.dx

        decay = xp.exp(-self.decay_alpha * r_meters)

        frequencies = xp.clip(frequencies, 1e-3, self.max_frequency)
        wavelengths = self.speed / frequencies
        phases = 2 * xp.pi * frequencies * t

        # for computing:
        #   ripple[n, k, h, w] = amplitude * decay[n, 1, h, w] * sin(phase[n, k, 1, 1] - 2π * r[n, 1, h, w] / wavelength[n, k, 1, 1])
        r = r_meters[:, None, :, :]
        decay = decay[:, None, :, :]
        wavelengths = wavelengths[:, :, None, None]
        phases = phases[:, :, None, None]

        propagation_limit = self.speed * self.time
        mask = r <= propagation_limit

        ripple = self.amplitude * decay * xp.sin(phases - 2 * xp.pi * r / wavelengths)
        # ripple *= mask
        
        self.propagator.add_excitation(ripple.sum(axis=(0, 1)))
        self.propagator.step()
        self.Z[:] = self.propagator.get_state()

    def update_visualization(self):
        self.time += self.dt

        if self.use_synthetic or self.processor is None:
            freqs = np.full((self.n_sources, 1), self.frequency, dtype=np.float32)
        else:
            top_k = self.processor.current_top_k_frequencies
            top_k = [f for f in top_k if f is not None and np.isfinite(f)]
            if len(top_k) == 0:
                return
            k = len(top_k)
            freqs = np.tile(top_k, (self.n_sources, 1))
            # freqs = map_audio_freq_to_visual_freq(freqs, self.max_frequency)

        self.compute_ripple(self.time, freqs)

        Z_vis = cp.asnumpy(self.Z) if self.use_gpu else self.Z
        max_abs = np.max(np.abs(Z_vis))
        self.image_item.setLevels([-max_abs, max_abs])
        self.image_item.setImage(Z_vis, autoLevels=False)

class WavePropagatorCPU:
    def __init__(self, shape, dx, dt, speed, damping):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping

        self.Z = np.zeros(shape, dtype=np.float32)
        self.Z_old = np.zeros_like(self.Z)
        self.Z_new = np.zeros_like(self.Z)

        self.c2_dt2 = (self.c * self.dt / self.dx)**2

    def add_excitation(self, excitation: np.ndarray):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        Z = self.Z
        laplacian = (
            -4 * Z +
            np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
        )
        self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian)
        self.Z_new *= self.damping
        self.Z_old = Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self):
        return self.Z

    def reset(self):
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0


class WavePropagatorGPU:
    def __init__(self, shape, dx, dt, speed, damping):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping

        self.Z = cp.zeros(shape, dtype=cp.float32)
        self.Z_old = cp.zeros_like(self.Z)
        self.Z_new = cp.zeros_like(self.Z)

        self.c2_dt2 = (self.c * self.dt / self.dx)**2

    def add_excitation(self, excitation: cp.ndarray):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        Z = self.Z
        laplacian = (
            -4 * Z +
            cp.roll(Z, 1, axis=0) + cp.roll(Z, -1, axis=0) +
            cp.roll(Z, 1, axis=1) + cp.roll(Z, -1, axis=1)
        )
        self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian)
        self.Z_new *= self.damping
        self.Z_old = Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self):
        return self.Z

    def reset(self):
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
