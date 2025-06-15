from typing import Optional, Tuple

import numpy as np
import matplotlib.cm as cm
from PyQt5 import QtWidgets
import pyqtgraph as pg
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.signal_processing import map_audio_freq_to_visual_freq

class RippleWaveVisualizer(VisualizerBase):
    """
    A visualizer that simulates ripple waves on a 2D plane, either using synthetic data or real-time audio input.
    """
    def __init__(self,
                 processor: Optional[AudioProcessor] = None,
                 n_sources: int = 1,
                 plane_size_m: Tuple[float, float] = (0.36, 0.62),  # meters
                 resolution: Tuple[int, int] = (400, 400),  # pixels (H, W)
                 frequency: float = 440.0,
                 amplitude: float = 1.0,
                 speed: float = 340.0,  # speed of wave propagation (m/s)
                 damping: float = 0.999,  # damping factor
                 use_synthetic: bool = True,
                 apply_gaussian_smoothing: bool = False,
                 **kwargs):

        super().__init__(processor, **kwargs)

        self.processor = processor
        self.use_synthetic: bool = processor is None or use_synthetic

        self.n_sources: int = n_sources
        self.plane_size_m: Tuple[float, float] = plane_size_m
        self.resolution: Tuple[int, int] = resolution
        self.frequency: float = frequency
        self.apply_gaussian_smoothing = apply_gaussian_smoothing
        self.amplitude: float = amplitude
        self.speed: float = speed
        self.time: float = 0.0

        self.Z = np.zeros(self.resolution, dtype=np.float32)

        self.dx = self.plane_size_m[0] / self.resolution[0]  # meters per pixel (x)
        self.dy = self.plane_size_m[1] / self.resolution[1]  # meters per pixel (y)

        # self.dt = (max(self.dx, self.dy) / speed) * 1/np.sqrt(2)
        self.dt: float = 1 / 30.0  # 60 FPS update

        self.propagator: WavePropagator = WavePropagator(
            shape=self.resolution, dx=self.dx, dt=self.dt, 
            speed=self.speed, damping=damping)

        self.max_frequency = self.speed / (2 * max(self.dx, self.dy))

        self.source_positions = []
        np.random.seed(42)  # for reproducibility
        for i in range(n_sources):
            x = np.random.randint(0, resolution[0])
            y = np.random.randint(0, resolution[1])
            self.source_positions.append((x, y))

        self.xs, self.ys = np.meshgrid(np.arange(self.resolution[1]), np.arange(self.resolution[0]))

        self.image_item = pg.ImageItem()

        cmap = cm.get_cmap("seismic")
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.image_item.setLookupTable(lut)

        self.plot = pg.PlotItem()
        self.plot.setTitle("Ripple Simulation")
        self.plot.addItem(self.image_item)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.addItem(self.plot)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

    def compute_ripple(self, t: float, frequencies: np.ndarray):
        N, k = frequencies.shape
        H, W = self.resolution
        decay_alpha = 0.0

        x0 = np.array([p[0] for p in self.source_positions]).reshape(N, 1, 1)
        y0 = np.array([p[1] for p in self.source_positions]).reshape(N, 1, 1)

        xs = self.xs[None, :, :]
        ys = self.ys[None, :, :]

        r_pixels = np.sqrt((xs - x0)**2 + (ys - y0)**2)
        r_meters = r_pixels * self.dx

        decay = np.exp(-decay_alpha * r_meters)

        freqs = np.clip(frequencies, 1e-3, self.max_frequency)
        wavelengths = self.speed / freqs
        phases = 2 * np.pi * freqs * t

        r = r_meters[:, None, :, :]
        decay = decay[:, None, :, :]
        wavelengths = wavelengths[:, :, None, None]
        phases = phases[:, :, None, None]

        # Zero out ripple far from the source (simulate realistic support)
        propagation_limit = self.speed * self.time  # max distance wave can travel
        mask = r <= propagation_limit

        ripple = self.amplitude * decay * np.sin(phases - 2 * np.pi * r / wavelengths)
        # ripple *= mask  # suppress non-physical early propagation

        self.propagator.add_excitation(ripple.sum(axis=(0,1)))
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
            freqs = map_audio_freq_to_visual_freq(freqs, self.max_frequency)

        self.compute_ripple(self.time, freqs)

        Z_vis = self.Z
        max_abs = np.max(np.abs(Z_vis))
        self.image_item.setLevels([-max_abs, max_abs])
        self.image_item.setImage(Z_vis, autoLevels=False)

class WavePropagator:
    def __init__(self, shape, dx=0.001, dt=0.0001, speed=1.0, damping=0.999):
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
        return self.Z.copy()

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
