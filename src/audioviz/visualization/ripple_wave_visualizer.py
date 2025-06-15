from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor

class RippleWaveVisualizer(VisualizerBase):
    """
    A visualizer that simulates ripple waves on a 2D plane, either using synthetic data or real-time audio input.
    """
    def __init__(self,
                 processor: Optional[AudioProcessor] = None,
                 n_sources: int = 1,
                 plane_size=(200, 200),
                 frequency: float = 440.0,
                 amplitude: float = 1.0,
                 speed: float = 340.0,  # speed of wave propagation (m/s)
                 use_synthetic: bool = True,
                 **kwargs):

        super().__init__(processor, **kwargs)

        self.processor = processor
        self.use_synthetic: bool = processor is None or use_synthetic

        self.n_sources: int = n_sources
        self.plane_size: Tuple[int, int] = plane_size
        self.frequency: float = frequency
        self.amplitude: float = amplitude
        self.speed: float = speed
        self.time: float = 0.0
        self.dt: float = 1 / 60.0  # 60 FPS update
        self.Z = np.zeros(self.plane_size, dtype=np.float32)

        self.source_positions = [
            (plane_size[0] // 2, plane_size[1] // 2)
        ]

        self.Z = np.zeros(plane_size, dtype=np.float32)
        self.xs, self.ys = np.meshgrid(np.arange(plane_size[1]), np.arange(plane_size[0]))

        self.image_item = pg.ImageItem()
        self.image_item.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.image_item.setLevels([-1, 1])

        self.plot = pg.PlotItem()
        self.plot.setTitle("Ripple Simulation")
        self.plot.addItem(self.image_item)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.addItem(self.plot)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

    def compute_ripple(self, t: float, frequency: float):
        if frequency <= 0:
            return
        wavelength = self.speed / (frequency if frequency != 0 else 1e-6)
        for (x0, y0) in self.source_positions:
            r = np.sqrt((self.xs - x0) ** 2 + (self.ys - y0) ** 2)
            self.Z[:] += self.amplitude * np.sin(2 * np.pi * self.frequency * t - 2 * np.pi * r / wavelength)

    def update_visualization(self):
        self.Z[:] = 0  # Reset Z for each update

        if self.use_synthetic:
            self.time += self.dt
            self.compute_ripple(self.time, frequency=self.frequency)

        if self.processor is not None and self.processor.current_top_k_frequencies:
            self.time += self.dt
            freqs = self.processor.current_top_k_frequencies
            print(f"Current top frequencies: {freqs}")
            if freqs:
                self.compute_ripple(self.time, frequency=freqs[0])
                self.compute_ripple(self.time, frequency=freqs[1])
                self.compute_ripple(self.time, frequency=freqs[2])

        if self.use_synthetic and self.processor is None:
            self.image_item.setImage(self.Z, autoLevels=True)

        self.image_item.setImage(self.Z, autoLevels=False)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
