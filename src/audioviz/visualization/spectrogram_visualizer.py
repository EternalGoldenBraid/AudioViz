from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np
import librosa as lr
from matplotlib import cm
from matplotlib.colors import Normalize

from .visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor

class SpectrogramVisualizer(VisualizerBase):
    def __init__(self,
                 processor: AudioProcessor,
                 cmap: cm.colors.Colormap,
                 norm: Normalize,
                 waveform_plot_duration: float,
                 parent: QtWidgets.QWidget = None):

        super().__init__(processor, parent=parent)

        self.cmap: cm.colors.Colormap = cmap
        self.norm: Normalize = norm
        self.waveform_plot_duration: float = waveform_plot_duration

        self.spectrogram_plot_item: pg.PlotItem
        self.spectrogram_view: pg.ImageView
        self.waveform_plot: pg.PlotWidget
        self.waveform_curve: pg.PlotDataItem
        
        self.waveform_y_range: tuple = (-1, 1)

        self.init_ui()

    def init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Spectrogram plot
        self.spectrogram_plot_item = pg.PlotItem(title="Spectrogram")
        self.spectrogram_view = pg.ImageView(view=self.spectrogram_plot_item)
        self.spectrogram_view.getView().setLabel("bottom", "Time (frames)")
        self.spectrogram_view.getView().setLabel("left", "Frequency bins")
        self.spectrogram_plot_item.getViewBox().invertY(False)
        layout.addWidget(self.spectrogram_view)

        # Waveform plot
        self.waveform_plot = pg.PlotWidget(title="Waveform")
        self.waveform_plot.setLabel("bottom", "Samples")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setYRange(
            self.waveform_y_range[0], self.waveform_y_range[1])
        self.waveform_curve = self.waveform_plot.plot(pen="y")
        layout.addWidget(self.waveform_plot)

    def update_visualization(self) -> None:
        # Update spectrogram
        spectrogram: np.ndarray = self.processor.get_spectrogram_buffer()
        spectrogram_db: np.ndarray = lr.power_to_db(spectrogram, ref=np.max)

        rgba_img: np.ndarray = self.cmap(
            self.norm(spectrogram_db.T))  # Shape (time, freq, rgba)
        rgb_img: np.ndarray = (rgba_img[:, :, :3] * 255).astype(np.uint8)

        self.spectrogram_view.setImage(
            rgb_img,
            autoLevels=False,
            autoRange=False,
            levels=(0, 255)
        )

        # Update waveform
        audio = self.processor.get_audio_buffer()
        mean_waveform = np.mean(audio, axis=1)

        n_samples = int(self.waveform_plot_duration * self.processor.sr)
        if len(mean_waveform) > n_samples:
            mean_waveform = mean_waveform[-n_samples:]

        self.waveform_curve.setData(mean_waveform)
