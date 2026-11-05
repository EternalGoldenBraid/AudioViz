import time
from collections.abc import Callable

from PyQt5 import QtCore, QtWidgets
from loguru import logger
import pyqtgraph as pg
# pg.setConfigOptions(useOpenGL=True) 
# pg.setConfigOptions(antialias=True)
import numpy as np
import librosa as lr
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize

from .visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.sources.orchestrator import AUDIO_MODELS, RippleSourceOrchestrator
from .audio_history_view import COCHLEA_COLORMAP, COCHLEA_DB_RANGE, cochlea_db, spectrum_db, waveform_envelope

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
        self.canvas_provider: Callable[[], np.ndarray] | None = None
        self.landscape = None
        self._landscape_failed = False
        self._last_frame = -1
        self._last_3d_update = -float("inf")
        self.audio_model_owner: RippleSourceOrchestrator | None = None

        self.spectrogram_plot_item: pg.PlotItem
        self.spectrogram_view: pg.ImageView
        self.spectrogram_y_labels = processor.freq_bins
        self.waveform_plot: pg.PlotItem
        self.waveform_curve: pg.PlotDataItem
        
        self.waveform_y_range: tuple = (-1, 1)

        if not hasattr(processor, 'n_spec_bins'):
            raise AttributeError(
                f"{self.__class__.__name__}: " \
                "requires the processor to have 'n_spec_bins' attribute.")
        self.spectrogram_y_range: tuple = (0, processor.n_spec_bins)

        self.init_ui()

    def init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.landscape_toggle = QtWidgets.QCheckBox("3D audio landscape")
        self.landscape_toggle.toggled.connect(self._toggle_landscape)
        controls.addWidget(self.landscape_toggle)
        reset = QtWidgets.QPushButton("Isometric view")
        reset.clicked.connect(self._reset_landscape)
        controls.addWidget(reset)
        controls.addStretch()
        controls.addWidget(QtWidgets.QLabel("Audio model"))
        self.audio_model_selector = QtWidgets.QComboBox()
        self.audio_model_selector.addItems(AUDIO_MODELS)
        self.audio_model_selector.setEnabled(False)
        self.audio_model_selector.setToolTip("Selects the running graph's audio representation.")
        controls.addWidget(self.audio_model_selector)
        layout.addLayout(controls)
        self.landscape_help = QtWidgets.QLabel(
            "Drag to orbit; wheel to zoom; Ctrl-drag to pan. "
            "Source releases tracers; signed canvas channels steer x/y/z, unchanged through depth.\n"
            "Waveform shapes the fixed source; the fading wake carries its past along the field. "
            f"Spectrogram: {'mel bands' if self.processor.n_mels is not None else 'frequency (Hz)'} "
            f"/ magnitude [{self.norm.vmin:g}, {self.norm.vmax:g}] dB. "
            "No canvas field means no drift. Display-only advection, not electromagnetic simulation."
        )
        self.landscape_help.setWordWrap(True)
        self.landscape_help.hide()
        layout.addWidget(self.landscape_help)
        self.landscape_error = QtWidgets.QLabel()
        self.landscape_error.setWordWrap(True)
        self.landscape_error.hide()
        layout.addWidget(self.landscape_error)
    
        # Spectrogram plot using ImageItem
        self.spectrogram_plot_item = pg.PlotItem(title="Spectrogram | diagnostic signal analysis")
        self.spectrogram_plot_item.getViewBox().invertY(False)
    
        # Set axis labels
        self.spectrogram_plot_item.setLabel('bottom', "Time relative to latest audio", units="s")
        self.spectrogram_plot_item.setLabel('left', "Frequency", units="Hz")
        self.spectrogram_plot_item.setYRange(*self.spectrogram_y_range)
        # Set freq labels
        self.spectrogram_plot_item.setYRange(0, self.processor.n_spec_bins)
    
        self.spectrogram_image_item = pg.ImageItem(axisOrder="row-major")
        self.spectrogram_plot_item.addItem(self.spectrogram_image_item)
    
        # Pack into GraphicsLayoutWidget
        spectrogram_widget = pg.GraphicsLayoutWidget()
        self.spectrogram_widget = spectrogram_widget
        spectrogram_widget.addItem(self.spectrogram_plot_item)
        spectrum_lut = self.cmap(np.linspace(0, 1, 256), bytes=True)
        self.spectrogram_colorbar = pg.ColorBarItem(
            values=(self.norm.vmin, self.norm.vmax),
            colorMap=pg.ColorMap(np.linspace(0, 1, 256), spectrum_lut),
            label="dB re 1", interactive=False,
        )
        self.spectrogram_colorbar.setImageItem(self.spectrogram_image_item)
        spectrogram_widget.addItem(self.spectrogram_colorbar, row=0, col=1)
        spectrogram_widget.nextRow()
        self.cochlea_plot = spectrogram_widget.addPlot(title="Cochlea | enable the cochlea audio model")
        self.cochlea_plot.setLabel("left", "Oscillator index")
        self.cochlea_plot.setLabel("bottom", "Time relative to latest audio", units="s")
        self.cochlea_image = pg.ImageItem(axisOrder="row-major")
        self.cochlea_plot.addItem(self.cochlea_image)
        lut = colormaps[COCHLEA_COLORMAP](np.linspace(0, 1, 256), bytes=True)
        self.cochlea_colorbar = pg.ColorBarItem(
            values=COCHLEA_DB_RANGE, colorMap=pg.ColorMap(np.linspace(0, 1, 256), lut),
            label="dB re 1", interactive=False,
        )
        self.cochlea_colorbar.setImageItem(self.cochlea_image)
        spectrogram_widget.addItem(self.cochlea_colorbar, row=1, col=1)
        for plot in (self.spectrogram_plot_item, self.cochlea_plot):
            plot.getAxis("left").setWidth(85)
            plot.getAxis("bottom").setStyle(hideOverlappingLabels=False)
        self.spectrogram_plot_item.setXLink(self.cochlea_plot)
    
        layout.addWidget(spectrogram_widget)
    
        # Waveform plot
        self.waveform_plot = spectrogram_widget.addPlot(row=2, col=0, title="Raw waveform | fixed amplitude, peak-preserving envelope")
        self.waveform_plot.setLabel("bottom", "Time relative to latest audio", units="s")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setYRange(-1, 1, padding=0)
        self.waveform_plot.getAxis("left").setWidth(85)
        self.waveform_plot.getAxis("bottom").setStyle(hideOverlappingLabels=False)
        self.waveform_plot.setXLink(self.spectrogram_plot_item)
        self.waveform_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#44546a")))
        self.waveform_curve = self.waveform_plot.plot(pen="#5adc8c")
        self.waveform_low = self.waveform_plot.plot(pen="#5adc8c")
        self.waveform_plot.addItem(pg.FillBetweenItem(
            self.waveform_low, self.waveform_curve, brush=pg.mkBrush(90, 220, 140, 110),
        ))
        for row, stretch in enumerate((3, 3, 2)):
            spectrogram_widget.ci.layout.setRowStretchFactor(row, stretch)

    def set_canvas_provider(self, provider: Callable[[], np.ndarray]) -> None:
        self.canvas_provider = provider

    def set_audio_model_controller(self, owner: RippleSourceOrchestrator) -> None:
        self.audio_model_owner = owner
        self.audio_model_selector.setCurrentText(owner.audio_model)
        self.audio_model_selector.setEnabled(not owner.engine.use_shader)
        self.audio_model_selector.activated[str].connect(owner.set_audio_model)

    def _update_cochlea(self) -> None:
        if self.audio_model_owner is not None:
            self.audio_model_selector.setCurrentText(self.audio_model_owner.audio_model)
        if getattr(self.processor, "cochlea", None) is None:
            self.cochlea_image.clear()
            self.cochlea_plot.setTitle("Cochlea | enable the cochlea audio model")
            return
        history = self.processor.get_audio_history()
        if history is None or not history.cochlea.samples.size:
            self.cochlea_image.clear()
            self.cochlea_plot.setTitle("Cochlea | waiting for audio")
            return
        columns = max(1, int(self.cochlea_plot.width()))
        levels = cochlea_db(history, columns)
        count = levels.shape[0]
        self.cochlea_image.setImage(levels, autoLevels=False)
        self.cochlea_image.setRect(QtCore.QRectF(-history.duration, 0.5, history.duration, count))
        self.cochlea_plot.setXRange(-history.duration, 0, padding=0)
        self.cochlea_plot.getAxis("bottom").setTicks([[(float(t), f"{t:g}") for t in np.linspace(-history.duration, 0, 5)]])
        self.cochlea_plot.setYRange(0.5, count + 0.5, padding=0)
        indices = np.unique(np.linspace(1, count, min(5, count)).astype(int))
        self.cochlea_plot.getAxis("left").setTicks([[(int(i), str(i)) for i in indices]])
        self.cochlea_plot.setTitle("Cochlea | RMS oscillator state, low index at bottom")

    def _update_spectrum(self) -> None:
        history = self.processor.get_audio_history()
        if history is None:
            return
        columns = max(1, int(self.spectrogram_plot_item.width()))
        self.spectrogram_image_item.setImage(spectrum_db(history, columns), autoLevels=False)
        count = len(history.frequencies)
        self.spectrogram_image_item.setRect(QtCore.QRectF(-history.duration, -.5, history.duration, count))
        self.spectrogram_plot_item.setXRange(-history.duration, 0, padding=0)
        self.spectrogram_plot_item.setYRange(-.5, count - .5, padding=0)
        self.spectrogram_plot_item.setLabel("left", "Hz (mel)" if history.mel_power else "Hz", units=None)
        indices = np.unique(np.linspace(0, count - 1, min(5, count)).astype(int))
        self.spectrogram_plot_item.getAxis("left").setTicks([
            [(int(i), f"{history.frequencies[i]:.0f}") for i in indices]
        ])
        self.spectrogram_plot_item.getAxis("bottom").setTicks([
            [(float(t), f"{t:g}") for t in np.linspace(-history.duration, 0, 5)]
        ])

    def _toggle_landscape(self, enabled: bool) -> None:
        if enabled and self.landscape is None:
            try:
                from .audio_landscape_view import AudioLandscapeView
            except ImportError as error:
                self._landscape_error(str(error))
                return
            history = self.processor.num_samples_in_buffer / self.processor.sr
            self.landscape = AudioLandscapeView(cmap=self.cmap, history_seconds=history, parent=self)
            self.landscape.rendering_failed.connect(self._landscape_error)
            self.layout().addWidget(self.landscape, stretch=1)
        if self.landscape is not None:
            self.landscape.setVisible(enabled)
            if not enabled:
                self.landscape.clear_audio()
        self.spectrogram_widget.setVisible(not enabled)
        self.waveform_plot.setVisible(not enabled)
        self.landscape_help.setVisible(enabled)
        self._last_frame = -1

    def _landscape_error(self, message: str) -> None:
        logger.error("3D audio landscape unavailable: {}", message)
        self._landscape_failed = True
        self.landscape_toggle.setChecked(False)
        self.landscape_toggle.setEnabled(False)
        self.landscape_error.setText(f"3D unavailable: {message}. The 2D plots remain available.")
        self.landscape_error.show()

    def _reset_landscape(self) -> None:
        if self.landscape is not None:
            self.landscape.reset_view()

    def update_visualization(self) -> None:
        """
        Currently averaging spectrograms across channels.
        """
        landscape_active = self.landscape_toggle.isChecked() and not self._landscape_failed
        now = time.monotonic()
        if landscape_active and (not self.isVisible() or now - self._last_3d_update < .1):
            return
        if landscape_active:
            self.landscape.set_canvas(self.canvas_provider() if self.canvas_provider is not None else None)
            self._last_3d_update = now
        self._update_cochlea()
        snapshot = self.processor.get_latest_snapshot()
        if snapshot is None:
            if landscape_active:
                self.landscape.advance_source(now)
            return  # No new data yet
        
        audio, spectrograms = snapshot
        
        spectrogram = np.mean(spectrograms, axis=0)
        if landscape_active:
            mean_waveform = np.mean(audio, axis=1)
            n_samples = max(1, int(self.waveform_plot_duration * self.processor.sr))
            mean_waveform = mean_waveform[-n_samples:]
            # STFT stores magnitudes; mel analysis stores power.
            to_db = lr.power_to_db if self.processor.n_mels is not None else lr.amplitude_to_db
            levels = np.asarray(self.norm(to_db(spectrogram, ref=1.0, top_db=None)))
            frame = self.processor.frame_counter
            self.landscape.set_audio(
                mean_waveform, levels, timestamp=now, new_frame=frame != self._last_frame,
            )
            self._last_frame = frame
            return
        self._update_spectrum()
        
        history = self.processor.get_audio_history()
        times, low, high = waveform_envelope(history, max(1, int(self.waveform_plot.width())))
        self.waveform_curve.setData(times, high)
        self.waveform_low.setData(times, low)
        self.waveform_plot.setYRange(-1, 1, padding=0)
        self.waveform_plot.getAxis("bottom").setTicks([
            [(float(t), f"{t:g}") for t in np.linspace(-history.duration, 0, 5)]
        ])
