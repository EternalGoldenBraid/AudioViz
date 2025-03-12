from typing import Optional
from pathlib import Path

from loguru import logger

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import librosa as lr
import sounddevice as sd
from matplotlib import cm
from matplotlib.colors import Normalize


class AudioVisualizer(QtWidgets.QMainWindow):
    def __init__(self,
            sr: int,
            n_fft: int,
            hop_length: int,
            n_mels: int,
            mel_filters: np.ndarray,
            window_samples: int,
            mel_spec_max: float,
            cmap: cm.colors.Colormap,
            norm: Normalize,
            plot_update_interval: int,
            data: Optional[np.ndarray] = None,
            is_streaming: bool = False,
            device_index: Optional[int] = None,
        ):

        super().__init__()
        self.sr: int = sr
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.n_mels: int = n_mels
        self.mel_filters: np.ndarray = mel_filters
        self.window_samples: int = window_samples
        self.mel_spec_max: Optional[float] = mel_spec_max
        self.cmap: cm.colors.Colormap = cmap
        self.norm: Normalize = norm
        self.data: np.ndarray = data

        self.is_streaming: bool = is_streaming
        self.device_index: Optional[int] = device_index
        self.audio_buffer: np.ndarray = np.zeros(10*window_samples)

        self.setWindowTitle("Real-Time Audio Visualization")

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Plot for mel spectrogram
        self.mel_plot_item = pg.PlotItem(title="Mel Spectrogram")
        # self.mel_plot_item.setYRange(0, n_mels)
        self.mel_plot = pg.ImageView(view=self.mel_plot_item)
        self.mel_plot.getView().setLabel("bottom", "Time (s)")
        self.mel_plot.getView().setLabel("left", "Mel Filters")
        layout.addWidget(self.mel_plot)
        self.mel_plot_item.getViewBox().invertY(False)
        self.mel_plot_item.getViewBox().setLimits(yMin=0, yMax=n_mels)
        # self.mel_plot_item.getViewBox().setLimits(xMin=-10*sr, xMax=10*sr)
        # self.mel_plot_item.getViewBox().setLimits(xMin=0, xMax=235)
        self.mel_plot_item.getViewBox().setAspectLocked(False)

        # Plot for waveform
        self.waveform_plot = pg.PlotWidget(title="Audio Signal")
        self.waveform_plot.setLabel("bottom", "Samples")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_curve = self.waveform_plot.plot(pen="y")
        layout.addWidget(self.waveform_plot)

        # Timer for updating plots
        self.timer = QtCore.QTimer()
        self.timer.setInterval(plot_update_interval)  # Update every 50 ms
        self.timer.timeout.connect(self.update_plot)

        # Audio playback variables
        self.current_time = 0
        self.stream = sd.OutputStream(
            samplerate=sr, channels=1,
            callback=self.audio_callback, blocksize=self.hop_length,
            # device=0,
        )
        if self.is_streaming:
            self.input_stream = sd.InputStream(
                device=self.device_index, channels=1, samplerate=self.sr,
                blocksize=self.hop_length, callback=self.audio_input_callback
            )

    def start(self):
        """Start the audio playback and visualization."""
        self.stream.start()
        self.timer.start()

        if self.is_streaming:
            self.input_stream.start()

    def stop(self):
        """Stop the audio playback and visualization."""
        self.stream.stop()
        self.timer.stop()

        if self.is_streaming:
            self.input_stream.stop()

    def update_plot(self):
        """Update the mel spectrogram and waveform."""

        # Compute mel spectrogram segment
        segment = self.audio_buffer
        mel_spectrogram = lr.feature.melspectrogram(
            y=segment, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        # mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=self.mel_spec_max)

        # Map to Matplotlib colormap
        rgba_img = self.cmap(self.norm(mel_spectrogram.T))  # Map dB values to RGBA
        rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit

        # Update mel spectrogram plot
        self.mel_plot.setImage(
            rgb_img, autoLevels=False, autoRange=False, levels=(0, 255)
        )

        # Update waveform plot
        self.waveform_curve.setData(segment)
        # self.waveform_plot.setXRange(0, len(segment))
        self.waveform_plot.setXRange(0, 0.0125*self.sr)

        # Advance current time
        self.current_time += self.hop_length / self.sr
        print(segment)

    def audio_input_callback(self, indata, frames, time, status):
        """Handles real-time audio input from the microphone."""
        if status:
            print(f"Input Stream Error: {status}")
    
        # Shift the buffer and store new mic data
        self.audio_buffer[:] = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata.flatten()
        logger.debug(f"indata: {indata}")

    def audio_callback(self, outdata, frames, time, status):
        """
        Handles real-time audio processing,
        whether from a device or a preloaded array.
        """
        
        # Shift the buffer to remove old data
        self.audio_buffer[:] = np.roll(self.audio_buffer, -frames)
    
        if not self.is_streaming:
            # If playing back preloaded audio
            start_idx = int(self.current_time * self.sr)
            end_idx = start_idx + frames
            if end_idx > len(self.data):
                indata = np.zeros(frames)  # Pad with silence if the audio is finished
                indata[: len(self.data[start_idx:])] = self.data[start_idx:]
                raise sd.CallbackStop()
            else:
                indata = self.data[start_idx:end_idx]
    
            # Store new audio data in the buffer
            self.audio_buffer[-frames:] = indata
    
        # Output the latest audio buffer segment
        outdata[:, 0] = self.audio_buffer[-frames:]
    
        # Advance playback time
        self.current_time += frames / self.sr
