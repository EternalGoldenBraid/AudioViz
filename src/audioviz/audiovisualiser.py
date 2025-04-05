from typing import Optional, Union
from pathlib import Path
from functools import partial

from loguru import logger

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import librosa as lr
import sounddevice as sd
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.fft import fft

def get_stft_spectrogram(segment: np.ndarray, hop_length: int,
                        stft_window: np.ndarray, n_fft: int) -> np.ndarray:

    spectrogram = np.abs(
        lr.stft(segment, n_fft=stft_window.shape[0],
                hop_length=hop_length, window=stft_window
        ))

    return spectrogram

def get_mel_spectrogram(segment: np.ndarray,
                         stft_window: np.ndarray, n_fft: int,
                         hop_length: int, n_mels: int, sr: int) -> np.ndarray:

    spectrogram = lr.feature.melspectrogram(
        y=segment, sr=sr, window=stft_window,
        n_fft=stft_window.shape[0], hop_length=hop_length, n_mels=n_mels
    )
    return spectrogram

class AudioVisualizer(QtWidgets.QMainWindow):
    def __init__(self,
            sr: int,
            n_fft: int,
            hop_length: int,
            num_samples_in_plot_window: int,
            mel_spec_max: float,
            stft_window: np.ndarray,
            cmap: cm.colors.Colormap,
            norm: Normalize,
            plot_update_interval: int,
            n_mels: Optional[int] = None,
            data: Optional[np.ndarray] = None,
            is_streaming: bool = False,
            input_device_index: Optional[int] = None,
            input_channels: Optional[int] = None,
            output_device_index: Optional[int] = None,
            output_channels: Optional[int] = None,
            io_blocksize: int = 512,
        ):

        super().__init__()
        self.sr: int = sr
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.n_mels: int = n_mels
        self.num_samples_in_plot_window: int = num_samples_in_plot_window
        self.mel_spec_max: Optional[float] = mel_spec_max
        self.cmap: cm.colors.Colormap = cmap
        self.norm: Normalize = norm
        self.data: np.ndarray = data
        self.stft_window: Union[str, tuple, np.ndarray] = stft_window

        self.is_streaming: bool = is_streaming
        self.io_blocksize: int = io_blocksize

        if None not in (input_device_index, input_channels, output_device_index, output_channels):
            self.input_channels = input_channels
            self.output_channels = output_channels
        elif data is not None:
            assert data.ndim == 2, f"Data shape is {data.shape}"
            self.input_channels = data.shape[1] 


        self.audio_buffer: np.ndarray = np.zeros((num_samples_in_plot_window, self.input_channels))

        n_spec_bins = n_mels if n_mels is not None else n_fft // 2 + 1
        n_spec_frames = num_samples_in_plot_window // hop_length
        self.spectrogram_buffer: np.ndarray = np.zeros(
                    (n_spec_bins, n_spec_frames))
        if n_mels is None:
            # self.get_spectrogram = get_stft_spectrogram
            self.get_spectrogram = partial( get_stft_spectrogram,
                stft_window=stft_window, n_fft=n_fft, hop_length=hop_length
            )
        else:
            # self.get_spectrogram = get_mel_spectrogram
            self.get_spectrogram = partial( get_mel_spectrogram,
                stft_window=stft_window, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, sr=sr
            )

        self.setWindowTitle("Real-Time Audio Visualization")

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Plot for mel spectrogram
        self.spectogram_plot_item = pg.PlotItem(title="Mel Spectrogram")
        # self.spectogram_plot_item.setYRange(0, n_mels)
        self.spectogram_plot = pg.ImageView(view=self.spectogram_plot_item)
        self.spectogram_plot.getView().setLabel("bottom", "Time (s)")
        self.spectogram_plot.getView().setLabel("left", "Mel Filters")
        layout.addWidget(self.spectogram_plot)
        self.spectogram_plot_item.getViewBox().invertY(False)
        self.spectogram_plot_item.getViewBox().setLimits(yMin=0, yMax=n_spec_bins)
        # self.spectogram_plot_item.getViewBox().setLimits(xMin=-10*sr, xMax=10*sr)
        # self.spectogram_plot_item.getViewBox().setLimits(xMin=0, xMax=235)
        self.spectogram_plot_item.getViewBox().setAspectLocked(False)

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
            samplerate=sr, device=output_device_index, channels=self.output_channels,
            callback=self.audio_output_callback, blocksize=self.io_blocksize,
            # channels=2,
            # device=8,
        )
        if self.is_streaming:
            assert input_device_index is not None,\
                    f"Input device index is {input_device_index}"
            self.input_stream = sd.InputStream(
                device=input_device_index, channels=self.input_channels, samplerate=self.sr,
                blocksize=self.io_blocksize, callback=self.audio_input_callback,
                dtype='float32'
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

        spectrogram = self.spectrogram_buffer.copy()
        spectrogram = lr.power_to_db(spectrogram, ref=np.max)

        # Map to Matplotlib colormap
        rgba_img = self.cmap(self.norm(spectrogram.T))  # Map dB values to RGBA
        rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit

        # Update mel spectrogram plot
        self.spectogram_plot.setImage(
            rgb_img, autoLevels=False, autoRange=False, levels=(0, 255)
        )

        # Update waveform plot
        self.waveform_curve.setData(np.mean(self.audio_buffer, axis=1)[-self.hop_length:])

        # Advance current time
        self.current_time += self.hop_length / self.sr

    def audio_input_callback(self, indata: np.ndarray, frames: int, time, status):
        """Handles real-time audio input from the microphone."""
        if status:
            print(f"Input Stream Error: {status}")

        # Shift the buffer and store new mic data
        self.audio_buffer[:] = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata
        logger.debug(f"indata {indata.shape}: {indata}")

        # Compute the spectrogram for the current audio frame
        segment = np.mean(indata, axis=1)
        spectrogram = self.get_spectrogram(segment=segment)
        logger.debug(f"Spectrogram shape: {spectrogram.shape}")
        # Store the spectrogram in the buffer
        frames_spec = spectrogram.shape[1]
        self.spectrogram_buffer[:] = np.roll(self.spectrogram_buffer, -frames_spec)
        self.spectrogram_buffer[:, -frames_spec:] = spectrogram

        # self.spectrogram_buffer[:] = lr.power_to_db(
        #     self.spectrogram_buffer, ref=np.max(self.spectrogram_buffer))

    def audio_output_callback(self, outdata, frames, time, status):
        """
        Handles real-time audio processing,
        whether from a device or a preloaded array.
        """
        
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
        outdata[:] = self.audio_buffer[-frames:]
    
        # Advance playback time
        self.current_time += frames / self.sr
