from pathlib import Path


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
            audio_len: int,
            mel_spec_max: float,
            cmap: cm.colors.Colormap,
            norm: Normalize,
            plot_update_interval: int,
        ):

        super().__init__()
        self.sr: int = sr
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.n_mels: int = n_mels
        self.mel_filters: np.ndarray = mel_filters
        self.window_samples: int = window_samples
        self.audio_len: int = audio_len
        self.mel_spec_max: float = mel_spec_max

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
            callback=self.audio_callback, blocksize=hop_length,
            # device=0,
        )

    def start(self):
        """Start the audio playback and visualization."""
        self.stream.start()
        self.timer.start()

    def stop(self):
        """Stop the audio playback and visualization."""
        self.stream.stop()
        self.timer.stop()

    def update_plot(self):
        """Update the mel spectrogram and waveform."""
        start_idx = int(self.current_time * self.sr)
        end_idx = min(start_idx + self.window_samples, self.audio_len)

        # Compute mel spectrogram segment
        segment = data[start_idx:end_idx]
        mel_spectrogram = lr.feature.melspectrogram(
            y=segment, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        # mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=self.mel_spec_max)

        # Map to Matplotlib colormap
        rgba_img = cmap(norm(mel_spectrogram.T))  # Map dB values to RGBA
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
        self.current_time += hop_length / sr

    def audio_callback(self, outdata, frames, time, status):
        """Stream audio data to the playback device."""
        start_idx = int(self.current_time * self.sr)
        end_idx = start_idx + frames
        if end_idx > len(data):
            outdata[: len(data[start_idx:]), 0] = data[start_idx:]
            raise sd.CallbackStop()
        else:
            outdata[:, 0] = data[start_idx:end_idx]
        self.current_time += frames / self.sr


if __name__ == "__main__":
    # Load audio file
    data_path: Path = Path("/home/nicklas/Projects/AudioViz/data")
    # audio_file = "estas_tonte.wav"
    # audio_file = "aaaa.wav"
    # audio_file = "savu.wav"
    # audio_file = "drums.wav"
    audio_file = data_path/"ex1.wav"
    data, sr = lr.load(audio_file, sr=None)
    
    # Spectrogram parameters
    stft_window = 20  # ms 
    # stft_window = 80  # ms 
    # stft_window = 40  # ms 
    # n_fft = int(0.140 * sr)  # x ms window
    n_fft = int(stft_window * sr / 1000)
    hop_length = n_fft // 2
    n_mels = 140
    mel_filters = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    
    # Windowing parameters
    window_duration = 2.0  # Duration of spectrogram view in seconds
    window_samples = int(window_duration * sr)
    audio_len = len(data)
    
    
    mel_spectrogram = lr.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
    mel_spec_max = np.max(mel_spectrogram)
    
    # Load colormap
    cmap = cm.get_cmap('viridis')  # Replace 'viridis' with your choice
    norm = Normalize(vmin=-80, vmax=0)  # Typical decibel range for spectrogram
    
    plot_update_interval = 50  # Update plot every 50 ms
    
    # Run the application
    app = QtWidgets.QApplication([])
    window = AudioVisualizer(
        sr=int(sr),
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        mel_filters=mel_filters,
        window_samples=window_samples,
        audio_len=audio_len,
        mel_spec_max=mel_spec_max,
        cmap=cmap,
        norm=norm,
        plot_update_interval=plot_update_interval,
    )
    window.show()
    window.start()
    app.exec()
