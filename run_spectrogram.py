from pathlib import Path

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import librosa as lr
import sounddevice as sd
from matplotlib import cm
from matplotlib.colors import Normalize

from audioviz.audiovisualiser import AudioVisualizer


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
    data=data,
)
window.show()
window.start()
app.exec()
