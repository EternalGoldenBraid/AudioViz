from pathlib import Path
from typing import Union, Optional

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import librosa as lr
import sounddevice as sd
from matplotlib import cm
from matplotlib.colors import Normalize

from audioviz.audiovisualiser import AudioVisualizer
from audioviz.utils.audio_devices import (
    AudioDeviceMSI,
    AudioDeviceDesktop,
)

# Spectrogram parameters
stft_window = 30  # ms 
# n_fft = int(stft_window * sr / 1000)
n_fft = 2048
hop_length = n_fft // 2
n_mels = 40

# Check if streaming from microphone or using file data
# is_streaming = False
is_streaming = True

if is_streaming:
    data = None  # No data loading needed for streaming
    device_enum = AudioDeviceDesktop
    device_index: int = device_enum.SCARLETT_SOLO_USB.value
    sr: Union[int,float] = 44100
else:
    data_path: Path = Path("/home/nicklas/Projects/AudioViz/data")
    # audio_file = "estas_tonte.wav"
    # audio_file = "aaaa.wav"
    # audio_file = "savu.wav"
    # audio_file = "drums.wav"
    audio_file = data_path/"ex1.wav"
    data, sr = lr.load(audio_file, sr=None)


mel_filters = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

# Windowing parameters
window_duration = 2.0  # Duration of spectrogram view in seconds
window_samples = int(window_duration * sr)

if not is_streaming:
    mel_spectrogram = lr.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
    mel_spec_max = np.max(mel_spectrogram)
else:
    mel_spec_max = 1.0

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
    mel_spec_max=mel_spec_max,
    cmap=cmap,
    norm=norm,
    plot_update_interval=plot_update_interval,
    data=data,
    is_streaming=is_streaming,
)
window.show()
window.start()
app.exec()
