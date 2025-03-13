from pathlib import Path
from typing import Union, Optional, Dict

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

# Check if streaming from microphone or using file data
# is_streaming = False
is_streaming = True

if is_streaming:
    data = None  # No data loading needed for streaming
    device_enum = AudioDeviceDesktop
    input_device_index: int = device_enum.SCARLETT_SOLO_USB.value
    output_device_index: int = device_enum.SCARLETT_SOLO_USB.value

    assert input_device_index == 8, "Input device index is not 8"
    assert output_device_index == 8, "output device index is not 8"
    sr: Union[int,float] = 44100
else:
    data_path: Path = Path("/home/nicklas/Projects/AudioViz/data")
    # audio_file = "estas_tonte.wav"
    # audio_file = "aaaa.wav"
    # audio_file = "savu.wav"
    # audio_file = "drums.wav"
    # audio_file = data_path/"ex1.wav"
    audio_file = data_path/"test.wav"
    data, sr = lr.load(audio_file, sr=None)

    input_device_index = None
    output_device_index = None

io_config: Dict = {
    "is_streaming": is_streaming,
    "input_device_index": input_device_index,
    "output_device_index": output_device_index,
    "io_blocksize": 128*8
}

# Spectrogram parameters
n_fft = 2048


spectrogram_params = {
    "stft_window" : 30, # ms
    "n_fft" : n_fft,
    "hop_length" : n_fft // 2,
    "n_mels" : 140,
    "stft_window" : lr.filters.get_window("hann", n_fft),
}

# mel_filters = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
spectrogram_params["mel_filters"] = lr.filters.mel(
    n_fft=spectrogram_params["n_fft"],
    n_mels=spectrogram_params["n_mels"],
    sr=sr, 
)


if not is_streaming:
    mel_spectrogram = lr.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
    spectrogram_params["mel_spec_max"] = np.max(mel_spectrogram)
else:
    # spectrogram_params["mel_spec_max"] = 1.0
    spectrogram_params["mel_spec_max"] = 0.5

# Load colormap
cmap = cm.get_cmap('viridis')  # Replace 'viridis' with your choice
norm = Normalize(vmin=-80, vmax=0)  # Typical decibel range for spectrogram
plot_update_interval = 50  # Update plot every 50 ms

# Spectrogram window parameters
plot_window_duration = .5  # Duration of spectrogram view in seconds
plot_window_samples = int(plot_window_duration * sr)

plotting_config = {
    "cmap": cmap,
    "norm": norm,
    "plot_update_interval": plot_update_interval,
    "window_samples" : plot_window_samples,
}

# Run the application
app = QtWidgets.QApplication([])
window = AudioVisualizer(
    sr=int(sr),
    data=data,
    # n_fft=n_fft,
    # hop_length=hop_length,
    # n_mels=n_mels,
    # mel_filters=mel_filters,
    # window=window,
    # window_samples=window_samples,
    # mel_spec_max=mel_spec_max,
    **spectrogram_params,

    # cmap=cmap,
    # norm=norm,
    # plot_update_interval=plot_update_interval,
    **plotting_config,
    # is_streaming=is_streaming,
    # input_device_index=input_device_index,
    # output_device_index=output_device_index,
    **io_config,
)
window.show()
window.start()
app.exec()
