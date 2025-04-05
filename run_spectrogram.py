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

from audioviz.utils.audio_devices import select_devices

# Check if streaming from microphone or using file data
# is_streaming = False
is_streaming = True

if is_streaming:
    data = None  # No data loading needed for streaming
    device_enum = AudioDeviceDesktop
    # input_device_index: int = device_enum.SCARLETT_SOLO_USB.value
    # output_device_index: int = device_enum.SCARLETT_SOLO_USB.value

    config = select_devices(config_file=Path("outputs/audio_devices.json"))

    sr: Union[int,float] = config["samplerate"]
else:
    data_path: Path = Path("/home/nicklas/Projects/AudioViz/data")
    # audio_file = "estas_tonte.wav"
    # audio_file = "aaaa.wav"
    # audio_file = "savu.wav"
    # audio_file = "drums.wav"
    # audio_file = data_path/"ex1.wav"
    audio_file = data_path/"test.wav"
    data, sr = lr.load(audio_file, sr=None)

    # Audio I/O configuration
    config = {
        "input_device_index": None,
        "input_channels": None,
        "output_device_index": None,
        "output_channels": None,
        "sample_rate": sr,
    }


io_config: Dict = {
    "is_streaming": is_streaming,
    "input_device_index": config["input_device_index"],
    "input_channels": config["input_channels"],
    "output_device_index": config["output_device_index"],
    "output_channels": config["output_channels"],
    # "io_blocksize": 128*8
    # "io_blocksize": 2**10,
    "io_blocksize": 4096,
    # "io_blocksize": 128*4
}

# Spectrogram parameters
# n_fft = 2048
n_fft = 512
window_duration = 20 # ms
window_length = int((window_duration / 1000) * sr)
# Quantize window_length to nearest power of 2
window_length = 2**int(np.log2(window_length))


spectrogram_params = {
    "n_fft" : n_fft,
    "hop_length" : window_length // 4,
    # "n_mels" : 140,
    "n_mels" : None,
    # "stft_window" : lr.filters.get_window("hann", n_fft),
    "stft_window" : lr.filters.get_window(
        "hann", window_length),
}

if not is_streaming:
    mel_spectrogram = lr.feature.melspectrogram(
        n_fft=spectrogram_params["n_fft"], hop_length=spectrogram_params["hop_length"], 
        y=data, sr=sr, n_mels=spectrogram_params["n_mels"]
    )
    spectrogram_params["mel_spec_max"] = np.max(mel_spectrogram)
else:
    # spectrogram_params["mel_spec_max"] = 1.0
    spectrogram_params["mel_spec_max"] = 0.0

# Load colormap
cmap = cm.get_cmap('viridis')  # Replace 'viridis' with your choice
norm = Normalize(vmin=-80, vmax=0)  # Typical decibel range for spectrogram
plot_update_interval = 100  # Update plot every 50 ms

# Spectrogram window parameters
plot_window_duration = 10.5 # Duration of spectrogram view in seconds
plot_window_samples = int(plot_window_duration * sr)

plotting_config = {
    "cmap": cmap,
    "norm": norm,
    "plot_update_interval": plot_update_interval,
    "num_samples_in_plot_window" : plot_window_samples,
}

# Run the application
app = QtWidgets.QApplication([])
window = AudioVisualizer(
    sr=int(sr),
    data=data,
    **spectrogram_params,
    **plotting_config,
    **io_config,
)
window.show()
window.start()
app.exec()
