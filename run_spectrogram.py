from pathlib import Path
from typing import Union, Optional, Dict

from PyQt5 import QtWidgets
import numpy as np
import librosa as lr
from matplotlib import cm
from matplotlib.colors import Normalize

from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer
from audioviz.utils.audio_devices import select_devices  # Assuming you keep this
from audioviz.utils.audio_devices import AudioDeviceDesktop  # Assuming you keep this

# --- Config Phase ---

is_streaming = True

if is_streaming:
    data = None
    device_enum = AudioDeviceDesktop
    config = select_devices(config_file=Path("outputs/audio_devices.json"))
    sr: Union[int, float] = config["samplerate"]
else:
    data_path: Path = Path("/home/nicklas/Projects/AudioViz/data")
    audio_file = data_path / "test.wav"
    data, sr = lr.load(audio_file, sr=None)

    config = {
        "input_device_index": None,
        "input_channels": None,
        "output_device_index": None,
        "output_channels": None,
        "samplerate": sr,
    }

io_config: Dict = {
    "is_streaming": is_streaming,
    "input_device_index": config["input_device_index"],
    "input_channels": config["input_channels"],
    "output_device_index": config["output_device_index"],
    "output_channels": config["output_channels"],
    "io_blocksize": 4096,
}

# Spectrogram parameters
n_fft = 512
window_duration = 20  # ms
window_length = int((window_duration / 1000) * sr)
window_length = 2**int(np.log2(window_length))

spectrogram_params = {
    "n_fft": n_fft,
    "hop_length": window_length // 4,
    "n_mels": None,
    "stft_window": lr.filters.get_window("hann", window_length),
}

# Spectrogram dynamic range setup
if not is_streaming:
    mel_spectrogram = lr.feature.melspectrogram(
        n_fft=spectrogram_params["n_fft"],
        hop_length=spectrogram_params["hop_length"],
        y=data,
        sr=sr,
        n_mels=spectrogram_params["n_mels"]
    )
    spectrogram_params["mel_spec_max"] = np.max(mel_spectrogram)
else:
    spectrogram_params["mel_spec_max"] = 0.0

# Plotting configs
cmap = cm.get_cmap('viridis')
norm = Normalize(vmin=-80, vmax=0)
plot_update_interval = 100  # ms

plotting_config = {
    "cmap": cmap,
    "norm": norm,
    "plot_update_interval": plot_update_interval,
    "num_samples_in_plot_window": int(10.5 * sr),
    "waveform_plot_duration": 0.5,
}

# --- Run Phase ---

app = QtWidgets.QApplication([])

# Audio processor
processor = AudioProcessor(
    sr=int(sr),
    data=data,
    n_fft=spectrogram_params["n_fft"],
    hop_length=spectrogram_params["hop_length"],
    n_mels=spectrogram_params["n_mels"],
    stft_window=spectrogram_params["stft_window"],
    num_samples_in_buffer=plotting_config["num_samples_in_plot_window"],
    is_streaming=io_config["is_streaming"],
    input_device_index=io_config["input_device_index"],
    input_channels=io_config["input_channels"] or 1,
    output_device_index=io_config["output_device_index"],
    output_channels=io_config["output_channels"] or 1,
    io_blocksize=io_config["io_blocksize"],
)

# Visualizer
visualizer = SpectrogramVisualizer(
    processor=processor,
    cmap=plotting_config["cmap"],
    norm=plotting_config["norm"],
    waveform_plot_duration=plotting_config["waveform_plot_duration"],
)
visualizer.setWindowTitle("Audio Visualizer")
visualizer.resize(800, 600)
visualizer.show()

# Start audio
processor.start()

app.exec()
