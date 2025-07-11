import sys
import signal
from pathlib import Path
from typing import Union, Optional, Dict

from PyQt5 import QtWidgets, QtCore
import numpy as np
import librosa as lr
from matplotlib import cm
from matplotlib.colors import Normalize
from loguru import logger

from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer
from audioviz.visualization.pitch_helix_visualizer import PitchHelixVisualizer
from audioviz.utils.audio_devices import select_devices
from audioviz.utils.audio_devices import AudioDeviceDesktop 
from audioviz.utils.guitar_profiles import GuitarProfile  

def main():
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
        # "io_blocksize": 2048,
        # "io_blocksize": 1024,
        # "io_blocksize": 512,
    }
    
    # Spectrogram parameters
    n_fft = 256
    window_duration = 20  # ms
    # window_duration = 500  # ms
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
        "num_samples_in_plot_window": int(5.0 * sr),
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
        number_top_k_frequencies=n_fft//2,
    )
    
    # Create a processing timer
    block_duration_ms = (io_config["io_blocksize"] / sr) * 1000
    processing_timer = QtCore.QTimer()
    # processing_timer.setInterval(20)  # e.g., 50 Hz
    processing_timer.setInterval(int(block_duration_ms*(1 - 1e-3))) 
    processing_timer.timeout.connect(processor.process_pending_audio)
    processing_timer.start()
    
    # Visualizer
    show_spectrogram = True 
    if show_spectrogram == True:
        visualizer = SpectrogramVisualizer(
            processor=processor,
            cmap=plotting_config["cmap"],
            norm=plotting_config["norm"],
            waveform_plot_duration=plotting_config["waveform_plot_duration"],
        )
        visualizer.setWindowTitle("Audio Visualizer")
        visualizer.resize(800, 600)
        visualizer.show()
    
    # Create Pitch Helix Visualizer
    show_helix = False
    if show_helix:
        standard_guitar = GuitarProfile(
            open_strings=[82.41, 110.00, 146.83, 196.00, 246.94, 329.63],
            num_frets=22
        )
        
        dadgad_guitar = GuitarProfile(
            open_strings=[73.42, 110.00, 146.83, 196.00, 220.00, 293.66],
            num_frets=22
        )
        
        helix_window = PitchHelixVisualizer(
            processor=processor,
            guitar_profile=standard_guitar,
        )
        helix_window.setWindowTitle("Pitch Helix Visualizer")
        helix_window.resize(800, 600)
        helix_window.show()
    
    # Create Ripple Wave Visualizer
    show_ripples = True
    ripple_config = {
        # "use_synthetic": True,  # Set to True for synthetic data
        "use_synthetic": False,  # Set to True for synthetic data
        "n_sources": 4,
        # "plane_size_m": (10.20, 10.20),  # meters
        "plane_size_m": (0.30, 0.30),  # meters
        "resolution":  (1440, 2560),  # pixels (H, W)
        "frequency": 1.0,  # Hz
        # "amplitude": 1.0,
        "amplitude": 10.0,
        # "speed": 1e-4,  # m/s
        "speed": 340.0,  # m/s
        # "speed": 34.0,  # m/s
        "damping": 0.90,  # damping factor
        # "damping": 0.1,  # damping factor
    }
    
    if show_ripples:
        ripple_window = RippleWaveVisualizer(
            processor=processor,
            **ripple_config
        )
        ripple_window.setWindowTitle("Ripple Wave Visualizer (Synthetic)")
        ripple_window.resize(600, 600)
        ripple_window.show()
    
    # Start audio
    processor_success = processor.start()
    if not processor_success:
        logger.error("Failed to start audio processing. Exiting.")
        return
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.aboutToQuit.connect(processor.stop)
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Exiting...")
        processor.stop()
        app.quit()

if __name__ == "__main__":
    main()
