import sys
import signal
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Dict

from PyQt5 import QtWidgets, QtCore
import numpy as np
import librosa as lr
from matplotlib import cm
from matplotlib.colors import Normalize
from loguru import logger
import qdarkstyle

from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer
from audioviz.visualization.pitch_helix_visualizer import PitchHelixVisualizer
from audioviz.sources import (
    AudioSourceConfig,
    CameraFrameSourceConfig,
    RipplePoseConfig,
    RippleSourceOrchestratorConfig,
    SyntheticSourceConfig,
)
from audioviz.transforms.prediction_error import (
    PredictionErrorTransformConfig,
)
from audioviz.readouts import GaussianSpatialPrecision, PredictiveCodingConfig
from audioviz.sources.gaze import ActiveVisionConfig
from audioviz.utils.audio_devices import select_devices
from audioviz.utils.audio_devices import AudioDeviceDesktop 
from audioviz.utils.guitar_profiles import GuitarProfile  


# --- App config ---
APP_CONFIG = {
    "is_streaming": True,
    "windows": {
        "spectrogram": {
            "enabled": True,
            # "enabled": False,
            "title": "Audio Visualizer",
            "size": (800, 600),
        },
        "helix": {
            "enabled": False,
            "title": "Pitch Helix Visualizer",
            "size": (800, 600),
        },
        "ripples": {
            "enabled": True,
            "title": "Ripple Wave Visualizer",
            # "size": (600, 600),
            "size": (1920, 1080),
        },
    },
}

AUDIO_CONFIG = {
    "data_path": Path("/home/nicklas/Projects/AudioViz/data"),
    "file_name": "test.wav",
    "io_blocksize": 2096,
    "analysis": {
        "n_fft": 256,
        "window_duration_ms": 20,
        "n_mels": None,
    },
    "plotting": {
        "update_interval_ms": 100,
        "waveform_duration_s": 0.5,
        "window_duration_s": 5.0,
        "colormap": "viridis",
        "db_range": (-80, 0),
    },
}

RIPPLE_CONFIG = {
    "active_vision": {
        "enabled": False,
        "mode": "mouse",
        "gaze": (0.5, 0.5),
        "sigma": 0.15,
        "baseline": 0.01,
        "peak": 1.0,
        "initial_variance": 1.0,
        "process_variance": 0.01,
        "memory_time_constant": 1/3,
        "audio_precision": None,
        "shared_sensory_graph": False,
    },
    "field": {
        "n_sources": 5,
        # "plane_size_m": (500.0, 500.0),
        "plane_size_m": (50.0, 50.0),
        # "resolution": (480, 640),
        # "resolution": (1080, 1920),
        # "resolution": (600, 800),
        # "resolution": (300, 400),
        "resolution": (100, 200),
        # "resolution": (48, 64),
    },
    "dynamics": {
        "amplitude": 1.,
        "decay_alpha": 0.01,
        "speed": 340.0,
        "damping": 0.988,
        "boundary_condition": "neumann",
    },
    "renderer": {
        "backend": "numpy",  # numpy | gpu | opengl
        "rgb_canvas_enabled": True,
        "auto_color_activation_threshold": 0.1,
        "auto_color_floor": 0.1,
    },
    "sources": {
        "synthetic": {
            "enabled": False,
            "frequency": 20.0,
        },
        "audio": {
            # "enabled": False,
            "enabled": True,
            "signal_gate_threshold": 0.60,
            "drive_amplitude": 0.05,
            "peak_picking": {
                "minimum_peak_magnitude": 0.1,
                "peak_prominence_ratio": 5.0,
                "top_k_count": 3,
            },
            "mapping": {
                # "legacy": alpha * log10(1 + f_audio / f0) * exp(-f_audio / fc)
                # "linear": linear_offset + linear_scale * f_audio
                "mode": "legacy",
                "alpha": 50.0,
                "f0": 50.0,
                "fc": 2000.0,
                "linear_scale": 0.05,
                "linear_offset": 0.0,
            },
        },
        "pose": {
            "enabled": False,
            # "render_mode": "standing-body",
            "render_mode": "overlay",
            "model_path": "models/pose_landmarker_lite.task",
            "camera_index": 0,
            "debug_view": False,
            "medium": {
                "graph_stiffness": 0.25,
                "field_width_fraction": 1.0,
                "field_height_fraction": 1.0,
            },
            "boundary": {
                "transmission": 0.54,
                "dissipation": 0.06,
            },
        },
        "camera_frame": {
            "enabled": True,
            # "enabled": True,
            "camera_index": "/dev/video0",
            "gain": 1.0,
        },
    },
    "transforms": {
        "prediction_error": {
            "enabled": True,
            "inputs": ("camera_frame",),
            "inference": {
                "dtype": "float32",
                "hidden_channels": 6,
                "steps": 8,
                "hidden_rate": 0.1,
                "canvas_rate": 0.01,
            },
            "learning": {
                "enabled": False,
                "learning_rate": 1.,
                "weight_decay": 1e-4,
                "weight_clip": 10.0,
                "gradient_clip": 1.0,
            },
        },
    },
}


def build_ripple_visualizer_config(config: Dict) -> Dict:
    sources = config["sources"]
    renderer = config["renderer"]
    audio = sources["audio"]
    audio_mapping = audio["mapping"]
    pose = sources["pose"]
    pose_medium = pose["medium"]
    pose_boundary = pose["boundary"]
    camera_frame = sources["camera_frame"]
    prediction_error = config["transforms"]["prediction_error"]
    prediction_error_inference = prediction_error["inference"]
    prediction_error_learning = prediction_error["learning"]
    backend = renderer["backend"]
    gaze = config.get("active_vision", {})
    active_vision = None
    if gaze.get("enabled", False):
        audio_precision = gaze.get("audio_precision")
        if audio["enabled"] and audio_precision is None:
            raise ValueError("Active vision requires ripple audio to be disabled unless audio_precision is configured")
        active_vision = ActiveVisionConfig(
            mode=gaze["mode"],
            precision=GaussianSpatialPrecision(
                gaze=tuple(gaze["gaze"]), sigma=gaze["sigma"], baseline=gaze["baseline"], peak=gaze["peak"],
            ),
            initial_variance=gaze["initial_variance"],
            process_variance=gaze["process_variance"],
            memory_time_constant=gaze["memory_time_constant"],
            audio_precision=audio_precision,
            shared_sensory_graph=gaze.get("shared_sensory_graph", False),
        )

    return {
        **config["field"],
        **config["dynamics"],
        "source_orchestrator_config": RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(
                enabled=sources["synthetic"]["enabled"],
                frequency=sources["synthetic"]["frequency"],
            ),
            audio=AudioSourceConfig(
                enabled=audio["enabled"],
                observation_gain=float(audio.get("observation_gain", 1.0)),
                signal_gate_threshold=audio["signal_gate_threshold"],
                drive_amplitude=audio["drive_amplitude"],
                mapping_mode=audio_mapping["mode"],
                mapping_alpha=audio_mapping["alpha"],
                mapping_f0=audio_mapping["f0"],
                mapping_fc=audio_mapping["fc"],
                linear_scale=audio_mapping["linear_scale"],
                linear_offset=audio_mapping["linear_offset"],
            ),
            camera_frame=CameraFrameSourceConfig(
                enabled=camera_frame["enabled"],
                camera_index=camera_frame["camera_index"],
                gain=camera_frame["gain"],
            ),
            prediction_error=PredictionErrorTransformConfig(
                enabled=prediction_error["enabled"],
                inputs=tuple(prediction_error["inputs"]),
            ),
            visual_pathway=PredictiveCodingConfig(
                hidden_channels=prediction_error_inference["hidden_channels"],
                dtype=prediction_error_inference.get("dtype", "float64"),
                inference_steps=prediction_error_inference["steps"],
                inference_rate=prediction_error_inference["hidden_rate"],
                canvas_rate=prediction_error_inference["canvas_rate"],
                cross_modal_enabled=prediction_error_inference.get("cross_modal_enabled", False),
                learning_enabled=prediction_error_learning["enabled"],
                learning_rate=prediction_error_learning["learning_rate"],
                weight_decay=prediction_error_learning["weight_decay"],
                weight_clip=prediction_error_learning["weight_clip"],
                gradient_clip=prediction_error_learning["gradient_clip"],
            ),
        ),
        "pose_config": RipplePoseConfig(
            enabled=pose["enabled"],
            render_mode=pose["render_mode"],
            model_path=pose["model_path"],
            camera_index=pose["camera_index"],
            debug_view=pose["debug_view"],
            graph_stiffness=pose_medium["graph_stiffness"],
            field_width_fraction=pose_medium["field_width_fraction"],
            field_height_fraction=pose_medium["field_height_fraction"],
            body_boundary_transmission=pose_boundary["transmission"],
            body_boundary_dissipation=pose_boundary["dissipation"],
        ),
        "use_gpu": backend == "gpu",
        "use_shader": backend == "opengl",
        "rgb_canvas_enabled": renderer["rgb_canvas_enabled"],
        "auto_color_activation_threshold": renderer["auto_color_activation_threshold"],
        "auto_color_floor": renderer["auto_color_floor"],
        "active_vision": active_vision,
    }


def configure_audio_source_processor(processor: AudioProcessor, config: Dict) -> None:
    peak_picking = config["sources"]["audio"]["peak_picking"]
    processor.minimum_frequency_peak_magnitude = float(
        peak_picking["minimum_peak_magnitude"]
    )
    processor.minimum_frequency_peak_to_median_ratio = float(
        peak_picking["peak_prominence_ratio"]
    )
    processor.minimum_signal_level = float(
        config["sources"]["audio"]["signal_gate_threshold"]
    )
    processor.set_num_top_frequencies(int(peak_picking["top_k_count"]))


def _create_audio_processor(ripple_config):
    # --- Config Phase ---

    is_streaming = APP_CONFIG["is_streaming"]
    audio_file = AUDIO_CONFIG["data_path"] / AUDIO_CONFIG["file_name"]
    audio_analysis_config = AUDIO_CONFIG["analysis"]
    audio_plotting_config = AUDIO_CONFIG["plotting"]
    
    if is_streaming:
        data = None
        device_enum = AudioDeviceDesktop
        config = select_devices(config_file=Path("config/audio_devices.json"))
        sr: Union[int, float] = config["samplerate"]
    else:
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
        "io_blocksize": AUDIO_CONFIG["io_blocksize"],
    }
    
    # Spectrogram parameters
    n_fft = audio_analysis_config["n_fft"]
    window_length = int((audio_analysis_config["window_duration_ms"] / 1000) * sr)
    window_length = 2**int(np.log2(window_length))
    
    spectrogram_params = {
        "n_fft": n_fft,
        "hop_length": window_length // 4,
        "n_mels": audio_analysis_config["n_mels"],
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
    cmap = cm.get_cmap(audio_plotting_config["colormap"])
    norm = Normalize(*audio_plotting_config["db_range"])
    plot_update_interval = audio_plotting_config["update_interval_ms"]
    
    plotting_config = {
        "cmap": cmap,
        "norm": norm,
        "plot_update_interval": plot_update_interval,
        "num_samples_in_plot_window": int(audio_plotting_config["window_duration_s"] * sr),
        "waveform_plot_duration": audio_plotting_config["waveform_duration_s"],
    }
    
    # --- Run Phase ---
    
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
        number_top_k_frequencies=n_fft // 2,
    )
    configure_audio_source_processor(processor, ripple_config)
    return processor, plotting_config, (io_config["io_blocksize"]/sr)*1000


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audio Ripple main application")
    parser.add_argument(
        "--active-vision", choices=("mouse", "autonomous", "full"),
        help="Camera-only session with shared gaze controls; audio windows and audio devices are disabled unless --active-audio is passed.",
    )
    parser.add_argument(
        "--active-audio", action="store_true",
        help="Opt into common-mode temporal Gaussian audio inference alongside --active-vision.",
    )
    parser.add_argument(
        "--shared-sensory-graph", action="store_true",
        help="Use one compact learned sensory/latent graph with hard evidence clamps; requires --active-vision.",
    )
    args = parser.parse_args(argv)
    if args.active_audio and args.active_vision is None:
        parser.error("--active-audio requires --active-vision")
    if args.shared_sensory_graph and args.active_vision is None:
        parser.error("--shared-sensory-graph requires --active-vision")
    app_config, source_config = deepcopy(APP_CONFIG), deepcopy(RIPPLE_CONFIG)
    if args.active_vision is not None:
        source_config["active_vision"].update(enabled=True, mode=args.active_vision)
        if args.shared_sensory_graph:
            source_config["active_vision"]["shared_sensory_graph"] = True
        if args.active_audio:
            if source_config["active_vision"].get("audio_precision") is None:
                source_config["active_vision"]["audio_precision"] = 1.0
            source_config["sources"]["audio"]["enabled"] = True
            logger.info(
                "Active vision + audio: shared learned graph enabled."
                if source_config["active_vision"].get("shared_sensory_graph", False) else
                "Active vision + audio: temporal Gaussian visual and common-mode audio inference enabled."
            )
        else:
            source_config["sources"]["audio"]["enabled"] = False
            logger.info("Active vision: camera-only session; audio windows and audio device initialization disabled.")
        source_config["sources"]["camera_frame"]["enabled"] = True
        for name in ("spectrogram", "helix"):
            app_config["windows"][name]["enabled"] = False
        app_config["windows"]["ripples"]["enabled"] = True
        if source_config["active_vision"].get("shared_sensory_graph", False):
            logger.info("Shared sensory graph: compact joint Gaussian learning with camera/audio evidence clamps.")
    windows = app_config["windows"]
    ripple_config = build_ripple_visualizer_config(source_config) if windows["ripples"]["enabled"] else None
    camera_only = (
        ripple_config is not None and ripple_config["active_vision"] is not None
        and ripple_config["active_vision"].audio_precision is None
        and not windows["spectrogram"]["enabled"] and not windows["helix"]["enabled"]
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    processor = None
    plotting_config = None
    if not camera_only:
        processor, plotting_config, block_duration_ms = _create_audio_processor(source_config)
    
        processing_timer = QtCore.QTimer()
        processing_timer.setInterval(int(block_duration_ms*(1 - 1e-3)))
        processing_timer.timeout.connect(processor.process_pending_audio)
        processing_timer.start()
    
    # Visualizer
    spectrogram_window_config = windows["spectrogram"]
    if spectrogram_window_config["enabled"]:
        assert processor is not None and plotting_config is not None
        visualizer = SpectrogramVisualizer(
            processor=processor,
            cmap=plotting_config["cmap"],
            norm=plotting_config["norm"],
            waveform_plot_duration=plotting_config["waveform_plot_duration"],
        )
        visualizer.setWindowTitle(spectrogram_window_config["title"])
        visualizer.resize(*spectrogram_window_config["size"])
        visualizer.show()
    
    # Create Pitch Helix Visualizer
    helix_window_config = windows["helix"]
    if helix_window_config["enabled"]:
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
        helix_window.setWindowTitle(helix_window_config["title"])
        helix_window.resize(*helix_window_config["size"])
        helix_window.show()
    
    # Create Ripple Wave Visualizer
    ripples_window_config = windows["ripples"]
    if ripples_window_config["enabled"]:
        active_vision = ripple_config["active_vision"]
        ripple_processor = (
            processor if active_vision is None or active_vision.audio_precision is not None else None
        )
        ripple_window = RippleWaveVisualizer(
            processor=ripple_processor,
            **ripple_config
        )
        ripple_window.setWindowTitle(ripples_window_config["title"])
        ripple_window.resize(*ripples_window_config["size"])
        ripple_window.show()
    
    # Start audio
    if processor is not None:
        processor_success = processor.start()
        if not processor_success:
            logger.error("Failed to start audio processing. Exiting.")
            return
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if processor is not None:
        app.aboutToQuit.connect(processor.stop)
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Exiting...")
        if processor is not None:
            processor.stop()
        app.quit()

if __name__ == "__main__":
    main()
