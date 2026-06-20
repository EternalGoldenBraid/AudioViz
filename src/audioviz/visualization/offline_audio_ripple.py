from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.visualization.offline_pose_ripple import (
    RecordingFieldRenderer,
    _ALLOWED_VIDEO_SUFFIXES,
    _write_render_artifacts,
)
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

DEFAULT_OUTPUT_DIR = Path("outputs/audio_ripple_validation")
DEFAULT_VIDEO_NAME = "audio_ripple_validation.gif"


@dataclass(frozen=True)
class OfflineAudioRippleResult:
    output_dir: Path
    video_path: Path
    frame_paths: tuple[Path, ...]
    max_abs_by_frame: tuple[float, ...]
    detected_audio_frequencies_by_frame: tuple[tuple[float, ...], ...]
    mapped_visual_frequencies_by_frame: tuple[tuple[float, ...], ...]


def run_offline_audio_ripple(
    *,
    audio_samples: np.ndarray,
    sr: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    video_path: Path | None = None,
    resolution: tuple[int, int] = (96, 128),
    plane_size_m: tuple[float, float] = (1.0, 1.0),
    fps: float = 12.0,
    io_blocksize: int = 2048,
    n_fft: int = 256,
    window_duration_ms: float = 20.0,
    damping: float = 0.995,
    amplitude: float = 1.0,
    decay_alpha: float = 0.1,
    speed: float = 1.0,
    audio_visual_mapping_mode: str = "legacy",
    audio_visual_mapping_alpha: float = 50.0,
    audio_visual_mapping_f0: float = 50.0,
    audio_visual_mapping_fc: float = 2000.0,
    audio_visual_linear_scale: float = 0.05,
    audio_visual_linear_offset: float = 0.0,
) -> OfflineAudioRippleResult:
    if sr <= 0:
        raise ValueError("sr must be positive")
    if io_blocksize <= 0:
        raise ValueError("io_blocksize must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    samples = _coerce_audio_samples(audio_samples)
    resolved_output_dir = Path(output_dir)
    resolved_video_path = (
        Path(video_path)
        if video_path is not None
        else resolved_output_dir / DEFAULT_VIDEO_NAME
    )
    if resolved_video_path.suffix.lower() not in _ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(
            f"video_path must use one of {sorted(_ALLOWED_VIDEO_SUFFIXES)}"
        )

    window_length = int((window_duration_ms / 1000.0) * sr)
    window_length = max(2, 2 ** int(np.log2(max(window_length, 2))))

    processor = AudioProcessor(
        sr=int(sr),
        n_fft=n_fft,
        hop_length=window_length // 4,
        num_samples_in_buffer=max(io_blocksize * 2, window_length * 4),
        stft_window=_hann_window(window_length),
        io_blocksize=io_blocksize,
        number_top_k_frequencies=n_fft // 2,
        n_mels=None,
        is_streaming=False,
        input_device_index=-1,
        input_channels=samples.shape[1],
        output_device_index=-1,
        output_channels=1,
        data=None,
    )

    renderer = RecordingFieldRenderer()

    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = RippleWaveVisualizer(
        processor=processor,
        n_sources=1,
        resolution=resolution,
        plane_size_m=plane_size_m,
        speed=speed,
        damping=damping,
        amplitude=amplitude,
        decay_alpha=decay_alpha,
        use_synthetic=False,
        use_pose_sources=False,
        audio_visual_mapping_mode=audio_visual_mapping_mode,
        audio_visual_mapping_alpha=audio_visual_mapping_alpha,
        audio_visual_mapping_f0=audio_visual_mapping_f0,
        audio_visual_mapping_fc=audio_visual_mapping_fc,
        audio_visual_linear_scale=audio_visual_linear_scale,
        audio_visual_linear_offset=audio_visual_linear_offset,
    )
    visualizer.timer.stop()
    visualizer.renderer = renderer

    detected_audio_frequencies: list[tuple[float, ...]] = []
    mapped_visual_frequencies: list[tuple[float, ...]] = []

    try:
        for block in _iter_audio_blocks(samples, io_blocksize=io_blocksize):
            processor.process_audio(block)
            detected_audio_frequencies.append(
                tuple(
                    float(freq)
                    for freq in processor.current_top_k_frequencies
                    if freq is not None
                )
            )
            resolved = visualizer._resolve_audio_frequencies()
            mapped_visual_frequencies.append(
                ()
                if resolved is None
                else tuple(float(value) for value in resolved[0])
            )
            visualizer.update_visualization()
            app.processEvents()
    finally:
        visualizer.close()
        app.processEvents()

    if not renderer.fields:
        raise RuntimeError("Offline audio ripple validation produced no frames.")

    max_abs_by_frame = tuple(float(np.max(np.abs(field))) for field in renderer.fields)
    if not any(value > 1e-9 for value in max_abs_by_frame):
        raise RuntimeError("Offline audio ripple validation stayed blank.")

    frame_paths = _write_render_artifacts(
        renderer.fields,
        rgb_frames=renderer.rgb_frames if renderer.rgb_frames else None,
        output_dir=resolved_output_dir,
        video_path=resolved_video_path,
        fps=fps,
    )
    return OfflineAudioRippleResult(
        output_dir=resolved_output_dir,
        video_path=resolved_video_path,
        frame_paths=frame_paths,
        max_abs_by_frame=max_abs_by_frame,
        detected_audio_frequencies_by_frame=tuple(detected_audio_frequencies),
        mapped_visual_frequencies_by_frame=tuple(mapped_visual_frequencies),
    )


def _coerce_audio_samples(audio_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio_samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("audio_samples must have shape (frames,) or (frames, channels)")
    return np.ascontiguousarray(samples)


def _hann_window(window_length: int) -> np.ndarray:
    import librosa as lr

    return lr.filters.get_window("hann", window_length)


def _iter_audio_blocks(
    samples: np.ndarray,
    *,
    io_blocksize: int,
):
    for start in range(0, samples.shape[0], io_blocksize):
        stop = min(start + io_blocksize, samples.shape[0])
        block = samples[start:stop]
        if block.shape[0] == io_blocksize:
            yield block
            continue
        padded = np.zeros((io_blocksize, samples.shape[1]), dtype=np.float32)
        padded[: block.shape[0]] = block
        yield padded
