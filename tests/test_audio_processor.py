import numpy as np
import librosa as lr

from audioviz.audio_processing.audio_processor import AudioProcessor


def _make_processor() -> AudioProcessor:
    sr = 48_000
    window_length = 512
    return AudioProcessor(
        sr=sr,
        n_fft=256,
        hop_length=64,
        num_samples_in_buffer=4096,
        stft_window=lr.filters.get_window("hann", window_length),
        io_blocksize=2048,
        number_top_k_frequencies=128,
        n_mels=None,
        is_streaming=False,
        input_device_index=-1,
        input_channels=1,
        output_device_index=-1,
        output_channels=1,
        data=None,
    )


def test_audio_processor_ignores_low_prominence_noise():
    processor = _make_processor()
    rng = np.random.default_rng(0)

    noise = (0.002 * rng.standard_normal((2048, 1))).astype(np.float32)
    processor.process_audio(noise)

    assert processor.current_top_k_frequencies == [None, None, None]


def test_audio_processor_detects_dominant_tone():
    processor = _make_processor()

    sr = int(processor.sr)
    time = np.arange(2048, dtype=np.float32) / sr
    tone_hz = 440.0
    signal = (0.02 * np.sin(2.0 * np.pi * tone_hz * time)).reshape(-1, 1).astype(
        np.float32
    )
    processor.process_audio(signal)

    detected = [freq for freq in processor.current_top_k_frequencies if freq is not None]

    assert detected
    assert min(abs(freq - tone_hz) for freq in detected) < 80.0


def test_audio_processor_signal_level_tracks_rms_strength():
    processor = _make_processor()

    quiet = np.full((2048, 1), 0.001, dtype=np.float32)
    loud = np.full((2048, 1), 0.05, dtype=np.float32)

    processor.process_audio(quiet)
    quiet_level = processor.current_signal_level
    processor.process_audio(loud)
    loud_level = processor.current_signal_level

    assert 0.0 <= quiet_level < loud_level <= 1.0
