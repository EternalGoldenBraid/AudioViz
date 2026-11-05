"""Audio input for the TUI: a live device or a synthetic tone, feeding one AudioProcessor."""

from collections import deque
from dataclasses import dataclass

import librosa as lr
import numpy as np

from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.audio_processing.cochlea import CochleaBlock

SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
WINDOW = 1024
HOP = 256
N_MELS = 64
HISTORY_SECONDS = 4
MAX_QUEUED_BLOCKS = 8


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    cochlea: CochleaBlock | None


class AudioFeed:
    """Owns an AudioProcessor without streams; blocks arrive via pump()."""

    def __init__(self) -> None:
        self.processor = AudioProcessor(
            sr=SAMPLE_RATE,
            n_fft=WINDOW,
            hop_length=HOP,
            num_samples_in_buffer=SAMPLE_RATE * HISTORY_SECONDS,
            stft_window=lr.filters.get_window("hann", WINDOW),
            io_blocksize=BLOCK_SIZE,
            number_top_k_frequencies=3,
            n_mels=N_MELS,
            is_streaming=True,
            input_device_index=-1,
            output_device_index=-1,
        )
        self.source = "synthetic"
        self._stream = None
        self._queue: deque[np.ndarray] = deque(maxlen=MAX_QUEUED_BLOCKS)
        self._phase = 0.0
        self._time = 0.0
        self.tone: float | None = None  # steady sine for testing; None is the gliding pattern

    @staticmethod
    def list_devices() -> list[str]:
        import sounddevice as sd

        return [
            f"{index}: {device['name']} ({device['max_input_channels']} in)"
            for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0
        ]

    def use_synthetic(self) -> None:
        self._close_stream()
        self._reset_cochlea()
        self.tone = None
        self.source = "synthetic"

    def use_device(self, device: int | None) -> None:
        """Open a live input (None selects the system default); raises on failure."""
        import sounddevice as sd

        self._close_stream()
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            device=device,
            dtype="float32",
            callback=self._on_block,
        )
        stream.start()
        self._stream = stream
        self._queue.clear()
        self._reset_cochlea()
        self.source = "default input" if device is None else f"device {device}"

    def close(self) -> None:
        self._close_stream()

    def pump(self) -> tuple[AudioFrame, ...]:
        """Process a bounded batch, retaining each block for ordered playback."""
        if self._stream is None:
            blocks = (self._synthetic_block(),)
        else:
            blocks = tuple(self._queue.popleft() for _ in range(len(self._queue)))
        frames = []
        for block in blocks:
            self.processor.process_audio(block)
            cochlea = self.processor.cochlea
            frames.append(AudioFrame(block[:, 0].copy(), None if cochlea is None else cochlea.last_block))
        return tuple(frames)

    def _reset_cochlea(self) -> None:
        if self.processor.cochlea is not None:
            self.processor.cochlea.reset(sample_offset=self.processor.samples_processed)

    def _on_block(self, indata: np.ndarray, frames: int, time, status) -> None:
        self._queue.append(indata.copy())

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def use_tone(self, frequency: float) -> None:
        self.use_synthetic()
        self.tone = frequency
        self.source = f"tone {frequency:g} Hz"

    def _synthetic_block(self) -> np.ndarray:
        if self.tone is not None:
            phase = self._phase + np.cumsum(np.full(BLOCK_SIZE, 2 * np.pi * self.tone / SAMPLE_RATE))
            self._phase = phase[-1]
            return (0.5 * np.sin(phase)).astype(np.float32)[:, None]
        t = self._time + np.arange(BLOCK_SIZE) / SAMPLE_RATE
        freq = 220.0 * 2.0 ** (1.5 * np.sin(2 * np.pi * 0.15 * t))
        phase = self._phase + np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
        self._phase = phase[-1]
        self._time += BLOCK_SIZE / SAMPLE_RATE
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
        block = envelope * (np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)) / 1.6
        return block.astype(np.float32)[:, None]
