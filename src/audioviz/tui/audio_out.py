"""Waveform listening taps and the legacy spectral sine-bank vocoder."""

from collections import deque

import librosa as lr
import numpy as np

from audioviz.tui.audio_feed import BLOCK_SIZE, N_MELS, SAMPLE_RATE
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS

MODES = ("off", "prediction", "error", "input", "evidence")
MAX_QUEUED_BLOCKS = 8


def band_frequencies() -> np.ndarray:
    """Centre of each band: the observation averages groups of mel bins."""
    mel_centres = lr.mel_frequencies(n_mels=N_MELS + 2, fmin=0.0, fmax=SAMPLE_RATE / 2)[1:-1]
    return np.array([group.mean() for group in np.array_split(mel_centres, AUDIO_SPECTRAL_BANDS)])


class Vocoder:
    """One sine per band. A sine of amplitude A reads as band value ~A*gain, so callers divide by gain."""

    def __init__(self) -> None:
        self.freqs = band_frequencies()
        self.phases = np.zeros(AUDIO_SPECTRAL_BANDS)
        self.amps = np.zeros(AUDIO_SPECTRAL_BANDS)

    def block(self, amps: np.ndarray, frames: int = BLOCK_SIZE) -> np.ndarray:
        ramp = np.linspace(0.0, 1.0, frames, endpoint=False)[:, None]  # avoids clicks between blocks
        envelope = self.amps + (amps - self.amps) * ramp
        steps = np.arange(1, frames + 1)[:, None] * 2 * np.pi * self.freqs / SAMPLE_RATE
        signal = (envelope * np.sin(self.phases + steps)).sum(axis=1)
        self.phases = (self.phases + steps[-1]) % (2 * np.pi)
        self.amps = amps
        return signal.astype(np.float32)


class AudioOut:
    """Mode and volume plus an optional speaker stream. last_block lets tests listen without hardware."""

    def __init__(self) -> None:
        self.mode = "off"
        self.device: int | None = None  # None: system default
        self.volume = 0.5
        self.vocoder = Vocoder()
        self.last_block: np.ndarray | None = None
        self._queue: deque[np.ndarray] = deque(maxlen=MAX_QUEUED_BLOCKS)
        self._stream = None

    def set_mode(self, mode: str, *, speakers: bool = True) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != self.mode:
            self._queue.clear()
            self.last_block = None
        self.mode = mode
        if mode == "off":
            self.close()
        elif speakers and self._stream is None:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
                dtype="float32", callback=self._fill, device=self.device,
            )
            self._stream.start()

    def use_device(self, device: int | None) -> None:
        """Switch output device; reopens the stream if sound is on."""
        self.device = device
        if self._stream is not None:
            self.close()
            self.set_mode(self.mode)

    def submit(self, bands: np.ndarray | None, gain: float) -> None:
        if self.mode == "off" or bands is None:
            return
        amps = np.asarray(bands, dtype=np.float64) / max(gain, 1e-9)
        if self.mode != "error":
            amps = np.clip(amps, 0.0, None)
        block = np.tanh(self.vocoder.block(amps) * self.volume)
        self.last_block = block
        self._queue.append(block)

    def submit_waveform(self, samples: np.ndarray) -> None:
        if self.mode == "off":
            return
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 1 or not 0 < samples.size <= BLOCK_SIZE or not np.all(np.isfinite(samples)):
            raise ValueError("output waveform must be finite, mono, and at most one audio block")
        block = np.clip(samples * self.volume, -1.0, 1.0).astype(np.float32)
        self.last_block = block
        self._queue.append(block)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue.clear()

    def _fill(self, outdata, frames, time, status) -> None:
        outdata[:] = 0.0
        if self._queue:
            block = self._queue.popleft()
            count = min(len(block), frames)
            outdata[:count, 0] = block[:count]
            if count < len(block):
                self._queue.appendleft(block[count:])
