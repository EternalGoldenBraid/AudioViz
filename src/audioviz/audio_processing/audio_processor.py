from collections import deque
from typing import Optional, Union, Tuple
import numpy as np
import sounddevice as sd
from functools import partial
import librosa as lr
from loguru import logger

def get_stft_spectrogram(segment: np.ndarray, hop_length: int,
                         stft_window: np.ndarray, n_fft: int) -> np.ndarray:
    spectrogram = np.abs(
        lr.stft(segment, n_fft=stft_window.shape[0],
                hop_length=hop_length, window=stft_window)
    )
    return spectrogram

def get_mel_spectrogram(segment: np.ndarray,
                        stft_window: np.ndarray, n_fft: int,
                        hop_length: int, n_mels: int, sr: int) -> np.ndarray:
    spectrogram = lr.feature.melspectrogram(
        y=segment, sr=sr, window=stft_window,
        n_fft=stft_window.shape[0], hop_length=hop_length, n_mels=n_mels
    )
    return spectrogram

class AudioProcessor:
    def __init__(self,
                 sr: int,
                 n_fft: int,
                 hop_length: int,
                 num_samples_in_buffer: int,
                 stft_window: np.ndarray,
                 io_blocksize: int,
                 n_mels: Optional[int] = None,
                 is_streaming: bool = False,
                 input_device_index: Optional[int] = None,
                 input_channels: int = 1,
                 output_device_index: Optional[int] = None,
                 output_channels: int = 1,
                 data: Optional[np.ndarray] = None):

        self.snapshot_queue: deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=5)

        self.sr: float = sr
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.n_mels: int = n_mels
        self.num_samples_in_buffer: int = num_samples_in_buffer
        self.stft_window: Union[str, tuple, np.ndarray] = stft_window
        self.is_streaming: bool = is_streaming
        self.io_blocksize: int = io_blocksize
        self.data = data

        self.input_channels: int = input_channels
        self.output_channels: int = output_channels

        self.audio_buffer: np.ndarray = np.zeros(
            (num_samples_in_buffer, input_channels), dtype=np.float32
        )
        self.n_spec_bins: int = n_mels if n_mels is not None else n_fft // 2 + 1
        n_spec_frames: int = num_samples_in_buffer // hop_length
        self.spectrogram_buffer: np.ndarray = np.zeros(
            (self.n_spec_bins, n_spec_frames), dtype=np.float32
        )

        if n_mels is None:
            self.get_spectrogram = partial(
                get_stft_spectrogram,
                stft_window=stft_window,
                n_fft=n_fft,
                hop_length=hop_length
            )
        else:
            self.get_spectrogram = partial(
                get_mel_spectrogram,
                stft_window=stft_window,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                sr=sr
            )

        self.current_time: float = 0.0  # in seconds

        self.stream = sd.OutputStream(
            samplerate=sr,
            device=output_device_index,
            channels=output_channels,
            callback=self.audio_output_callback,
            blocksize=io_blocksize
        )

        if self.is_streaming:
            self.input_stream = sd.InputStream(
                device=input_device_index,
                channels=input_channels,
                samplerate=sr,
                callback=self.audio_input_callback,
                blocksize=io_blocksize,
                dtype='float32'
            )

    def start(self):
        self.stream.start()
        if self.is_streaming:
            self.input_stream.start()

    def stop(self):
        self.stream.stop()
        if self.is_streaming:
            self.input_stream.stop()

    def audio_input_callback(self, indata: np.ndarray, frames: int, time, status):
        if status:
            logger.warning(f"Input stream error: {status}")

        if indata.ndim == 2 and indata.shape[1] > 1:
            # Optional: cross-modulation
            indata[:, 1] = indata[:, 1]*indata[:, 0]*0.5 + indata[:, 1]*0.5

        self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
        self.audio_buffer[-frames:] = indata

        segment = np.mean(indata, axis=1)  # average across channels
        spectrogram = self.get_spectrogram(segment=segment)

        frames_spec = spectrogram.shape[1]
        self.spectrogram_buffer = np.roll(self.spectrogram_buffer, -frames_spec, axis=1)
        self.spectrogram_buffer[:, -frames_spec:] = spectrogram

        self.snapshot_queue.append((self.audio_buffer.copy(), self.spectrogram_buffer.copy()))

    def audio_output_callback(self, outdata, frames, time, status):
        if status:
            logger.warning(f"Output stream error: {status}")

        if self.data is not None and not self.is_streaming:
            start_idx = int(self.current_time * self.sr)
            end_idx = start_idx + frames
            if end_idx > len(self.data):
                indata = np.zeros(frames, dtype=np.float32)
                indata[:len(self.data[start_idx:])] = self.data[start_idx:]
                raise sd.CallbackStop()
            else:
                indata = self.data[start_idx:end_idx]

            if self.audio_buffer.shape[1] > 1:
                indata = np.tile(indata[:, np.newaxis], (1, self.audio_buffer.shape[1]))

            self.audio_buffer[-frames:] = indata

        outdata[:] = self.audio_buffer[-frames:]
        self.current_time += frames / self.sr

    def get_audio_buffer(self) -> np.ndarray:
        return self.audio_buffer.copy()

    def get_spectrogram_buffer(self) -> np.ndarray:
        return self.spectrogram_buffer.copy()

    def get_latest_snapshot(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the latest buffered (audio, spectrogram) snapshot."""
        if self.snapshot_queue:
            return self.snapshot_queue[-1]
        else:
            return None
