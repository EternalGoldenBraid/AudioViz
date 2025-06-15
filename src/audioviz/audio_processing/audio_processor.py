from collections import deque
from typing import Optional, Union, Tuple, List

import numpy as np
import sounddevice as sd
from functools import partial
import librosa as lr
from loguru import logger

def get_stft_spectrogram(segment: np.ndarray, hop_length: int,
                         stft_window: np.ndarray, n_fft: int) -> np.ndarray:
    spectrogram = np.abs(
        lr.stft(segment, n_fft=stft_window.shape[0],
                hop_length=hop_length, window=stft_window)[:n_fft // 2 + 1, :]
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
        """
        TODO 
        - [ ] Channel wise spectrogram config (window size etc...)
        """


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
        # self.spectrogram_buffer: np.ndarray = np.zeros(
        #     (self.n_spec_bins, n_spec_frames), dtype=np.float32
        # )
        self.spectrogram_buffers: List[np.ndarray] = [
            np.zeros((self.n_spec_bins, n_spec_frames), dtype=np.float32)
            for _ in range(self.input_channels)
        ]

        self.snapshot_queue: deque[Tuple[np.ndarray, List[np.ndarray]]] = deque(maxlen=5)
        self.num_top_frequencies: int = 3
        self.current_top_k_frequencies: list[float] = [None] * self.num_top_frequencies
        self.freq_bins = np.fft.rfftfreq(self.n_fft, d=1/self.sr)

        self.raw_input_queue: deque[np.ndarray] = deque(maxlen=32)

        if n_mels is None:
            self.compute_spectrogram = partial(
                get_stft_spectrogram,
                stft_window=stft_window,
                n_fft=n_fft,
                hop_length=hop_length
            )
        else:
            self.compute_spectrogram = partial(
                get_mel_spectrogram,
                stft_window=stft_window,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                sr=sr
            )

        self.current_time: float = 0.0  # in seconds
        self.frame_counter: int = 0
        self.input_overflow_count: int = 0

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
                dtype='float32', latency='low',
            )

    def start(self):
        self.stream.start()
        if self.is_streaming:
            self.input_stream.start()

    def get_smoothed_top_k_peak_frequency(self, 
            channel_idx: int = 1,
            k:int = 3, window_frames: int = 3) -> Optional[Tuple[List[int], List[float]]]:
        """
        Return dominant frequency smoothed over last `window_frames` spectrogram frames.
        """
        if self.spectrogram_buffers is None:
            return None

        # Average over last `window_frames` frames
        averaged_frame = np.mean(self.spectrogram_buffers[channel_idx][:, -window_frames:], axis=1)

        freq_bins = np.fft.rfftfreq(self.n_fft, d=1/self.sr)
        # top_idx = np.argmax(averaged_frame)
        # return freq_bins[top_idx]
        top_k_idxs = np.argsort(averaged_frame)[-k:]
        return top_k_idxs.tolist(), freq_bins[top_k_idxs].tolist()

    def update_spectrogram_buffer(self, indata: np.ndarray) -> None:
        for channel_idx in range(indata.shape[1]):
            segment = indata[:, channel_idx]
            # Discard the symmetric half
            spectrogram = self.compute_spectrogram(segment=segment)
        
            frames_spec = spectrogram.shape[1]
            self.spectrogram_buffers[channel_idx] = np.roll(
                self.spectrogram_buffers[channel_idx], -frames_spec, axis=1
            )
            self.spectrogram_buffers[channel_idx][:, -frames_spec:] = spectrogram

    def process_audio(self, indata: np.ndarray):
        frames = indata.shape[0]

        self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
        self.audio_buffer[-frames:] = indata

        self.update_spectrogram_buffer(indata)

        idxs_, self.current_top_k_frequencies[:] = self.get_smoothed_top_k_peak_frequency(
            window_frames=10, k=self.num_top_frequencies, channel_idx=0)

        # # Set non -peak frequencies to 0
        # zero_idxs = np.setdiff1d(np.arange(self.n_spec_bins), idxs_)
        # self.spectrogram_buffers[1][zero_idxs, -frames:] = 0.0

        self.snapshot_queue.append((
            self.audio_buffer.copy(),
            [spec.copy() for spec in self.spectrogram_buffers]
        ))

        # Only optional print every few frames
        if self.frame_counter % 50 == 0:
            print(f"Top {self.num_top_frequencies} frequencies: {self.current_top_k_frequencies}")

        self.frame_counter += 1

    def process_pending_audio(self):
        while self.raw_input_queue:
            audio_chunk = self.raw_input_queue.popleft()
            self.process_audio(audio_chunk)


    def audio_input_callback(self, indata: np.ndarray, frames: int, time, status):

        # if status and self.frame_counter % 10 == 0:
        #     logger.warning(f"Input stream error: {status}")
        if status:
            self.input_overflow_count += 1
            if self.input_overflow_count % 10 == 0:
                logger.warning(f"Input overflows: {self.input_overflow_count}, {status}")

        # # Optional: fast hard thresholding directly (optional)
        threshold = 0.001
        mask = np.abs(indata) < threshold
        indata[mask] = 0.0

        # Push a copy into a fast queue
        self.raw_input_queue.append(indata.copy())

    def audio_output_callback(self, outdata, frames, time, status):
        if status and self.frame_counter % 10 == 0:
            logger.warning(f"Output stream error: {status}")

        # For file playback, we can use the data directly
        # if self.data is not None and not self.is_streaming:
        #     start_idx = int(self.current_time * self.sr)
        #     end_idx = start_idx + frames
        #     if end_idx > len(self.data):
        #         indata = np.zeros(frames, dtype=np.float32)
        #         indata[:len(self.data[start_idx:])] = self.data[start_idx:]
        #         raise sd.CallbackStop()
        #     else:
        #         indata = self.data[start_idx:end_idx]
        #
        #     if self.audio_buffer.shape[1] > 1:
        #         indata = np.tile(indata[:, np.newaxis], (1, self.audio_buffer.shape[1]))
        #
        #     self.audio_buffer[-frames:] = indata

        outdata[:] = self.audio_buffer[-frames:]
        self.current_time += frames / self.sr

    def get_channel_spectrogram(self, channel_idx: int) -> np.ndarray:
        """Return the spectrogram of the specified channel."""
        if 0 <= channel_idx < len(self.spectrogram_buffers):
            return self.snapshot_queue[-1][1][channel_idx]
        else:
            raise IndexError("Channel index out of range.")

    def get_latest_snapshot(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the latest buffered (audio, spectrogram) snapshot."""
        if self.snapshot_queue:
            return self.snapshot_queue[-1]
        else:
            return None

    def stop(self) -> None:
        """Stop audio processing and release resources."""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                print("Output stream stopped.")
            except Exception as e:
                print(f"Error stopping output stream: {e}")

        if self.is_streaming and hasattr(self, 'input_stream') and self.input_stream is not None:
            try:
                self.input_stream.stop()
                self.input_stream.close()
                print("Input stream stopped.")
            except Exception as e:
                print(f"Error stopping input stream: {e}")

        # Stop any other resources if needed (timers etc.)
        print("AudioProcessor stopped cleanly.")
