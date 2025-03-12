
import numpy as np
import librosa as lr
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Function to compute the mel spectrogram for a segment
def compute_mel_segment(start_idx: int, data: np.ndarray,
                        sr: int, n_fft: int, hop_length: int,
                        mel_filters: np.ndarray, window_samples: int,
                        audio_len: int) -> np.ndarray:

    end_idx = min(start_idx + window_samples, audio_len)
    segment = data[start_idx:end_idx]
    stft = np.abs(lr.stft(segment, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel_spectrogram = np.dot(mel_filters, stft)
    return mel_spectrogram

