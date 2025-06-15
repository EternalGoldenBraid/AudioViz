import numpy as np
import librosa as lr
import sounddevice as sd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
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

def map_audio_freq_to_visual_freq(f_audio: np.ndarray,
                                  alpha: float = 50.0,
                                  f0: float = 50.0,
                                  fc: float = 2000.0) -> np.ndarray:
    """
    Map audio frequency (Hz) to visual frequency (Hz) for spatial ripple rendering.

    Parameters:
        f_audio (np.ndarray): Input array of audio frequencies (Hz).
        alpha (float): Overall output scaling factor.
        f0 (float): Pivot frequency for log compression.
        fc (float): Cutoff frequency for exponential damping (viscosity).

    Returns:
        np.ndarray: Mapped visual frequencies (Hz).
    """
    f_audio = np.clip(f_audio, 1e-3, None)
    log_scaled = np.log10(1 + f_audio / f0)
    damping = np.exp(-f_audio / fc)
    return alpha * log_scaled * damping

def inspect_frequency_mapping():
    """
    Visual inspection utility that plots the mapping from audio frequency to visual frequency.
    """
    f_audio = np.linspace(0, 10000, 1000)
    f_visual = map_audio_freq_to_visual_freq(f_audio)

    plt.figure(figsize=(8, 4))
    plt.plot(f_audio, f_visual)
    plt.title("Audio Frequency to Visual Frequency Mapping")
    plt.xlabel("Audio Frequency (Hz)")
    plt.ylabel("Visual Frequency (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inspect_frequency_mapping()
