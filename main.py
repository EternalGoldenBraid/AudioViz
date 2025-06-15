import numpy as np
import librosa as lr
import sounddevice as sd
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from matplotlib.animation import FuncAnimation

from audioviz.utils.signal_processing import compute_mel_segment

# Load the audio file
audio_file = "aaaa.wav"
data, sr = lr.load(audio_file, sr=None)

# Define parameters for the spectrogram
n_fft = int(0.02 * sr)  # 20 ms window
hop_length = n_fft // 2
n_mels = 40
mel_filters = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

# Initialize playback variables
window_duration = 2.0  # Display 2 seconds of the spectrogram
window_samples = int(window_duration * sr)
audio_len = len(data)

# Real-time plotting and playback
# fig, ax = plt.subplots()
fig: plt.Figure = plt.figure()
spec_mel = fig.add_gridspec(ncols=1, nrows=3)
current_time = 0

ax_mel: plt.Axes = fig.add_subplot(spec_mel[:2, :])
spec_img = ax_mel.imshow(
    np.zeros((n_mels, 1)), aspect="auto", origin="lower",
    interpolation="nearest", extent=[0, window_duration, 0, n_mels]
)
ax_mel.set_title("Real-Time Mel Spectrogram")
ax_mel.set_xlabel("Time (s)")
ax_mel.set_ylabel("Mel Filter")

ax_audio = fig.add_subplot(spec_mel[2, :])
ax_audio.set_title("Audio Signal")
ax_audio.set_xlabel("Samples")
ax_audio.set_ylabel("Amplitude")

def update(frame):
    global current_time
    start_idx = int(current_time * sr)
    mel_spectrogram = compute_mel_segment(start_idx=start_idx, data=data,
                                          sr=sr, n_fft=n_fft, hop_length=hop_length,
                                          mel_filters=mel_filters, window_samples=window_samples,
                                          audio_len=audio_len,)
    spec_img.set_array(mel_spectrogram)
    spec_img.set_extent([current_time, current_time + window_duration, 0, n_mels])

    ax_audio.clear()
    ax_audio.plot(data[start_idx:start_idx + window_samples])
    ax_audio.set_xlim(0, window_samples)
    ax_audio.set_ylim(-1, 1)

    current_time += hop_length / sr
    return spec_img,

# Play the audio while updating the plot
def audio_callback(outdata, frames, time, status):
    global current_time
    start_idx = int(current_time * sr)
    end_idx = start_idx + frames
    if end_idx > len(data):
        outdata[:len(data[start_idx:]), 0] = data[start_idx:]
        raise sd.CallbackStop()
    else:
        outdata[:, 0] = data[start_idx:end_idx]
    current_time += frames / sr

# Start audio playback and plotting
stream = sd.OutputStream(
    samplerate=sr, channels=1, callback=audio_callback, blocksize=hop_length
)
ani = FuncAnimation(fig, update, interval=50, blit=True)

with stream:
    plt.show()
