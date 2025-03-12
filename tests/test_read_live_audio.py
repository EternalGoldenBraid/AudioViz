import sounddevice as sd

from audioviz.utils.audio_devices import (
    AudioDeviceMSI,
    AudioDeviceDesktop,
)

# Set the device index (Scarlett Solo is now index 0)
# device_enum = AudioDeviceMSI
device_enum = AudioDeviceDesktop

device_index: int = device_enum.SCARLETT_SOLO_USB.value
channels: int = 2  # Assuming stereo input (you can change it depending on your setup)

# Set the sample rate
samplerate = 44100

print(sd.query_devices())

# Function to process the audio stream
def callback(indata, frames, time, status):
    if status:
        print(f"{time} s: {status}")
    print(indata)

# Open the audio stream
# with sd.InputStream(device=device_index, channels=channels, samplerate=samplerate, callback=callback):
#     print("Recording... Press Ctrl+C to stop.")
#     sd.sleep(1)  # Record for 10 seconds
