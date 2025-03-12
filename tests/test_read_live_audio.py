from enum import Enum
import sounddevice as sd

class AudioDeviceMSI(Enum):
    SCARLETT_SOLO_USB = 0
    ALC298_ANALOG = 1
    HDMI_0 = 2
    HDMI_1 = 3
    HDMI_2 = 4
    SYSDEFAULT = 5
    FRONT = 6
    SURROUND40 = 7
    DMIX = 8

# Example usage:
# print(f"Scarlett Solo USB device index: {AudioDevice.SCARLETT_SOLO_USB.value}")

# Set the device index (Scarlett Solo is now index 0)
device_index: int = AudioDeviceMSI.SCARLETT_SOLO_USB.value
channels: int = 2  # Assuming stereo input (you can change it depending on your setup)

# Set the sample rate
samplerate = 44100



# Function to process the audio stream
def callback(indata, frames, time, status):
    if status:
        print(f"{time} s: {status}")
    print(indata)

# Open the audio stream
with sd.InputStream(device=device_index, channels=channels, samplerate=samplerate, callback=callback):
    print("Recording... Press Ctrl+C to stop.")
    sd.sleep(10000)  # Record for 10 seconds
