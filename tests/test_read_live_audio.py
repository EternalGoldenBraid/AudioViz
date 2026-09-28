from typing import Dict, Union
from json import dumps

import sounddevice as sd

from audioviz.utils.audio_devices import (
    AudioDeviceMSI,
    AudioDeviceDesktop,
)
from audioviz.utils.audio_devices import select_devices

class TestAudioDeviceSetup:

    def test_read_live_audio(self):
        # Set the device index (Scarlett Solo is now index 0)
        # device_enum = AudioDeviceMSI
        device_enum = AudioDeviceDesktop
        
        device_index: int = device_enum.SCARLETT_SOLO_USB.value
        channels: int = 2  # Assuming stereo input (you can change it depending on your setup)
        
        # Set the sample rate
        samplerate = 44100
        
        print(sd.query_devices())

    def test_select_devices_reads_existing_config(self, tmp_path):
        config_file = tmp_path / "audio_devices.json"
        expected = {
            "input_device_index": -1,
            "output_device_index": -1,
            "input_channels": 0,
            "output_channels": 0,
            "samplerate": 44100,
        }
        config_file.write_text(dumps(expected))

        config = select_devices(config_file=config_file)

        assert config == expected
