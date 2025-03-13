from typing import Dict, Any
from enum import Enum

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


class AudioDeviceDesktop(Enum):
    SAMSUNG = 0
    SPEAKERS = 4
    ALCS1200A_ANALOG = 5
    ALCS1200A_DIGITAL = 6
    ALCS1200A_ALT_ANALOG = 7
    SCARLETT_SOLO_USB = 8

DesktopDevicesConfigs: Dict[str, Dict[str, Any]] = {
    "SCARLETT_SOLO_USB": {
        "channels": 2,
        "samplerate": 44100,
        "channel_metadata": {"in": "1: Mic, 2: Guitar"},
    }
}
