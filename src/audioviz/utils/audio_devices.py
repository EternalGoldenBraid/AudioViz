from typing import Dict, Any
from enum import Enum
from json import dumps, loads
from pathlib import Path

import sounddevice as sd

NULL_DEVICE = {
    "name": "Null Device",
    "max_input_channels": 0,
    "max_output_channels": 0,
    "default_samplerate": 44100,
}


def validate_device_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Audio configuration must be an object.")
    for name in ("input_device_index", "output_device_index", "input_channels", "output_channels", "samplerate"):
        if type(config.get(name)) is not int:
            raise ValueError(f"Audio configuration needs an integer {name}.")
    if config["samplerate"] <= 0:
        raise ValueError("Audio sample rate must be positive.")
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as error:
        raise ValueError(f"Audio devices are unavailable: {error}") from error
    for kind in ("input", "output"):
        index, channels = config[f"{kind}_device_index"], config[f"{kind}_channels"]
        if channels < 0:
            raise ValueError(f"Audio {kind} channels cannot be negative.")
        if index == -1:
            continue
        if not 0 <= index < len(devices):
            raise ValueError(f"Saved {kind} device {index} is no longer available.")
        device = devices[index]
        saved_name = config.get(f"{kind}_device_name")
        if saved_name is not None and saved_name != device["name"]:
            raise ValueError(f"Saved {kind} device {saved_name!r} has changed; select it again.")
        if not 0 < channels <= device[f"max_{kind}_channels"]:
            raise ValueError(f"{device['name']} cannot provide {channels} {kind} channels.")
        check = sd.check_input_settings if kind == "input" else sd.check_output_settings
        try:
            check(device=index, channels=channels, dtype="float32", samplerate=config["samplerate"])
        except sd.PortAudioError as error:
            raise ValueError(
                f"{kind}: {device['name']} cannot use {config['samplerate']} Hz: {error}\n"
                "Choose compatible devices in the TUI; the system default route can provide sample-rate conversion."
            ) from error


def load_device_config(config_file: Path) -> dict:
    config = loads(config_file.read_text())
    validate_device_config(config)
    return config


def make_device_config(devices, input_index: int, output_index: int) -> dict:
    input_dev = NULL_DEVICE if input_index == -1 else devices[input_index]
    output_dev = NULL_DEVICE if output_index == -1 else devices[output_index]
    return {
        "input_device_index": input_index,
        "output_device_index": output_index,
        "input_channels": input_dev["max_input_channels"],
        "output_channels": output_dev["max_output_channels"],
        "samplerate": int(input_dev["default_samplerate"] if input_index != -1 else output_dev["default_samplerate"]),
        "input_device_name": input_dev["name"],
        "output_device_name": output_dev["name"],
    }

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


def list_devices():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        io_flag = []
        if dev['max_input_channels'] > 0:
            io_flag.append("IN")
        if dev['max_output_channels'] > 0:
            io_flag.append("OUT")
        print(f"[{idx}] {dev['name']} ({', '.join(io_flag)})")

    # Null device is always available
    print(f"[{len(devices)}] Null Device (no audio input/output)")
    return devices

def prompt_for_device(devices, kind: str = "input"):
    while True:
        try:
            index = int(input(f"Select {kind} device index: "))

            if index == len(devices):
                print("Using Null Device (no audio input/output)")
                index = -1
                return index, NULL_DEVICE

            device = devices[index]
            if kind == "input" and device['max_input_channels'] == 0:
                print("Selected device has no input channels.")
                continue
            if kind == "output" and device['max_output_channels'] == 0:
                print("Selected device has no output channels.")
                continue
            return index, device
        except (ValueError, IndexError):
            print("Invalid index. Try again.")

def select_devices(config_file: Path = Path("config.json")):
    if config_file.exists():
        print(f"Reading configuration from {config_file}")
        with open(config_file, "r") as f:
            return loads(f.read())
    else:
        print(f"Configuration file not found. Creating new configuration at {config_file}")
        if not config_file.parent.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)

    print("=== Available Audio Devices ===")
    devices = list_devices()
    
    print("\n--- Select Input Device ---")
    input_index, _ = prompt_for_device(devices, kind="input")
    
    print("\n--- Select Output Device ---")
    output_index, _ = prompt_for_device(devices, kind="output")
    
    config = make_device_config(devices, input_index, output_index)

    with open(config_file, "w") as f:
        f.write(dumps(config, indent=4))

    return config
