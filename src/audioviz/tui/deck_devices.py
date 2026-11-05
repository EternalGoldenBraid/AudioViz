"""Device knobs for a running window: audio input, audio output and camera.

Choices are labelled "<index>: <name>"; a saved choice is matched by name so it survives a changed index.
"""

import glob
import re

from audioviz.tui.mixer import Control


def label(index: int, name: str) -> str:
    return f"{index}: {name}"


def audio_choices(devices: list[dict], kind: str, channels: int) -> tuple[str, ...]:
    """Devices that offer at least `channels` channels in the given direction ('input' or 'output')."""
    return tuple(label(i, d["name"]) for i, d in enumerate(devices) if d[f"max_{kind}_channels"] >= channels)


def camera_choices() -> tuple[str, ...]:
    paths = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0))
    return tuple(label(int(re.sub(r"\D", "", p)), p) for p in paths)


def _index(choice: str) -> int:
    return int(choice.split(":", 1)[0])


def _current(choices: tuple[str, ...], index) -> str:
    return next((c for c in choices if _index(c) == index), choices[0])


def _choice_control(name: str, choices: tuple[str, ...], current, apply) -> Control:
    """A choice knob that keeps the selected entry when the device fails to open, so h/l can step past it."""
    selected: list[str] = []

    def get(_) -> str:
        return selected[0] if selected else _current(choices, current())

    def set_(_, value: str) -> None:
        selected[:] = [value]
        apply(_index(value))

    return Control(name, 0, 0, get, set_, choices=choices)


def device_controls(window, devices: list[dict]) -> dict[str, Control]:
    """Controls for the devices the window has: audio in/out per the processor's channel counts, and the camera."""
    processor, camera = window.processor, window.source_orchestrator.camera_source
    controls: dict[str, Control] = {}
    inputs = audio_choices(devices, "input", processor.input_channels)
    outputs = audio_choices(devices, "output", processor.output_channels)
    cameras = camera_choices()
    if inputs:
        controls["audio in"] = _choice_control(
            "audio in", inputs, lambda: getattr(getattr(processor, "input_stream", None), "device", None),
            processor.set_input_device)
    if outputs:
        controls["audio out"] = _choice_control(
            "audio out", outputs, lambda: getattr(getattr(processor, "stream", None), "device", None),
            processor.set_output_device)
    if cameras:
        controls["camera"] = _choice_control("camera", cameras, lambda: camera.camera_index, camera.set_camera_index)
    return controls
