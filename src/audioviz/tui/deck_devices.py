"""Device knobs for a running window: audio input, audio output and camera.

Choices are labelled "<index>: <name>"; a saved choice is matched by name so it survives a changed index.
"""

import glob
import re
from pathlib import Path


def label(index: int, name: str) -> str:
    return f"{index}: {name}"


def audio_choices(devices: list[dict], kind: str, channels: int) -> tuple[str, ...]:
    """Devices that offer at least `channels` channels in the given direction ('input' or 'output')."""
    return tuple(label(i, d["name"]) for i, d in enumerate(devices) if d[f"max_{kind}_channels"] >= channels)


def camera_choices() -> tuple[str, ...]:
    """Capture nodes only: a camera also exposes metadata nodes (sysfs `index` > 0) that deliver no frames."""
    choices = []
    for path in glob.glob("/dev/video*"):
        number = int(re.sub(r"\D", "", path) or 0)
        sysfs = Path(f"/sys/class/video4linux/video{number}")
        if (sysfs / "index").exists() and (sysfs / "index").read_text().strip() != "0":
            continue
        name = (sysfs / "name").read_text().strip() if (sysfs / "name").exists() else path
        choices.append((number, label(number, name)))
    return tuple(text for _, text in sorted(choices))


def _index(choice: str) -> int:
    return int(choice.split(":", 1)[0])


def _current(choices: tuple[str, ...], index) -> str:
    """The choice for the open device; "off" while no stream is open, which a saved config never reopens."""
    return next((c for c in choices if _index(c) == index), "off")


class DeviceControl:
    """Picks devices from a list. `select` takes the whole selection; a control with capacity 1 keeps the last entry,
    so choosing a second device replaces the first. Nothing opens until `select` is called."""

    toggle = False

    def __init__(self, name: str, choices: tuple[str, ...], current, apply, capacity: int = 1) -> None:
        self.name, self.choices, self.capacity = name, choices, capacity
        self._current, self._apply = current, apply

    def selected(self, context=None) -> list[str]:
        label_ = _current(self.choices, self._current())
        return [] if label_ == "off" else [label_]

    def select(self, context, values: list[str]) -> None:
        values = [v for v in values if v in self.choices][-self.capacity:]
        if not values:
            raise ValueError(f"{self.name}: closing a device is not supported yet; choose another")
        for value in values:
            self._apply(_index(value))

    def get(self, context=None) -> str:
        return ", ".join(self.selected(context)) or "off"

    def set(self, context, value: str) -> None:
        self.select(context, [value])

    def format(self, context=None) -> str:
        return self.get(context)

    def position(self, context=None) -> float:
        return 0.0

    def nudge(self, context, steps: int) -> None:
        raise ValueError("press Enter to choose a device")


class CameraDevices(DeviceControl):
    """Two cameras at once: the first selection feeds the canvas, the second runs beside it in its own scene tile.
    Selecting one camera frees the second device; selecting none closes both (eyes closed).

    The list is read from the system on every access, so a camera plugged in or renumbered after start is offered
    under its current index; a choice made from an older list is matched by name. A selection that fails to open
    leaves the previous cameras running."""

    def __init__(self, choices, primary, secondary) -> None:
        self._choices = choices if callable(choices) else (lambda: choices)
        super().__init__("camera", self._choices(), lambda: _camera_number(primary.camera_index), primary.set_camera_index, capacity=2)
        self.primary, self.secondary = primary, secondary

    @property
    def choices(self) -> tuple[str, ...]:
        return self._choices()

    @choices.setter
    def choices(self, _value) -> None:
        pass

    def selected(self, context=None) -> list[str]:
        chosen = [_current(self.choices, _camera_number(source.camera_index)) for source in (self.primary, self.secondary) if source.enabled]
        return [c for c in chosen if c != "off"]

    def _resolve(self, values: list[str]) -> list[str]:
        choices = self.choices
        by_name = {c.split(": ", 1)[-1]: c for c in choices}
        resolved = (v if v in choices else by_name.get(v.split(": ", 1)[-1]) for v in values)
        return list(dict.fromkeys(v for v in resolved if v))[-self.capacity:]

    def select(self, context, values: list[str]) -> None:
        previous = self.selected()
        try:
            self._open(self._resolve(values))
        except RuntimeError:
            try:
                self._open(previous)
            except RuntimeError:
                self._open([])
            raise

    def _open(self, values: list[str]) -> None:
        self.secondary.set_enabled_closing(False)  # frees the device a new primary may take
        if not values:
            self.primary.set_enabled_closing(False)
            return
        self.primary.set_camera_index(_index(values[0]))
        self.primary.set_enabled_closing(True)  # reopens after the eyes were closed
        if len(values) > 1:
            self.secondary.camera_index = _index(values[1])
            self.secondary.set_enabled_closing(True)


def _camera_number(camera_index) -> int | None:
    """The config may name the camera by index or by device path."""
    digits = re.sub(r"\D", "", str(camera_index))
    return int(digits) if digits else None


def device_controls(window, devices: list[dict]) -> dict[str, DeviceControl]:
    """Controls for the devices the window has: audio in/out per the processor's channel counts, and the cameras."""
    orchestrator = window.source_orchestrator
    controls = audio_controls(window.processor, devices)
    if camera_choices():  # only decides whether the control exists; its list is refreshed on use
        controls["camera"] = CameraDevices(camera_choices, orchestrator.camera_source, orchestrator.secondary_camera_source)
    return controls


def audio_controls(processor, devices: list[dict]) -> dict[str, DeviceControl]:
    controls: dict[str, DeviceControl] = {}
    inputs = audio_choices(devices, "input", processor.input_channels)
    outputs = audio_choices(devices, "output", processor.output_channels)
    if inputs:
        controls["audio in"] = DeviceControl(
            "audio in", inputs, lambda: getattr(getattr(processor, "input_stream", None), "device", None),
            processor.set_input_device)
    if outputs:
        controls["audio out"] = DeviceControl(
            "audio out", outputs, lambda: getattr(getattr(processor, "stream", None), "device", None),
            processor.set_output_device)
    return controls
