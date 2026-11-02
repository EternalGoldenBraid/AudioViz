"""Audio device picker: inputs on the left, outputs on the right."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


def list_devices() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(inputs, outputs) as (value, label); inputs always offer the built-in synthetic source."""
    import sounddevice as sd

    inputs = [("synthetic", "synthetic pattern"), ("default", "system default input")]
    outputs = [("default", "system default output")]
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            inputs.append((str(index), f"{index}: {device['name']}"))
        if device["max_output_channels"] > 0:
            outputs.append((str(index), f"{index}: {device['name']}"))
    return inputs, outputs


class DevicePicker(ModalScreen):
    """Enter picks in the focused list, Tab switches list, Esc closes. Dismisses with (kind, value)."""

    DEFAULT_CSS = """
    DevicePicker { align: center middle; }
    DevicePicker > Horizontal { width: 90%; height: 80%; border: heavy $accent; background: $surface; }
    DevicePicker OptionList { width: 1fr; border: round $primary-darken-2; }
    DevicePicker OptionList:focus { border: heavy $accent; }
    """
    BINDINGS = [
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("escape", "dismiss(None)", show=False),
        Binding("h", "focus_list('inputs')", show=False),
        Binding("l", "focus_list('outputs')", show=False),
    ]

    def __init__(self, inputs, outputs, current_input: str, current_output: str) -> None:
        super().__init__()
        self.inputs, self.outputs = inputs, outputs
        self.current = {"inputs": current_input, "outputs": current_output}

    def compose(self) -> ComposeResult:
        with Horizontal():
            for kind, devices in (("inputs", self.inputs), ("outputs", self.outputs)):
                options = OptionList(*(Option(label, id=value) for value, label in devices), id=kind)
                options.border_title = f"{kind[:-1]} (now: {self.current[kind]})"
                yield options

    def on_mount(self) -> None:
        self.query_one("#inputs", OptionList).focus()

    def _focused(self) -> OptionList:
        return self.focused if isinstance(self.focused, OptionList) else self.query_one("#inputs", OptionList)

    def action_cursor_down(self) -> None:
        self._focused().action_cursor_down()

    def action_cursor_up(self) -> None:
        self._focused().action_cursor_up()

    def action_focus_list(self, kind: str) -> None:
        self.query_one(f"#{kind}", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss((event.option_list.id[:-1], event.option.id))
