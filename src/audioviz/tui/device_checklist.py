"""Device checklist: a popup with one checkbox per device.

j/k move, Space checks or unchecks, Enter applies, Esc cancels. When the control holds fewer devices
than are listed, checking one more unchecks the earliest. Dismisses with the checked list, or None.
"""

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


class DeviceChecklist(ModalScreen):
    AUTO_FOCUS = "OptionList"
    DEFAULT_CSS = """
    DeviceChecklist { align: center middle; background: $background 60%; }
    DeviceChecklist > Vertical {
        width: 80%; max-width: 110; height: auto; max-height: 80%;
        border: heavy $accent; border-title-color: $accent; border-title-style: bold;
        border-subtitle-color: $text-muted; background: $surface; padding: 0 1;
    }
    DeviceChecklist OptionList { height: auto; max-height: 22; border: none; background: $surface; padding: 0; }
    DeviceChecklist OptionList > .option-list--option-highlighted { background: $accent 30%; }
    DeviceChecklist #capacity { color: $text-muted; padding: 0 1; }
    DeviceChecklist #hint { color: $text-muted; padding: 1 1 0 1; }
    """
    BINDINGS = [
        Binding("j,down", "cursor_down", show=False),
        Binding("k,up", "cursor_up", show=False),
        Binding("space", "toggle", show=False),
        Binding("enter", "apply", show=False),
        Binding("escape,q", "dismiss(None)", show=False),
    ]

    def __init__(self, title: str, choices: list[str], checked: list[str], capacity: int, *, live: bool = True) -> None:
        super().__init__()
        self.title_text, self.choices, self.capacity = title, choices, capacity
        self.opened = list(checked) if live else []  # only running devices are labelled live
        self.checked = list(checked)

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = f" {self.title_text} "
            box.border_subtitle = " Space check · Enter apply · Esc cancel "
            options = OptionList(*(Option(self._row(c), id=str(i)) for i, c in enumerate(self.choices)))
            if self.checked and self.checked[-1] in self.choices:
                options.highlighted = self.choices.index(self.checked[-1])
            yield options
            yield Static(self._capacity_text(), id="capacity")

    def _capacity_text(self) -> str:
        if self.capacity == 1:
            return "one device at a time: checking another replaces it"
        return f"up to {self.capacity} devices: {len(self.checked)} checked"

    def on_resize(self, event: events.Resize) -> None:
        # Leave room for the frame and wrapped hints in short terminals.
        self.query_one(OptionList).styles.max_height = max(1, min(22, int(event.size.height * 0.8) - 6))

    def _row(self, choice: str) -> Text:
        on = choice in self.checked
        text = Text()
        text.append("◉ " if on else "○ ", style="bold green" if on else "grey50")
        index, _, name = choice.partition(": ")
        text.append(f"{index:>3}  ", style="dim")
        text.append(name, style="bold" if on else "")
        if choice in self.opened:
            text.append("   ● live", style="bold yellow")
        return text

    def _redraw(self) -> None:
        options = self.query_one(OptionList)
        for i, choice in enumerate(self.choices):
            options.replace_option_prompt_at_index(i, self._row(choice))
        self.query_one("#capacity", Static).update(self._capacity_text())

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_toggle(self) -> None:
        index = self.query_one(OptionList).highlighted
        if index is None:
            return
        choice = self.choices[index]
        if choice in self.checked:
            self.checked.remove(choice)
        else:
            self.checked = (self.checked + [choice])[-self.capacity:]
        self._redraw()

    def action_apply(self) -> None:
        self.dismiss(self.checked)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()  # the list consumes Enter before the screen binding sees it
        self.action_apply()
