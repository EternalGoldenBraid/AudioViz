"""Saved application settings from the checkout's config directory."""

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


class PresetPicker(ModalScreen[Path | None]):
    AUTO_FOCUS = "OptionList"
    DEFAULT_CSS = """
    PresetPicker { align: center middle; background: $background 60%; }
    PresetPicker > Vertical {
        width: 90%; max-width: 90; height: auto; max-height: 90%;
        border: heavy $accent; background: $surface; padding: 0 1;
    }
    PresetPicker Static {
        height: 1; color: $text-muted; text-wrap: nowrap; text-overflow: ellipsis;
    }
    PresetPicker OptionList { height: auto; border: none; background: $surface; }
    """
    BINDINGS = [
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("escape,q", "dismiss(None)", show=False),
    ]

    def __init__(self, directory: Path, paths: list[Path], current: Path) -> None:
        super().__init__()
        self.directory, self.paths, self.current = directory, paths, current

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = " Load settings "
            box.border_subtitle = " Enter load | Esc cancel "
            yield Static(Text(str(self.directory)))
            yield OptionList(*(
                Option(Text(str(path.relative_to(self.directory)) + (
                    " (current)" if path.resolve() == self.current.resolve() else ""
                )), id=str(index))
                for index, path in enumerate(self.paths)
            ))

    def on_mount(self) -> None:
        self.query_one(OptionList).highlighted = next(
            (i for i, path in enumerate(self.paths) if path.resolve() == self.current.resolve()), 0,
        )
        self.on_resize()

    def on_resize(self) -> None:
        self.query_one(OptionList).styles.max_height = max(1, int(self.size.height * 0.9) - 5)

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self.paths[int(event.option.id)])
