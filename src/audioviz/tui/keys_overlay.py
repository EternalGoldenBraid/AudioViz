"""The `?` overlay: bindings grouped by section."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from audioviz.tui.keys import Binding, label


class KeysScreen(ModalScreen):
    """Any of Esc, q or ? closes."""

    DEFAULT_CSS = """
    KeysScreen { align: center middle; }
    KeysScreen > VerticalScroll { width: 90%; height: 80%; border: heavy $accent; background: $surface; padding: 0 1; }
    KeysScreen Grid { height: auto; grid-size: 3; grid-gutter: 0 2; }
    KeysScreen.-narrow Grid { grid-size: 1; }
    KeysScreen Static { height: auto; margin-bottom: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    """

    def __init__(self, table: tuple[Binding, ...], problems: list[str]) -> None:
        super().__init__()
        self.table = table
        self.problems = problems

    def compose(self) -> ComposeResult:
        sections: dict[str, list[Binding]] = {}
        for binding in self.table:
            sections.setdefault(binding.section, []).append(binding)
        with VerticalScroll(), Grid():
            for section, rows in sections.items():
                text = Text(section.upper() + "\n", style="bold", no_wrap=True, overflow="ellipsis")
                for row in rows:
                    keys = " ".join(label(k) for k in row.keys)
                    hint = " (select)" if row.mode == "visual" else ""
                    text.append(f"{keys:<12}", style="bold yellow")
                    text.append(f"{row.description}{hint}\n")
                yield Static(text)
            if self.problems:
                yield Static(Text("\n".join(self.problems), style="red"))

    def on_mount(self) -> None:
        self.set_class(self.app.size.width < 110, "-narrow")

    def on_key(self, event) -> None:
        if event.key in ("escape", "q", "question_mark"):
            self.dismiss(None)
            event.stop()
