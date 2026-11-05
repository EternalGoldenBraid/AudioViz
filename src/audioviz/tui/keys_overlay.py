"""The `?` overlay: bindings grouped by section; `/` filters them."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from audioviz.tui.keys import Binding, label


class KeysScreen(ModalScreen):
    """Any of Esc, q or ? closes; / filters the rows by key, description or section, Enter keeps the filter, Esc drops it."""

    DEFAULT_CSS = """
    KeysScreen { align: center middle; }
    KeysScreen > VerticalScroll { width: 90%; height: 80%; border: heavy $accent; background: $surface; padding: 0 1; }
    KeysScreen Grid { height: auto; grid-size: 3; grid-gutter: 0 2; }
    KeysScreen.-narrow Grid { grid-size: 1; }
    KeysScreen Input { display: none; dock: top; }
    KeysScreen.-searching Input { display: block; }
    KeysScreen Static { height: auto; margin-bottom: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    """

    def __init__(self, table: tuple[Binding, ...], problems: list[str]) -> None:
        super().__init__()
        self.table = table
        self.problems = problems

    def compose(self) -> ComposeResult:
        yield Input(placeholder="/ filter keys", id="search")
        with VerticalScroll(), Grid():
            yield from self.cards("")

    def cards(self, query: str) -> list[Static]:
        words = query.lower().split()
        sections: dict[str, list[Binding]] = {}
        for binding in self.table:
            sections.setdefault(binding.section, []).append(binding)
        cards = []
        for section, rows in sections.items():
            text = Text(section.upper() + "\n", style="bold", no_wrap=True, overflow="ellipsis")
            shown = 0
            for row in rows:
                keys = " ".join(label(k) for k in row.keys)
                haystack = f"{keys} {row.description} {section}".lower()
                if not all(word in haystack for word in words):
                    continue
                hint = {"visual": " (select)", "edit": " (entered)"}.get(row.mode, "")
                text.append(f"{keys:<12}", style="bold yellow")
                text.append(f"{row.description}{hint}\n")
                shown += 1
            if shown:
                cards.append(Static(text))
        if self.problems and not words:
            cards.append(Static(Text("\n".join(self.problems), style="red")))
        if not cards:
            cards.append(Static(Text("no bindings match", style="grey50")))
        return cards

    def filter(self, query: str) -> None:
        grid = self.query_one(Grid)
        grid.remove_children()
        grid.mount(*self.cards(query))

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.search(False)
        event.stop()

    def search(self, on: bool) -> None:
        """The input is disabled while not searching, so it never takes the keys that close or search."""
        box = self.query_one(Input)
        box.disabled = not on
        self.set_class(on, "-searching")
        if on:
            box.focus()
        else:
            self.focus()

    def on_mount(self) -> None:
        self.query_one(Input).disabled = True
        self.set_class(self.app.size.width < 110, "-narrow")

    def on_key(self, event) -> None:
        searching = self.has_class("-searching")
        if event.key == "escape" and (searching or self.query_one(Input).value):
            self.query_one(Input).value = ""
            self.search(False)
            event.stop()
        elif searching:
            return
        elif event.character == "/":
            self.set_class(True, "-searching")
            self.search(True)
            event.stop()
        elif event.key in ("escape", "q", "question_mark"):
            self.dismiss(None)
            event.stop()
