"""Tile list: a popup with one section per group and one checkbox per tile.

j/k move, Space shows or hides the row (a section row toggles its whole group), Enter applies, Esc cancels.
Dismisses with the set of hidden tile names, or None.
"""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList
from textual.widgets.option_list import Option


class TileList(ModalScreen):
    AUTO_FOCUS = "OptionList"
    DEFAULT_CSS = """
    TileList { align: center middle; background: $background 60%; }
    TileList > Vertical {
        width: 60; height: auto; max-height: 80%;
        border: heavy $accent; border-title-color: $accent; border-title-style: bold;
        border-subtitle-color: $text-muted; background: $surface; padding: 0 1;
    }
    TileList OptionList { height: auto; border: none; background: $surface; padding: 0; }
    TileList OptionList > .option-list--option-highlighted { background: $accent 30%; }
    """
    BINDINGS = [
        Binding("j,down", "cursor_down", show=False),
        Binding("k,up", "cursor_up", show=False),
        Binding("space", "toggle", show=False),
        Binding("enter", "apply", show=False),
        Binding("escape,q", "dismiss(None)", show=False),
    ]

    def __init__(self, sections: list[tuple[str, list[str]]], hidden: set[str]) -> None:
        super().__init__()
        self.rows: list[tuple[str, list[str], bool]] = []  # (label, tiles it toggles, is a section)
        for title, tiles in sections:
            self.rows.append((title, tiles, True))
            self.rows += [(tile, [tile], False) for tile in tiles]
        self.hidden = set(hidden)

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = " tiles "
            box.border_subtitle = " Space show/hide · Enter apply · Esc cancel "
            yield OptionList(*(Option(self._row(row), id=str(i)) for i, row in enumerate(self.rows)))

    def on_mount(self) -> None:
        self._fit()

    def on_resize(self) -> None:
        self._fit()

    def _fit(self) -> None:
        # The box is height:auto, so a percentage cap on the list cannot resolve; size it from the screen, leaving the border.
        self.query_one(OptionList).styles.max_height = max(3, int(self.size.height * 0.8) - 2)

    def _row(self, row: tuple[str, list[str], bool]) -> Text:
        label, tiles, section = row
        shown = sum(t not in self.hidden for t in tiles)
        glyph = "○" if shown == 0 else "◉" if shown == len(tiles) else "◐"
        text = Text()
        text.append(f"{glyph} ", style="grey50" if shown == 0 else "bold green")
        if section:
            text.append(f"{label}  ({shown}/{len(tiles)})", style="bold")
        else:
            text.append(f"   {label}", style="" if shown else "grey50")
        return text

    def _redraw(self) -> None:
        options = self.query_one(OptionList)
        for i, row in enumerate(self.rows):
            options.replace_option_prompt_at_index(i, self._row(row))

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_toggle(self) -> None:
        index = self.query_one(OptionList).highlighted
        if index is None:
            return
        tiles = self.rows[index][1]
        if any(t not in self.hidden for t in tiles):
            self.hidden |= set(tiles)
        else:
            self.hidden -= set(tiles)
        self._redraw()

    def action_apply(self) -> None:
        self.dismiss(self.hidden)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()  # the list consumes Enter before the screen binding sees it
        self.action_apply()
