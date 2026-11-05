"""Key bindings as data: one table feeds dispatch, the `?` overlay and `keys.toml`.

`keys.toml` mirrors the sections below and rebinds an action by name:

    [navigate]
    focus_left = "h"
    next_tile = ["tab", "w"]
"""

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "audioviz" / "keys.toml"


@dataclass(frozen=True)
class Binding:
    section: str
    action: str
    keys: tuple[str, ...]
    description: str
    mode: str = "normal"  # "visual": while selecting tiles to group; "knob": while a knob is focused; "edit": while a knob is entered
    custom: bool = False  # set by keys.toml; a custom binding takes a key over from a default one


DEFAULTS: tuple[Binding, ...] = (
    Binding("navigate", "focus_left", ("h",), "focus the tile to the left"),
    Binding("navigate", "focus_down", ("j",), "focus the tile below"),
    Binding("navigate", "focus_up", ("k",), "focus the tile above"),
    Binding("navigate", "focus_right", ("l",), "focus the tile to the right"),
    Binding("navigate", "next_tile", ("tab", "w"), "focus the next tile"),
    Binding("navigate", "prev_tile", ("shift+tab",), "focus the previous tile"),
    Binding("navigate", "rise", ("escape",), "focus the whole group"),
    Binding("navigate", "dive", ("enter",), "enter a knob, or leave group focus"),
    Binding("arrange", "move_left", ("H",), "move the tile or group left"),
    Binding("arrange", "move_down", ("J",), "move down"),
    Binding("arrange", "move_up", ("K",), "move up"),
    Binding("arrange", "move_right", ("L",), "move right"),
    Binding("arrange", "lift", ("right_square_bracket",), "raise one layer"),
    Binding("arrange", "lower", ("left_square_bracket",), "lower one layer"),
    Binding("arrange", "opacity_up", ("full_stop",), "more opaque"),
    Binding("arrange", "opacity_down", ("comma",), "more transparent"),
    Binding("group", "visual", ("v",), "start selecting tiles to group"),
    Binding("group", "ungroup", ("U",), "ungroup the focused group"),
    Binding("group", "page_next", ("n",), "next page of the group"),
    Binding("group", "page_prev", ("p",), "previous page of the group"),
    Binding("group", "select_left", ("h",), "add the tile to the left", "visual"),
    Binding("group", "select_down", ("j",), "add the tile below", "visual"),
    Binding("group", "select_up", ("k",), "add the tile above", "visual"),
    Binding("group", "select_right", ("l",), "add the tile to the right", "visual"),
    Binding("group", "commit_group", ("g", "enter"), "group the selection", "visual"),
    Binding("group", "cancel_group", ("escape",), "cancel the selection", "visual"),
    Binding("view", "zoom", ("z",), "zoom the tile"),
    Binding("view", "immersive", ("F",), "fullscreen"),
    Binding("view", "render", ("r",), "cycle the render mode"),
    Binding("view", "hide", ("d d",), "hide the tile, or the whole group when it is focused"),
    Binding("view", "unhide", ("u",), "bring back what was hidden last"),
    Binding("view", "tiles", ("T",), "list the tiles to show or hide"),
    Binding("view", "devices", ("a",), "pick audio devices"),
    Binding("knob", "turn_down", ("minus", "left"), "turn the focused knob down", "knob"),
    Binding("knob", "turn_up", ("plus", "equals_sign", "right"), "turn the focused knob up", "knob"),
    Binding("knob", "slide_down", ("h",), "slide down while entered", "edit"),
    Binding("knob", "slide_up", ("l",), "slide up while entered", "edit"),
    Binding("knob", "exit_edit", ("enter", "escape"), "leave the knob", "edit"),
    Binding("app", "command", ("colon",), "open the command line"),
    Binding("app", "help", ("question_mark",), "show these bindings"),
    Binding("app", "quit", ("ctrl+q",), "quit"),
)

SYMBOLS = {
    "right_square_bracket": "]", "left_square_bracket": "[", "full_stop": ".", "comma": ",", "colon": ":",
    "minus": "-", "plus": "+", "equals_sign": "=", "question_mark": "?", "escape": "Esc", "enter": "Enter",
    "tab": "Tab", "shift+tab": "S-Tab", "left": "←", "right": "→", "ctrl+q": "C-q",
}
NAMES = {symbol: name for name, symbol in SYMBOLS.items()}


def label(key: str) -> str:
    return SYMBOLS.get(key, key)


def resolve_key(text: str) -> str:
    """Accepts a Textual key name or the symbol shown in the overlay."""
    return NAMES.get(text, text)


def load_overrides(path: Path) -> tuple[dict[tuple[str, str], tuple[str, ...]], list[str]]:
    """Returns ({(section, action): keys}, problems); a missing file overrides nothing."""
    if not path.exists():
        return {}, []
    known = {(b.section, b.action) for b in DEFAULTS}
    overrides, problems = {}, []
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        return {}, [f"{path}: {error}"]
    for section, table in data.items():
        for action, keys in (table.items() if isinstance(table, dict) else []):
            if (section, action) not in known:
                problems.append(f"unknown binding [{section}] {action}")
            elif isinstance(keys, (str, list)) and all(isinstance(k, str) for k in ([keys] if isinstance(keys, str) else keys)):
                overrides[(section, action)] = tuple(resolve_key(k) for k in ([keys] if isinstance(keys, str) else keys))
            else:
                problems.append(f"[{section}] {action}: expected a key or a list of keys")
    return overrides, problems


def bindings(overrides: dict[tuple[str, str], tuple[str, ...]] | None = None) -> tuple[Binding, ...]:
    overrides = overrides or {}
    return tuple(
        replace(b, keys=overrides[(b.section, b.action)], custom=True) if (b.section, b.action) in overrides else b
        for b in DEFAULTS
    )


def keymap(table: tuple[Binding, ...]) -> dict[tuple[str, str], str]:
    """(mode, key) to action; keys.toml bindings take precedence over the defaults."""
    return {(b.mode, key): b.action for b in sorted(table, key=lambda b: b.custom) for key in b.keys}


def default_toml() -> str:
    lines = ["# AudioViz key bindings. Keys are Textual names (escape, enter, tab, ctrl+q) or the symbols shown by `?`."]
    section = None
    for b in DEFAULTS:
        if b.section != section:
            section = b.section
            lines += ["", f"[{section}]"]
        suffix = {"visual": " (selection mode)", "edit": " (entered knob)"}.get(b.mode, "")
        lines.append(f"{b.action} = {list(b.keys)!r}".replace("'", '"') + f"  # {b.description}{suffix}")
    return "\n".join(lines) + "\n"
