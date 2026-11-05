"""The record tile: a blinking REC mark, where the video goes, and a path prompt that completes folders."""

from __future__ import annotations

import math
import time
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.widgets import Input, Static

BLINK_SECONDS = 2.0


class PathSuggester(Suggester):
    """Completes the path being typed to the first matching entry; folders end in a slash."""

    def __init__(self) -> None:
        super().__init__(use_cache=False, case_sensitive=True)

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None
        typed = Path(value).expanduser()
        folder, prefix = (typed, "") if value.endswith("/") else (typed.parent, typed.name)
        try:
            names = sorted(p for p in folder.iterdir() if p.name.startswith(prefix) and (prefix or not p.name.startswith(".")))
        except OSError:
            return None
        names = [p for p in names if p.is_dir() or p.suffix == ".mp4"]
        if not names:
            return None
        base = value[: len(value) - len(prefix)]
        return base + names[0].name + ("/" if names[0].is_dir() else "")


class RecordPrompt(ModalScreen):
    """Asks where to write the video; Enter starts, Esc cancels. Tab accepts the suggested completion."""

    DEFAULT_CSS = """
    RecordPrompt { align: center middle; }
    RecordPrompt > Vertical { width: 80%; height: auto; border: heavy $accent; background: $surface; padding: 0 1; }
    """

    def __init__(self, default: Path) -> None:
        super().__init__()
        self.default = default

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Record the shown tiles to (Tab completes folders, Enter starts, Esc cancels)")
            yield Input(value=str(self.default), suggester=PathSuggester(), id="record-path")

    def on_mount(self) -> None:
        box = self.query_one(Input)
        box.focus()
        box.cursor_position = len(box.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or str(self.default))
        event.stop()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class RecordTile(Static, can_focus=True):
    """Idle: where recordings go and how to start. Recording: a softly blinking red dot, time, frames and the file."""

    DEFAULT_CSS = """
    RecordTile { position: absolute; border: round $primary-darken-2; padding: 0 1; background: $surface; }
    RecordTile:focus { border: heavy $accent; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "record"
        self.started = 0.0

    def refresh_view(self) -> None:
        recorder = self.app.recorder
        text = Text(no_wrap=True, overflow="ellipsis")
        if recorder is None:
            text.append("○ ", style="grey50")
            text.append("not recording  ")
            text.append("Enter: choose path and start\n", style="grey50")
            text.append(str(self.app.recordings_dir()), style="cyan")
        else:
            glow = 0.5 + 0.5 * math.sin(2 * math.pi * time.monotonic() / BLINK_SECONDS)  # a slow swell, never off
            level = round(90 + 165 * glow)
            elapsed = int(time.monotonic() - self.started)
            text.append("● ", style=f"bold rgb({level},{level // 6},{level // 6})")
            text.append(f"REC {elapsed // 60:02d}:{elapsed % 60:02d}  {recorder.frames} frames  ", style="bold")
            text.append("Enter: stop\n", style="grey50")
            text.append(str(recorder.path), style="cyan")
        self.update(text)

    def on_key(self, event) -> None:
        if event.key == "enter":
            self.app.record_toggle_or_prompt()
            event.stop()

    def on_click(self) -> None:
        self.focus()
