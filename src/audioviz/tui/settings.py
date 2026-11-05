"""Settings rows: interface and output options, as Controls over the app."""

from audioviz.tui.audio_out import MODES
from audioviz.tui.graph_runtime import format_resolution
from audioviz.tui.mixer import Control

THEME_NAMES = ("black", "amber", "phosphor", "paper", "nord", "gruvbox", "dracula", "textual-dark")
TILE_BACKGROUNDS = ("theme", "#000000", "#0b0b10", "#1a1a1a", "#10202a")
RESOLUTIONS = ("48x48", "64x64", "96x96", "128x128", "100x200", "192x192", "256x256", "300x400", "480x640")


def _set_theme(app, name: str) -> None:
    if name in app.available_themes:
        app.theme = name


def settings_controls() -> tuple[Control, ...]:
    return (
        Control("theme", 0, 0, lambda a: a.theme, _set_theme, choices=THEME_NAMES),
        Control("tile background", 0, 0, lambda a: a.tile_background or "theme",
                lambda a, v: a.set_tile_background(None if v == "theme" else v), choices=TILE_BACKGROUNDS),
        Control("resolution", 0, 0, lambda a: format_resolution(a.runtime.resolution), lambda a, v: a.runtime.rebuild(v), choices=RESOLUTIONS),
        Control("sound out", 0, 0, lambda a: a.runtime.out.mode, lambda a, v: a.runtime.out.set_mode(v), choices=MODES),
        Control("volume", 0.0, 1.0, lambda a: a.runtime.out.volume, lambda a, v: setattr(a.runtime.out, "volume", v)),
    )
