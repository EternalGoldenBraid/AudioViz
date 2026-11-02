"""Colour palettes for the interface chrome and tile backgrounds."""

from textual.theme import Theme

THEMES = (
    Theme(name="black", primary="#3b82f6", secondary="#6b7280", accent="#f59e0b",
          background="#000000", surface="#000000", panel="#0a0a0a", dark=True),
    Theme(name="amber", primary="#ffb000", secondary="#b37a00", accent="#ffd966",
          background="#0d0800", surface="#140d00", panel="#1c1200", dark=True),
    Theme(name="phosphor", primary="#33ff66", secondary="#1a9940", accent="#b3ffcc",
          background="#000a03", surface="#001205", panel="#001a08", dark=True),
    Theme(name="paper", primary="#1d4ed8", secondary="#6b7280", accent="#c2410c",
          background="#f5f5f0", surface="#fbfbf8", panel="#e8e8e0", dark=False),
)
