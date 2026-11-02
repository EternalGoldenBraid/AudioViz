"""Render fields and images into terminal cells using sub-cell glyphs."""

from functools import lru_cache

import numpy as np
from rich.color import Color
from rich.segment import Segment
from rich.style import Style


def field_to_rgb(field: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Map a signed field to a blue-black-orange diverging uint8 image."""
    field = np.asarray(field, dtype=np.float32)
    peak = float(np.max(np.abs(field))) if scale is None else scale
    unit = np.clip(field / max(peak, 1e-6), -1.0, 1.0)
    pos = np.clip(unit, 0.0, 1.0)
    neg = np.clip(-unit, 0.0, 1.0)
    rgb = np.stack([pos, 0.5 * pos + 0.4 * neg, neg], axis=-1)
    return (rgb * 255).astype(np.uint8)


_HEAT_STOPS = np.array(
    [[0, 0, 0], [60, 10, 90], [190, 40, 80], [250, 150, 30], [255, 250, 200]], dtype=np.float32
)


def heat_to_rgb(unit: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] onto a black-purple-orange-cream ramp."""
    position = np.clip(unit, 0.0, 1.0) * (len(_HEAT_STOPS) - 1)
    low = np.minimum(position.astype(int), len(_HEAT_STOPS) - 2)
    frac = (position - low)[..., None]
    return (_HEAT_STOPS[low] * (1 - frac) + _HEAT_STOPS[low + 1] * frac).astype(np.uint8)


def waveform_to_rgb(samples: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Draw per-column min/max of a waveform in [-1, 1] as a filled envelope."""
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    chunks = np.array_split(np.asarray(samples, dtype=np.float32), cols)
    for col, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        low = int((1 - np.clip(chunk.max(), -1, 1)) / 2 * (rows - 1))
        high = int((1 - np.clip(chunk.min(), -1, 1)) / 2 * (rows - 1))
        image[low : high + 1, col] = (90, 220, 140)
    return image


def resample(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Nearest-neighbour resample to rows x cols pixels."""
    r = np.linspace(0, image.shape[0] - 1, rows).astype(int)
    c = np.linspace(0, image.shape[1] - 1, cols).astype(int)
    return image[r[:, None], c[None, :]]


# (pixel rows, pixel columns) packed into one terminal cell
RENDER_MODES = {"half": (2, 1), "quad": (2, 2), "braille": (4, 2)}
_QUAD_GLYPHS = np.array(list(" ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"))
_QUAD_BITS = np.array([1, 2, 4, 8])
_BRAILLE_BITS = np.array([0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80])  # row-major, 4x2
_BRAILLE_GLYPHS = np.array([chr(0x2800 + bits) for bits in range(256)])


@lru_cache(maxsize=65536)
def _style(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> Style:
    return Style(color=Color.from_rgb(*fg), bgcolor=Color.from_rgb(*bg))


def _quantize(colors: np.ndarray) -> np.ndarray:
    """Drop low bits so the style cache stays effective."""
    return colors.astype(np.uint8) & 0xF8


class CellImage:
    """Rich renderable: one glyph per cell with foreground and background colours."""

    def __init__(self, glyphs: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> None:
        self.glyphs, self.fg, self.bg = glyphs, fg, bg

    def __rich_console__(self, console, options):
        fg, bg = self.fg.tolist(), self.bg.tolist()
        for row, glyph_row in enumerate(self.glyphs.tolist()):
            for col, glyph in enumerate(glyph_row):
                yield Segment(glyph, _style(tuple(fg[row][col]), tuple(bg[row][col])))
            yield Segment.line()


def render_cells(image: np.ndarray, rows: int, cols: int, mode: str = "half") -> CellImage:
    """Resample to the cell grid at the mode's sub-cell density and pack into glyphs.

    half keeps exact colours (top pixel fg, bottom bg); quad and braille split each cell
    into a bright and a dark group, trading colour fidelity for spatial resolution.
    """
    pixel_rows, pixel_cols = RENDER_MODES[mode]
    pixels = resample(image, rows * pixel_rows, cols * pixel_cols).astype(np.float32)
    cells = pixels.reshape(rows, pixel_rows, cols, pixel_cols, 3).transpose(0, 2, 1, 3, 4)
    cells = cells.reshape(rows, cols, pixel_rows * pixel_cols, 3)
    if mode == "half":
        glyphs = np.full((rows, cols), "▀")
        return CellImage(glyphs, _quantize(cells[:, :, 0]), _quantize(cells[:, :, 1]))
    luminance = cells.mean(axis=-1)
    lit = luminance > luminance.mean(axis=-1, keepdims=True)
    lit_count = lit.sum(axis=-1, keepdims=True)
    dark_count = lit.shape[-1] - lit_count
    cell_mean = cells.mean(axis=2)
    fg = np.where(lit_count > 0, (cells * lit[..., None]).sum(axis=2) / np.maximum(lit_count, 1), cell_mean)
    bg = np.where(dark_count > 0, (cells * ~lit[..., None]).sum(axis=2) / np.maximum(dark_count, 1), cell_mean)
    if mode == "quad":
        glyphs = _QUAD_GLYPHS[(lit * _QUAD_BITS).sum(axis=-1)]
    else:
        glyphs = _BRAILLE_GLYPHS[(lit * _BRAILLE_BITS).sum(axis=-1)]
    return CellImage(glyphs, _quantize(fg), _quantize(bg))
