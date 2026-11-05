"""Records the tiles that are on screen into one video, laid out and stacked as the TUI shows them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CELL_PX = (16, 8)  # pixels per terminal cell, (height, width)


@dataclass(frozen=True)
class RecordedTile:
    """One tile with an image: its cell rectangle and the image it is showing."""

    x: int
    y: int
    width: int
    height: int
    image: np.ndarray
    name: str = ""


def compose(tiles: list[RecordedTile], size: tuple[int, int] | None = None) -> np.ndarray:
    """Paints tiles in the given order (later ones on top) on one canvas; size is (height, width) in pixels."""
    cell_h, cell_w = CELL_PX
    x0, y0 = min(t.x for t in tiles), min(t.y for t in tiles)
    if size is None:
        size = (max(t.y + t.height for t in tiles) - y0) * cell_h, (max(t.x + t.width for t in tiles) - x0) * cell_w
    canvas = np.zeros((*size, 3), dtype=np.uint8)
    for tile in tiles:
        top, left = (tile.y - y0) * cell_h, (tile.x - x0) * cell_w
        height, width = min(tile.height * cell_h, size[0] - top), min(tile.width * cell_w, size[1] - left)
        if height <= 0 or width <= 0:
            continue
        rows = np.arange(height) * tile.image.shape[0] // (tile.height * cell_h)
        cols = np.arange(width) * tile.image.shape[1] // (tile.width * cell_w)
        canvas[top : top + height, left : left + width] = tile.image[rows][:, cols]
        if tile.name:
            _title(canvas, tile.name, top, left, width)
    return canvas


def _title(canvas: np.ndarray, text: str, top: int, left: int, width: int) -> None:
    """A dark label strip with the tile's name over its top-left corner."""
    import cv2

    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    right = min(left + text_w + 8, left + width)
    bottom = min(top + text_h + 8, canvas.shape[0])
    canvas[top:bottom, left:right] //= 4
    cv2.putText(canvas, text, (left + 4, top + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


class Recorder:
    """Streams composed frames to a video file; the frame size is fixed by the first frame."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self.frames = 0
        self.size: tuple[int, int] | None = None
        self.writer = None

    def capture(self, tiles: list[RecordedTile]) -> None:
        if not tiles:
            return
        if self.writer is None:
            import cv2

            frame = compose(tiles)
            self.size = frame.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.size[1], self.size[0]))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"cannot write video to {self.path}")
        else:
            frame = compose(tiles, self.size)
        self.writer.write(np.ascontiguousarray(frame[..., ::-1]))  # OpenCV wants BGR
        self.frames += 1

    def stop(self) -> Path:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.path
