from __future__ import annotations

from collections import deque

import numpy as np
from PyQt5 import QtGui

from audioviz.visualization.prediction_layer_view import PredictionLayerView, gl, rgb_texture


class AudioLandscapeView(PredictionLayerView):
    """Bounded audio history over a sampled copy of the live canvas."""

    MAX_FREQUENCIES = 128
    MAX_TIMES = 96
    MAX_SAMPLES = 256
    MAX_WAVEFORMS = 48

    def __init__(self, *, cmap, history_seconds: float, parent=None) -> None:
        super().__init__(parent)
        self.cmap = cmap
        self.history_seconds = history_seconds
        self.waveforms: deque[tuple[float, np.ndarray]] = deque(maxlen=self.MAX_WAVEFORMS)
        self.surface = gl.GLSurfacePlotItem(
            smooth=False, shader="shaded", glOptions="opaque",
        )
        self.surface.setVisible(False)
        self.addItem(self.surface)
        self.wave_lines = []
        for _ in range(self.MAX_WAVEFORMS):
            line = gl.GLLinePlotItem(width=1.5, antialias=True, glOptions="translucent")
            self.addItem(line)
            line.setVisible(False)
            self.wave_lines.append(line)
        for left, right in ((-6.6, -.6), (.6, 6.6)):
            self._set_lines(f"axes/{left}", np.array([
                [left, 6, 0], [left, 0, 0], [right, 0, 0],
            ]), mode="line_strip")
        self._set_label("wave-title", (-6.6, -3, 0), "Waveform")
        self._set_label("spec-title", (.6, -2, 0), "Spectrogram")
        self._set_label("history", (-6.6, 6.8, 0), f"Past {history_seconds:g} s")
        self._set_label("now", (-8, 0, 0), "Now")

    def reset_view(self) -> None:
        self.opts["fov"] = 45
        self.setCameraPosition(
            pos=QtGui.QVector3D(0, 2.5, 0),
            distance=32, elevation=35.264, azimuth=-45,
        )

    def set_audio(
        self, waveform: np.ndarray, levels: np.ndarray, *, timestamp: float,
        new_frame: bool,
    ) -> None:
        if new_frame:
            samples = np.linspace(0, len(waveform) - 1, min(len(waveform), self.MAX_SAMPLES), dtype=int)
            self.waveforms.append((timestamp, waveform[samples].copy()))
        while self.waveforms and timestamp - self.waveforms[0][0] > self.history_seconds:
            self.waveforms.popleft()
        for line in self.wave_lines:
            line.setVisible(False)
        for line, (recorded, values) in zip(self.wave_lines, self.waveforms):
            age = (timestamp - recorded) / self.history_seconds
            line.setData(
                pos=np.column_stack((
                    np.linspace(-6.6, -.6, len(values)),
                    np.full(len(values), age * 6),
                    np.clip(values, -1, 1) * 1.3,
                )),
                color=(1, .8, .25, 1 - .7 * age),
            )
            line.setVisible(True)
        rows = np.linspace(0, levels.shape[0] - 1, min(levels.shape[0], self.MAX_FREQUENCIES), dtype=int)
        columns = np.linspace(0, levels.shape[1] - 1, min(levels.shape[1], self.MAX_TIMES), dtype=int)
        # The newest spectrogram column meets the front (Now) edge.
        height = np.clip(levels[np.ix_(rows, columns)][:, ::-1], 0, 1)
        self.surface.setData(
            x=np.linspace(.6, 6.6, len(rows)), y=np.linspace(0, 6, len(columns)),
            z=height * 2.5, colors=self.cmap(height).astype(np.float32).reshape(-1, 4),
        )
        self.surface.setVisible(True)
        self.update()

    def set_canvas(self, field: np.ndarray | None) -> None:
        if field is None:
            for items in (self.planes, self.labels, self.borders):
                if "canvas" in items:
                    items["canvas"].setVisible(False)
            return
        self._set_plane("canvas", rgb_texture(field), (-7.5, -1, -1.6), 15,
                        text="Live canvas")
        self.planes["canvas"].setDepthValue(-10)
