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
    MAX_WAVEFORMS = 12
    MAX_SHELLS = 4
    WAVEFRONT_SECONDS = 1.2
    WAVEFRONT_CENTER = np.array([-3.6, 2.8, .9])

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
        sphere = gl.MeshData.sphere(rows=16, cols=48)
        self._sphere_directions = sphere.vertexes()
        self._sphere_faces = sphere.faces()
        self.wavefront = gl.GLMeshItem(
            smooth=True, shader="shaded", glOptions="opaque", color=(1, .72, .18, 1),
        )
        self.wavefront.setVisible(False)
        self.addItem(self.wavefront)
        self._shell_directions = self._make_shell_directions()
        self.shell_lines = []
        for _ in range(self.MAX_SHELLS):
            line = gl.GLLinePlotItem(width=1, antialias=True, glOptions="translucent")
            self.addItem(line)
            line.setVisible(False)
            self.shell_lines.append(line)
        self._set_lines("spectrogram-axes", np.array([
            [.6, 6, 0], [.6, 0, 0], [6.6, 0, 0],
        ]), mode="line_strip")
        self.lines["spectrogram-axes"].setGLOptions("opaque")
        self._set_label("wave-title", (-6.6, -3, 0), "Waveform wavefront")
        self._set_label("spec-title", (.6, -2, 0), "Spectrogram")
        self._set_label("history", (.6, 6.8, 3.2), f"Past {history_seconds:g} s")
        self._set_label("now", (.6, -.6, 0), "Now")

    @staticmethod
    def _make_shell_directions() -> np.ndarray:
        rings = []
        for latitude in np.linspace(np.pi / 6, 5 * np.pi / 6, 5):
            longitude = np.linspace(-np.pi, np.pi, 49)
            rings.append(np.column_stack((
                np.sin(latitude) * np.cos(longitude),
                np.sin(latitude) * np.sin(longitude),
                np.full(len(longitude), np.cos(latitude)),
            )))
        for longitude in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            latitude = np.linspace(0, np.pi, 17)
            rings.append(np.column_stack((
                np.sin(latitude) * np.cos(longitude),
                np.sin(latitude) * np.sin(longitude),
                np.cos(latitude),
            )))
        return np.concatenate([
            np.stack((ring[:-1], ring[1:]), axis=1).reshape(-1, 3) for ring in rings
        ])

    def _wavefront_vertices(
        self, directions: np.ndarray, values: np.ndarray, *, age: float = 0,
    ) -> np.ndarray:
        phase = (np.arctan2(directions[:, 1], directions[:, 0]) + np.pi) / (2 * np.pi)
        # Wrap sample time around longitude; poles remain single shared points.
        displacement = np.interp(
            phase, np.linspace(0, 1, len(values) + 1), np.r_[values, values[0]],
        ) * (1 - directions[:, 2]**2)
        rms = float(np.sqrt(np.mean(values**2)))
        radius = 1 + .15 * rms + .3 * displacement * (1 - age) + .75 * age
        return self.WAVEFRONT_CENTER + directions * radius[:, None]

    def advance_wavefront(self, timestamp: float) -> None:
        while self.waveforms and timestamp - self.waveforms[0][0] >= self.WAVEFRONT_SECONDS:
            self.waveforms.popleft()
        self.wavefront.setVisible(bool(self.waveforms))
        for line in self.shell_lines:
            line.setVisible(False)
        if not self.waveforms:
            return
        self.wavefront.setMeshData(
            vertexes=self._wavefront_vertices(self._sphere_directions, self.waveforms[-1][1]),
            faces=self._sphere_faces,
        )
        older = list(self.waveforms)[:-1]
        if not older:
            return
        indices = np.unique(np.linspace(0, len(older) - 1, min(len(older), self.MAX_SHELLS), dtype=int))
        for line, index in zip(self.shell_lines, indices):
            recorded, values = older[index]
            age = (timestamp - recorded) / self.WAVEFRONT_SECONDS
            line.setData(
                pos=self._wavefront_vertices(self._shell_directions, values, age=age),
                color=(1, .8, .35, .55 * (1 - age)**1.5), mode="lines",
            )
            line.setVisible(True)

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
            values = np.clip(waveform[-self.MAX_SAMPLES:], -1, 1)
            self.waveforms.append((timestamp, values.copy()))
        self.advance_wavefront(timestamp)
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
        self.borders["canvas"].setGLOptions("opaque")
