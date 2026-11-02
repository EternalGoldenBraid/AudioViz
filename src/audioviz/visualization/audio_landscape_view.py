from __future__ import annotations

import numpy as np
from PyQt5 import QtGui

from audioviz.visualization.canvas_tracer_wake import CanvasTracerWake
from audioviz.visualization.prediction_layer_view import PredictionLayerView, gl, rgb_texture


class AudioLandscapeView(PredictionLayerView):
    """Bounded audio history over a sampled copy of the live canvas."""

    MAX_FREQUENCIES = 128
    MAX_TIMES = 96
    MAX_SAMPLES = 256
    SOURCE_HOLD_SECONDS = 1.2
    SOURCE_CENTER = np.array([-3.6, 2.8, .9])

    def __init__(self, *, cmap, history_seconds: float, parent=None) -> None:
        super().__init__(parent)
        self.cmap = cmap
        self.history_seconds = history_seconds
        self.waveform: np.ndarray | None = None
        self._waveform_at = 0.0
        self.wake = CanvasTracerWake()
        self.surface = gl.GLSurfacePlotItem(
            smooth=False, shader="shaded", glOptions="opaque",
        )
        self.surface.setVisible(False)
        self.addItem(self.surface)
        sphere = gl.MeshData.sphere(rows=16, cols=48)
        self._sphere_directions = sphere.vertexes()
        self._sphere_faces = sphere.faces()
        self.source = gl.GLMeshItem(
            smooth=True, shader="shaded", glOptions="opaque", color=(1, .72, .18, 1),
        )
        self.source.setVisible(False)
        self.addItem(self.source)
        z = np.linspace(-.95, .95, self.wake.EMISSION_COUNT)
        longitude = np.arange(len(z)) * np.pi * (3 - np.sqrt(5))
        self._emission_directions = np.column_stack((
            np.sqrt(1 - z**2) * np.cos(longitude),
            np.sqrt(1 - z**2) * np.sin(longitude), z,
        ))
        self.tracers = gl.GLScatterPlotItem(size=2.5, glOptions="translucent")
        self.trails = gl.GLLinePlotItem(width=1.5, antialias=True, glOptions="translucent")
        for item in (self.tracers, self.trails):
            item.setVisible(False)
            self.addItem(item)
        self._set_lines("spectrogram-axes", np.array([
            [.6, 6, 0], [.6, 0, 0], [6.6, 0, 0],
        ]), mode="line_strip")
        self.lines["spectrogram-axes"].setGLOptions("opaque")
        self._set_label("wave-title", (-6.6, -4, 0), "Source / wake")
        self._set_label("spec-title", (.6, -2, 0), "Spectrogram")
        self._set_label("history", (.6, 6.8, 3.2), f"Past {history_seconds:g} s")
        self._set_label("now", (.6, -.6, 0), "Now")

    def _source_vertices(self, directions: np.ndarray, values: np.ndarray) -> np.ndarray:
        phase = (np.arctan2(directions[:, 1], directions[:, 0]) + np.pi) / (2 * np.pi)
        # Wrap sample time around longitude; poles remain single shared points.
        displacement = np.interp(
            phase, np.linspace(0, 1, len(values) + 1), np.r_[values, values[0]],
        ) * (1 - directions[:, 2]**2)
        rms = float(np.sqrt(np.mean(values**2)))
        radius = 1 + .15 * rms + .3 * displacement
        return self.SOURCE_CENTER + directions * radius[:, None]

    def advance_source(self, timestamp: float, waveform: np.ndarray | None = None) -> None:
        self.wake.advance(timestamp)
        if waveform is not None:
            self.waveform = np.clip(waveform[-self.MAX_SAMPLES:], -1, 1).copy()
            self._waveform_at = timestamp
            self.wake.emit(
                self._source_vertices(self._emission_directions, self.waveform),
                float(np.sqrt(np.mean(self.waveform**2))),
            )
        if timestamp - self._waveform_at >= self.SOURCE_HOLD_SECONDS:
            self.waveform = None
        self.source.setVisible(self.waveform is not None)
        if self.waveform is not None:
            self.source.setMeshData(
                vertexes=self._source_vertices(self._sphere_directions, self.waveform),
                faces=self._sphere_faces,
            )
        indices = np.flatnonzero(self.wake.active)
        for item in (self.tracers, self.trails):
            item.setVisible(bool(len(indices)))
        if not len(indices):
            return
        age = (timestamp - self.wake.born[indices]) / self.wake.LIFETIME
        colors = np.tile([1., .72, .2, 1.], (len(indices), 1))
        colors[:, 3] = (1 - age)**1.3 * (.35 + .65 * self.wake.strength[indices])
        self.tracers.setData(pos=self.wake.positions[indices], color=colors)
        segments = np.stack(
            (self.wake.trails[indices, :-1], self.wake.trails[indices, 1:]), axis=2,
        ).reshape(-1, 3)
        colors[:, 3] *= .75
        self.trails.setData(
            pos=segments, color=np.repeat(colors, 2 * (self.wake.TRAIL_POINTS - 1), axis=0),
            mode="lines",
        )

    def clear_audio(self) -> None:
        self.waveform = None
        self.wake.clear()
        for item in (self.source, self.tracers, self.trails):
            item.setVisible(False)

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
        self.advance_source(timestamp, waveform=waveform if new_frame else None)
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
        self.wake.set_field(field)
        field = self.wake.field
        if field is None:
            for items in (self.planes, self.labels, self.borders):
                if "canvas" in items:
                    items["canvas"].setVisible(False)
            return
        self._set_plane(
            "canvas", rgb_texture(field), (*self.wake.CANVAS_ORIGIN, -1.6),
            self.wake.CANVAS_WIDTH, text="Live canvas",
        )
        self.planes["canvas"].setDepthValue(-10)
        self.borders["canvas"].setGLOptions("opaque")
