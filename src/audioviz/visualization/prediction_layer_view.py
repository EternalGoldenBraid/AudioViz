from __future__ import annotations

import math
import os
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


def _configure_opengl_platform() -> None:
    app = QtWidgets.QApplication.instance()
    if (
        app is not None
        and app.platformName() == "xcb"
        and "OpenGL.platform" not in sys.modules
    ):
        # Match Qt's X11 integration, including inside a Wayland desktop session.
        integration = os.environ.get("QT_XCB_GL_INTEGRATION")
        os.environ.setdefault(
            "PYOPENGL_PLATFORM", "egl" if integration == "xcb_egl" else "glx"
        )


_configure_opengl_platform()

from OpenGL import platform
from pyqtgraph import opengl as gl

from audioviz.utils.source_preview import SourceScenePreview
from audioviz.utils.continuous_graph_preview import ContinuousGraphPreview
from audioviz.visualization.ripple_renderers import _canvas_to_rgb


def rgb_texture(field: np.ndarray) -> np.ndarray:
    rgb = _canvas_to_rgb(field, limit=1.0)
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=2).transpose(1, 0, 2)[:, ::-1]


def spectrum_mesh(
    values: np.ndarray, position: tuple[float, float, float], width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One signed bar per band, using the same fixed scale for both spectra."""
    values = np.clip(values, -1, 1)
    cube = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.uint32)
    step = width / len(values)
    vertices = cube[None, :, :] * np.column_stack((
        np.full(len(values), step * 0.8), values * 1.6, np.where(values == 0, 0, 0.08)
    ))[:, None, :]
    vertices += np.asarray(position) + np.column_stack((
        np.arange(len(values)) * step, np.zeros(len(values)), np.zeros(len(values))
    ))[:, None, :]
    colors = np.where(values[:, None] >= 0, [0.33, 0.78, 1, 1], [1, 0.61, 0.33, 1])
    return (
        vertices.reshape(-1, 3),
        (faces[None, :, :] + np.arange(len(values))[:, None, None] * 8).reshape(-1, 3),
        np.repeat(colors, len(faces), axis=0),
    )


def hidden_texture(field: np.ndarray) -> np.ndarray:
    signed = np.tanh(field)[..., None]
    color = np.where(
        signed >= 0,
        np.array([255, 155, 85]),
        np.array([85, 200, 255]),
    )
    rgb = np.rint(25 + np.abs(signed) * (color - 25)).astype(np.uint8)
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=2).transpose(1, 0, 2)[:, ::-1]


class PredictionLayerView(gl.GLViewWidget):
    """Small textured planes; GL work and interactive camera stay on the GUI thread."""

    rendering_failed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setBackgroundColor((20, 25, 30))
        self.planes: dict[str, gl.GLImageItem] = {}
        self.labels: dict[str, gl.GLTextItem] = {}
        self.borders: dict[str, gl.GLLinePlotItem] = {}
        self.histograms: dict[str, gl.GLMeshItem] = {}
        self.lines: dict[str, gl.GLLinePlotItem] = {}
        self.nodes: dict[str, gl.GLScatterPlotItem] = {}
        self._context_ready = False
        self.reset_view()

    def reset_view(self) -> None:
        self.opts["fov"] = 45
        self.setCameraPosition(
            pos=QtGui.QVector3D(0, 0, 0.3),
            distance=24,
            elevation=25,
            azimuth=-90,
        )

    def set_scene(self, canvas: np.ndarray, sources: tuple[SourceScenePreview, ...]) -> None:
        self._hide_items()
        aspect = canvas.shape[0] / canvas.shape[1]
        self._set_plane("canvas", rgb_texture(canvas), (-3, -3 * aspect, 3.5), 6,
                        text="Canvas (corrected)")
        branches = [source for source in sources if source.mapping.role == "sensory" and source.hidden is not None]
        centers = {
            source.mapping.key: (index - (len(branches) - 1) / 2) * 4.8
            for index, source in enumerate(branches)
        }
        for index, source in enumerate(branches):
            center = (index - (len(branches) - 1) / 2) * 4.8
            self._set_hidden(source, center, aspect)
            if source.stream_state is not None:
                if source.stream_state_prediction is None:
                    raise ValueError("A stream-state scene node requires its prediction.")
                name = f"{source.mapping.key}/stream-state"
                position = (center - 2.6, 2 * aspect + 1.0, -0.8)
                if name not in self.nodes:
                    self.nodes[name] = gl.GLScatterPlotItem(size=12)
                    self.addItem(self.nodes[name])
                self.nodes[name].setData(
                    pos=np.array([position]),
                    color=(.5, .85, .6, 1) if source.stream_state else (.9, .65, .35, 1),
                )
                self.nodes[name].setVisible(True)
                self._set_label(
                    name, (position[0], position[1] + .3, position[2]),
                    f"{source.mapping.label} state: {'ON' if source.stream_state else 'OFF'}"
                    f"; p(on)={source.stream_state_prediction:.2f}",
                )
                self._set_lines(name, np.array([position, (center, 0, 1.2)]), mode="lines")
            for stage, values, depth in (
                ("prediction", source.prediction, -0.8),
                ("observation", source.observation, -3.0),
            ):
                name = f"{source.mapping.key}/{stage}"
                position = (center - 2, -2 * aspect, depth)
                text = source.mapping.label + (
                    " prediction (pre-evidence)" if stage == "prediction" else " observed"
                )
                if values is not None:
                    self._set_source_value(source, name, values, position, 4, text)
                else:
                    self._set_label(name, (center - 2, 0, depth),
                                    f"{source.mapping.label}: " + ("disabled" if not source.enabled else "no evidence"))
            self._set_lines(f"{source.mapping.key}/pathway", np.array([
                [0, 0, 3.5], [center, 0, 1.2], [center, 0, -0.8], [center, 0, -3.0],
            ]), mode="line_strip")
            if source.recurrent_parent in centers:
                parent_center = centers[source.recurrent_parent]
                depth = 2.9 + 0.35 * index
                y = -2 * aspect - 0.6
                self._set_lines(f"{source.mapping.key}/recurrent", np.array([
                    [parent_center, 0, 1.2], [parent_center, y, depth],
                    [center, y, depth], [center, 0, 1.2],
                ]), mode="line_strip")
                self._set_label(
                    f"{source.mapping.key}/recurrent-label", (-1.5, y, depth),
                    f"{source.recurrent_parent.capitalize()} -> {source.mapping.label} hidden",
                )
        attachments = [source for source in sources if source.mapping.role != "sensory" and source.enabled]
        for index, source in enumerate(attachments):
            center = -7.0 if index % 2 == 0 else 7.0
            name = f"{source.mapping.key}/observation"
            text = f"{source.mapping.label} ({source.mapping.role})"
            if source.observation is not None:
                self._set_source_value(source, name, source.observation, (center - 1.5, -1, 0.5), 3, text)
                self._set_lines(f"{source.mapping.key}/attachment", np.array([
                    [center, 0, 0.5], [center, 0, 3.5], [0, 0, 3.5],
                ]), mode="line_strip")
            else:
                self._set_label(name, (center - 1.5, 0, 0.5), f"{text}: no evidence")
        self.update()

    def set_graph_snapshot(self, snapshot: ContinuousGraphPreview) -> None:
        self._hide_items()
        height = 6 * snapshot.visual_shape[0] / snapshot.visual_shape[1]
        origin = (-3, -height / 2)
        for name, image, depth, title in (
            ("world/camera", snapshot.reference, -3.5, "Full reference; only outlined evidence enters model"),
            ("world/prediction", snapshot.prediction, -1.0, "Whole-world prediction: sigmoid(z)"),
        ):
            if image is not None:
                self._set_plane(name, rgb_texture(image), (*origin, depth), 6, text=title)
            else:
                self._set_label(name, (-3, height / 2 + .15, depth), "Camera reference unavailable")

        error_texture = (
            hidden_texture(np.sqrt(np.mean(snapshot.error**2, axis=-1)))
            if snapshot.error is not None else rgb_texture(np.full_like(snapshot.prediction, 65 / 255))
        )
        self._set_plane(
            "world/error", error_texture, (*origin, .5), 6,
            text=(
                "Whole-frame RGB RMSE (diagnostic only)"
                if snapshot.error is not None else "Reference error unavailable (no current camera frame)"
            ),
        )
        self._set_plane(
            "world/root", rgb_texture(snapshot.canvas), (5, -height / 2, 1), 3,
            text="Canvas / root input",
        )
        for name, mask, depth in (
            ("camera", snapshot.fovea, -3.47),
            ("prediction", snapshot.fovea, -.97),
            ("error", snapshot.observed, .53),
        ):
            # Cell outlines retain the exact sampled support, including clipped apertures.
            points = []
            nrows, ncols = mask.shape
            for row, col in np.argwhere(mask):
                x, y = -3 + 6 * col / ncols, height / 2 - height * (row + 1) / nrows
                dx, dy = 6 / ncols, height / nrows
                if row == nrows - 1 or not mask[row + 1, col]:
                    points.extend(((x, y, depth), (x+dx, y, depth)))
                if row == 0 or not mask[row - 1, col]:
                    points.extend(((x, y+dy, depth), (x+dx, y+dy, depth)))
                if col == ncols - 1 or not mask[row, col + 1]:
                    points.extend(((x+dx, y, depth), (x+dx, y+dy, depth)))
                if col == 0 or not mask[row, col - 1]:
                    points.extend(((x, y, depth), (x, y+dy, depth)))
            self._set_lines(f"support/{name}", np.asarray(points).reshape(-1, 3), mode="lines")
            self.lines[f"support/{name}"].setData(color=(1, .85, .2, 1), width=2)

        indices = snapshot.visual_indices
        rows, cols = snapshot.visual_shape
        pixels = indices // 3
        sensory = np.column_stack((
            -3 + 6 * (pixels % cols + .5) / cols,
            height / 2 - height * (pixels // cols + .5) / rows,
            np.full(len(indices), -0.94),
        ))
        root = sensory.copy()
        root[:, 0] = 5 + (sensory[:, 0] + 3) / 2
        root[:, 1] /= 2
        root[:, 2] = 1.04
        count = len(snapshot.hidden)
        hidden = np.column_stack((
            np.linspace(-3, 3, count), np.full(count, -.2), np.full(count, 3.8),
        ))
        audio = np.column_stack((
            np.full(len(snapshot.audio), -5),
            np.linspace(-1, 1, len(snapshot.audio)), np.full(len(snapshot.audio), -.94),
        ))
        sensory_all = np.concatenate((sensory, audio))
        for name, positions, values in (
            ("hidden", hidden, snapshot.hidden), ("sensory", sensory, np.zeros(len(sensory))),
            ("root", root, np.zeros(len(root))), ("audio", audio, snapshot.audio),
        ):
            if name not in self.nodes:
                self.nodes[name] = gl.GLScatterPlotItem(size=9, pxMode=True)
                self.addItem(self.nodes[name])
            colors = np.tile((.7, .7, .7, 1), (len(values), 1))
            colors[values > 0] = (1, .6, .3, 1)
            colors[values < 0] = (.3, .7, 1, 1)
            self.nodes[name].setData(pos=positions, color=colors)
            self.nodes[name].setVisible(True)
        for index, position in zip(snapshot.hidden_indices, hidden):
            self._set_label(f"h/{index}", tuple(position + (0, .25, .1)), f"h{index}")
        for index, position, value in zip(snapshot.audio_indices, audio, snapshot.audio):
            self._set_label(f"a/{index}", tuple(position + (-1.8, 0, 0)), f"audio{index}: {value:+.2f}")
        self._set_label("population", (-3, 3.1, 3.8),
                        f"One recurrent population: showing {count}/{snapshot.hidden_count} real neurons")
        self._set_weight_edges("Ws: h -> z/audio", hidden, sensory_all, snapshot.sensory_weights)
        self._set_weight_edges("Wc: root -> h", root, hidden, snapshot.canvas_weights)
        self._set_weight_edges("Wh: h -> h", hidden, hidden, snapshot.recurrent_weights, recurrent=True)
        self.update()

    def _set_weight_edges(self, name, sources, targets, weights, *, recurrent=False) -> None:
        flat = np.flatnonzero(weights)
        selected = flat[np.argsort(np.abs(weights.reshape(-1)[flat]), kind="stable")[-24:]]
        points, colors = [], []
        for index in selected:
            target, source = np.unravel_index(index, weights.shape)
            start, end = sources[source], targets[target]
            color = (1, .55, .3, .65) if weights[target, source] > 0 else (.3, .65, 1, .65)
            if recurrent:
                middle = (start + end) / 2 + (0, .6 + .1 * source, .4)
                points.extend((start, middle))
                colors.extend((color, color))
                start = middle
            direction = end - start
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            unit = direction / length
            side = np.cross(unit, (0, 0, 1))
            if np.linalg.norm(side) < .01:
                side = np.cross(unit, (0, 1, 0))
            side *= .08 / np.linalg.norm(side)
            base = end - .2 * unit
            points.extend((start, end, base + side, end, base - side, end))
            colors.extend((color,) * 6)
        self._set_lines(name, np.asarray(points).reshape(-1, 3), mode="lines")
        self.lines[name].setData(color=np.asarray(colors).reshape(-1, 4))

    def _set_hidden(self, source: SourceScenePreview, center: float, aspect: float) -> None:
        hidden = source.hidden
        if hidden is None:
            raise ValueError("A hidden-layer display requires hidden states.")
        count = hidden.shape[2]
        columns = min(3, count)
        rows = math.ceil(count / columns)
        width = 4 / columns - 0.15
        step_y = width * aspect + 0.55
        for channel in range(count):
            column, row = channel % columns, channel // columns
            x = center - 2 + column * (4 / columns)
            y = (rows / 2 - row - 1) * step_y
            text = f"{source.mapping.label[0]} H{channel}"
            if source.hidden_energies:
                text += f":{source.hidden_energies[channel]:.1g}"
            self._set_plane(
                f"{source.mapping.key}/hidden/{channel}",
                hidden_texture(hidden[:, :, channel]),
                (x, y, 1.2),
                width,
                text=text,
            )
        self._set_label(f"{source.mapping.key}/hidden-title",
                        (center - 2, (rows / 2) * step_y + 0.3, 1.2),
                        f"{source.mapping.label} hidden states")

    def _set_source_value(
        self, source: SourceScenePreview, name: str, values: np.ndarray,
        position: tuple[float, float, float], width: float, text: str,
    ) -> None:
        representation = source.mapping.representation
        if representation == "rgb":
            self._set_plane(name, rgb_texture(values), position, width, text=text)
        elif representation == "signed":
            self._set_plane(name, hidden_texture(values.mean(axis=2)), position, width, text=text)
        elif representation == "spectrum":
            vertices, faces, colors = spectrum_mesh(values, position, width)
            if name not in self.histograms:
                self.histograms[name] = gl.GLMeshItem(smooth=False, glOptions="opaque")
                self.addItem(self.histograms[name])
            self.histograms[name].setMeshData(vertexes=vertices, faces=faces, faceColors=colors)
            self.histograms[name].setVisible(True)
            x, y, z = position
            self._set_lines(name, np.array([[x, y, z], [x + width, y, z]]), mode="lines")
            self._set_label(name, (x, y + 1.8, z), text)
        else:
            raise ValueError(f"Unsupported scene representation: {representation}")

    def _set_label(self, name: str, position: tuple[float, float, float], text: str) -> None:
        if name not in self.labels:
            self.labels[name] = gl.GLTextItem(
                color=(225, 230, 235, 255), font=QtGui.QFont("Sans Serif", 9)
            )
            self.labels[name].setDepthValue(1000)
            self.addItem(self.labels[name])
        self.labels[name].setData(pos=position, text=text)
        self.labels[name].setVisible(True)

    def _set_lines(self, name: str, points: np.ndarray, *, mode: str) -> None:
        if name not in self.lines:
            self.lines[name] = gl.GLLinePlotItem(color=(0.4, 0.5, 0.55, 1), width=1)
            self.addItem(self.lines[name])
        self.lines[name].setData(pos=points, mode=mode)
        self.lines[name].setVisible(True)

    def _set_plane(
        self,
        name: str,
        texture: np.ndarray,
        position: tuple[float, float, float],
        width: float,
        *,
        text: str | None = None,
    ) -> None:
        if name not in self.planes:
            self.planes[name] = gl.GLImageItem(texture, glOptions="opaque")
            self.borders[name] = gl.GLLinePlotItem(
                color=(0.4, 0.5, 0.55, 1), width=1, mode="line_strip"
            )
            for item in (self.planes[name], self.borders[name]):
                self.addItem(item)
        plane = self.planes[name]
        plane.setData(texture)
        plane.setVisible(True)
        height = width * texture.shape[1] / texture.shape[0]
        x, y, z = position
        plane.resetTransform()
        plane.scale(width / texture.shape[0], height / texture.shape[1], 1)
        plane.translate(x, y, z)
        self._set_label(name, (x, y + height + 0.15, z), text or name)
        self.borders[name].setData(
            pos=np.array(
                [
                    (x, y, z), (x + width, y, z),
                    (x + width, y + height, z), (x, y + height, z), (x, y, z),
                ]
            )
        )
        self.borders[name].setVisible(True)

    def _hide_items(self) -> None:
        for items in (self.planes, self.labels, self.borders, self.histograms, self.lines, self.nodes):
            for item in items.values():
                item.setVisible(False)

    def clear_evidence(self, key: str) -> None:
        name = f"{key}/observation"
        for items in (self.planes, self.labels, self.borders, self.histograms, self.lines, self.nodes):
            if name in items:
                items[name].setVisible(False)
        if name in self.planes:
            self.planes[name].setData(np.zeros((1, 1, 4), dtype=np.uint8))

    def clear_preview(self) -> None:
        self._hide_items()
        for plane in self.planes.values():
            plane.setData(np.zeros((1, 1, 4), dtype=np.uint8))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._check_context)

    def initializeGL(self) -> None:
        self._context_ready = bool(platform.GetCurrentContext())
        if not self._context_ready:
            return
        super().initializeGL()

    def paintGL(self) -> None:
        if self._context_ready:
            super().paintGL()

    def _check_context(self) -> None:
        if not self.isVisible():
            return
        if not self.isValid():
            self.rendering_failed.emit(
                "OpenGL context creation failed"
            )
        elif not self._context_ready:
            expected = (
                "egl" if os.environ.get("QT_XCB_GL_INTEGRATION") == "xcb_egl" else "glx"
            )
            self.rendering_failed.emit(
                "Qt and PyOpenGL use incompatible contexts. On X11, restart "
                f"with PYOPENGL_PLATFORM={expected}"
            )
