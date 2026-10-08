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

from audioviz.readouts.diagnostics import PredictiveCodingDiagnostics
from audioviz.visualization.ripple_renderers import _canvas_to_rgb


def rgb_texture(field: np.ndarray) -> np.ndarray:
    if field.ndim == 1:
        field = np.broadcast_to(field[None, :, None], (8, field.size, 3))
    rgb = _canvas_to_rgb(field, limit=1.0)
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=2).transpose(1, 0, 2)[:, ::-1]


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

    def __init__(self, parent=None, *, modality: str = "Camera") -> None:
        super().__init__(parent)
        self.modality = modality
        self.setMinimumSize(360, 360)
        self.setBackgroundColor((20, 25, 30))
        self.planes: dict[str, gl.GLImageItem] = {}
        self.labels: dict[str, gl.GLTextItem] = {}
        self.borders: dict[str, gl.GLLinePlotItem] = {}
        self._context_ready = False
        self.reset_view()

    def reset_view(self) -> None:
        self.opts["fov"] = 45
        self.setCameraPosition(
            pos=QtGui.QVector3D(0, 0, 0.3),
            distance=10,
            elevation=25,
            azimuth=-90,
        )

    def set_preview(
        self, canvas: np.ndarray, snapshot: PredictiveCodingDiagnostics
    ) -> None:
        preview = snapshot.spatial
        if preview is None:
            raise ValueError("A 3D layer view requires a fresh spatial preview.")
        aspect = canvas.shape[0] / canvas.shape[1]
        for name, field, depth in (
            ("Canvas (corrected)", canvas, 3.0),
            (
                "Prediction (before evidence)" if self.modality == "Camera"
                else "Audio prediction (before evidence)",
                preview.prediction, -0.6,
            ),
            (f"{self.modality} (observed)", preview.observation, -2.4),
        ):
            self._set_plane(name, rgb_texture(field), (-2, -2 * aspect, depth), 4)
        count = preview.hidden.shape[2]
        columns = min(3, count)
        rows = math.ceil(count / columns)
        width = 4 / columns - 0.15
        step_y = width * aspect + 0.55
        for channel in range(count):
            column, row = channel % columns, channel // columns
            x = -2 + column * (4 / columns)
            y = (rows / 2 - row - 1) * step_y
            energy = snapshot.hidden.channel_means[channel]
            self._set_plane(
                f"H{channel}",
                hidden_texture(preview.hidden[:, :, channel]),
                (x, y, 1.2),
                width,
                text=f"H{channel}:{energy:.1g}",
            )
        for name, plane in self.planes.items():
            active = not name.startswith("H") or int(name[1:]) < count
            plane.setVisible(active)
            self.labels[name].setVisible(active)
            self.borders[name].setVisible(active)

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
            self.labels[name] = gl.GLTextItem(
                color=(225, 230, 235, 255), font=QtGui.QFont("Sans Serif", 9)
            )
            self.borders[name] = gl.GLLinePlotItem(
                color=(0.4, 0.5, 0.55, 1), width=1, mode="line_strip"
            )
            for item in (self.planes[name], self.labels[name], self.borders[name]):
                self.addItem(item)
        plane = self.planes[name]
        plane.setData(texture)
        height = width * texture.shape[1] / texture.shape[0]
        x, y, z = position
        plane.resetTransform()
        plane.scale(width / texture.shape[0], height / texture.shape[1], 1)
        plane.translate(x, y, z)
        self.labels[name].setData(pos=(x, y + height + 0.15, z), text=text or name)
        self.borders[name].setData(
            pos=np.array(
                [
                    (x, y, z), (x + width, y, z),
                    (x + width, y + height, z), (x, y + height, z), (x, y, z),
                ]
            )
        )

    def clear_preview(self) -> None:
        for name, plane in self.planes.items():
            plane.setVisible(False)
            plane.setData(np.zeros((1, 1, 4), dtype=np.uint8))
            self.labels[name].setVisible(False)
            self.borders[name].setVisible(False)

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
                "OpenGL context creation failed. The 2D graph remains available."
            )
        elif not self._context_ready:
            expected = (
                "egl" if os.environ.get("QT_XCB_GL_INTEGRATION") == "xcb_egl" else "glx"
            )
            self.rendering_failed.emit(
                "Qt and PyOpenGL use incompatible contexts. On X11, restart "
                f"with PYOPENGL_PLATFORM={expected}. The 2D graph remains available."
            )
