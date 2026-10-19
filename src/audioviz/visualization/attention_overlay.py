from collections.abc import Callable

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from audioviz.readouts.spatial_precision import GaussianSpatialPrecision


def fit_image_rect(width: float, height: float, shape: tuple[int, int]) -> QtCore.QRectF:
    rows, cols = shape
    scale = min(width / cols, height / rows)
    return QtCore.QRectF((width-cols*scale)/2, (height-rows*scale)/2, cols*scale, rows*scale)


def preferred_gain_position(scores: np.ndarray, gaze: tuple[float, float]) -> tuple[float, float] | None:
    """Choose the nearest tied maximum only when it improves on the nearest-grid gaze."""
    rows, cols = scores.shape
    row, col = round(gaze[1]*(rows-1)), round(gaze[0]*(cols-1))
    best = float(scores.max())
    tolerance = 16*np.finfo(scores.dtype).eps*max(abs(best), abs(float(scores.min())))
    if best-float(scores[row, col]) <= tolerance:
        return None
    candidates = np.argwhere(best-scores <= tolerance)
    positions = candidates[:, ::-1] / np.array((cols-1, rows-1))
    chosen = positions[np.square(positions-np.asarray(gaze)).sum(axis=1).argmin()]
    return float(chosen[0]), float(chosen[1])


class AttentionOverlay(QtWidgets.QWidget):
    """Mouse-transparent model gaze and image-space information-seeking advice."""

    def __init__(
        self, parent: QtWidgets.QWidget, image_rect: Callable[[], QtCore.QRectF],
        shape: tuple[int, int],
    ):
        super().__init__(parent)
        self.image_rect = image_rect
        self.shape = shape
        self.precision: GaussianSpatialPrecision | None = None
        self.target: tuple[float, float] | None = None
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setGeometry(parent.rect())
        parent.installEventFilter(self)
        self.show()

    @classmethod
    def on_image(cls, view, image, shape):
        return cls(
            view.viewport(),
            lambda: view.viewportTransform().map(image.mapToScene(image.boundingRect())).boundingRect(),
            shape,
        )

    def set_attention(
        self, precision: GaussianSpatialPrecision | None, scores: np.ndarray | None = None,
        *, available: bool = False,
    ) -> None:
        self.precision = precision
        self.target = (
            preferred_gain_position(scores, precision.gaze)
            if precision is not None and precision.peak > 0 and available and scores is not None else None
        )
        self.update()

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Resize:
            self.setGeometry(watched.rect())
        return super().eventFilter(watched, event)

    def paintEvent(self, event):
        if self.precision is None:
            return
        rect = self.image_rect()
        if rect.isEmpty():
            return
        rows, cols = self.shape
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setClipRect(rect)

        def position(gaze):
            return QtCore.QPointF(
                rect.left()+(.5+gaze[0]*(cols-1))*rect.width()/cols,
                rect.top()+(.5+gaze[1]*(rows-1))*rect.height()/rows,
            )

        color = QtGui.QColor(255, 100, 225)
        center = position(self.precision.gaze)
        if self.precision.peak > 0:
            radius = self.precision.sigma*min(rows, cols)
            for pen_color, width in ((QtCore.Qt.black, 4), (color, 2)):
                painter.setPen(QtGui.QPen(pen_color, width, QtCore.Qt.DashLine))
                painter.drawEllipse(center, radius*rect.width()/cols, radius*rect.height()/rows)
                painter.drawLine(center+QtCore.QPointF(-9, 0), center+QtCore.QPointF(9, 0))
                painter.drawLine(center+QtCore.QPointF(0, -9), center+QtCore.QPointF(0, 9))
            if self.target is not None:
                end = position(self.target)
                line = QtCore.QLineF(center, end)
                if line.length() > 20:
                    unit = (end-center)/line.length()
                    side = QtCore.QPointF(-unit.y(), unit.x())
                    for pen_color, width in ((QtCore.Qt.black, 5), (QtCore.Qt.white, 2)):
                        painter.setPen(QtGui.QPen(pen_color, width))
                        painter.drawLine(center+unit*12, end)
                        painter.drawLine(end, end-unit*12+side*6)
                        painter.drawLine(end, end-unit*12-side*6)
        caption = (
            "Model gaze / 1-sigma fovea"
            if self.precision.peak > 0 else "Full-canvas attention"
        )
        if not self.precision.sample:
            caption += " | vision not sampled"
        if self.target is not None:
            caption += " | arrow: potential visual gain"
        text_rect = QtCore.QRectF(rect.left()+8, rect.top()+8, max(0, rect.width()-16), 44)
        painter.fillRect(text_rect, QtGui.QColor(0, 0, 0, 180))
        painter.setPen(QtCore.Qt.white)
        painter.drawText(text_rect.adjusted(6, 3, -6, -3), QtCore.Qt.TextWordWrap, caption)
