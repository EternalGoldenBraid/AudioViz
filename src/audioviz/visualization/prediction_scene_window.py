from __future__ import annotations

from dataclasses import replace
from time import monotonic

import numpy as np
from loguru import logger
from PyQt5 import QtCore, QtWidgets

from audioviz.utils.source_preview import SourceScenePreview


class PredictionSceneWindow(QtWidgets.QWidget):
    visibility_changed = QtCore.pyqtSignal(bool)
    open_catalogue_requested = QtCore.pyqtSignal()
    PREVIEW_INTERVAL = 0.1

    def __init__(self, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Multimodal Canvas - 3D Scene")
        self.resize(1280, 900)
        self.layer_view = None
        self.canvas: np.ndarray | None = None
        self.sources: tuple[SourceScenePreview, ...] = ()
        self._last_preview_at: float | None = None
        self._failed = False
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("Reset 3D camera")
        reset.clicked.connect(self._reset_camera)
        controls.addWidget(reset)
        catalogue = QtWidgets.QPushButton("Open inference / learning plots")
        catalogue.clicked.connect(self.open_catalogue_requested)
        controls.addWidget(catalogue)
        controls.addStretch()
        layout.addLayout(controls)
        legend = QtWidgets.QLabel(
            "Drag: orbit; wheel: zoom; Ctrl-drag: pan. One corrected canvas, independent sensory branches.\n"
            "RGB clipped [0,1]; hidden / drive colors: tanh, blue - / orange +. "
            "Audio bars: signed band magnitude, fixed [-1,1] display range.\n"
            "Hidden labels include mean error energy where available. Links show pathways, not individual weights. "
            "Labeled hidden-to-hidden links show the optional camera/audio loop. "
            "Stream-state nodes: clamped ON/OFF and predicted p(on). "
            "Previews: max 64 per side / pose nodes, 10 Hz."
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.status = QtWidgets.QLabel("Waiting for a canvas sample.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)

    def _create_layer_view(self):
        from audioviz.visualization.prediction_layer_view import PredictionLayerView

        return PredictionLayerView(self)

    def preview_active(self) -> bool:
        return self.isVisible() and not self.isMinimized() and not self._failed

    def take_spatial_preview_request(self) -> bool:
        if not self.preview_active():
            return False
        now = monotonic()
        if self._last_preview_at is not None and now - self._last_preview_at < self.PREVIEW_INTERVAL:
            return False
        self._last_preview_at = now
        return True

    def record(self, canvas: np.ndarray, sources: tuple[SourceScenePreview, ...]) -> None:
        if not self.preview_active():
            return
        self.canvas, self.sources = canvas, sources
        self.layer_view.set_scene(canvas, sources)
        self._update_status()

    def invalidate_evidence(self, key: str) -> None:
        if not any(source.mapping.key == key and source.observation is not None for source in self.sources):
            return
        self.sources = tuple(
            replace(source, observation=None) if source.mapping.key == key else source
            for source in self.sources
        )
        self.layer_view.clear_evidence(key)
        self._update_status()

    def _update_status(self) -> None:
        self.status.setText(" | ".join(
            f"{source.mapping.label}: "
            + ("disabled" if not source.enabled else
               "no current evidence" if source.observation is None else source.mapping.role)
            for source in self.sources
        ))

    def clear(self) -> None:
        self.canvas = None
        self.sources = ()
        self._last_preview_at = None
        if self.layer_view is not None:
            self.layer_view.clear_preview()
        self.status.setText("Waiting for a canvas sample.")

    def _reset_camera(self) -> None:
        if self.layer_view is not None:
            self.layer_view.reset_view()

    def _show_layer_error(self, message: str) -> None:
        logger.error("3D scene unavailable: {}", message)
        self._failed = True
        self.error.setText(f"3D unavailable: {message}. The plot catalogue remains available.")
        self.error.show()
        self.clear()
        self.visibility_changed.emit(False)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.layer_view is None:
            try:
                self.layer_view = self._create_layer_view()
            except ImportError as error:
                self._show_layer_error(str(error))
                return
            self.layer_view.rendering_failed.connect(self._show_layer_error)
            self.layout().addWidget(self.layer_view, stretch=1)
        self.visibility_changed.emit(self.preview_active())

    def hideEvent(self, event) -> None:
        self.clear()
        self.visibility_changed.emit(False)
        super().hideEvent(event)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange:
            if self.isMinimized():
                self.clear()
            self.visibility_changed.emit(self.preview_active())
