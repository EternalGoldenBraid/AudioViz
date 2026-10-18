from __future__ import annotations

from dataclasses import replace
from time import monotonic

import numpy as np
from loguru import logger
from PyQt5 import QtCore, QtWidgets

from audioviz.utils.source_preview import SourceScenePreview
from audioviz.utils.spatial_preview import PREVIEW_MAX_SIDE
from audioviz.visualization.prediction_image_view import PredictionImageView


class PredictionSceneWindow(QtWidgets.QWidget):
    visibility_changed = QtCore.pyqtSignal(bool)
    open_catalogue_requested = QtCore.pyqtSignal()
    PREVIEW_INTERVAL = 0.1
    IMAGE_PREVIEW_MAX_SIDE = 512

    def __init__(self, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Multimodal Canvas - Prediction Views")
        self.resize(1280, 900)
        self.layer_view = None
        self.canvas: np.ndarray | None = None
        self.sources: tuple[SourceScenePreview, ...] = ()
        self._last_preview_at: float | None = None
        self._failed = False
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.layout_selector = QtWidgets.QComboBox()
        for label, key in (("3D graph", "3d"), ("2D comparison", "comparison"), ("Prediction only", "prediction")):
            self.layout_selector.addItem(label, key)
        controls.addWidget(QtWidgets.QLabel("Layout"))
        controls.addWidget(self.layout_selector)
        self.reset_button = QtWidgets.QPushButton("Reset 3D camera")
        self.reset_button.clicked.connect(self._reset_camera)
        controls.addWidget(self.reset_button)
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
        self.legend = legend
        self._graph_legend = legend.text()
        layout.addWidget(legend)
        self.status = QtWidgets.QLabel("Waiting for a canvas sample.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)
        self.views = QtWidgets.QStackedWidget()
        self.graph_container = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self.graph_container).setContentsMargins(0, 0, 0, 0)
        self.views.addWidget(self.graph_container)
        self.image_view = PredictionImageView()
        self.views.addWidget(self.image_view)
        layout.addWidget(self.views, stretch=1)
        self.layout_selector.currentIndexChanged.connect(self._change_layout)

    @property
    def needs_hidden_preview(self) -> bool:
        return self.layout_selector.currentData() == "3d"

    @property
    def preview_max_side(self) -> int:
        return PREVIEW_MAX_SIDE if self.needs_hidden_preview else self.IMAGE_PREVIEW_MAX_SIDE

    def _change_layout(self, _index: int) -> None:
        graph = self.needs_hidden_preview
        self.views.setCurrentIndex(0 if graph else 1)
        self.reset_button.setEnabled(graph)
        self.error.setVisible(graph and self._failed)
        self.legend.setText(self._graph_legend if graph else (
            "Pre-evidence camera prediction, not the corrected canvas. RGB clipped to [0,1] at a fixed scale. "
            "Comparison includes the corrected canvas, current camera evidence and audio spectra. "
            "Image previews: at most 512 pixels per side, 10 Hz. Missing evidence is cleared."
        ))
        if graph and self.isVisible():
            self._ensure_layer_view()
        elif not graph:
            self.image_view.set_layout(self.layout_selector.currentData())
        self.clear()
        self.visibility_changed.emit(self.preview_active())

    def _create_layer_view(self):
        from audioviz.visualization.prediction_layer_view import PredictionLayerView

        return PredictionLayerView(self)

    def preview_active(self) -> bool:
        return self.isVisible() and not self.isMinimized() and not (self.needs_hidden_preview and self._failed)

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
        if self.needs_hidden_preview:
            self.layer_view.set_scene(canvas, sources)
        else:
            self.image_view.set_scene(canvas, sources)
        self._update_status()

    def invalidate_evidence(self, key: str) -> None:
        if not any(source.mapping.key == key and source.observation is not None for source in self.sources):
            return
        self.sources = tuple(
            replace(source, observation=None) if source.mapping.key == key else source
            for source in self.sources
        )
        if self.needs_hidden_preview:
            self.layer_view.clear_evidence(key)
        elif self.canvas is not None:
            self.image_view.set_scene(self.canvas, self.sources)
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
        self.image_view.clear_preview()
        self.status.setText("Waiting for a canvas sample.")

    def _reset_camera(self) -> None:
        if self.layer_view is not None:
            self.layer_view.reset_view()

    def _show_layer_error(self, message: str) -> None:
        logger.error("3D scene unavailable: {}", message)
        self._failed = True
        self.error.setText(f"3D unavailable: {message}. Select a 2D layout or open the plot catalogue.")
        self.error.setVisible(self.needs_hidden_preview)
        if self.needs_hidden_preview:
            self.clear()
            self.visibility_changed.emit(False)

    def _ensure_layer_view(self) -> None:
        if self.layer_view is None and not self._failed:
            try:
                self.layer_view = self._create_layer_view()
            except ImportError as error:
                self._show_layer_error(str(error))
                return
            self.layer_view.rendering_failed.connect(self._show_layer_error)
            self.graph_container.layout().addWidget(self.layer_view)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.needs_hidden_preview:
            self._ensure_layer_view()
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
