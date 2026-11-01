from __future__ import annotations

from dataclasses import replace
from time import monotonic

import numpy as np
from loguru import logger
from PyQt5 import QtCore, QtWidgets

from audioviz.utils.source_preview import SourceScenePreview
from audioviz.utils.continuous_graph_preview import ContinuousGraphPreview
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
        self.continuous_graph = False
        self.graph_snapshot: ContinuousGraphPreview | None = None
        self._snapshot_requested = True
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
        self.live_graph = QtWidgets.QCheckBox("Live")
        self.live_graph.setChecked(True)
        self.live_graph.setToolTip("Follow model steps at up to 10 Hz. Uncheck to freeze only this visualization.")
        self.live_graph.toggled.connect(self._set_graph_live)
        self.live_graph.hide()
        controls.addWidget(self.live_graph)
        self.snapshot_button = QtWidgets.QPushButton("Capture frame")
        self.snapshot_button.setToolTip("Refresh the frozen visualization on the next model step.")
        self.snapshot_button.clicked.connect(self._request_snapshot)
        self.snapshot_button.hide()
        controls.addWidget(self.snapshot_button)
        self.catalogue_button = catalogue = QtWidgets.QPushButton("Open inference / learning plots")
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
            "Modeled stream-state nodes: clamped ON/OFF and predicted p(on). "
            "Previews: max 64 per side, 10 Hz."
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

    def configure_continuous_graph(self) -> None:
        self.continuous_graph = True
        self.layout_selector.setItemText(self.layout_selector.findData("3d"), "3D graph")
        self._graph_legend = (
            "Live settled-state graph, up to 10 Hz as the model steps. Uncheck Live to freeze this view; "
            "Capture frame refreshes it on the next model step. "
            "Drag: orbit; wheel: zoom; Ctrl-drag: pan.\n"
            "Bottom: full reference. Above: whole-world sigmoid(z) prediction, then whole-frame RGB RMSE "
            "(diagnostic only, never fed back; tanh color scale). Gray error means reference unavailable. "
            "Yellow outline: selected fovea; error outline marks actual evidence used by the model.\n"
            "Top: one recurrent hidden population (up to 12 actual neurons). Right: separate canvas/root input. "
            "Dots on images: up to 25 actual RGB coordinates, channels cycle R/G/B. Audio: up to 4 actual nodes.\n"
            "Arrows: top 24 nonzero |weights| per displayed block: Ws hidden->sensory, Wc root->hidden, "
            "Wh hidden->hidden. Orange + / blue -. Other connections are omitted, not absent. "
            "Error gradients return through these same weights; no separate learned feedback links. "
            "Depth is display layout, not additional model layers. Image samples <=64 per side."
        )
        self.layout_selector.setToolTip("The graph shows real indexed neurons and sampled weight blocks.")
        self._change_layout(self.layout_selector.currentIndex())

    def _request_snapshot(self) -> None:
        self._snapshot_requested = True

    def _set_graph_live(self, enabled: bool) -> None:
        self.snapshot_button.setEnabled(not enabled)
        if enabled:
            self._snapshot_requested = True
            self._last_preview_at = None
        self._update_status()

    def _change_layout(self, _index: int) -> None:
        graph = self.needs_hidden_preview
        self.views.setCurrentIndex(0 if graph else 1)
        self.reset_button.setEnabled(graph)
        self.snapshot_button.setVisible(graph and self.continuous_graph)
        self.snapshot_button.setEnabled(not self.live_graph.isChecked())
        self.live_graph.setVisible(graph and self.continuous_graph)
        self.error.setVisible(graph and self._failed)
        self.legend.setText(self._graph_legend if graph else (
            "Pre-evidence camera prediction, not the corrected canvas. RGB clipped to [0,1] at a fixed scale. "
            "Comparison includes the corrected canvas, current camera evidence and audio spectra. "
            "The camera frame is shown in full; a foveal circle marks observation support when present. "
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
        if self.continuous_graph and self.needs_hidden_preview:
            if self._snapshot_requested:
                return True
            if not self.live_graph.isChecked():
                return False
        now = monotonic()
        if self._last_preview_at is not None and now - self._last_preview_at < self.PREVIEW_INTERVAL:
            return False
        self._last_preview_at = now
        return True

    def record(
        self, canvas: np.ndarray, sources: tuple[SourceScenePreview, ...],
        *, graph_snapshot: ContinuousGraphPreview | None = None,
    ) -> None:
        if not self.preview_active():
            return
        self.canvas, self.sources = canvas, sources
        if self.needs_hidden_preview:
            if self.continuous_graph:
                if graph_snapshot is None:
                    raise ValueError("A continuous graph view requires a graph snapshot")
                self.graph_snapshot = graph_snapshot
                self.layer_view.set_graph_snapshot(graph_snapshot)
                self._snapshot_requested = False
                self._last_preview_at = monotonic()
            else:
                self.layer_view.set_scene(canvas, sources)
        else:
            self.image_view.set_scene(canvas, sources)
        self._update_status()

    def invalidate_evidence(self, key: str) -> None:
        if self.continuous_graph and self.needs_hidden_preview:
            if (
                self.live_graph.isChecked() and key == "camera"
                and self.graph_snapshot is not None and self.graph_snapshot.reference is not None
                and not self.graph_snapshot.independent_reference
            ):
                self.graph_snapshot = replace(
                    self.graph_snapshot, reference=None,
                    observed=np.zeros_like(self.graph_snapshot.observed),
                    error=None,
                )
                self.layer_view.set_graph_snapshot(self.graph_snapshot)
                self._update_status()
            return
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
        if self.continuous_graph and self.needs_hidden_preview and self.graph_snapshot is not None:
            snapshot = self.graph_snapshot
            mode = "Live graph" if self.live_graph.isChecked() else "Frozen snapshot"
            self.status.setText(
                f"{mode} | world {snapshot.visual_shape[0]} x {snapshot.visual_shape[1]} | "
                f"{int(snapshot.observed.sum())}/{snapshot.observed.size} preview pixels observed | "
                "states and weights captured before the displayed step's learning.\n"
                + "Hidden state: " + ", ".join(
                    f"h{index}={value:+.3f}"
                    for index, value in zip(snapshot.hidden_indices, snapshot.hidden)
                )
            )
            return
        self.status.setText(" | ".join(
            f"{source.mapping.label}: "
            + ("disabled" if not source.enabled else
               "no current evidence" if source.observation is None else source.mapping.role)
            for source in self.sources
        ))

    def clear(self) -> None:
        self.canvas = None
        self.sources = ()
        self.graph_snapshot = None
        self._snapshot_requested = True
        self._last_preview_at = None
        if self.layer_view is not None:
            self.layer_view.clear_preview()
        self.image_view.clear_preview()
        self.status.setText(
            "Waiting for the next model step. Run or step the simulation to update the graph."
            if self.continuous_graph and self.needs_hidden_preview else "Waiting for a canvas sample."
        )

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
