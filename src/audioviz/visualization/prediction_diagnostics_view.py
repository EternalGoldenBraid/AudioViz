from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from audioviz.readouts.diagnostics import PredictiveCodingDiagnostics


HIDDEN_COLOR = (255, 185, 75)
VISUAL_COLOR = (85, 200, 255)
POSITIVE_WEIGHT_COLOR = (85, 200, 255)
NEGATIVE_WEIGHT_COLOR = (255, 155, 85)


class PredictionDiagnosticsView(QtWidgets.QWidget):
    visibility_changed = QtCore.pyqtSignal(bool)
    open_scene_requested = QtCore.pyqtSignal()
    HISTORY_LIMIT = 300
    PREVIEW_INTERVAL = 0.1

    def __init__(self, parent=None, *, modality: str = "Camera") -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.modality = modality
        self.audio_view = None
        self.setWindowTitle("Inference and Learning - Plot Catalogue")
        self.resize(1100, 760)
        self.history: deque[tuple[float, ...]] = deque(maxlen=self.HISTORY_LIMIT)
        self.latest: PredictiveCodingDiagnostics | None = None
        self._dirty = True
        self._graph_channels = 0
        self._observation_channels = 3
        self._cross_channels = 0
        self._context_graph = False
        self._graph_edges: list[tuple[pg.PlotDataItem, int, int, int]] = []
        self._graph_labels: list[pg.TextItem] = []

        layout = QtWidgets.QVBoxLayout(self)
        definition = self._compact_label(
            "Energy = 0.5 * prediction_error^2. Layer mean and population variance "
            "are over pixels and channels, after inference and before weight learning."
        )
        if modality == "Audio":
            definition.setText(
                "Energy = 0.5 * prediction_error^2. Hidden statistics: over pixels and channels; "
                "audio statistics: over spectral bands. Errors: after inference, before learning."
            )
        layout.addWidget(definition)
        self.status = self._compact_label(f"Waiting for {modality.lower()} evidence.")
        layout.addWidget(self.status)
        catalogue = QtWidgets.QHBoxLayout()
        self.energy_toggle = QtWidgets.QCheckBox("Energy and inference plots")
        self.energy_toggle.setChecked(True)
        self.energy_toggle.toggled.connect(self._toggle_energy)
        catalogue.addWidget(self.energy_toggle)
        self.graph_toggle = QtWidgets.QCheckBox("2D weights and hidden states")
        self.graph_toggle.toggled.connect(self._toggle_graph)
        catalogue.addWidget(self.graph_toggle)
        scene_button = QtWidgets.QPushButton("Open multimodal 3D scene")
        scene_button.clicked.connect(self.open_scene_requested)
        catalogue.addWidget(scene_button)
        layout.addLayout(catalogue)
        self.empty_catalogue = self._compact_label("Select a plot from the catalogue above.")
        self.empty_catalogue.hide()
        layout.addWidget(self.empty_catalogue)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter = splitter
        layout.addWidget(splitter, stretch=1)
        charts = pg.GraphicsLayoutWidget()
        self.charts = charts
        splitter.addWidget(charts)
        mean_plot = charts.addPlot(row=0, col=0, title="Mean node energy")
        variance_plot = charts.addPlot(
            row=1, col=0, title="Population variance of node energy"
        )
        inference_plot = charts.addPlot(
            row=2, col=0, title="Latest inference trajectory (weights fixed)"
        )
        self.mean_curves = self._layer_curves(mean_plot)
        self.variance_curves = self._layer_curves(variance_plot)
        self.stream_mean_curve = mean_plot.plot(
            name="Stream state", pen=pg.mkPen((140, 220, 155), width=2), connect="finite",
        )
        self.stream_variance_curve = variance_plot.plot(
            name="Stream state", pen=pg.mkPen((140, 220, 155), width=2), connect="finite",
        )
        mean_plot.setLabel("left", "Energy")
        variance_plot.setLabel("left", "Variance")
        variance_plot.setLabel("bottom", "Simulation time", units="s")
        variance_plot.setXLink(mean_plot)
        inference_plot.setLabel("left", "Total energy")
        inference_plot.setLabel("bottom", "State updates this frame")
        self.inference_curve = inference_plot.plot(pen=pg.mkPen((200, 235, 180), width=2))
        for plot in (mean_plot, variance_plot, inference_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)

        self.graph_container = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        self.graph_legend = self._compact_label(
            "Shared channel graph, repeated at each pixel.\n"
            "Predictions -> (tanh); local errors <- (dashed).\n"
            "Edges: blue + / orange -; width = |weight|.\n"
            "Nodes: state mean; fill = local mean energy.\n"
            "Canvas has no own error. Weights: after learning."
        )
        if modality == "Audio":
            self.graph_legend.setText(
                "Canvas -> audio hidden field -> spectral bands.\n"
                "The sensory readout pools tanh(hidden) over space.\n"
                "Predictions ->; local errors <- (dashed).\n"
                "Edges: blue + / orange -; width = |weight|.\n"
                "Nodes: mean state and local mean energy."
            )
        self.graph_legend.setText(
            self.graph_legend.text()
            + "\nLoop on: P = peer hidden. Incoming loop weights shown; reciprocal weights in the other tab."
            + "\nS = clamped stream ON/OFF state; pooled hidden -> sigmoid."
        )
        policy = self.graph_legend.sizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Preferred)
        self.graph_legend.setSizePolicy(policy)
        graph_layout.addWidget(self.graph_legend)
        self.graph_plot = pg.PlotWidget()
        self.graph_plot.hideAxis("left")
        self.graph_plot.hideAxis("bottom")
        self.graph_plot.setMouseEnabled(x=False, y=False)
        graph_layout.addWidget(self.graph_plot, stretch=1)
        splitter.addWidget(self.graph_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.graph_container.hide()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.refresh_timer.setInterval(round(self.PREVIEW_INTERVAL * 1000))
        self.refresh_timer.timeout.connect(self.refresh)

    @staticmethod
    def _compact_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        policy = label.sizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        label.setSizePolicy(policy)
        return label

    def add_audio_tab(self) -> None:
        body = QtWidgets.QWidget()
        body.setLayout(self.layout())
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(body, "Camera")
        self.audio_view = PredictionDiagnosticsView(self, modality="Audio")
        self.audio_view.setWindowFlags(QtCore.Qt.Widget)
        self.tabs.addTab(self.audio_view, "Audio")
        self.audio_view.open_scene_requested.connect(self.open_scene_requested)

    def _layer_curves(self, plot) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
        plot.addLegend(offset=(-10, 10), brush=pg.mkBrush(20, 25, 30, 210))
        return (
            plot.plot(name="Hidden", pen=pg.mkPen(HIDDEN_COLOR, width=2), connect="finite"),
            plot.plot(name=self.modality, pen=pg.mkPen(VISUAL_COLOR, width=2), connect="finite"),
        )

    def record(
        self,
        simulation_time: float,
        snapshot: PredictiveCodingDiagnostics | None,
    ) -> None:
        if snapshot is None:
            if self.latest is not None:
                self.history.append((simulation_time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            self.history.append(
                (
                    simulation_time,
                    snapshot.hidden.mean,
                    snapshot.visual.mean,
                    snapshot.hidden.variance,
                    snapshot.visual.variance,
                    snapshot.stream_state.energy if snapshot.stream_state is not None else np.nan,
                    0.0 if snapshot.stream_state is not None else np.nan,
                )
            )
        self.latest = (
            replace(snapshot, spatial=None)
            if snapshot is not None and snapshot.spatial is not None else snapshot
        )
        self._dirty = True

    def clear(self) -> None:
        self.history.clear()
        self.latest = None
        self._dirty = True
        self.refresh()

    def refresh(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        if self.history:
            values = np.asarray(self.history)
            for curve, column in zip(
                (*self.mean_curves, *self.variance_curves), range(1, 5)
            ):
                curve.setData(values[:, 0], values[:, column])
            self.stream_mean_curve.setData(values[:, 0], values[:, 5])
            self.stream_variance_curve.setData(values[:, 0], values[:, 6])
        else:
            for curve in (*self.mean_curves, *self.variance_curves, self.stream_mean_curve, self.stream_variance_curve):
                curve.clear()
        if self.latest is None:
            self.status.setText(
                f"No fresh {self.modality.lower()} evidence: inference and weight learning are not sampled."
            )
            self.inference_curve.clear()
            self.graph_plot.clear()
            self._graph_channels = 0
            return
        snapshot = self.latest
        learning = "on" if snapshot.learning_enabled else "off"
        sensory_status = (
            f"{self.modality} mean {snapshot.visual.mean:.4g}, variance {snapshot.visual.variance:.4g}"
            if snapshot.observation_present else f"{self.modality}: missing evidence (unclamped)"
        )
        context = snapshot.stream_state
        context_status = (
            f"\nStream state {'ON' if context.observed else 'OFF'} (clamped)"
            f" | p(on)={context.prediction:.3g} | E={context.energy:.3g}"
            if context is not None else ""
        )
        self.status.setText(
            f"Hidden mean {snapshot.hidden.mean:.4g}, variance {snapshot.hidden.variance:.4g}"
            f" | {sensory_status}"
            f"\nWeight learning {learning} (rate {snapshot.learning_rate:g})"
            f" | actual ||delta W|| = {snapshot.weight_update_norm:.4g}"
            + (" | missing evidence: sensory nodes show predictions, sensory energy is unavailable"
               if not snapshot.observation_present else "")
            + context_status
        )
        self.inference_curve.setData(
            np.arange(len(snapshot.inference_energy)), snapshot.inference_energy
        )
        if self.graph_toggle.isChecked():
            self._update_graph(snapshot)

    def _toggle_energy(self, enabled: bool) -> None:
        self.charts.setVisible(enabled)
        self.empty_catalogue.setVisible(not enabled and not self.graph_toggle.isChecked())

    def _toggle_graph(self, enabled: bool) -> None:
        self.graph_container.setVisible(enabled)
        if enabled:
            width = self.splitter.width() // 2
            self.splitter.setSizes([width, width])
        self.empty_catalogue.setVisible(not enabled and not self.energy_toggle.isChecked())
        self._dirty = True
        self.refresh()

    def _build_graph(self, hidden_channels: int, cross_channels: int = 0, context: bool = False) -> None:
        self.graph_plot.clear()
        self._graph_edges.clear()
        self._graph_labels.clear()
        self._graph_channels = hidden_channels
        self._cross_channels = cross_channels
        self._context_graph = context
        height = max(hidden_channels - 1, self._observation_channels - 1, 2)
        positions = [
            np.column_stack(
                (np.full(count, column * 1.4), np.linspace(0, height, count))
            )
            for column, count in enumerate((3, hidden_channels, self._observation_channels))
        ]
        self._node_positions = np.concatenate(positions)
        for layer, (sources, destinations) in enumerate(zip(positions, positions[1:])):
            for destination, end in enumerate(destinations):
                for source, start in enumerate(sources):
                    edge = pg.PlotDataItem(
                        [start[0], end[0]], [start[1], end[1]]
                    )
                    self.graph_plot.addItem(edge)
                    self._graph_edges.append((edge, layer, destination, source))
        if cross_channels:
            peers = np.column_stack(
                (np.full(cross_channels, -1.4), np.linspace(0, height, cross_channels))
            )
            self._node_positions = np.concatenate((self._node_positions, peers))
            for destination, end in enumerate(positions[1]):
                for source, start in enumerate(peers):
                    edge = pg.PlotDataItem(
                        [start[0], -0.7, 1.0, end[0]],
                        [start[1], -1.5 - .05 * source, -1.5 - .05 * destination, end[1]],
                    )
                    self.graph_plot.addItem(edge)
                    self._graph_edges.append((edge, 2, destination, source))
            label = pg.TextItem("Peer hidden -> hidden", color=(225, 230, 235), anchor=(0.5, 0.5))
            label.setPos(-1.1, height + 0.7)
            self.graph_plot.addItem(label)
            recurrent = pg.TextItem("recurrent predictions", color=(225, 230, 235), anchor=(0.5, 0.5))
            recurrent.setPos(0.2, -2.05)
            self.graph_plot.addItem(recurrent)
        if context:
            state_position = np.array([2.8, -2.2])
            self._node_positions = np.concatenate((self._node_positions, state_position[None, :]))
            for source, start in enumerate(positions[1]):
                edge = pg.PlotDataItem(
                    [start[0], 2.0, state_position[0]], [start[1], -1.3, state_position[1]]
                )
                self.graph_plot.addItem(edge)
                self._graph_edges.append((edge, 3, 0, source))
        self._graph_nodes = pg.ScatterPlotItem(size=16, pen=pg.mkPen("w", width=1))
        self.graph_plot.addItem(self._graph_nodes)
        for x, y in self._node_positions:
            label = pg.TextItem(
                color=(225, 230, 235), anchor=(0, 0.5),
                fill=pg.mkBrush(25, 35, 44, 220),
            )
            label.setPos(x + 0.1, y)
            self.graph_plot.addItem(label)
            self._graph_labels.append(label)
        for column, title in enumerate(("Wave canvas", "Hidden belief", self.modality)):
            label = pg.TextItem(title, color=(225, 230, 235), anchor=(0.5, 0.5))
            label.setPos(column * 1.4 + 0.2, height + 0.7)
            self.graph_plot.addItem(label)
            if column == 2:
                self._sensory_title = label
        for column in (0, 1):
            left = column * 1.4
            right = left + 1.4
            feedback = pg.PlotDataItem(
                [left + 0.2, right - 0.1],
                [-0.65, -0.65],
                pen=pg.mkPen((185, 215, 185), width=2, style=QtCore.Qt.DashLine),
            )
            self.graph_plot.addItem(feedback)
            self.graph_plot.addItem(
                pg.ArrowItem(
                    pos=(left + 0.2, -0.65), angle=0, headLen=12,
                    pen=None, brush=(185, 215, 185),
                )
            )
            label = pg.TextItem("local error", color=(185, 215, 185), anchor=(0.5, 0.5))
            label.setPos(left + 0.7, -1.05)
            self.graph_plot.addItem(label)
        self.graph_plot.setRange(
            xRange=(-1.8 if cross_channels else -0.3, 5.0 if context else 4.3 if cross_channels else 3.9),
            yRange=(-4.2 if context else -2.5 if cross_channels else -1.4, height + 1.2), padding=0
        )

    def _update_graph(self, snapshot: PredictiveCodingDiagnostics) -> None:
        count = len(snapshot.hidden_means)
        observation_channels = len(snapshot.observation_means)
        cross_channels = len(snapshot.cross_parent_means)
        context = snapshot.stream_state
        if (self._graph_channels != count or self._observation_channels != observation_channels
                or self._cross_channels != cross_channels or self._context_graph != (context is not None)):
            self._observation_channels = observation_channels
            self._build_graph(count, cross_channels, context is not None)
        weights = (
            snapshot.canvas_weights, snapshot.visual_weights, snapshot.cross_modal_weights,
            context.weights if context is not None else None,
        )
        self._sensory_title.setText(
            f"{self.modality} ({'observed' if snapshot.observation_present else 'predicted'})"
        )
        for edge, layer, destination, source in self._graph_edges:
            matrix = weights[layer]
            if matrix is None:
                raise ValueError("Recurrent graph nodes require incoming recurrent weights.")
            weight = float(matrix[destination, source])
            color = POSITIVE_WEIGHT_COLOR if weight >= 0 else NEGATIVE_WEIGHT_COLOR
            edge.setPen(pg.mkPen((*color, 110), width=0.5 + min(abs(weight), 3.0)))
            edge.setVisible(weight != 0.0)
        energies = ((None,) * 3 + snapshot.hidden.channel_means
                    + tuple(value if np.isfinite(value) else None for value in snapshot.visual.channel_means)
                    + (None,) * cross_channels)
        means = snapshot.canvas_means + snapshot.hidden_means + snapshot.observation_means + snapshot.cross_parent_means
        names = (
            tuple(f"C{index}" for index in range(3))
            + tuple(f"H{index}" for index in range(count))
            + tuple(f"Y{index}" for index in range(self._observation_channels))
            + tuple(f"P{index}" for index in range(cross_channels))
        )
        if context is not None:
            energies += (context.energy,)
            means += (float(context.observed),)
            names += ("S",)
        maximum = max((value for value in energies if value is not None), default=1e-12)
        maximum = max(maximum, 1e-12)
        brushes = []
        for label, name, value, energy in zip(self._graph_labels, names, means, energies):
            level = 0.0 if energy is None else energy / maximum
            brushes.append(pg.mkBrush(40 + int(210 * level), 100 + int(60 * level), 155 - int(90 * level)))
            text = f"{name}: {value:+.3g}"
            if name == "S" and context is not None:
                text = (
                    f"S: {'ON' if context.observed else 'OFF'} (clamped)\np(on)={context.prediction:.3g}"
                    f"\nE={context.energy:.2g}; b={context.bias:.2g}"
                )
            elif energy is not None:
                separator = " " if name.startswith("Y") and self._observation_channels > 8 else "\n"
                text += f"{separator}E={energy:.2g}"
            label.setText(text)
        self._graph_nodes.setData(
            self._node_positions[:, 0], self._node_positions[:, 1], brush=brushes
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_timer.start()
        self.visibility_changed.emit(True)

    def hideEvent(self, event) -> None:
        self.refresh_timer.stop()
        self.visibility_changed.emit(False)
        super().hideEvent(event)
