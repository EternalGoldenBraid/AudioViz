from __future__ import annotations

from collections import deque

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
    HISTORY_LIMIT = 300

    def __init__(self, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Predictive Coding Diagnostics")
        self.resize(1100, 760)
        self.history: deque[tuple[float, ...]] = deque(maxlen=self.HISTORY_LIMIT)
        self.latest: PredictiveCodingDiagnostics | None = None
        self._dirty = True
        self._graph_channels = 0
        self._graph_edges: list[tuple[pg.PlotDataItem, int, int, int]] = []
        self._graph_labels: list[pg.TextItem] = []

        layout = QtWidgets.QVBoxLayout(self)
        definition = self._compact_label(
            "Energy = 0.5 * prediction_error^2. Layer mean and population variance "
            "are over pixels and channels, after inference and before weight learning."
        )
        layout.addWidget(definition)
        self.status = self._compact_label("Waiting for camera evidence.")
        layout.addWidget(self.status)
        self.graph_toggle = QtWidgets.QCheckBox("Show shared-channel computational graph")
        self.graph_toggle.toggled.connect(self._toggle_graph)
        layout.addWidget(self.graph_toggle)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)
        charts = pg.GraphicsLayoutWidget()
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
        legend = self._compact_label(
            "Shared channel graph, repeated at each pixel.\n"
            "Predictions -> (tanh); local errors <- (dashed).\n"
            "Edges: blue + / orange -; width = |weight|.\n"
            "Nodes: state mean; fill = local mean energy.\n"
            "Canvas has no own error. Weights: after learning."
        )
        graph_layout.addWidget(legend)
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
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh)

    @staticmethod
    def _compact_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        policy = label.sizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        label.setSizePolicy(policy)
        return label

    @staticmethod
    def _layer_curves(plot) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
        plot.addLegend(offset=(-10, 10), brush=pg.mkBrush(20, 25, 30, 210))
        return (
            plot.plot(name="Hidden", pen=pg.mkPen(HIDDEN_COLOR, width=2), connect="finite"),
            plot.plot(name="Camera", pen=pg.mkPen(VISUAL_COLOR, width=2), connect="finite"),
        )

    def record(
        self, simulation_time: float, snapshot: PredictiveCodingDiagnostics | None
    ) -> None:
        if snapshot is None:
            if self.latest is not None:
                self.history.append((simulation_time, np.nan, np.nan, np.nan, np.nan))
        else:
            self.history.append(
                (
                    simulation_time,
                    snapshot.hidden.mean,
                    snapshot.visual.mean,
                    snapshot.hidden.variance,
                    snapshot.visual.variance,
                )
            )
        self.latest = snapshot
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
        else:
            for curve in (*self.mean_curves, *self.variance_curves):
                curve.clear()
        if self.latest is None:
            self.status.setText(
                "No fresh camera evidence: inference and weight learning are not sampled."
            )
            self.inference_curve.clear()
            self.graph_plot.clear()
            self._graph_channels = 0
            return
        snapshot = self.latest
        learning = "on" if snapshot.learning_enabled else "off"
        self.status.setText(
            f"Hidden mean {snapshot.hidden.mean:.4g}, variance {snapshot.hidden.variance:.4g}"
            f" | Camera mean {snapshot.visual.mean:.4g}, variance {snapshot.visual.variance:.4g}"
            f"\nWeight learning {learning} (rate {snapshot.learning_rate:g})"
            f" | actual ||delta W|| = {snapshot.weight_update_norm:.4g}"
        )
        self.inference_curve.setData(
            np.arange(len(snapshot.inference_energy)), snapshot.inference_energy
        )
        if self.graph_toggle.isChecked():
            self._update_graph(snapshot)

    def _toggle_graph(self, enabled: bool) -> None:
        self.graph_container.setVisible(enabled)
        self._dirty = True
        self.refresh()

    def _build_graph(self, hidden_channels: int) -> None:
        self.graph_plot.clear()
        self._graph_edges.clear()
        self._graph_labels.clear()
        self._graph_channels = hidden_channels
        height = max(hidden_channels - 1, 2)
        positions = [
            np.column_stack(
                (np.full(count, column * 1.4), np.linspace(0, height, count))
            )
            for column, count in enumerate((3, hidden_channels, 3))
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
        for column, title in enumerate(("Wave canvas", "Hidden belief", "Camera (clamped)")):
            label = pg.TextItem(title, color=(225, 230, 235), anchor=(0.5, 0.5))
            label.setPos(column * 1.4 + 0.2, height + 0.7)
            self.graph_plot.addItem(label)
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
            xRange=(-0.3, 3.9), yRange=(-1.4, height + 1.2), padding=0
        )

    def _update_graph(self, snapshot: PredictiveCodingDiagnostics) -> None:
        count = len(snapshot.hidden_means)
        if self._graph_channels != count:
            self._build_graph(count)
        weights = snapshot.canvas_weights, snapshot.visual_weights
        for edge, layer, destination, source in self._graph_edges:
            weight = float(weights[layer][destination, source])
            color = POSITIVE_WEIGHT_COLOR if weight >= 0 else NEGATIVE_WEIGHT_COLOR
            edge.setPen(pg.mkPen((*color, 110), width=0.5 + min(abs(weight), 3.0)))
            edge.setVisible(weight != 0.0)
        energies = (None,) * 3 + snapshot.hidden.channel_means + snapshot.visual.channel_means
        means = snapshot.canvas_means + snapshot.hidden_means + snapshot.observation_means
        names = (
            tuple(f"C{index}" for index in range(3))
            + tuple(f"H{index}" for index in range(count))
            + tuple(f"Y{index}" for index in range(3))
        )
        maximum = max((*snapshot.hidden.channel_means, *snapshot.visual.channel_means, 1e-12))
        brushes = []
        for label, name, value, energy in zip(self._graph_labels, names, means, energies):
            level = 0.0 if energy is None else energy / maximum
            brushes.append(pg.mkBrush(40 + int(210 * level), 100 + int(60 * level), 155 - int(90 * level)))
            text = f"{name}: {value:+.3g}"
            if energy is not None:
                text += f"\nE={energy:.2g}"
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
