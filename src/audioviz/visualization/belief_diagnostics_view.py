from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from audioviz.visualization.attention_overlay import preferred_gain_position


@dataclass(frozen=True)
class BeliefDiagnosticsFrame:
    scores: np.ndarray
    gaze: tuple[float, float]
    dt: float
    sampling: bool
    visual: dict[str, np.ndarray]
    audio: dict[str, np.ndarray] | None = None
    visual_location: str = ""
    audio_gain: float | None = None


class BeliefDiagnosticsView(QtWidgets.QWidget):
    """Visible-only previews of the stored Gaussian belief and prospective gain."""

    visibility_changed = QtCore.pyqtSignal(bool)
    open_energy_requested = QtCore.pyqtSignal()

    def __init__(self, snapshot: Callable[[], BeliefDiagnosticsFrame | None], parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.snapshot = snapshot
        self.latest: BeliefDiagnosticsFrame | None = None
        self.setWindowTitle("Belief Diagnostics - Attention and Uncertainty")
        self.resize(1000, 760)
        layout = QtWidgets.QVBoxLayout(self)
        definition = QtWidgets.QLabel(
            "Current stored Gaussian beliefs, conditional on the known canvas and fixed weights. "
            "Hidden covariance is projected diagonal; sensory covariance is locally linearized signal covariance, excluding observation noise. "
            "No wave covariance or audiovisual cross-covariance is inferred. Not calibrated confidence."
        )
        definition.setWordWrap(True)
        layout.addWidget(definition)
        self.status = QtWidgets.QLabel("Waiting for the first model forecast.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        attention = QtWidgets.QWidget()
        attention_layout = QtWidgets.QVBoxLayout(attention)
        explanation = QtWidgets.QLabel(
            "Expected information gain for a hypothetical next visual observation, in nats. "
            "This score surface is not an action-probability distribution. "
            "Observe off still previews potential evidence. Pixel-area quadrature is approximate."
        )
        explanation.setWordWrap(True)
        attention_layout.addWidget(explanation)
        self.gain_status = QtWidgets.QLabel()
        self.gain_status.setWordWrap(True)
        attention_layout.addWidget(self.gain_status)
        self.gain_plot = pg.PlotWidget()
        self.gain_plot.setLabel("bottom", "Normalized gaze X")
        self.gain_plot.setLabel("left", "Normalized gaze Y")
        self.gain_plot.invertY(True)
        self.gain_image = pg.ImageItem(axisOrder="row-major")
        self.gain_image.setLookupTable(pg.colormap.get("viridis").getLookupTable(nPts=256))
        self.gain_plot.addItem(self.gain_image)
        self.current_marker = pg.ScatterPlotItem(symbol="+", size=18, pen=pg.mkPen((255, 80, 220), width=3))
        self.best_marker = pg.ScatterPlotItem(symbol="o", size=16, pen=pg.mkPen("k", width=2), brush=None)
        self.gain_plot.addItem(self.best_marker)
        self.gain_plot.addItem(self.current_marker)
        attention_layout.addWidget(self.gain_plot, 1)
        self.candidate_status = QtWidgets.QLabel("Pink +: current gaze. Black circle: best grid candidate. Hover for a score.")
        self.candidate_status.setWordWrap(True)
        attention_layout.addWidget(self.candidate_status)
        self.gain_plot.scene().sigMouseMoved.connect(self._hover_gain)
        self.tabs.addTab(attention, "Attention / EIG")

        self.belief_tables = {}
        for key, title, description in (
            ("visual", "Vision", "Nearest image pixel to gaze. Marginals are N(mean, variance)."),
            ("audio", "Sound", "One common auditory perturbation N(0, P), shared across pixels. This is not sound direction."),
        ):
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            label = QtWidgets.QLabel(description)
            label.setWordWrap(True)
            page_layout.addWidget(label)
            sensory_mean = QtWidgets.QLabel()
            sensory_mean.setWordWrap(True)
            page_layout.addWidget(sensory_mean)
            marginals = self._table(("Mean", "Std. dev.", "Variance"))
            page_layout.addWidget(marginals, 1)
            covariances = QtWidgets.QTabWidget()
            hidden_covariance = self._table(())
            sensory_covariance = self._table(())
            covariances.addTab(hidden_covariance, "Stored hidden covariance")
            covariances.addTab(sensory_covariance, "Linearized sensory covariance")
            page_layout.addWidget(covariances, 1)
            self.belief_tables[key] = (marginals, hidden_covariance, sensory_covariance, sensory_mean)
            self.tabs.addTab(page, title)
        self.tabs.setTabEnabled(2, False)
        energy = QtWidgets.QPushButton("Open energy / weight diagnostics")
        energy.clicked.connect(self.open_energy_requested)
        layout.addWidget(energy)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh)

    @staticmethod
    def _table(headers):
        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        return table

    @staticmethod
    def _set_matrix(table, values, *, columns=None):
        table.setRowCount(values.shape[0])
        table.setColumnCount(values.shape[1])
        table.setHorizontalHeaderLabels(columns or [str(index) for index in range(values.shape[1])])
        table.setVerticalHeaderLabels([str(index) for index in range(values.shape[0])])
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                item = QtWidgets.QTableWidgetItem(f"{values[row, col]:.6g}")
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                table.setItem(row, col, item)

    def refresh(self):
        if not self.isVisible() or self.isMinimized():
            return
        frame = self.snapshot()
        if frame is None:
            self.clear()
            return
        self.latest = frame
        self.status.setText(
            f"{frame.visual_location} | prospective held-canvas step {frame.dt:.6g} s | "
            f"visual sampling {'on' if frame.sampling else 'off'} | refresh at most 10 Hz"
            + (f" | global sound potential EIG: {frame.audio_gain:.6g} nats" if frame.audio_gain is not None else "")
        )
        scores = frame.scores
        rows, cols = scores.shape
        target = preferred_gain_position(scores, frame.gaze)
        row, col = round(frame.gaze[1]*(rows-1)), round(frame.gaze[0]*(cols-1))
        best = float(scores.max())
        current = float(scores[row, col])
        self.gain_status.setText(
            f"Nearest grid gaze ({col/(cols-1):.3f}, {row/(rows-1):.3f}): {current:.6g} nats | "
            f"best grid value: {best:.6g} nats | "
            f"grid margin: {best-current:.6g} | range {scores.min():.6g} to {best:.6g}"
        )
        self.gain_image.setImage(scores, autoLevels=False, levels=(float(scores.min()), max(best, float(scores.min())+1e-12)))
        self.gain_image.setRect(QtCore.QRectF(-.5/(cols-1), -.5/(rows-1), cols/(cols-1), rows/(rows-1)))
        self.gain_plot.setRange(xRange=(-.5/(cols-1), 1+.5/(cols-1)), yRange=(-.5/(rows-1), 1+.5/(rows-1)), padding=0)
        self.current_marker.setData([frame.gaze[0]], [frame.gaze[1]])
        if target is None:
            self.best_marker.clear()
        else:
            self.best_marker.setData([target[0]], [target[1]])
        self.candidate_status.setText(
            "Pink +: model gaze. Black circle: higher-scoring grid candidate; ties choose the nearest."
            if target is not None else "Pink +: model gaze. No higher-scoring grid candidate."
        )
        self.tabs.setTabEnabled(2, frame.audio is not None)
        for key, belief in (("visual", frame.visual), ("audio", frame.audio)):
            tables = self.belief_tables[key]
            if belief is None:
                for table in tables[:3]:
                    table.setRowCount(0)
                tables[3].clear()
                continue
            mean, variance = belief["mean"], belief["variance"]
            self._set_matrix(tables[0], np.column_stack((mean, np.sqrt(variance), variance)), columns=["Mean", "Std. dev.", "Variance"])
            self._set_matrix(tables[1], np.diag(variance))
            self._set_matrix(tables[2], belief["sensory_covariance"])
            values = ", ".join(f"{value:.5g}" for value in belief["sensory_mean"])
            tables[3].setText(f"Decoded sensory mean: [{values}]")

    def _hover_gain(self, position):
        if self.latest is None:
            return
        point = self.gain_plot.getViewBox().mapSceneToView(position)
        if not 0 <= point.x() <= 1 or not 0 <= point.y() <= 1:
            return
        scores = self.latest.scores
        row, col = round(point.y()*(scores.shape[0]-1)), round(point.x()*(scores.shape[1]-1))
        self.candidate_status.setText(
            f"Candidate ({col/(scores.shape[1]-1):.3f}, {row/(scores.shape[0]-1):.3f}): {scores[row,col]:.6g} nats. "
            "Pink +: current gaze; black circle: best grid candidate."
        )

    def clear(self):
        self.latest = None
        self.gain_image.clear()
        self.current_marker.clear()
        self.best_marker.clear()
        self.gain_status.clear()
        self.status.setText("Waiting for the first model forecast.")
        self.candidate_status.clear()
        self.tabs.setTabEnabled(2, False)
        for tables in self.belief_tables.values():
            for table in tables[:3]:
                table.setRowCount(0)
            tables[3].clear()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_timer.start()
        self.refresh()
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        self.refresh_timer.stop()
        self.clear()
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange:
            if self.isMinimized():
                self.refresh_timer.stop()
                self.clear()
            elif self.isVisible():
                self.refresh_timer.start()
                self.refresh()
            self.visibility_changed.emit(self.isVisible() and not self.isMinimized())
