from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from audioviz.visualization.attention_overlay import preferred_gain_position
from audioviz.readouts.shared_predictive_coding import SharedPredictiveCodingSnapshot


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
    joint: SharedPredictiveCodingSnapshot | None = None


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
        self.definition = definition
        self._separate_definition = definition.text()
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
        self.gain_explanation = explanation
        self._separate_gain_explanation = explanation.text()
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
        self.belief_descriptions = {}
        self.covariance_tabs = {}
        for key, title, description in (
            ("visual", "Vision", "Nearest image pixel to gaze. Marginals are N(mean, variance)."),
            ("audio", "Sound", "One common auditory perturbation N(0, P), shared across pixels. This is not sound direction."),
        ):
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            label = QtWidgets.QLabel(description)
            label.setWordWrap(True)
            self.belief_descriptions[key] = (label, description)
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
            self.covariance_tabs[key] = covariances
            self.belief_tables[key] = (marginals, hidden_covariance, sensory_covariance, sensory_mean)
            self.tabs.addTab(page, title)
        self.tabs.setTabEnabled(2, False)
        joint_page = QtWidgets.QWidget()
        joint_layout = QtWidgets.QVBoxLayout(joint_page)
        self.joint_status = QtWidgets.QLabel()
        self.joint_status.setWordWrap(True)
        joint_layout.addWidget(self.joint_status)
        self.joint_tabs = QtWidgets.QTabWidget()
        self.joint_tables = []
        for title, columns in (
            ("States / clamps", ("Node", "Clamped", "Evidence", "State", "Local variance")),
            ("Local GN covariance", ()),
            ("Canvas -> hidden", ()),
            ("Hidden -> hidden", ()),
            ("Hidden -> sensory", ()),
            ("Prediction errors", ()),
            ("Canvas gradient", ()),
            ("Inference energy", ()),
            ("Generative biases", ()),
        ):
            table = self._table(columns)
            self.joint_tables.append(table)
            self.joint_tabs.addTab(table, title)
        joint_layout.addWidget(self.joint_tabs, 1)
        self.tabs.addTab(joint_page, "Joint Graph")
        self.tabs.setTabEnabled(3, False)
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        self.joint_tabs.currentChanged.connect(lambda _: self.refresh())
        energy = QtWidgets.QPushButton("Open energy / weight diagnostics")
        energy.clicked.connect(self.open_energy_requested)
        layout.addWidget(energy)
        self.energy_button = energy
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
                item = table.item(row, col)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                    table.setItem(row, col, item)
                item.setText(f"{values[row, col]:.6g}")

    def refresh(self):
        if not self.isVisible() or self.isMinimized():
            return
        frame = self.snapshot()
        if frame is None:
            self.clear()
            return
        self.latest = frame
        shared = frame.joint is not None
        self.definition.setText(
            "Generative predictive coding: canvas root -> dense recurrent hidden -> sensory terminals. "
            "Inference, weight learning and canvas feedback use the same residual energy. "
            "Covariance is a local Gauss-Newton approximation with fixed canvas/weights and assigned unit residual noise, "
            "not an exact temporal posterior or calibrated confidence. No wave covariance or sound direction is inferred."
            if shared else self._separate_definition
        )
        self.gain_explanation.setText(
            "Local linearized information gain about shared hidden nodes, using the pre-evidence generative "
            "Jacobian and unit sensory residual noise, in nats. This is a curvature-based counterfactual, "
            "not exact nonlinear information gain. Each candidate routes a hard circular patch of the canvas; "
            "no current camera evidence is used to choose gaze."
            if shared else self._separate_gain_explanation
        )
        self.tabs.setTabEnabled(3, shared)
        self.energy_button.setEnabled(not shared)
        self.energy_button.setToolTip("Generative weights, prediction errors and energy are in Joint Graph." if shared else "")
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
            label, description = self.belief_descriptions[key]
            label.setText(
                "Both tabs show the same shared hidden state and its local Gaussian approximation. "
                + ("RGB statistics refer to the foveal center pixel." if key == "visual" else
                   "Audio statistics refer to the global spectral nodes, not a located source.")
                if shared else description
            )
            self.covariance_tabs[key].setTabText(
                1, "Local sensory covariance" if shared else "Linearized sensory covariance",
            )
            tables = self.belief_tables[key]
            if belief is None:
                for table in tables[:3]:
                    table.setRowCount(0)
                tables[3].clear()
                continue
            mean, variance = belief["mean"], belief["variance"]
            self._set_matrix(
                tables[0], np.column_stack((mean, np.sqrt(variance), variance)),
                columns=["Settled state" if shared else "Mean", "Std. dev.", "Variance"],
            )
            self._set_matrix(tables[1], belief["covariance"] if "covariance" in belief else np.diag(variance))
            self._set_matrix(tables[2], belief["sensory_covariance"])
            values = ", ".join(f"{value:.5g}" for value in belief["sensory_mean"])
            tables[3].setText(
                f"{'Sensory terminal state' if shared else 'Decoded sensory mean'}: [{values}]",
            )
        if shared:
            self._refresh_joint(frame.joint)

    def _refresh_joint(self, joint):
        clamped = joint["clamped"]
        covariance = joint["hidden_covariance"]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise FloatingPointError("Local hidden covariance must be positive definite")
        entropy = .5*(covariance.shape[0]*(1+np.log(2*np.pi))+logdet)
        self.joint_status.setText(
            f"{len(joint['mean'])} nodes | {np.count_nonzero(clamped)} clamped | "
            f"learning {'on' if joint['learning_enabled'] else 'frozen'}, rate {joint['learning_rate']:.4g}, "
            f"{joint['updates']} weight updates | local Gaussian hidden entropy {entropy:.6g} nats. "
            f"Residual energy {joint['energy']:.6g}; {joint['inference_steps']} settling steps, rate {joint['inference_rate']:.4g}. "
            "Weight rows are destinations; columns are sources. Root feedback is -dE/d(canvas), sampled-port adjoint."
        )
        if self.tabs.currentIndex() != 3:
            return
        index = self.joint_tabs.currentIndex()
        table = self.joint_tables[index]
        if index:
            if index <= 4:
                key = ("hidden_covariance", "canvas_weights", "recurrent_weights", "sensory_weights")[index-1]
                values = joint[key]
            elif index == 5:
                values = np.concatenate((joint["sensory_error"], joint["hidden_error"]))[:, None]
            elif index == 6:
                values = joint["canvas_gradient"].reshape(-1, 3)
            elif index == 7:
                values = np.asarray(joint["inference_energy"])[:, None]
            else:
                values = np.concatenate((joint["sensory_bias"], joint["hidden_bias"]))[:, None]
            self._set_matrix(table, values)
            return
        table.setRowCount(len(joint["mean"]))
        cols = joint["visual_shape"][1]
        for node, mean in enumerate(joint["mean"]):
            if node < joint["audio_slice"][0]:
                pixel, channel = divmod(node, 3)
                row, col = divmod(pixel, cols)
                label = f"V({row},{col},{'RGB'[channel]})"
            elif node < joint["hidden_slice"][0]:
                label = f"A[{node-joint['audio_slice'][0]}]"
            else:
                label = f"H[{node-joint['hidden_slice'][0]}]"
            values = (
                label, "yes" if clamped[node] else "free",
                f"{joint['evidence'][node]:.6g}" if clamped[node] else "",
                f"{mean:.6g}", f"{joint['variance'][node]:.6g}",
            )
            for col, text in enumerate(values):
                item = table.item(node, col)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    table.setItem(node, col, item)
                item.setText(text)

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
        self.tabs.setTabEnabled(3, False)
        self.joint_status.clear()
        for table in self.joint_tables:
            table.setRowCount(0)
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
