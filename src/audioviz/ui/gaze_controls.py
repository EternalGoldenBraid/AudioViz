from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


class GazeControls(QtWidgets.QWidget):
    """Shared gaze inputs; the owner applies precision and advances the belief."""

    precision_changed = QtCore.pyqtSignal()
    mode_changed = QtCore.pyqtSignal()

    def __init__(self, precision, *, mode="mouse", compact=False, parent=None):
        super().__init__(parent)
        if mode not in ("mouse", "autonomous", "full"):
            raise ValueError("gaze mode must be mouse, autonomous or full")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.coordinates = QtWidgets.QHBoxLayout()
        self.gaze_x = self._coordinate(precision.gaze[0])
        self.gaze_y = self._coordinate(precision.gaze[1])
        for label, control in (("Gaze X", self.gaze_x), ("Gaze Y", self.gaze_y)):
            self.coordinates.addWidget(QtWidgets.QLabel(label))
            self.coordinates.addWidget(control)
        self.observe = QtWidgets.QCheckBox("Observe vision" if compact else "Observe")
        self.observe.setToolTip(
            "Condition visual beliefs on camera frames. Sound has its own Audio source toggle."
        )
        self.observe.setChecked(precision.sample)
        if not compact:
            self.coordinates.addWidget(self.observe)
        self.coordinates.addStretch()
        layout.addLayout(self.coordinates)
        if compact:
            layout.addWidget(self.observe)
        modes = QtWidgets.QVBoxLayout() if compact else QtWidgets.QHBoxLayout()
        modes.addWidget(QtWidgets.QLabel("Gaze control"))
        self.gaze_modes = QtWidgets.QButtonGroup(self)
        self.autonomous_gaze = QtWidgets.QRadioButton("Autonomous gaze")
        self.follow_mouse = QtWidgets.QRadioButton("Mouse gaze")
        self.full_canvas = QtWidgets.QRadioButton("Full canvas (gaze disabled)")
        for button in (self.autonomous_gaze, self.follow_mouse, self.full_canvas):
            self.gaze_modes.addButton(button)
            modes.addWidget(button)
        modes.addStretch()
        layout.addLayout(modes)
        parameters = QtWidgets.QFormLayout()
        self.parameter_form = parameters
        parameters.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.sigma = pg.SpinBox(value=precision.sigma, bounds=(np.nextafter(0., 1.), None), step=.01)
        self.baseline = pg.SpinBox(value=precision.baseline, bounds=(0, None), step=.01)
        self.peak = pg.SpinBox(value=precision.peak, bounds=(0, None), step=.1)
        for label, control, tooltip in (
            ("Width (sigma)", self.sigma, "Gaussian width as a fraction of the shorter image side. Narrow foveae can need denser scoring quadrature."),
            ("Peripheral precision", self.baseline, "Baseline inverse observation variance across the image. Precision is not a probability."),
            ("Foveal increment", self.peak, "Precision added by the Gaussian fovea. At its center, total precision is baseline plus increment."),
        ):
            control.setToolTip(tooltip)
            control.sigValueChanged.connect(lambda *_: self.precision_changed.emit())
            parameters.addRow(label, control)
        layout.addLayout(parameters)
        {"mouse": self.follow_mouse, "autonomous": self.autonomous_gaze, "full": self.full_canvas}[mode].setChecked(True)
        self.follow_mouse.setToolTip("Move over the image, or enter normalized gaze coordinates.")
        self.autonomous_gaze.setToolTip("Choose information gain before capture; hold gaze when Observe is off.")
        self.full_canvas.setToolTip("Observe every pixel with unit precision.")
        self.gaze_x.valueChanged.connect(self.precision_changed)
        self.gaze_y.valueChanged.connect(self.precision_changed)
        self.observe.toggled.connect(self.precision_changed)
        self.gaze_modes.buttonToggled.connect(lambda _button, checked: self._mode_changed() if checked else None)
        self._mode_changed()

    def use_hard_clamps(self):
        for control in (self.baseline, self.peak):
            control.hide()
            self.parameter_form.labelForField(control).hide()
        self.sigma.setToolTip(
            "Clamp the nearest RGB sample and sample centers within one sigma. "
            "Full canvas clamps every compact RGB sample; assigned precision is not used."
        )

    @staticmethod
    def _coordinate(value):
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(0, 1)
        control.setDecimals(6)
        control.setSingleStep(.05)
        control.setValue(value)
        return control

    def _mode_changed(self):
        self.gaze_x.setEnabled(self.follow_mouse.isChecked())
        self.gaze_y.setEnabled(self.follow_mouse.isChecked())
        for control in (self.sigma, self.baseline, self.peak):
            control.setEnabled(not self.full_canvas.isChecked())
        self.mode_changed.emit()

    def gaze_precision(self, profile):
        return replace(
            profile,
            gaze=(self.gaze_x.value(), self.gaze_y.value()) if self.follow_mouse.isChecked() else profile.gaze,
            sample=self.observe.isChecked(),
            sigma=self.sigma.value(), baseline=self.baseline.value(), peak=self.peak.value(),
        )

    def observation_precision(self, profile):
        return replace(profile, baseline=1, peak=0) if self.full_canvas.isChecked() else profile

    def display_gaze(self, gaze):
        with QtCore.QSignalBlocker(self.gaze_x), QtCore.QSignalBlocker(self.gaze_y):
            self.gaze_x.setValue(gaze[0])
            self.gaze_y.setValue(gaze[1])
