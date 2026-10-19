from time import monotonic

from PyQt5 import QtCore, QtWidgets


class FrameRateDisplay(QtWidgets.QWidget):
    """Wall-clock rate of newly presented canvas frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.toggle = QtWidgets.QCheckBox("Show FPS")
        self.toggle.setChecked(True)
        self.readout = QtWidgets.QLabel("FPS: -- | -- ms/frame")
        self.readout.setMinimumWidth(
            self.fontMetrics().horizontalAdvance("FPS: 999.9 | 9999.9 ms/frame")
        )
        self.readout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.setToolTip(
            "New canvas frames presented per wall-clock second, averaged over about one second. "
            "OpenGL counts swaps after a fresh canvas draw; NumPy/CuPy count fresh image viewport paints. "
            "ms/frame is the average presentation interval, not GPU kernel time. "
            "Redraws of unchanged data are not counted."
        )
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle)
        layout.addStretch()
        layout.addWidget(self.readout)
        self._frames = 0
        self._started = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh)
        self.toggle.toggled.connect(self._restart)
        if parent is not None:
            parent.installEventFilter(self)

    def _restart(self):
        self._timer.stop()
        self._frames = 0
        self._started = None
        self.readout.setText("FPS: -- | -- ms/frame")
        self.readout.setVisible(self.toggle.isChecked())
        if self.toggle.isChecked() and self.isVisible() and not self.window().isMinimized():
            self._started = monotonic()
            self._timer.start()

    def record_frame(self):
        if self._started is not None:
            self._frames += 1

    def _refresh(self):
        if self._started is None:
            return
        now = monotonic()
        elapsed = now - self._started
        if elapsed <= 0:
            return
        fps = self._frames / elapsed
        interval = f"{elapsed * 1000 / self._frames:.1f}" if self._frames else "--"
        self.readout.setText(f"FPS: {fps:.1f} | {interval} ms/frame")
        self._frames = 0
        self._started = now

    def showEvent(self, event):
        super().showEvent(event)
        self._restart()

    def hideEvent(self, event):
        self._restart()
        super().hideEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.WindowStateChange:
            self._restart()
        return super().eventFilter(watched, event)
