import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

collect_ignore = [
    "test_input_stream.py",
    "test_pyqt5.py",
    "test_read_live_audio.py",
    "test_ripple_engine_performance.py",
]


@pytest.fixture
def qapp():
    qt_widgets = pytest.importorskip("PyQt5.QtWidgets")
    return qt_widgets.QApplication.instance() or qt_widgets.QApplication([])


@pytest.fixture
def layer_view_stub(qapp, monkeypatch):
    from PyQt5 import QtCore, QtWidgets
    from audioviz.visualization.prediction_diagnostics_view import PredictionDiagnosticsView

    class LayerViewStub(QtWidgets.QWidget):
        rendering_failed = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.preview = None
            self.updates = 0
            self.clears = 0
            self.resets = 0

        def set_preview(self, canvas, snapshot):
            self.preview = canvas, snapshot
            self.updates += 1

        def clear_preview(self):
            self.preview = None
            self.clears += 1

        def reset_view(self):
            self.resets += 1

    monkeypatch.setattr(
        PredictionDiagnosticsView, "_create_layer_view",
        lambda self: LayerViewStub(self),
    )
    return LayerViewStub
