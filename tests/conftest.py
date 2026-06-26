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
