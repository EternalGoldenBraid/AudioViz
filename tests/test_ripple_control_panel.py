import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from audioviz.engine import RippleEngine


@pytest.fixture
def ripple_panel_deps():
    qt_widgets = pytest.importorskip("PyQt5.QtWidgets")
    from audioviz.visualization.ripple_control_panel import RippleControlPanel

    return qt_widgets, RippleControlPanel


def test_ripple_control_panel_updates_wave_physics(ripple_panel_deps):
    QtWidgets, RippleControlPanel = ripple_panel_deps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = RippleEngine(
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        speed=10.0,
        damping=0.999,
        amplitude=1.0,
        use_gpu=False,
    )
    panel = RippleControlPanel(engine)

    panel.damping_slider.setValue(500)
    panel.speed_slider.setValue(20)
    panel.amplitude_slider.setValue(250)
    panel.decay_slider.setValue(42)
    app.processEvents()

    assert engine.damping == 0.5
    assert engine.propagator.damping == 0.5
    assert engine.speed == 20.0
    assert engine.propagator.c == 20.0
    assert engine.amplitude == 2.5
    assert engine.decay_alpha == 4.2


def test_ripple_control_panel_amplitude_slider_covers_initial_value(ripple_panel_deps):
    QtWidgets, RippleControlPanel = ripple_panel_deps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = RippleEngine(
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        speed=10.0,
        damping=0.999,
        amplitude=10.0,
        use_gpu=False,
    )
    panel = RippleControlPanel(engine)
    app.processEvents()

    assert panel.amplitude_slider.maximum() >= 1000
    assert panel.amplitude_slider.value() == 1000


def test_ripple_control_panel_resets_field(ripple_panel_deps):
    QtWidgets, RippleControlPanel = ripple_panel_deps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = RippleEngine(
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        speed=10.0,
        damping=0.999,
        amplitude=1.0,
        use_gpu=False,
    )
    panel = RippleControlPanel(engine)
    engine.Z[:] = 1
    engine.propagator.add_excitation(np.ones(engine.resolution, dtype=np.float32))
    engine.time = 3.0

    panel.reset_field()
    app.processEvents()

    assert engine.time == 0.0
    assert np.all(engine.Z == 0)
    assert np.all(engine.propagator.Z == 0)
