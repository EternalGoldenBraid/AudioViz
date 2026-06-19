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


def test_ripple_control_panel_renders_per_source_synthetic_frequency_controls(
    ripple_panel_deps,
):
    QtWidgets, RippleControlPanel = ripple_panel_deps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = RippleEngine(
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        use_gpu=False,
    )
    updates = []
    from audioviz.source_controls import SyntheticFrequencySource
    from audioviz.visualization.ripple_control_panel import ControlPanelSection

    panel = RippleControlPanel(
        engine,
        source_sections=(
            ControlPanelSection(
                key="synthetic-source-0",
                title="Synthetic Source 1",
                controls=SyntheticFrequencySource(frequency_hz=110.0, n_sources=1).get_controls(),
            ),
            ControlPanelSection(
                key="synthetic-source-1",
                title="Synthetic Source 2",
                controls=SyntheticFrequencySource(frequency_hz=220.0, n_sources=1).get_controls(),
            ),
        ),
        on_source_control_changed=lambda section, key, value: updates.append(
            (section, key, value)
        ),
    )

    panel.source_control_widgets[("synthetic-source-1", "frequency_hz")].setValue(330.0)
    app.processEvents()

    assert updates[-1] == ("synthetic-source-1", "frequency_hz", 330.0)


def test_ripple_control_panel_explains_when_no_source_controls_exist(ripple_panel_deps):
    QtWidgets, RippleControlPanel = ripple_panel_deps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = RippleEngine(
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        use_gpu=False,
    )

    panel = RippleControlPanel(engine, source_sections=())
    app.processEvents()

    labels = [label.text() for label in panel.findChildren(QtWidgets.QLabel)]
    assert any("No source-specific controls" in text for text in labels)


def test_ripple_control_panel_updates_checkable_source_dropdown(ripple_panel_deps):
    QtWidgets, RippleControlPanel = ripple_panel_deps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = RippleEngine(
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        use_gpu=False,
    )
    from audioviz.visualization.ripple_control_panel import SourceToggle

    updates = []
    panel = RippleControlPanel(
        engine,
        source_toggles=(
            SourceToggle(key="synthetic", label="Synthetic", enabled=True),
            SourceToggle(key="audio", label="Audio", enabled=False),
        ),
        on_source_toggle_changed=lambda key, enabled: updates.append((key, enabled)),
    )
    app.processEvents()

    assert panel.source_toggle_button.text() == "Active Sources: Synthetic"

    panel.source_toggle_actions["audio"].trigger()
    app.processEvents()

    assert updates[-1] == ("audio", True)
    assert panel.source_toggle_button.text() == "Active Sources: Synthetic, Audio"


def test_ripple_control_panel_updates_auto_color_scaling_toggle(ripple_panel_deps):
    QtWidgets, RippleControlPanel = ripple_panel_deps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = RippleEngine(
        resolution=(8, 8),
        plane_size_m=(1.0, 1.0),
        use_gpu=False,
    )
    updates = []
    panel = RippleControlPanel(
        engine,
        auto_color_levels_enabled=True,
        on_auto_color_levels_changed=lambda enabled: updates.append(enabled),
    )
    app.processEvents()

    assert panel.auto_color_levels_checkbox.isChecked() is True

    panel.auto_color_levels_checkbox.setChecked(False)
    app.processEvents()

    assert updates[-1] is False
