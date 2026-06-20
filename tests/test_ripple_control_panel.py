import pytest

from audioviz.engine import RippleEngine
from audioviz.source_controls import SourceControl
from audioviz.visualization.ripple_control_panel import ControlPanelSection


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
    panel.boundary_transmission_slider.setValue(25)
    panel.boundary_dissipation_slider.setValue(60)
    app.processEvents()

    assert engine.damping == 0.5
    assert engine.propagator.damping == 0.5
    assert engine.speed == 20.0
    assert engine.propagator.c == 20.0
    assert engine.amplitude == 2.5
    assert engine.decay_alpha == 4.2
    assert engine.body_boundary_transmission == 0.25
    assert engine.body_boundary_dissipation == 0.6


def test_ripple_control_panel_supports_choice_text_and_auto_floor(ripple_panel_deps):
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
    source_events = []
    auto_floor_values = []
    panel = RippleControlPanel(
        engine,
        auto_color_floor=0.1,
        on_auto_color_floor_changed=auto_floor_values.append,
        source_sections=(
            ControlPanelSection(
                key="audio-source",
                title="Audio Source",
                controls=(
                    SourceControl(
                        key="mapping_mode",
                        label="Mapping Mode",
                        default="legacy",
                        kind="choice",
                        choices=("legacy", "linear"),
                    ),
                    SourceControl(
                        key="signal_level",
                        label="Signal Level",
                        default="0.00",
                        kind="text",
                    ),
                ),
            ),
        ),
        on_source_control_changed=lambda section, key, value: source_events.append(
            (section, key, value)
        ),
    )

    combo = panel.source_control_widgets[("audio-source", "mapping_mode")]
    combo.setCurrentText("linear")
    panel.set_source_control_value("audio-source", "signal_level", "0.42")
    panel.auto_color_floor_slider.setValue(25)
    app.processEvents()

    assert source_events[-1] == ("audio-source", "mapping_mode", "linear")
    assert panel.source_control_widgets[("audio-source", "signal_level")].text() == "0.42"
    assert auto_floor_values[-1] == 0.25
