import pytest

from audioviz.engine import RippleEngine
from audioviz.source_controls import PredictionLearningControls, SourceControl
from audioviz.ui.ripple_control_panel import ControlPanelSection, SourceToggle


@pytest.fixture
def ripple_panel_deps():
    qt_widgets = pytest.importorskip("PyQt5.QtWidgets")
    from audioviz.ui.ripple_control_panel import RippleControlPanel

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
    panel.decay_slider.setValue(420)
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
    assert panel.section_widgets["wave-physics"].is_expanded()


def test_inference_settings_explain_behavior_on_both_labels_and_editors(qapp):
    from types import SimpleNamespace
    from PyQt5 import QtWidgets
    from audioviz.sources.audio import AudioRippleSource
    from audioviz.ui.ripple_control_panel import RippleControlPanel

    inference = PredictionLearningControls().get_controls()
    audio = AudioRippleSource(processor=SimpleNamespace(), predictive=True).controls()
    panel = RippleControlPanel(
        RippleEngine(resolution=(8, 8), plane_size_m=(1, 1), speed=1),
        source_sections=(
            ControlPanelSection("inference", "Sensory Inference", inference),
            ControlPanelSection("audio", "Audio Evidence", audio),
        ),
    )
    for section, controls in (("inference", inference), ("audio", audio)):
        form = panel.section_widgets[section].content_layout.itemAt(0).layout()
        for control in controls:
            widget = panel.source_control_widgets[(section, control.key)]
            assert control.tooltip
            expected = panel._tooltip_text(control.tooltip)
            assert widget.toolTip() == expected
            assert form.labelForField(widget).toolTip() == expected
            for child in widget.findChildren(QtWidgets.QWidget):
                assert child.toolTip() == expected
    canvas = panel.source_control_widgets[("inference", "canvas_rate")]
    assert "not a blend percentage" in canvas.toolTip()
    assert "Zero disables sensory canvas correction" in canvas.toolTip()
    assert "Off freezes weights, not hidden inference" in (
        panel.source_control_widgets[("inference", "learning_enabled")].toolTip()
    )
    assert "uniform canvas shift" in panel.source_control_widgets[("audio", "observation_gain")].toolTip()
    assert panel.amplitude_slider.toolTip() == panel.amplitude_label.toolTip()
    assert "not audio evidence" in panel.amplitude_slider.toolTip()
    assert "negative values clip to black" in panel.rgb_canvas_checkbox.toolTip()
    panel.close()


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
    auto_threshold_values = []
    rgb_canvas_values = []
    panel = RippleControlPanel(
        engine,
        on_rgb_canvas_changed=rgb_canvas_values.append,
        auto_color_activation_threshold=0.1,
        on_auto_color_activation_threshold_changed=auto_threshold_values.append,
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

    choice = panel.source_control_widgets[("audio-source", "mapping_mode")]
    buttons = choice.findChildren(QtWidgets.QRadioButton)
    assert [button.text() for button in buttons] == ["legacy", "linear"]
    buttons[1].click()
    panel.set_source_control_value("audio-source", "signal_level", "0.42")
    panel.auto_color_threshold_slider.setValue(30)
    panel.auto_color_floor_slider.setValue(25)
    panel.rgb_canvas_checkbox.click()
    app.processEvents()

    assert source_events[-1] == ("audio-source", "mapping_mode", "linear")
    assert panel.source_control_widgets[("audio-source", "signal_level")].text() == "0.42"
    assert auto_threshold_values[-1] == 0.3
    assert auto_floor_values[-1] == 0.25
    assert rgb_canvas_values == [True]


def test_ripple_control_panel_supports_toggle_controls_and_section_expansion(
    ripple_panel_deps,
):
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
    panel = RippleControlPanel(
        engine,
        source_sections=(
            ControlPanelSection(
                key="learning-dynamics",
                title="Learning Dynamics",
                controls=(
                    SourceControl(
                        key="learning_enabled",
                        label="Learning Enabled",
                        default=False,
                        kind="toggle",
                    ),
                ),
                expanded=False,
            ),
        ),
        on_source_control_changed=lambda section, key, value: source_events.append(
            (section, key, value)
        ),
    )

    section = panel.section_widgets["learning-dynamics"]
    checkbox = panel.source_control_widgets[("learning-dynamics", "learning_enabled")]

    assert not section.is_expanded()
    checkbox.click()
    app.processEvents()

    assert section.is_expanded()
    assert source_events[-1] == ("learning-dynamics", "learning_enabled", True)


def test_ripple_control_panel_expands_learning_overlay_section_on_toggle(
    ripple_panel_deps,
):
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
    panel = RippleControlPanel(
        engine,
        source_sections=(
            ControlPanelSection(
                key="learning-overlay",
                title="Learning Overlay",
                controls=(
                    SourceControl(
                        key="show_learning_overlay",
                        label="Show Overlay",
                        default=False,
                        kind="toggle",
                    ),
                ),
                expanded=False,
            ),
        ),
        on_source_control_changed=lambda section, key, value: source_events.append(
            (section, key, value)
        ),
    )

    section = panel.section_widgets["learning-overlay"]
    checkbox = panel.source_control_widgets[("learning-overlay", "show_learning_overlay")]

    assert not section.is_expanded()
    checkbox.click()
    app.processEvents()

    assert section.is_expanded()
    assert source_events[-1] == ("learning-overlay", "show_learning_overlay", True)


def test_ripple_control_panel_expands_active_source_sections_by_default(
    ripple_panel_deps,
):
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
    toggle_events = []
    panel = RippleControlPanel(
        engine,
        source_toggles=(
            SourceToggle("audio", "Audio", enabled=False, available=True),
            SourceToggle("pose", "Pose Graph", enabled=True, available=False),
        ),
        source_sections=(
            ControlPanelSection(
                key="audio-source",
                title="Audio Source",
                controls=(),
                toggle_key="audio",
            ),
            ControlPanelSection(
                key="pose-source",
                title="Pose Graph Source",
                controls=(),
                toggle_key="pose",
            ),
        ),
        on_source_toggle_changed=lambda key, enabled: toggle_events.append(
            (key, enabled)
        ),
    )

    audio_toggle = panel.source_toggle_checkboxes["audio"]
    pose_toggle = panel.source_toggle_checkboxes["pose"]
    audio_section = panel.section_widgets["audio-source"]
    pose_section = panel.section_widgets["pose-source"]

    assert not audio_section.is_expanded()
    assert pose_section.is_expanded()

    audio_toggle.click()
    app.processEvents()

    assert audio_toggle.isChecked()
    assert not pose_toggle.isEnabled()
    assert audio_section.is_expanded()
    assert toggle_events[-1] == ("audio", True)


def test_ripple_control_panel_reverts_source_toggle_when_callback_fails(
    ripple_panel_deps,
):
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

    def fail_toggle(_key, _enabled):
        raise RuntimeError("camera unavailable")

    panel = RippleControlPanel(
        engine,
        source_toggles=(SourceToggle("camera", "Camera Frame", enabled=False),),
        source_sections=(
            ControlPanelSection(
                key="camera-source",
                title="Camera Source",
                controls=(),
                toggle_key="camera",
            ),
        ),
        on_source_toggle_changed=fail_toggle,
    )

    camera_toggle = panel.source_toggle_checkboxes["camera"]
    camera_section = panel.section_widgets["camera-source"]
    camera_toggle.click()
    app.processEvents()

    assert not camera_toggle.isChecked()
    assert not camera_section.is_expanded()
