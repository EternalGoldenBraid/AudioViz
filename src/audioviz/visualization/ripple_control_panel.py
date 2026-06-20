from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMenu,
    QScrollArea,
    QSlider,
    QToolButton,
)

from audioviz.engine import RippleEngine
from audioviz.source_controls import ControlValue, SourceControl


@dataclass(frozen=True)
class ControlPanelSection:
    key: str
    title: str
    controls: Sequence[SourceControl]


@dataclass(frozen=True)
class SourceToggle:
    key: str
    label: str
    enabled: bool
    available: bool = True


class RippleControlPanel(QtWidgets.QWidget):
    def __init__(
        self,
        engine: RippleEngine,
        *,
        on_speed_changed: Optional[Callable[[float], None]] = None,
        on_amplitude_changed: Optional[Callable[[float], None]] = None,
        on_decay_alpha_changed: Optional[Callable[[float], None]] = None,
        on_damping_changed: Optional[Callable[[float], None]] = None,
        on_boundary_transmission_changed: Optional[Callable[[float], None]] = None,
        on_boundary_dissipation_changed: Optional[Callable[[float], None]] = None,
        auto_color_levels_enabled: bool = True,
        on_auto_color_levels_changed: Optional[Callable[[bool], None]] = None,
        auto_color_activation_threshold: float = 0.1,
        on_auto_color_activation_threshold_changed: Optional[Callable[[float], None]] = None,
        auto_color_floor: float = 0.1,
        on_auto_color_floor_changed: Optional[Callable[[float], None]] = None,
        source_toggles: Sequence[SourceToggle] = (),
        source_sections: Sequence[ControlPanelSection] = (),
        on_source_control_changed: Optional[Callable[[str, str, ControlValue], None]] = None,
        on_source_toggle_changed: Optional[Callable[[str, bool], None]] = None,
        before_reset: Optional[Callable[[], bool]] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.engine = engine
        self.on_speed_changed = on_speed_changed
        self.on_amplitude_changed = on_amplitude_changed
        self.on_decay_alpha_changed = on_decay_alpha_changed
        self.on_damping_changed = on_damping_changed
        self.on_boundary_transmission_changed = on_boundary_transmission_changed
        self.on_boundary_dissipation_changed = on_boundary_dissipation_changed
        self.on_auto_color_levels_changed = on_auto_color_levels_changed
        self.on_auto_color_activation_threshold_changed = (
            on_auto_color_activation_threshold_changed
        )
        self.on_auto_color_floor_changed = on_auto_color_floor_changed
        self.source_toggles = tuple(source_toggles)
        self.source_sections = tuple(source_sections)
        self.on_source_control_changed = on_source_control_changed
        self.on_source_toggle_changed = on_source_toggle_changed
        self.before_reset = before_reset
        self.on_reset = on_reset
        self.source_control_widgets: dict[tuple[str, str], QtWidgets.QWidget] = {}
        self.source_toggle_actions: dict[str, QAction] = {}

        self.setWindowTitle("Ripple Controls")
        layout = QtWidgets.QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        scroll_area.setWidget(content)
        layout.addWidget(scroll_area)

        info_label = QLabel("Wave physics controls for the ripple field.")
        content_layout.addWidget(info_label)

        group = QtWidgets.QGroupBox("Wave Physics")
        group_layout = QtWidgets.QVBoxLayout(group)

        reset_button = QtWidgets.QPushButton("Reset Field")
        reset_button.clicked.connect(self.reset_field)
        group_layout.addWidget(reset_button)

        self.decay_label, self.decay_slider = self._add_slider(
            group_layout,
            title="Excitation Falloff (α)",
            tooltip=(
                "Controls how quickly the excitation decays away from the source point. "
                "Higher α = more localized excitation; lower α = more spread out excitation."
            ),
            value_label=f"Decay α: {self.engine.decay_alpha:.1f}",
            minimum=0,
            maximum=1000,
            value=int(self.engine.decay_alpha * 10),
            on_change=lambda raw: self.update_decay_alpha(raw / 10.0),
        )

        self.damping_label, self.damping_slider = self._add_slider(
            group_layout,
            title="Wave Damping",
            tooltip="Controls how quickly the wave loses energy as it propagates. 1.0 = no damping.",
            value_label=f"Damping: {self.engine.damping:.3f}",
            minimum=0,
            maximum=1000,
            value=int(self.engine.damping * 1000),
            on_change=lambda raw: self.update_damping(raw / 1000.0),
        )

        self.speed_label, self.speed_slider = self._add_slider(
            group_layout,
            title="Wave Speed (m/s)",
            tooltip="Controls how fast the wave propagates across the surface.",
            value_label=f"Speed: {self.engine.speed:.1f}",
            minimum=1,
            maximum=1000,
            value=int(self.engine.speed),
            on_change=lambda raw: self.update_speed(float(raw)),
        )

        self.amplitude_label, self.amplitude_slider = self._add_slider(
            group_layout,
            title="Excitation Amplitude",
            tooltip=(
                "Controls the strength of excitation added to the wave field. "
                "Set to 0 to mute audio-driven ripples."
            ),
            value_label=f"Amplitude: {self.engine.amplitude:.2f}",
            minimum=0,
            maximum=max(1000, int(self.engine.amplitude * 100)),
            value=int(self.engine.amplitude * 100),
            on_change=lambda raw: self.update_amplitude(raw / 100.0),
        )

        self.boundary_transmission_label, self.boundary_transmission_slider = self._add_slider(
            group_layout,
            title="Boundary Transmission",
            tooltip=(
                "Controls how much wave coupling is allowed across segmentation "
                "boundary crossings. Lower values are more reflective."
            ),
            value_label=(
                f"Boundary Transmission: {self.engine.body_boundary_transmission:.2f}"
            ),
            minimum=0,
            maximum=100,
            value=int(round(self.engine.body_boundary_transmission * 100)),
            on_change=lambda raw: self.update_boundary_transmission(raw / 100.0),
        )

        self.boundary_dissipation_label, self.boundary_dissipation_slider = self._add_slider(
            group_layout,
            title="Masked Interior Dissipation",
            tooltip=(
                "Controls how strongly the full masked segmentation interior is "
                "damped during the state update."
            ),
            value_label=(
                "Masked Interior Dissipation: "
                f"{self.engine.body_boundary_dissipation:.2f}"
            ),
            minimum=0,
            maximum=100,
            value=int(round(self.engine.body_boundary_dissipation * 100)),
            on_change=lambda raw: self.update_boundary_dissipation(raw / 100.0),
        )

        self.auto_color_levels_checkbox = QCheckBox(
            "Auto Color Scaling (98th percentile)"
        )
        self.auto_color_levels_checkbox.setToolTip(
            "Continuously remap the ripple colormap to the current 98th percentile "
            "of active field magnitudes. Disable it to keep manual or fixed levels."
        )
        self.auto_color_levels_checkbox.setChecked(bool(auto_color_levels_enabled))
        self.auto_color_levels_checkbox.toggled.connect(self.update_auto_color_levels)
        group_layout.addWidget(self.auto_color_levels_checkbox)

        (
            self.auto_color_threshold_label,
            self.auto_color_threshold_slider,
        ) = self._add_slider(
            group_layout,
            title="Auto Scale Activation Threshold",
            tooltip=(
                "Adaptive color scaling only activates once the current field energy "
                "exceeds this threshold. Below it, the display stays on the fixed "
                "low-level reference scale."
            ),
            value_label=(
                "Auto Scale Threshold: "
                f"{auto_color_activation_threshold:.2f}"
            ),
            minimum=0,
            maximum=200,
            value=int(round(auto_color_activation_threshold * 100)),
            on_change=lambda raw: self.update_auto_color_activation_threshold(raw / 100.0),
        )

        self.auto_color_floor_label, self.auto_color_floor_slider = self._add_slider(
            group_layout,
            title="Low-Level Reference Scale",
            tooltip=(
                "Fixed display scale used while the field stays below the auto-scale "
                "activation threshold. Lower values make subtle fields more visible; "
                "higher values keep weak fields visually subtle."
            ),
            value_label=f"Reference Scale: {auto_color_floor:.2f}",
            minimum=0,
            maximum=200,
            value=int(round(auto_color_floor * 100)),
            on_change=lambda raw: self.update_auto_color_floor(raw / 100.0),
        )

        content_layout.addWidget(group)

        source_controls_group = QtWidgets.QGroupBox("Source Controls")
        source_controls_layout = QtWidgets.QVBoxLayout(source_controls_group)
        if self.source_toggles:
            self.source_toggle_button = QToolButton()
            self.source_toggle_button.setPopupMode(QToolButton.InstantPopup)
            self.source_toggle_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.source_toggle_menu = QMenu(self.source_toggle_button)
            self.source_toggle_button.setMenu(self.source_toggle_menu)
            self._populate_source_toggle_menu()
            source_controls_layout.addWidget(self.source_toggle_button)
        if self.source_sections:
            for section in self.source_sections:
                source_group = QtWidgets.QGroupBox(section.title)
                source_group_layout = QtWidgets.QFormLayout(source_group)
                for control in section.controls:
                    widget = self._create_source_control_widget(section.key, control)
                    source_group_layout.addRow(control.label, widget)
                    self.source_control_widgets[(section.key, control.key)] = widget
                source_controls_layout.addWidget(source_group)
        else:
            source_controls_layout.addWidget(
                QLabel(
                    "No source-specific controls are available for the current visualizer."
                )
            )
        content_layout.addWidget(source_controls_group)

        content_layout.addStretch(1)

    def _add_slider(
        self,
        layout: QtWidgets.QVBoxLayout,
        *,
        title: str,
        tooltip: str,
        value_label: str,
        minimum: int,
        maximum: int,
        value: int,
        on_change: Callable[[int], None],
    ) -> tuple[QLabel, QSlider]:
        title_label = QLabel(title)
        title_label.setToolTip(tooltip)
        label = QLabel(value_label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.valueChanged.connect(on_change)

        layout.addWidget(title_label)
        layout.addWidget(label)
        layout.addWidget(slider)
        return label, slider

    def reset_field(self) -> None:
        if self.before_reset is not None and not self.before_reset():
            return
        self.engine.reset()
        if self.on_reset is not None:
            self.on_reset()

    def update_speed(self, val: float) -> None:
        self.engine.set_speed(val)
        self.speed_label.setText(f"Speed: {val:.1f}")
        if self.on_speed_changed is not None:
            self.on_speed_changed(val)

    def update_amplitude(self, val: float) -> None:
        self.engine.amplitude = val
        self.amplitude_label.setText(f"Amplitude: {val:.2f}")
        if self.on_amplitude_changed is not None:
            self.on_amplitude_changed(val)

    def update_decay_alpha(self, val: float) -> None:
        self.engine.decay_alpha = val
        self.decay_label.setText(f"Decay α: {val:.1f}")
        if self.on_decay_alpha_changed is not None:
            self.on_decay_alpha_changed(val)

    def update_damping(self, val: float) -> None:
        self.engine.set_damping(val)
        self.damping_label.setText(f"Damping: {val:.3f}")
        if self.on_damping_changed is not None:
            self.on_damping_changed(val)

    def update_auto_color_levels(self, enabled: bool) -> None:
        if self.on_auto_color_levels_changed is not None:
            self.on_auto_color_levels_changed(enabled)

    def update_auto_color_activation_threshold(self, val: float) -> None:
        self.auto_color_threshold_label.setText(f"Auto Scale Threshold: {val:.2f}")
        if self.on_auto_color_activation_threshold_changed is not None:
            self.on_auto_color_activation_threshold_changed(val)

    def update_auto_color_floor(self, val: float) -> None:
        self.auto_color_floor_label.setText(f"Reference Scale: {val:.2f}")
        if self.on_auto_color_floor_changed is not None:
            self.on_auto_color_floor_changed(val)

    def update_boundary_transmission(self, val: float) -> None:
        self.engine.set_body_boundary_transmission(val)
        self.boundary_transmission_label.setText(f"Boundary Transmission: {val:.2f}")
        if self.on_boundary_transmission_changed is not None:
            self.on_boundary_transmission_changed(val)

    def update_boundary_dissipation(self, val: float) -> None:
        self.engine.set_body_boundary_dissipation(val)
        self.boundary_dissipation_label.setText(
            f"Masked Interior Dissipation: {val:.2f}"
        )
        if self.on_boundary_dissipation_changed is not None:
            self.on_boundary_dissipation_changed(val)

    def _create_source_control_widget(
        self,
        section_key: str,
        control: SourceControl,
    ) -> QtWidgets.QWidget:
        if control.kind == "number":
            return self._create_number_control(section_key, control)
        if control.kind == "choice":
            return self._create_choice_control(section_key, control)
        if control.kind == "text":
            return self._create_text_control(control)
        raise NotImplementedError(f"Unsupported source control kind: {control.kind!r}")

    def _create_number_control(
        self,
        section_key: str,
        control: SourceControl,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setDecimals(self._decimals_for_step(control.step))
        widget.setSingleStep(float(control.step) if control.step is not None else 1.0)
        widget.setMinimum(float(control.minimum) if control.minimum is not None else -1e12)
        widget.setMaximum(float(control.maximum) if control.maximum is not None else 1e12)
        widget.setValue(float(control.default))
        if control.unit:
            widget.setSuffix(f" {control.unit}")
        widget.valueChanged.connect(
            lambda value, section_key=section_key, control_key=control.key: self._emit_source_control_change(
                section_key, control_key, float(value)
            )
        )
        return widget

    def _create_choice_control(
        self,
        section_key: str,
        control: SourceControl,
    ) -> QComboBox:
        widget = QComboBox()
        for choice in control.choices:
            widget.addItem(str(choice), choice)
        default = str(control.default)
        index = widget.findText(default)
        if index >= 0:
            widget.setCurrentIndex(index)
        widget.currentTextChanged.connect(
            lambda value, section_key=section_key, control_key=control.key: self._emit_source_control_change(
                section_key, control_key, str(value)
            )
        )
        return widget

    @staticmethod
    def _create_text_control(control: SourceControl) -> QLabel:
        widget = QLabel(str(control.default))
        widget.setWordWrap(True)
        return widget

    @staticmethod
    def _decimals_for_step(step: float | None) -> int:
        if step is None or step >= 1.0:
            return 0
        text = f"{step:.8f}".rstrip("0")
        if "." not in text:
            return 0
        return len(text.split(".", 1)[1])

    def _emit_source_control_change(
        self,
        section_key: str,
        control_key: str,
        value: ControlValue,
    ) -> None:
        if self.on_source_control_changed is not None:
            self.on_source_control_changed(section_key, control_key, value)

    def set_source_control_value(
        self,
        section_key: str,
        control_key: str,
        value: ControlValue,
    ) -> None:
        widget = self.source_control_widgets.get((section_key, control_key))
        if widget is None:
            raise KeyError(f"Unknown source control widget: {section_key}.{control_key}")
        if isinstance(widget, QLabel):
            widget.setText(str(value))
            return
        if isinstance(widget, QComboBox):
            index = widget.findText(str(value))
            if index >= 0 and index != widget.currentIndex():
                widget.blockSignals(True)
                widget.setCurrentIndex(index)
                widget.blockSignals(False)
            return
        if isinstance(widget, QDoubleSpinBox):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
            return
        raise TypeError(f"Unsupported source control widget type: {type(widget)!r}")

    def _populate_source_toggle_menu(self) -> None:
        self.source_toggle_menu.clear()
        self.source_toggle_actions.clear()
        for toggle in self.source_toggles:
            action = self.source_toggle_menu.addAction(toggle.label)
            action.setCheckable(True)
            action.setChecked(toggle.enabled)
            action.setEnabled(toggle.available)
            action.toggled.connect(
                lambda checked, key=toggle.key: self._emit_source_toggle_change(key, checked)
            )
            self.source_toggle_actions[toggle.key] = action
        self._update_source_toggle_button_text()

    def _update_source_toggle_button_text(self) -> None:
        enabled_labels = [
            action.text()
            for action in self.source_toggle_actions.values()
            if action.isEnabled() and action.isChecked()
        ]
        label = ", ".join(enabled_labels) if enabled_labels else "None"
        self.source_toggle_button.setText(f"Active Sources: {label}")

    def _emit_source_toggle_change(self, source_key: str, enabled: bool) -> None:
        self._update_source_toggle_button_text()
        if self.on_source_toggle_changed is not None:
            self.on_source_toggle_changed(source_key, enabled)
