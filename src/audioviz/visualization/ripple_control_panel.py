from typing import Callable, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSlider

from audioviz.engine import RippleEngine


class RippleControlPanel(QtWidgets.QWidget):
    def __init__(
        self,
        engine: RippleEngine,
        *,
        on_speed_changed: Optional[Callable[[float], None]] = None,
        on_amplitude_changed: Optional[Callable[[float], None]] = None,
        on_decay_alpha_changed: Optional[Callable[[float], None]] = None,
        on_damping_changed: Optional[Callable[[float], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.engine = engine
        self.on_speed_changed = on_speed_changed
        self.on_amplitude_changed = on_amplitude_changed
        self.on_decay_alpha_changed = on_decay_alpha_changed
        self.on_damping_changed = on_damping_changed
        self.on_reset = on_reset

        self.setWindowTitle("Ripple Controls")
        layout = QtWidgets.QVBoxLayout(self)

        info_label = QLabel("Wave physics controls for the ripple field.")
        layout.addWidget(info_label)

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
            tooltip="Controls the strength of the input excitation added to the wave field.",
            value_label=f"Amplitude: {self.engine.amplitude:.2f}",
            minimum=0,
            maximum=500,
            value=int(self.engine.amplitude * 100),
            on_change=lambda raw: self.update_amplitude(raw / 100.0),
        )

        layout.addWidget(group)

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
