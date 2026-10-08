from __future__ import annotations

from audioviz.source_controls import (
    CameraFrameSourceControls,
    ControlValue,
    PredictionLearningControls,
    PredictionOverlayControls,
)
from audioviz.ui.ripple_control_panel import (
    ControlPanelSection,
    RippleControlPanel,
    SourceToggle,
)


def format_frequency_readout(frequencies: tuple[float, ...]) -> str:
    if not frequencies:
        return "—"
    return ", ".join(f"{value:.1f}" for value in frequencies)


class RippleSourceControlBinding:
    def __init__(self, visualizer) -> None:
        self.visualizer = visualizer

    def build_sections(self) -> tuple[ControlPanelSection, ...]:
        visualizer = self.visualizer
        sections: list[ControlPanelSection] = []
        controls = visualizer.audio_source.controls()
        sections.append(
            ControlPanelSection(
                key="audio-source",
                title="Audio Source",
                controls=controls,
                toggle_key="audio",
                empty_message=(
                    "Audio source controls are unavailable without an audio processor."
                ),
            )
        )
        sections.append(
            ControlPanelSection(
                key="camera-source",
                title="Camera Source",
                controls=CameraFrameSourceControls(
                    gain=visualizer.camera_source.gain,
                ).get_controls(),
                toggle_key="camera",
            )
        )
        for index in range(visualizer.n_sources):
            sections.append(
                ControlPanelSection(
                    key=f"synthetic-source-{index}",
                    title=f"Synthetic Source {index + 1}",
                    controls=visualizer.synthetic_source.controls_for_index(index),
                    toggle_key="synthetic",
                )
            )
        sections.append(
            ControlPanelSection(
                key="pose-source",
                title="Pose Graph Source",
                controls=(),
                toggle_key="pose",
                empty_message="No pose-specific controls are available yet.",
            )
        )
        transform_config = visualizer.prediction_error_transform.config
        if transform_config.enabled or visualizer.audio_source.processor is not None or not visualizer.use_shader:
            pathway_config = visualizer.source_orchestrator.visual_readout.config
            sections.append(
                ControlPanelSection(
                    key="learning-dynamics",
                    title="Sensory Inference and Learning",
                    controls=PredictionLearningControls(
                        enabled=pathway_config.learning_enabled,
                        inference_steps=pathway_config.inference_steps,
                        inference_rate=pathway_config.inference_rate,
                        canvas_rate=pathway_config.canvas_rate,
                        learning_rate=pathway_config.learning_rate,
                        weight_decay=pathway_config.weight_decay,
                        weight_clip=pathway_config.weight_clip,
                        gradient_clip=pathway_config.gradient_clip,
                        cross_modal_enabled=pathway_config.cross_modal_enabled,
                        cross_modal_available=(
                            visualizer.source_orchestrator.hidden_coupling is not None and not visualizer.use_shader
                        ),
                    ).get_controls(),
                    expanded=True,
                )
            )
            if not visualizer.use_shader:
                sections.append(
                    ControlPanelSection(
                        key="learning-overlay",
                        title="Substrate Conductances (Fixed)",
                        controls=PredictionOverlayControls(
                            enabled=visualizer.show_learning_overlay,
                            stride=visualizer.learning_overlay_stride,
                            threshold=visualizer.learning_overlay_threshold,
                            scale=visualizer.learning_overlay_scale,
                        ).get_controls(),
                        expanded=visualizer.show_learning_overlay,
                    )
                )
        return tuple(sections)

    def build_toggles(self) -> tuple[SourceToggle, ...]:
        visualizer = self.visualizer
        return (
            SourceToggle(
                key="synthetic",
                label="Synthetic",
                enabled=visualizer.use_synthetic,
                available=True,
            ),
            SourceToggle(
                key="audio",
                label="Audio",
                enabled=visualizer.use_audio_source,
                available=visualizer.audio_source.processor is not None and not visualizer.use_shader,
            ),
            SourceToggle(
                key="camera",
                label="Camera Frame",
                enabled=visualizer.use_camera_source,
                available=not visualizer.use_shader,
            ),
            SourceToggle(
                key="pose",
                label="Pose Graph",
                enabled=visualizer.use_pose_sources,
                available=not (visualizer.use_gpu or visualizer.use_shader),
            ),
        )

    def update_control(
        self,
        section_key: str,
        control_key: str,
        value: ControlValue,
    ) -> None:
        if section_key == "audio-source":
            self._update_audio_control(control_key, value)
            return
        if section_key == "camera-source":
            self._update_camera_control(control_key, value)
            return
        if section_key == "learning-dynamics":
            self._update_learning_control(control_key, value)
            return
        if section_key == "learning-overlay":
            self._update_learning_overlay_control(control_key, value)
            return
        self._update_synthetic_control(section_key, control_key, value)

    def update_toggle(self, source_key: str, enabled: bool) -> None:
        visualizer = self.visualizer
        if source_key == "synthetic":
            visualizer.use_synthetic = enabled
            return
        if source_key == "audio":
            if enabled and visualizer.use_shader:
                raise NotImplementedError("Audio predictive inference requires the CPU/GPU ripple backend.")
            visualizer.audio_source.set_enabled(enabled)
            return
        if source_key == "camera":
            if enabled and visualizer.use_shader:
                raise NotImplementedError(
                    "Camera-frame source currently requires the CPU/GPU ripple backend."
                )
            visualizer.camera_source.set_enabled(enabled)
            return
        if source_key == "pose":
            visualizer._set_pose_sources_enabled(enabled)
            return
        raise KeyError(f"Unknown source toggle: {source_key}")

    def sync_audio_panel(
        self,
        control_panel: RippleControlPanel | None,
        freqs: np.ndarray | None,
    ) -> None:
        visualizer = self.visualizer
        audio_source = visualizer.audio_source
        if control_panel is None or audio_source.processor is None:
            return
        if audio_source.predictive:
            control_panel.set_source_control_value(
                "audio-source", "signal_level", f"{audio_source.signal_level():.2f}"
            )
            control_panel.set_source_control_value(
                "audio-source", "observation_status",
                "receiving" if audio_source.has_recent_observation() else "waiting",
            )
            return
        gate_open = audio_source.gate_open(freqs)
        detected = audio_source.detected_frequencies()
        mapped = () if freqs is None else tuple(float(freq) for freq in freqs[0])
        control_panel.set_source_control_value(
            "audio-source",
            "signal_level",
            f"{audio_source.signal_level():.2f}",
        )
        control_panel.set_source_control_value(
            "audio-source",
            "gate_open",
            "open" if gate_open else "closed",
        )
        control_panel.set_source_control_value(
            "audio-source",
            "detected_frequencies",
            format_frequency_readout(detected),
        )
        control_panel.set_source_control_value(
            "audio-source",
            "mapped_frequencies",
            format_frequency_readout(mapped),
        )

    def _update_audio_control(
        self,
        control_key: str,
        value: ControlValue,
    ) -> None:
        visualizer = self.visualizer
        visualizer.audio_source.update_control(control_key, value)

    def _update_camera_control(
        self,
        control_key: str,
        value: ControlValue,
    ) -> None:
        if control_key == "gain":
            self.visualizer.camera_source.set_gain(float(value))
            return
        raise KeyError(f"Unknown camera source control: {control_key}")

    def _update_synthetic_control(
        self,
        section_key: str,
        control_key: str,
        value: ControlValue,
    ) -> None:
        visualizer = self.visualizer
        if control_key != "frequency_hz" or not section_key.startswith(
            "synthetic-source-"
        ):
            raise KeyError(f"Unknown source control: {section_key}.{control_key}")
        index = int(section_key.removeprefix("synthetic-source-"))
        visualizer.synthetic_source.set_frequency(index, float(value))

    def _update_learning_control(
        self,
        control_key: str,
        value: ControlValue,
    ) -> None:
        readout = self.visualizer.source_orchestrator
        if control_key == "inference_steps":
            readout.update_pathway_config(inference_steps=int(value))
            return
        if control_key in ("inference_rate", "canvas_rate"):
            readout.update_pathway_config(**{control_key: float(value)})
            return
        if control_key in ("learning_enabled", "cross_modal_enabled"):
            readout.update_pathway_config(**{control_key: bool(value)})
            return
        if control_key == "learning_rate":
            readout.update_pathway_config(learning_rate=float(value))
            return
        if control_key == "learning_weight_decay":
            readout.update_pathway_config(weight_decay=float(value))
            return
        if control_key == "learning_weight_clip":
            readout.update_pathway_config(weight_clip=float(value))
            return
        if control_key == "learning_gradient_clip":
            readout.update_pathway_config(gradient_clip=float(value))
            return
        raise KeyError(f"Unknown visual pathway control: {control_key}")

    def _update_learning_overlay_control(
        self,
        control_key: str,
        value: ControlValue,
    ) -> None:
        visualizer = self.visualizer
        if control_key == "show_learning_overlay":
            visualizer.set_learning_overlay_enabled(bool(value))
            return
        if control_key == "overlay_stride":
            visualizer.set_learning_overlay_stride(int(round(float(value))))
            return
        if control_key == "overlay_threshold":
            visualizer.set_learning_overlay_threshold(float(value))
            return
        if control_key == "overlay_scale":
            visualizer.set_learning_overlay_scale(float(value))
            return
        raise KeyError(f"Unknown learning overlay control: {control_key}")
