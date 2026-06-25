from __future__ import annotations

import numpy as np

from audioviz.source_controls import (
    AudioSourceControls,
    CameraFrameSourceControls,
    ControlValue,
    SyntheticFrequencySource,
)
from audioviz.utils.signal_processing import normalize_audio_visual_mapping_mode
from audioviz.visualization.ripple_control_panel import (
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
        if visualizer.processor is not None:
            controls = AudioSourceControls(
                signal_gate_threshold=visualizer.audio_signal_gate_threshold,
                drive_amplitude=visualizer.audio_drive_amplitude,
                minimum_peak_magnitude=float(
                    getattr(
                        visualizer.processor,
                        "minimum_frequency_peak_magnitude",
                        0.1,
                    )
                ),
                peak_prominence_ratio=float(
                    getattr(
                        visualizer.processor,
                        "minimum_frequency_peak_to_median_ratio",
                        5.0,
                    )
                ),
                top_k_count=int(
                    getattr(visualizer.processor, "num_top_frequencies", 3)
                ),
                mapping_mode=visualizer.audio_visual_mapping_mode,
                mapping_alpha=visualizer.audio_visual_mapping_alpha,
                mapping_f0=visualizer.audio_visual_mapping_f0,
                mapping_fc=visualizer.audio_visual_mapping_fc,
                linear_scale=visualizer.audio_visual_linear_scale,
                linear_offset=visualizer.audio_visual_linear_offset,
            ).get_controls()
            sections.append(
                ControlPanelSection(
                    key="audio-source",
                    title="Audio Source",
                    controls=controls,
                )
            )
        sections.append(
            ControlPanelSection(
                key="camera-source",
                title="Camera Source",
                controls=CameraFrameSourceControls(
                    gain=visualizer.camera_source.gain,
                ).get_controls(),
            )
        )
        for index, frequency_hz in enumerate(visualizer.synthetic_frequencies[:, 0]):
            sections.append(
                ControlPanelSection(
                    key=f"synthetic-source-{index}",
                    title=f"Synthetic Source {index + 1}",
                    controls=SyntheticFrequencySource(
                        frequency_hz=float(frequency_hz),
                        n_sources=1,
                    ).get_controls(),
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
                available=visualizer.processor is not None,
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
        self._update_synthetic_control(section_key, control_key, value)

    def update_toggle(self, source_key: str, enabled: bool) -> None:
        visualizer = self.visualizer
        if source_key == "synthetic":
            visualizer.use_synthetic = enabled
            return
        if source_key == "audio":
            if visualizer.processor is None:
                raise RuntimeError("Audio source toggles require an audio processor.")
            visualizer.use_audio_source = enabled
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
        if control_panel is None or visualizer.processor is None:
            return
        gate_open = (
            visualizer.use_audio_source
            and float(getattr(visualizer.processor, "current_signal_level", 0.0))
            >= visualizer.audio_signal_gate_threshold
            and freqs is not None
        )
        detected = tuple(
            float(freq)
            for freq in getattr(visualizer.processor, "current_top_k_frequencies", [])
            if freq is not None and np.isfinite(freq)
        )
        mapped = () if freqs is None else tuple(float(freq) for freq in freqs[0])
        control_panel.set_source_control_value(
            "audio-source",
            "signal_level",
            f"{float(getattr(visualizer.processor, 'current_signal_level', 0.0)):.2f}",
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
        if visualizer.processor is None:
            raise RuntimeError("Audio source controls require an audio processor.")
        if control_key == "signal_gate_threshold":
            visualizer.audio_signal_gate_threshold = float(value)
            visualizer.processor.minimum_signal_level = float(value)
            return
        if control_key == "drive_amplitude":
            visualizer.audio_drive_amplitude = float(value)
            return
        if control_key == "minimum_peak_magnitude":
            visualizer.processor.minimum_frequency_peak_magnitude = float(value)
            return
        if control_key == "peak_prominence_ratio":
            visualizer.processor.minimum_frequency_peak_to_median_ratio = float(value)
            return
        if control_key == "top_k_count":
            visualizer.processor.set_num_top_frequencies(int(round(float(value))))
            return
        if control_key == "mapping_mode":
            visualizer.audio_visual_mapping_mode = normalize_audio_visual_mapping_mode(
                str(value)
            )
            return
        if control_key == "mapping_alpha":
            visualizer.audio_visual_mapping_alpha = float(value)
            return
        if control_key == "mapping_f0":
            visualizer.audio_visual_mapping_f0 = float(value)
            return
        if control_key == "mapping_fc":
            visualizer.audio_visual_mapping_fc = float(value)
            return
        if control_key == "linear_scale":
            visualizer.audio_visual_linear_scale = float(value)
            return
        if control_key == "linear_offset":
            visualizer.audio_visual_linear_offset = float(value)
            return
        if control_key in {
            "signal_level",
            "gate_open",
            "detected_frequencies",
            "mapped_frequencies",
        }:
            return
        raise KeyError(f"Unknown audio source control: {control_key}")

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
        visualizer.synthetic_frequencies[index, 0] = float(value)
        visualizer.frequency = float(visualizer.synthetic_frequencies[0, 0])
