from __future__ import annotations

import numpy as np

from audioviz.source_controls import AudioSourceControls, ControlValue
from audioviz.utils.signal_processing import (
    map_audio_freq_to_visual_freq,
    normalize_audio_visual_mapping_mode,
)


class AudioRippleSource:
    def __init__(
        self,
        *,
        processor=None,
        enabled: bool = False,
        signal_gate_threshold: float = 0.05,
        drive_amplitude: float = 1.0,
        mapping_mode: str = "legacy",
        mapping_alpha: float = 50.0,
        mapping_f0: float = 50.0,
        mapping_fc: float = 2000.0,
        linear_scale: float = 0.05,
        linear_offset: float = 0.0,
    ) -> None:
        self.processor = processor
        self.enabled = processor is not None and bool(enabled)
        self.signal_gate_threshold = float(signal_gate_threshold)
        self.drive_amplitude = float(drive_amplitude)
        self.mapping_mode = normalize_audio_visual_mapping_mode(mapping_mode)
        self.mapping_alpha = float(mapping_alpha)
        self.mapping_f0 = float(mapping_f0)
        self.mapping_fc = float(mapping_fc)
        self.linear_scale = float(linear_scale)
        self.linear_offset = float(linear_offset)
        if self.processor is not None:
            self.processor.minimum_signal_level = self.signal_gate_threshold

    def set_enabled(self, enabled: bool) -> None:
        if enabled and self.processor is None:
            raise RuntimeError("Audio source toggles require an audio processor.")
        self.enabled = bool(enabled)

    def frequencies(self, *, n_sources: int) -> np.ndarray | None:
        if not self.enabled or self.processor is None:
            return None
        if self.signal_level() < self.signal_gate_threshold:
            return None
        top_k = self.detected_frequencies()
        if len(top_k) == 0:
            return None
        visual_frequencies = map_audio_freq_to_visual_freq(
            np.asarray(top_k, dtype=np.float32),
            mode=self.mapping_mode,
            alpha=self.mapping_alpha,
            f0=self.mapping_f0,
            fc=self.mapping_fc,
            linear_scale=self.linear_scale,
            linear_offset=self.linear_offset,
        ).astype(np.float32, copy=False)
        return np.tile(visual_frequencies, (n_sources, 1))

    def excitation_amplitude(
        self,
        *,
        base_amplitude: float,
        frequencies: np.ndarray | None,
        synthetic_enabled: bool,
    ) -> float:
        if (
            frequencies is None
            or self.processor is None
            or not self.enabled
            or synthetic_enabled
        ):
            return float(base_amplitude)
        signal_level = self.signal_level()
        if not np.isfinite(signal_level):
            return float(base_amplitude)
        return float(base_amplitude) * self.drive_amplitude * max(signal_level, 0.0)

    def controls(self):
        if self.processor is None:
            return ()
        return AudioSourceControls(
            signal_gate_threshold=self.signal_gate_threshold,
            drive_amplitude=self.drive_amplitude,
            minimum_peak_magnitude=float(
                getattr(
                    self.processor,
                    "minimum_frequency_peak_magnitude",
                    0.1,
                )
            ),
            peak_prominence_ratio=float(
                getattr(
                    self.processor,
                    "minimum_frequency_peak_to_median_ratio",
                    5.0,
                )
            ),
            top_k_count=int(getattr(self.processor, "num_top_frequencies", 3)),
            mapping_mode=self.mapping_mode,
            mapping_alpha=self.mapping_alpha,
            mapping_f0=self.mapping_f0,
            mapping_fc=self.mapping_fc,
            linear_scale=self.linear_scale,
            linear_offset=self.linear_offset,
        ).get_controls()

    def update_control(self, control_key: str, value: ControlValue) -> None:
        if self.processor is None:
            raise RuntimeError("Audio source controls require an audio processor.")
        if control_key == "signal_gate_threshold":
            self.signal_gate_threshold = float(value)
            self.processor.minimum_signal_level = float(value)
            return
        if control_key == "drive_amplitude":
            self.drive_amplitude = float(value)
            return
        if control_key == "minimum_peak_magnitude":
            self.processor.minimum_frequency_peak_magnitude = float(value)
            return
        if control_key == "peak_prominence_ratio":
            self.processor.minimum_frequency_peak_to_median_ratio = float(value)
            return
        if control_key == "top_k_count":
            self.processor.set_num_top_frequencies(int(round(float(value))))
            return
        if control_key == "mapping_mode":
            self.mapping_mode = normalize_audio_visual_mapping_mode(str(value))
            return
        if control_key == "mapping_alpha":
            self.mapping_alpha = float(value)
            return
        if control_key == "mapping_f0":
            self.mapping_f0 = float(value)
            return
        if control_key == "mapping_fc":
            self.mapping_fc = float(value)
            return
        if control_key == "linear_scale":
            self.linear_scale = float(value)
            return
        if control_key == "linear_offset":
            self.linear_offset = float(value)
            return
        if control_key in {
            "signal_level",
            "gate_open",
            "detected_frequencies",
            "mapped_frequencies",
        }:
            return
        raise KeyError(f"Unknown audio source control: {control_key}")

    def signal_level(self) -> float:
        return float(getattr(self.processor, "current_signal_level", 0.0))

    def detected_frequencies(self) -> tuple[float, ...]:
        if self.processor is None:
            return ()
        return tuple(
            float(freq)
            for freq in getattr(self.processor, "current_top_k_frequencies", [])
            if freq is not None and np.isfinite(freq)
        )

    def gate_open(self, frequencies: np.ndarray | None) -> bool:
        return (
            self.enabled
            and self.signal_level() >= self.signal_gate_threshold
            and frequencies is not None
        )
