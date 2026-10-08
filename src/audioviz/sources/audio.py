from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from audioviz.source_controls import AudioSourceControls, ControlValue
from audioviz.source_controls import SourceControl
from audioviz.sources.ripple_grid import frequency_excitation_grid
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS, spectral_observation
from audioviz.utils.source_preview import SourceSceneMapping
from audioviz.utils.signal_processing import (
    map_audio_freq_to_visual_freq,
    normalize_audio_visual_mapping_mode,
)


@dataclass(frozen=True)
class AudioSourceConfig:
    enabled: bool | None = None
    signal_gate_threshold: float = 0.05
    drive_amplitude: float = 1.0
    mapping_mode: str = "legacy"
    mapping_alpha: float = 50.0
    mapping_f0: float = 50.0
    mapping_fc: float = 2000.0
    linear_scale: float = 0.05
    linear_offset: float = 0.0
    observation_gain: float = 1.0


class AudioRippleSource:
    scene_mapping = SourceSceneMapping("audio", "Audio", "spectrum", "sensory")

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
        observation_gain: float = 1.0,
        predictive: bool = False,
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
        self.observation_gain = float(observation_gain)
        if not np.isfinite(self.observation_gain) or self.observation_gain < 0:
            raise ValueError("Audio evidence gain must be finite and non-negative.")
        self.predictive = predictive
        self._last_observation_frame = 0
        self._last_observation_time: float | None = None
        self._latest_observation: np.ndarray | None = None
        if self.processor is not None:
            self.processor.minimum_signal_level = self.signal_gate_threshold

    def set_enabled(self, enabled: bool) -> None:
        if enabled and self.processor is None:
            raise RuntimeError("Audio source toggles require an audio processor.")
        self.enabled = bool(enabled)
        if not enabled:
            self._last_observation_time = None
            self._latest_observation = None

    def observation(self) -> np.ndarray | None:
        if not self.has_new_observation():
            return None
        frame = self.processor.frame_counter
        evidence = spectral_observation(
            self.processor.spectrogram_buffers,
            window_sum=float(np.sum(self.processor.stft_window)),
            mel_power=self.processor.n_mels is not None,
            gain=self.observation_gain,
        )
        self._last_observation_frame = frame
        self._last_observation_time = monotonic()
        self._latest_observation = evidence
        return evidence

    def recent_observation(self) -> np.ndarray | None:
        return self._latest_observation if self.has_recent_observation() else None

    def has_new_observation(self) -> bool:
        return self.enabled and self.processor.frame_counter > self._last_observation_frame

    def has_recent_observation(self) -> bool:
        return (
            self.enabled
            and self._last_observation_time is not None
            and monotonic() - self._last_observation_time < 0.5
        )

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

    def excitation_grid(
        self,
        *,
        t: float,
        n_sources: int,
        frequencies: np.ndarray | None = None,
        source_positions: np.ndarray,
        resolution: tuple[int, int],
        grid_spacing: float,
        speed: float,
        max_frequency: float,
        decay_alpha: float,
    ) -> np.ndarray | None:
        if frequencies is None:
            frequencies = self.frequencies(n_sources=n_sources)
        if frequencies is None:
            return None
        return frequency_excitation_grid(
            t=t,
            frequencies=frequencies,
            source_positions=source_positions,
            resolution=resolution,
            grid_spacing=grid_spacing,
            speed=speed,
            max_frequency=max_frequency,
            decay_alpha=decay_alpha,
        )

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
        if self.predictive:
            return (
                SourceControl("signal_level", "Signal Level", "0.00", kind="text", tooltip=(
                    "Level of the latest input block. Predictive evidence is not suppressed by the signal or peak gates."
                )),
                SourceControl("observation_status", "Spectral Evidence", "waiting", kind="text", tooltip=(
                    "Receiving means a genuine observation arrived within half a second. "
                    "Silent blocks are valid observations; unchanged analysis generations are not inferred again."
                )),
                SourceControl("spectral_bands", "Spectral Bands (mono magnitude)", AUDIO_SPECTRAL_BANDS, kind="text", tooltip=(
                    "Native STFT/mel magnitude bands averaged across input channels. "
                    "The readout pools hidden activity over space, so audio has no assigned canvas position."
                )),
                SourceControl("observation_gain", "Audio Evidence Gain", self.observation_gain,
                              minimum=0.0, maximum=100.0, step=0.1, tooltip=(
                                  "Multiply fixed-scale spectral magnitudes before inference. Increase to expose weak audio "
                                  "evidence; large values can destabilize inference/learning. Zero means silent evidence, not "
                                  "disabled audio. Global pooling can produce a uniform canvas shift rather than local ripples."
                              )),
            )
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
        if control_key == "observation_gain":
            gain = float(value)
            if not np.isfinite(gain) or gain < 0:
                raise ValueError("Audio evidence gain must be finite and non-negative.")
            self.observation_gain = gain
            return
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
