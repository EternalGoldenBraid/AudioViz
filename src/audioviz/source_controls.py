from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np


ControlKind = Literal["number", "toggle", "choice", "text"]
ControlValue = bool | float | int | str


@dataclass(frozen=True)
class SourceControl:
    key: str
    label: str
    default: ControlValue
    kind: ControlKind = "number"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str | None = None
    choices: tuple[ControlValue, ...] = ()
    tooltip: str = ""


class SourceControls(Protocol):
    def get_controls(self) -> Sequence[SourceControl]:
        """Return metadata for source-specific controls."""
        ...


class SourceControlProvider:
    def get_controls(self) -> Sequence[SourceControl]:
        return ()


@dataclass
class SyntheticFrequencySource(SourceControlProvider):
    frequency_hz: float = 1.0
    n_sources: int = 1

    def get_controls(self) -> Sequence[SourceControl]:
        return (
            SourceControl(
                key="frequency_hz",
                label="Frequency",
                default=self.frequency_hz,
                minimum=0.1,
                maximum=20_000.0,
                step=0.1,
                unit="Hz",
            ),
        )

    def frequencies(self) -> np.ndarray:
        return np.full((self.n_sources, 1), self.frequency_hz, dtype=np.float32)


@dataclass
class AudioSourceControls(SourceControlProvider):
    signal_gate_threshold: float = 0.05
    drive_amplitude: float = 1.0
    minimum_peak_magnitude: float = 0.1
    peak_prominence_ratio: float = 5.0
    top_k_count: int = 3
    mapping_mode: str = "legacy"
    mapping_alpha: float = 50.0
    mapping_f0: float = 50.0
    mapping_fc: float = 2000.0
    linear_scale: float = 0.05
    linear_offset: float = 0.0

    def get_controls(self) -> Sequence[SourceControl]:
        return (
            SourceControl(
                key="signal_level",
                label="Signal Level",
                default="0.00",
                kind="text",
            ),
            SourceControl(
                key="gate_open",
                label="Gate",
                default="closed",
                kind="text",
            ),
            SourceControl(
                key="detected_frequencies",
                label="Detected Audio Hz",
                default="—",
                kind="text",
            ),
            SourceControl(
                key="mapped_frequencies",
                label="Mapped Ripple Hz",
                default="—",
                kind="text",
            ),
            SourceControl(
                key="signal_gate_threshold",
                label="Signal Gate Threshold",
                default=self.signal_gate_threshold,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
            ),
            SourceControl(
                key="drive_amplitude",
                label="Audio Drive Amplitude",
                default=self.drive_amplitude,
                minimum=0.0,
                maximum=10.0,
                step=0.05,
            ),
            SourceControl(
                key="minimum_peak_magnitude",
                label="Min Peak Magnitude",
                default=self.minimum_peak_magnitude,
                minimum=0.0,
                maximum=10.0,
                step=0.01,
            ),
            SourceControl(
                key="peak_prominence_ratio",
                label="Peak Prominence Ratio",
                default=self.peak_prominence_ratio,
                minimum=1.0,
                maximum=50.0,
                step=0.1,
            ),
            SourceControl(
                key="top_k_count",
                label="Top-K Peaks",
                default=int(self.top_k_count),
                minimum=1,
                maximum=8,
                step=1,
            ),
            SourceControl(
                key="mapping_mode",
                label="Mapping Mode",
                default=self.mapping_mode,
                kind="choice",
                choices=("legacy", "linear"),
            ),
            SourceControl(
                key="mapping_alpha",
                label="Mapping Alpha",
                default=self.mapping_alpha,
                minimum=0.0,
                maximum=200.0,
                step=1.0,
            ),
            SourceControl(
                key="mapping_f0",
                label="Mapping f0",
                default=self.mapping_f0,
                minimum=1.0,
                maximum=2000.0,
                step=1.0,
                unit="Hz",
            ),
            SourceControl(
                key="mapping_fc",
                label="Mapping fc",
                default=self.mapping_fc,
                minimum=1.0,
                maximum=20000.0,
                step=10.0,
                unit="Hz",
            ),
            SourceControl(
                key="linear_scale",
                label="Linear Scale",
                default=self.linear_scale,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
            ),
            SourceControl(
                key="linear_offset",
                label="Linear Offset",
                default=self.linear_offset,
                minimum=0.0,
                maximum=200.0,
                step=0.1,
            ),
        )


@dataclass
class CameraFrameSourceControls(SourceControlProvider):
    gain: float = 1.0

    def get_controls(self) -> Sequence[SourceControl]:
        return (
            SourceControl(
                key="gain",
                label="Camera Evidence Gain",
                default=self.gain,
                minimum=0.0,
                maximum=10.0,
                step=0.05,
                tooltip=(
                    "Multiply observed camera RGB values before inference. This changes evidence, not wave forcing. "
                    "Zero supplies black evidence; use the Camera toggle to disable observation."
                ),
            ),
        )


@dataclass
class PredictionLearningControls(SourceControlProvider):
    enabled: bool = False
    inference_steps: int = 8
    inference_rate: float = 0.1
    canvas_rate: float = 0.01
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    weight_clip: float = 10.0
    gradient_clip: float = 1.0
    cross_modal_enabled: bool = False
    cross_modal_available: bool = True
    stream_state_enabled: bool = False
    stream_state_available: bool = True

    def get_controls(self) -> Sequence[SourceControl]:
        return (
            SourceControl(
                key="inference_steps",
                label="Inference Steps",
                default=self.inference_steps,
                minimum=1,
                maximum=64,
                step=1,
                tooltip=(
                    "Number of fixed-weight settling steps on sensory evidence, or every frame with the hidden loop or stream context on. "
                    "More steps increase computation and total state correction. Weights learn once after settling."
                ),
            ),
            SourceControl(
                key="inference_rate",
                label="Hidden State Rate",
                default=self.inference_rate,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                tooltip=(
                    "Hidden-state step size, also used to relax persistent hidden states toward the wave prior "
                    "before observing evidence. Zero freezes hidden states; large values can cause oscillation."
                ),
            ),
            SourceControl(
                key="canvas_rate",
                label="Canvas Correction Rate",
                default=self.canvas_rate,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                tooltip=(
                    "Per-inference-step gradient rate for correcting the shared canvas from camera/audio hidden errors, "
                    "not a blend percentage. Zero disables sensory canvas correction, including loop-only correction; "
                    "waves and hidden inference continue. "
                    "Larger rates or more steps can destabilize the surface."
                ),
            ),
            SourceControl(
                key="learning_enabled",
                label="Pathway Learning",
                default=self.enabled,
                kind="toggle",
                tooltip=(
                    "Learn both sensory pathways' predictive weights from final local errors. "
                    "Off freezes weights, not hidden inference or canvas correction. Wave conductances remain fixed."
                    " Enabled stream-state context is evidence even with camera and microphone off."
                ),
            ),
            SourceControl(
                key="learning_rate",
                label="Weight Learning Rate",
                default=self.learning_rate,
                minimum=0.0,
                maximum=1.0,
                step=1e-4,
                tooltip=(
                    "Step size for one local weight update after inference on new evidence. "
                    "This changes sensory predictions, not wave physics. Large values can destabilize learning."
                ),
            ),
            SourceControl(
                key="learning_weight_decay",
                label="Weight Decay",
                default=self.weight_decay,
                minimum=0.0,
                maximum=1.0,
                step=1e-4,
                tooltip=(
                    "L2 coefficient that subtracts weight_decay * weight from each learning gradient before clipping. "
                    "Encourages smaller weights and acts only while pathway learning is enabled."
                ),
            ),
            SourceControl(
                key="learning_weight_clip",
                label="Weight Clip",
                default=self.weight_clip,
                minimum=1e-4,
                maximum=1000.0,
                step=0.01,
                tooltip="Maximum absolute predictive weight after learning. This bounds weights, not hidden states or wave conductances.",
            ),
            SourceControl(
                key="learning_gradient_clip",
                label="Gradient Clip",
                default=self.gradient_clip,
                minimum=1e-4,
                maximum=1000.0,
                step=0.01,
                tooltip=(
                    "Maximum absolute entry of the learning gradient, including decay, before multiplication by "
                    "Weight Learning Rate. This limits weight steps, not sensory errors or state inference."
                ),
            ),
        ) + (
            (SourceControl(
                key="cross_modal_enabled",
                label="Camera / Audio Hidden Loop",
                default=self.cross_modal_enabled,
                kind="toggle",
                tooltip=(
                    "Enable reciprocal learned predictions between camera and audio hidden states. "
                    "Connections start at zero; enable Pathway Learning with paired evidence to learn them. "
                    "Missing sensors remain unclamped and hidden/canvas inference continues. "
                    "With stream context off, no evidence means no weight learning. "
                    "Off retains the loop weights and restores independent pathways. Recurrence can destabilize large state rates."
                ),
            ),) if self.cross_modal_available else ()
        ) + (
            (SourceControl(
                key="stream_state_enabled",
                label="Stream State Context",
                default=self.stream_state_enabled,
                kind="toggle",
                tooltip=(
                    "Add one clamped ON/OFF context node per modality, predicted from pooled hidden states. "
                    "Source toggles define this evidence; waiting for samples or a failed read does not mean OFF. "
                    "Hidden/canvas inference and enabled pathway learning continue with streams off. "
                    "Missing images and audio remain unclamped and their sensory weights stay frozen. "
                    "Turn Pathway Learning off to freeze all weights during a probe."
                ),
            ),) if self.stream_state_available else ()
        )


@dataclass
class PredictionOverlayControls(SourceControlProvider):
    enabled: bool = False
    stride: int = 8
    threshold: float = 0.0
    scale: float = 6.0

    def get_controls(self) -> Sequence[SourceControl]:
        return (
            SourceControl(
                key="show_learning_overlay",
                label="Show Fixed Conductances",
                default=self.enabled,
                kind="toggle",
                tooltip="Show fixed wave-neighbor conductances on the canvas. These are not the learned sensory-pathway weights.",
            ),
            SourceControl(
                key="overlay_stride",
                label="Overlay Stride",
                default=int(self.stride),
                minimum=1,
                maximum=64,
                step=1,
                tooltip="Sample one wave-grid connection group every this many pixels. Larger strides reduce overlay clutter and work.",
            ),
            SourceControl(
                key="overlay_threshold",
                label="Overlay Threshold",
                default=self.threshold,
                minimum=0.0,
                maximum=10.0,
                step=0.01,
                tooltip="Hide displayed conductance edges weaker than this absolute threshold. No weights or graph connections are pruned.",
            ),
            SourceControl(
                key="overlay_scale",
                label="Overlay Scale",
                default=self.scale,
                minimum=0.1,
                maximum=50.0,
                step=0.1,
                tooltip="Visual gain for overlay edge strength and opacity. Does not change physical conductances or learned weights.",
            ),
        )
