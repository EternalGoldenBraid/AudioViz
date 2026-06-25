from __future__ import annotations

from dataclasses import dataclass, field

from audioviz.visualization.prediction_error_transform import (
    PredictionErrorTransformConfig,
)


@dataclass(frozen=True)
class SyntheticSourceConfig:
    enabled: bool = True
    frequency: float | list[float] | tuple[float, ...] = 440.0


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


@dataclass(frozen=True)
class CameraFrameSourceConfig:
    enabled: bool = False
    camera_index: int = 0
    gain: float = 1.0


@dataclass(frozen=True)
class RippleSourceOrchestratorConfig:
    synthetic: SyntheticSourceConfig = field(default_factory=SyntheticSourceConfig)
    audio: AudioSourceConfig = field(default_factory=AudioSourceConfig)
    camera_frame: CameraFrameSourceConfig = field(default_factory=CameraFrameSourceConfig)
    prediction_error: PredictionErrorTransformConfig = field(
        default_factory=PredictionErrorTransformConfig
    )


@dataclass(frozen=True)
class RipplePoseConfig:
    enabled: bool = False
    render_mode: str = "overlay"
    model_path: str | None = None
    camera_index: int = 0
    debug_view: bool = False
    graph_stiffness: float = 0.25
    field_width_fraction: float = 1.0
    field_height_fraction: float = 1.0
    body_boundary_transmission: float = 0.0
    body_boundary_dissipation: float = 1.0
