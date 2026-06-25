from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


_LEGACY_SOURCE_KEYS = {
    "frequency",
    "use_synthetic",
    "use_audio_source",
    "audio_signal_gate_threshold",
    "audio_drive_amplitude",
    "audio_visual_mapping_mode",
    "audio_visual_mapping_alpha",
    "audio_visual_mapping_f0",
    "audio_visual_mapping_fc",
    "audio_visual_linear_scale",
    "audio_visual_linear_offset",
    "use_camera_source",
    "camera_source_index",
    "camera_source_gain",
    "prediction_error_transform_enabled",
    "prediction_error_inputs",
    "prediction_error_predictor_source",
    "prediction_error_sigma",
    "prediction_error_output_mode",
    "prediction_error_activation_function",
    "prediction_error_activation_scale",
    "prediction_error_gain",
    "prediction_error_prediction_clip",
    "prediction_error_max_output",
    "prediction_error_learning_enabled",
    "prediction_error_learning_rate",
    "prediction_error_learning_weight_decay",
    "prediction_error_learning_weight_clip",
    "prediction_error_learning_gradient_clip",
}

_LEGACY_POSE_KEYS = {
    "use_pose_sources",
    "pose_model_path",
    "pose_camera_index",
    "pose_graph_stiffness",
    "body_boundary_transmission",
    "body_boundary_dissipation",
    "pose_render_mode",
    "pose_field_width_fraction",
    "pose_field_height_fraction",
    "pose_debug_view",
    "pose_acceleration_scale",
    "pose_max_excitation",
    "pose_drive_scale",
}


def pop_legacy_source_orchestrator_config(
    kwargs: dict[str, Any],
) -> RippleSourceOrchestratorConfig:
    return RippleSourceOrchestratorConfig(
        synthetic=SyntheticSourceConfig(
            enabled=bool(kwargs.pop("use_synthetic", True)),
            frequency=kwargs.pop("frequency", 440.0),
        ),
        audio=AudioSourceConfig(
            enabled=kwargs.pop("use_audio_source", None),
            signal_gate_threshold=float(
                kwargs.pop("audio_signal_gate_threshold", 0.05)
            ),
            drive_amplitude=float(kwargs.pop("audio_drive_amplitude", 1.0)),
            mapping_mode=str(kwargs.pop("audio_visual_mapping_mode", "legacy")),
            mapping_alpha=float(kwargs.pop("audio_visual_mapping_alpha", 50.0)),
            mapping_f0=float(kwargs.pop("audio_visual_mapping_f0", 50.0)),
            mapping_fc=float(kwargs.pop("audio_visual_mapping_fc", 2000.0)),
            linear_scale=float(kwargs.pop("audio_visual_linear_scale", 0.05)),
            linear_offset=float(kwargs.pop("audio_visual_linear_offset", 0.0)),
        ),
        camera_frame=CameraFrameSourceConfig(
            enabled=bool(kwargs.pop("use_camera_source", False)),
            camera_index=int(kwargs.pop("camera_source_index", 0)),
            gain=float(kwargs.pop("camera_source_gain", 1.0)),
        ),
        prediction_error=PredictionErrorTransformConfig(
            enabled=bool(kwargs.pop("prediction_error_transform_enabled", False)),
            inputs=tuple(kwargs.pop("prediction_error_inputs", ("camera_frame",))),
            predictor_source=str(
                kwargs.pop("prediction_error_predictor_source", "ripple_state")
            ),
            sigma=float(kwargs.pop("prediction_error_sigma", 0.1)),
            output_mode=str(kwargs.pop("prediction_error_output_mode", "bits")),
            activation_function=str(
                kwargs.pop("prediction_error_activation_function", "softsign")
            ),
            activation_scale=float(
                kwargs.pop("prediction_error_activation_scale", 1.0)
            ),
            gain=float(kwargs.pop("prediction_error_gain", 1.0)),
            prediction_clip=float(kwargs.pop("prediction_error_prediction_clip", 10.0)),
            max_output=float(kwargs.pop("prediction_error_max_output", 10.0)),
            learning_enabled=bool(kwargs.pop("prediction_error_learning_enabled", False)),
            learning_rate=float(kwargs.pop("prediction_error_learning_rate", 1e-4)),
            learning_weight_decay=float(
                kwargs.pop("prediction_error_learning_weight_decay", 1e-4)
            ),
            learning_weight_clip=float(
                kwargs.pop("prediction_error_learning_weight_clip", 1.0)
            ),
            learning_gradient_clip=float(
                kwargs.pop("prediction_error_learning_gradient_clip", 1.0)
            ),
        ),
    )


def pop_legacy_pose_config(kwargs: dict[str, Any]) -> RipplePoseConfig:
    kwargs.pop("pose_acceleration_scale", 1.0)
    kwargs.pop("pose_max_excitation", None)
    kwargs.pop("pose_drive_scale", 0.1)
    return RipplePoseConfig(
        enabled=bool(kwargs.pop("use_pose_sources", False)),
        render_mode=str(kwargs.pop("pose_render_mode", "overlay")),
        model_path=kwargs.pop("pose_model_path", None),
        camera_index=int(kwargs.pop("pose_camera_index", 0)),
        debug_view=bool(kwargs.pop("pose_debug_view", False)),
        graph_stiffness=float(kwargs.pop("pose_graph_stiffness", 0.25)),
        field_width_fraction=float(kwargs.pop("pose_field_width_fraction", 1.0)),
        field_height_fraction=float(kwargs.pop("pose_field_height_fraction", 1.0)),
        body_boundary_transmission=float(
            kwargs.pop("body_boundary_transmission", 0.0)
        ),
        body_boundary_dissipation=float(kwargs.pop("body_boundary_dissipation", 1.0)),
    )


def reject_mixed_legacy_config(
    kwargs: dict[str, Any],
    *,
    source_orchestrator_config: RippleSourceOrchestratorConfig | None,
    pose_config: RipplePoseConfig | None,
) -> None:
    if source_orchestrator_config is not None:
        overlap = sorted(_LEGACY_SOURCE_KEYS.intersection(kwargs))
        if overlap:
            raise ValueError(
                "source_orchestrator_config cannot be combined with legacy source "
                f"kwargs: {', '.join(overlap)}"
            )
    if pose_config is not None:
        overlap = sorted(_LEGACY_POSE_KEYS.intersection(kwargs))
        if overlap:
            raise ValueError(
                "pose_config cannot be combined with legacy pose kwargs: "
                f"{', '.join(overlap)}"
            )
