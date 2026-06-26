from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audioviz.sources.audio import AudioRippleSource, AudioSourceConfig
from audioviz.sources.camera import CameraFrameSource, CameraFrameSourceConfig
from audioviz.sources.synthetic import SyntheticRippleSource, SyntheticSourceConfig
from audioviz.transforms.prediction_error import PredictionErrorTransform
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig


@dataclass(frozen=True)
class RippleSourceOrchestratorConfig:
    synthetic: SyntheticSourceConfig = field(default_factory=SyntheticSourceConfig)
    audio: AudioSourceConfig = field(default_factory=AudioSourceConfig)
    camera_frame: CameraFrameSourceConfig = field(
        default_factory=CameraFrameSourceConfig
    )
    prediction_error: PredictionErrorTransformConfig = field(
        default_factory=PredictionErrorTransformConfig
    )


@dataclass(frozen=True)
class ResolvedSourceFrame:
    audio_frequencies: np.ndarray | None
    grid_excitation: np.ndarray | None
    amplitude: float


class RippleSourceOrchestrator:
    def __init__(
        self,
        *,
        config: RippleSourceOrchestratorConfig,
        processor,
        engine,
        resolution: tuple[int, int],
        n_sources: int,
        camera_capture=None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.resolution = resolution
        self.n_sources = int(n_sources)
        self.base_amplitude = 1.0
        self.synthetic_source = SyntheticRippleSource(
            frequency=config.synthetic.frequency,
            n_sources=self.n_sources,
            enabled=bool(config.synthetic.enabled),
        )
        audio_enabled = (
            processor is not None and not config.synthetic.enabled
            if config.audio.enabled is None
            else processor is not None and bool(config.audio.enabled)
        )
        self.audio_source = AudioRippleSource(
            processor=processor,
            enabled=audio_enabled,
            signal_gate_threshold=config.audio.signal_gate_threshold,
            drive_amplitude=config.audio.drive_amplitude,
            mapping_mode=config.audio.mapping_mode,
            mapping_alpha=config.audio.mapping_alpha,
            mapping_f0=config.audio.mapping_f0,
            mapping_fc=config.audio.mapping_fc,
            linear_scale=config.audio.linear_scale,
            linear_offset=config.audio.linear_offset,
        )
        self.camera_source = CameraFrameSource(
            resolution=resolution,
            camera_index=config.camera_frame.camera_index,
            gain=config.camera_frame.gain,
            enabled=bool(config.camera_frame.enabled),
            capture=camera_capture,
        )
        self.prediction_error_transform = PredictionErrorTransform(
            config=config.prediction_error,
            resolution=resolution,
        )

    def set_base_amplitude(self, amplitude: float) -> None:
        self.base_amplitude = float(amplitude)

    def resolve(self) -> ResolvedSourceFrame:
        audio_frequencies = self.audio_source.frequencies(n_sources=self.n_sources)
        grid_excitation = self.resolve_excitation_grid(
            audio_frequencies=audio_frequencies,
        )
        amplitude = self.audio_source.excitation_amplitude(
            base_amplitude=self.base_amplitude,
            frequencies=audio_frequencies,
            synthetic_enabled=self.synthetic_source.enabled,
        )
        return ResolvedSourceFrame(
            audio_frequencies=audio_frequencies,
            grid_excitation=grid_excitation,
            amplitude=amplitude,
        )

    def resolve_excitation_grid(
        self,
        *,
        audio_frequencies: np.ndarray | None,
    ) -> np.ndarray | None:
        next_time = self.engine.time + self.engine.dt
        grids = [
            self.synthetic_source.excitation_grid(
                t=next_time,
                source_positions=self.engine.source_positions,
                resolution=self.resolution,
                grid_spacing=self.engine.grid_spacing,
                speed=self.engine.speed,
                max_frequency=self.engine.max_frequency,
                decay_alpha=self.engine.decay_alpha,
            ),
            self.audio_source.excitation_grid(
                t=next_time,
                n_sources=self.n_sources,
                frequencies=audio_frequencies,
                source_positions=self.engine.source_positions,
                resolution=self.resolution,
                grid_spacing=self.engine.grid_spacing,
                speed=self.engine.speed,
                max_frequency=self.engine.max_frequency,
                decay_alpha=self.engine.decay_alpha,
            ),
            self.resolve_camera_frame_excitation(),
        ]
        active = [grid for grid in grids if grid is not None]
        if not active:
            return None
        return np.sum(np.stack(active, axis=0), axis=0, dtype=np.float32)

    def resolve_camera_frame_excitation(self) -> np.ndarray | None:
        excitation = self.camera_source.excitation()
        if excitation is None:
            return None
        return self.apply_source_transforms("camera_frame", excitation)

    def apply_source_transforms(
        self,
        source_key: str,
        excitation: np.ndarray,
    ) -> np.ndarray:
        return self.engine.observe(
            source_key=source_key,
            observation=excitation,
            transform=self.prediction_error_transform,
        )
