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
    drive_grid: np.ndarray | None
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
        self.canvas_shape = engine.canvas_shape
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
            resolution=self.canvas_shape,
        )

    def set_base_amplitude(self, amplitude: float) -> None:
        self.base_amplitude = float(amplitude)

    def resolve(self) -> ResolvedSourceFrame:
        audio_frequencies = self.audio_source.frequencies(n_sources=self.n_sources)
        drive_grid = self.resolve_drive_grid(
            audio_frequencies=audio_frequencies,
        )
        amplitude = self.audio_source.excitation_amplitude(
            base_amplitude=self.base_amplitude,
            frequencies=audio_frequencies,
            synthetic_enabled=self.synthetic_source.enabled,
        )
        return ResolvedSourceFrame(
            audio_frequencies=audio_frequencies,
            drive_grid=drive_grid,
            amplitude=amplitude,
        )

    def resolve_drive_grid(
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
        ]
        active = [grid for grid in grids if grid is not None]
        if not active:
            return None
        canvas_grids = [self._to_canvas_grid(grid) for grid in active]
        return np.sum(np.stack(canvas_grids, axis=0), axis=0, dtype=np.float32)

    def resolve_observation_correction(self) -> np.ndarray | None:
        if not self.prediction_error_transform.applies_to("camera_frame"):
            return None
        observation = self.camera_source.excitation()
        if observation is None:
            return None
        return self.engine.compute_observation_correction(
            source_key="camera_frame",
            observation=observation,
            transform=self.prediction_error_transform,
        )

    def correct_from_observations(self) -> np.ndarray | None:
        correction = self.resolve_observation_correction()
        if correction is not None:
            self.engine.apply_observation_correction(correction)
        return correction

    def _to_canvas_grid(self, grid: np.ndarray) -> np.ndarray:
        values = np.asarray(grid, dtype=np.float32)
        if values.shape == self.resolution:
            return np.broadcast_to(
                values[..., None],
                self.canvas_shape,
            ).copy()
        if values.shape != self.canvas_shape:
            raise ValueError("source grid must match spatial resolution or canvas shape")
        return values
