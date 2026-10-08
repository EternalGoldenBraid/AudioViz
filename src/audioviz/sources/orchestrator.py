from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audioviz.readouts import (
    PredictiveCodingConfig,
    PredictiveCodingRGBReadout,
    VisualReadout,
)
from audioviz.sources.audio import AudioRippleSource, AudioSourceConfig
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.predictive_coding import infer_joint
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
    visual_pathway: PredictiveCodingConfig = field(
        default_factory=PredictiveCodingConfig
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
        visual_readout: VisualReadout | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.resolution = resolution
        self.canvas_shape = engine.canvas_shape
        self.n_sources = int(n_sources)
        self.base_amplitude = 1.0
        self.visual_readout = (
            PredictiveCodingRGBReadout(
                canvas_shape=self.canvas_shape,
                config=config.visual_pathway,
            )
            if visual_readout is None
            else visual_readout
        )
        self.audio_readout = (
            PredictiveCodingAudioReadout(
                canvas_shape=self.canvas_shape, config=config.visual_pathway
            )
            if processor is not None else None
        )
        self.audio_prediction: np.ndarray | None = None
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
        if audio_enabled and engine.use_shader:
            raise NotImplementedError("Audio predictive inference requires an array-backed canvas.")
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
            observation_gain=config.audio.observation_gain,
            predictive=True,
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
        amplitude = self.base_amplitude
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
        grid = self.synthetic_source.excitation_grid(
            t=next_time,
            source_positions=self.engine.source_positions,
            resolution=self.resolution,
            grid_spacing=self.engine.grid_spacing,
            speed=self.engine.speed,
            max_frequency=self.engine.max_frequency,
            decay_alpha=self.engine.decay_alpha,
        )
        return None if grid is None else self._to_canvas_grid(grid)

    def predict_visual(self) -> np.ndarray:
        prior = self.engine.get_prior_numpy()
        self.audio_prediction = None
        if self.audio_readout is not None:
            self.audio_prediction = self.audio_readout.predict(prior)
        return self.visual_readout.predict(prior)

    def resolve_visual_observation_correction(self) -> np.ndarray | None:
        if not self.prediction_error_transform.applies_to("camera_frame"):
            return None
        observation = self.camera_source.excitation()
        if observation is None:
            return None
        return self.visual_readout.infer(
            self.engine.get_prior_numpy(),
            observation,
        )

    def correct_from_observations(self) -> np.ndarray | None:
        audio = self.audio_source.observation()
        if audio is None:
            correction = self.resolve_visual_observation_correction()
        else:
            if self.engine.use_shader:
                raise NotImplementedError("Audio predictive inference requires an array-backed canvas.")
            prior = self.engine.get_prior_numpy()
            image = (
                self.camera_source.excitation()
                if self.prediction_error_transform.applies_to("camera_frame") else None
            )
            if self.audio_readout is None:
                raise RuntimeError("Audio evidence requires an audio predictive readout.")
            if image is None:
                correction = self.audio_readout.infer(prior, audio)
            else:
                if not isinstance(self.visual_readout, PredictiveCodingRGBReadout):
                    raise TypeError("Joint inference requires a predictive-coding visual readout.")
                correction = infer_joint(prior, ((self.visual_readout, image), (self.audio_readout, audio)))
        if correction is not None:
            self.engine.apply_observation_correction(correction)
        return correction

    def update_pathway_config(self, **changes) -> None:
        self.visual_readout.update_config(**changes)
        if self.audio_readout is not None:
            self.audio_readout.update_config(**changes)

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self.visual_readout.set_diagnostics_enabled(enabled)
        if self.audio_readout is not None:
            self.audio_readout.set_diagnostics_enabled(enabled)

    def reset_readouts(self) -> None:
        self.visual_readout.set_spatial_preview_enabled(False)
        self.visual_readout.reset()
        if self.audio_readout is not None:
            self.audio_readout.set_spatial_preview_enabled(False)
            self.audio_readout.reset()
        self.audio_prediction = None

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
