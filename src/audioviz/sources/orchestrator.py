from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from audioviz.readouts import (
    PredictiveCodingConfig,
    PredictiveCodingRGBReadout,
    VisualReadout,
)
from audioviz.sources.audio import AudioRippleSource, AudioSourceConfig
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.predictive_coding import PredictiveCodingHiddenCoupling, infer_joint
from audioviz.readouts.gaussian_audio_uncertainty import GaussianAudioUncertainty
from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty
from audioviz.readouts.shared_predictive_coding import SharedPredictiveCodingGraph
from audioviz.readouts.shared_sensory_io import SharedSensoryBranch, SharedSensoryIO
from audioviz.sources.camera import CameraFrameSource, CameraFrameSourceConfig
from audioviz.sources.gaze import ActiveVisionConfig, GazePolicy
from audioviz.sources.synthetic import SyntheticRippleSource, SyntheticSourceConfig
from audioviz.readouts.spatial_precision import GaussianSpatialPrecision
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
        shared_sensory_graph: ActiveVisionConfig | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.resolution = resolution
        self.canvas_shape = engine.canvas_shape
        self.n_sources = int(n_sources)
        self.base_amplitude = 1.0
        self.shader_model = None
        if shared_sensory_graph is not None:
            if not shared_sensory_graph.shared_sensory_graph:
                raise ValueError("Shared sensory graph configuration must explicitly opt in")
            if config.visual_pathway.cross_modal_enabled or visual_readout is not None:
                raise ValueError("Shared sensory graph cannot use legacy cross-modal coupling or a separate readout")
        if engine.use_shader:
            if visual_readout is not None:
                raise ValueError("OpenGL requires its native predictive readouts")
            from audioviz.readouts.opengl_predictive import OpenGLPredictiveModel

            self.shader_model = OpenGLPredictiveModel(
                engine, config.visual_pathway, audio_enabled=processor is not None,
            )
        self.sensory_graph = None
        self.graph_io = None
        if shared_sensory_graph is not None:
            self.graph_io = SharedSensoryIO(
                self.canvas_shape, radius=shared_sensory_graph.foveal_radius,
                backend=engine.backend,
                device=self.shader_model.device if self.shader_model is not None else None,
            )
            self.sensory_graph = SharedPredictiveCodingGraph(
                visual_shape=self.graph_io.visual_shape, backend=engine.backend,
                hidden_nodes=config.visual_pathway.hidden_channels,
                config=replace(config.visual_pathway, dtype="float64", spatial_precision=None),
                memory_time_constant=shared_sensory_graph.memory_time_constant,
            )
        self.visual_readout = (
            SharedSensoryBranch(self.sensory_graph, replace(config.visual_pathway, dtype="float64"), self.canvas_shape, engine.backend)
            if self.sensory_graph is not None else
            self.shader_model.camera if self.shader_model is not None else
            PredictiveCodingRGBReadout(
                canvas_shape=self.canvas_shape,
                config=config.visual_pathway,
                backend=engine.backend,
            )
            if visual_readout is None
            else visual_readout
        )
        if self.shader_model is None and self.visual_readout.backend is not engine.backend:
            raise ValueError("Visual readout must use the engine's array backend.")
        self.audio_readout = (
            SharedSensoryBranch(self.sensory_graph, replace(config.visual_pathway, dtype="float64"), self.canvas_shape, engine.backend)
            if self.sensory_graph is not None else
            self.shader_model.audio if self.shader_model is not None else
            PredictiveCodingAudioReadout(
                canvas_shape=self.canvas_shape, config=config.visual_pathway,
                backend=engine.backend,
            )
            if processor is not None else None
        )
        self.audio_prediction: np.ndarray | None = None
        self.hidden_coupling = (
            PredictiveCodingHiddenCoupling(self.visual_readout, self.audio_readout)
            if self.shader_model is None and isinstance(self.visual_readout, PredictiveCodingRGBReadout) and self.audio_readout is not None
            else None
        )
        self._validate_cross_modal(config.visual_pathway.cross_modal_enabled)
        self.visual_observation: np.ndarray | None = None
        self.visual_uncertainty: GaussianSensoryUncertainty | None = None
        self.audio_uncertainty: GaussianAudioUncertainty | None = None
        self._uncertainty_dt = 0.0
        self.gaze_policy: GazePolicy | None = None
        self.gaze_scores: np.ndarray | None = None
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
            observation_gain=config.audio.observation_gain,
            predictive=True,
        )
        self.camera_source = CameraFrameSource(
            resolution=resolution,
            camera_index=config.camera_frame.camera_index,
            gain=config.camera_frame.gain,
            enabled=bool(config.camera_frame.enabled),
            capture=camera_capture,
            backend=engine.backend,
            frame_mapper=self.shader_model.preprocess_camera if self.shader_model is not None else None,
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

    def predict_visual(self, *, dt: float | None = None) -> np.ndarray:
        if self.has_temporal_beliefs and dt is not None:
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("visual uncertainty requires a finite, positive timestep")
            self._uncertainty_dt = float(dt)
        if self.sensory_graph is not None:
            if self.gaze_policy is not None:
                self._select_visual_gaze(self.gaze_policy)
            precision = self.visual_readout.config.spatial_precision
            self.graph_io.set_gaze(precision.gaze if precision is not None else (.5, .5))
            canvas = self.graph_io.sample(self.engine.get_prior())
            visual, self.audio_prediction = self.sensory_graph.forecast(
                canvas, self._uncertainty_dt,
                aperture_mask=self.graph_io.valid.reshape(self.graph_io.visual_shape),
            )
            return self.graph_io.render(visual)
        if self.gaze_policy is not None:
            self._select_visual_gaze(self.gaze_policy)
        if self.visual_uncertainty is not None:
            prior = self.engine.get_prior()
            prediction = self.visual_uncertainty.predict(prior, dt=self._uncertainty_dt)
            self.audio_prediction = (
                self.audio_uncertainty.predict(prior, dt=self._uncertainty_dt)
                if self.audio_uncertainty is not None else None
            )
            return prediction
        return self._predict_visual_mean()

    def _select_visual_gaze(self, policy: GazePolicy) -> None:
        if not self.has_temporal_beliefs:
            raise RuntimeError("Gaze policy requires visual uncertainty to be enabled")
        precision = self.visual_readout.config.spatial_precision
        if precision is None:
            precision = GaussianSpatialPrecision(baseline=1.0, peak=0.0)
        self.gaze_scores = None
        if not precision.sample:
            return
        self.gaze_scores = self.visual_information_gain(precision)
        gaze = policy.select(self.gaze_scores, precision.gaze)
        if gaze != precision.gaze:
            self.update_pathway_config(spatial_precision=replace(precision, gaze=gaze))

    @property
    def has_temporal_beliefs(self) -> bool:
        return self.visual_uncertainty is not None or self.sensory_graph is not None

    def visual_information_gain(self, precision):
        if self.sensory_graph is None:
            if self.visual_uncertainty is None:
                raise RuntimeError("Information gain requires a Gaussian belief model")
            return self.visual_uncertainty.expected_information_gain(
                precision, prior=self.engine.get_prior(), dt=self._uncertainty_dt,
            )
        score_shape = (9, 13)
        gazes = [(float(x), float(y)) for y in np.linspace(0, 1, score_shape[0])
                 for x in np.linspace(0, 1, score_shape[1])]
        patches, masks = self.graph_io.sample_candidates(self.engine.get_prior(), gazes)
        return self.sensory_graph.candidate_information_gains(
            patches.reshape(*score_shape, *self.graph_io.visual_shape, 3), self._uncertainty_dt,
            aperture_masks=masks.reshape(*score_shape, *self.graph_io.visual_shape),
        )

    def enable_visual_uncertainty(
        self, *, dt, initial_variance=1.0, process_variance=0.01, memory_time_constant=1/3,
        audio_precision: float | None = None,
    ):
        """Opt into shared temporal Gaussian inference for visual and optional common-mode audio."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("visual uncertainty requires a finite, positive timestep")
        if self.shader_model is None and not isinstance(self.visual_readout, PredictiveCodingRGBReadout):
            raise ValueError("visual uncertainty requires the predictive RGB readout")
        if self.visual_uncertainty is not None:
            raise RuntimeError("Visual uncertainty is already enabled")
        if self.audio_readout is not None and audio_precision is None:
            raise ValueError("Temporal Gaussian visual inference requires a visual-only model")
        if audio_precision is not None and self.audio_readout is None:
            raise ValueError("Temporal Gaussian audio inference requires an audio readout")
        device = self.shader_model.device if self.shader_model is not None else None
        self.visual_uncertainty = GaussianSensoryUncertainty(
            self.visual_readout, initial_variance=initial_variance, process_variance=process_variance,
            memory_time_constant=memory_time_constant,
            backend=self.engine.backend,
            device=device,
        )
        if audio_precision is not None:
            self.audio_uncertainty = GaussianAudioUncertainty(
                self.audio_readout, initial_variance=initial_variance, process_variance=process_variance,
                memory_time_constant=memory_time_constant, precision=audio_precision,
                backend=self.engine.backend,
                device=device,
            )
        self._uncertainty_dt = float(dt)
        return self.visual_uncertainty

    def _predict_visual_mean(self) -> np.ndarray:
        prior = self.engine.get_prior()
        if self.shader_model is not None:
            prediction, self.audio_prediction = self.shader_model.predict(prior)
            return prediction
        if self.visual_readout.config.cross_modal_enabled:
            if self.hidden_coupling is None:
                raise RuntimeError("Cross-modal prediction requires both predictive readouts.")
            prediction, self.audio_prediction = self.hidden_coupling.predict(prior)
            return prediction
        self.audio_prediction = None
        if self.audio_readout is not None:
            self.audio_prediction = self.audio_readout.predict(prior)
        return self.visual_readout.predict(prior)

    def resolve_visual_observation_correction(self) -> np.ndarray:
        observation = (
            self.camera_source.excitation()
            if self.prediction_error_transform.applies_to("camera_frame") else None
        )
        self.visual_observation = observation
        return self.visual_readout.infer(
            self.engine.get_prior(),
            observation,
            stream_state=self.camera_source.enabled,
        )

    def correct_from_observations(self) -> np.ndarray:
        if self.sensory_graph is not None:
            precision = self.visual_readout.config.spatial_precision
            observing = precision is None or precision.sample
            self.visual_observation = (
                self.camera_source.excitation()
                if observing and self.prediction_error_transform.applies_to("camera_frame") else None
            )
            visual = (
                self.graph_io.sample(self.visual_observation)
                if self.visual_observation is not None else None
            )
            mask = self.graph_io.valid.reshape(self.graph_io.visual_shape)
            if not observing:
                mask = np.zeros_like(mask)
            canvas_direction = self.sensory_graph.condition(
                visual, self.audio_source.observation(),
                visual_mask=mask,
            )
            correction = self.graph_io.scatter(
                canvas_direction*self.visual_readout.config.canvas_rate,
            )
            self.engine.apply_observation_correction(correction)
            return correction
        if self.visual_uncertainty is not None:
            self.visual_observation = (
                self.camera_source.excitation()
                if self.prediction_error_transform.applies_to("camera_frame") else None
            )
            correction = self.visual_uncertainty.condition(
                self.visual_observation, self.visual_readout.config.spatial_precision,
            )
            if self.audio_uncertainty is not None:
                audio = self.audio_source.observation()
                audio_correction = self.audio_uncertainty.condition(audio)
                if self.shader_model is not None:
                    self.shader_model.device.run(
                        """layout(std430,binding=0) buffer Visual {float visual[];};
                        layout(std430,binding=1) readonly buffer Audio {float audio[];};
                        uniform int count;
                        void main(){int i=int(gl_GlobalInvocationID.x);if(i<count)visual[i]+=audio[i];}""",
                        [correction, audio_correction], correction.size, count=correction.size,
                    )
                    for branch, unc in (
                        (self.shader_model.camera, self.visual_uncertainty),
                        (self.shader_model.audio, self.audio_uncertainty),
                    ):
                        if branch.diagnostics_enabled:
                            self.shader_model._trajectory = getattr(branch, "_trajectory", ())
                            self.shader_model._diagnostics(
                                None, None, branches=(branch,), audio_precision=unc.precision if branch.index == 1 else 1.0,
                            )
                else:
                    correction = correction + audio_correction
                    final_canvas = self.engine.get_prior() + correction
                    canvas_means = tuple(
                        float(v) for v in self.engine.backend.asnumpy(final_canvas.mean(axis=(0, 1)))
                        if self.engine.use_gpu
                    ) if self.engine.use_gpu else tuple(float(v) for v in final_canvas.mean(axis=(0, 1)))
                    for branch in (self.visual_readout, self.audio_readout):
                        if branch.diagnostics is not None:
                            branch.diagnostics = replace(branch.diagnostics, canvas_means=canvas_means)
            self.engine.apply_observation_correction(correction)
            return correction
        return self._correct_from_observations_mean()

    def _correct_from_observations_mean(self) -> np.ndarray:
        self.visual_observation = None
        audio = self.audio_source.observation()
        if self.shader_model is not None:
            image = (
                self.camera_source.excitation()
                if self.prediction_error_transform.applies_to("camera_frame") else None
            )
            self.visual_observation = image
            correction = self.shader_model.infer(
                self.engine.get_prior(), image, audio,
                states=(self.camera_source.enabled, self.audio_source.enabled),
            )
            self.engine.apply_observation_correction(correction)
            return correction
        if self.audio_readout is None:
            correction = self.resolve_visual_observation_correction()
        else:
            image = (
                self.camera_source.excitation()
                if self.prediction_error_transform.applies_to("camera_frame") else None
            )
            self.visual_observation = image
            if not isinstance(self.visual_readout, PredictiveCodingRGBReadout):
                raise TypeError("Stream-state inference requires a predictive-coding readout.")
            correction = infer_joint(
                self.engine.get_prior(),
                ((self.visual_readout, image), (self.audio_readout, audio)),
                coupling=self.hidden_coupling if self.visual_readout.config.cross_modal_enabled else None,
                stream_states={
                    self.visual_readout: self.camera_source.enabled,
                    self.audio_readout: self.audio_source.enabled,
                },
            )
        self.engine.apply_observation_correction(correction)
        return correction

    def update_pathway_config(self, **changes) -> None:
        if self.sensory_graph is not None and changes.get("cross_modal_enabled", False):
            raise ValueError("The shared graph already has joint connectivity; legacy coupling is not used")
        if self.visual_uncertainty is not None:
            self.visual_uncertainty.validate_config(replace(self.visual_readout.config, **changes))
        if self.audio_uncertainty is not None:
            self.audio_uncertainty.validate_config(replace(self.audio_readout.config, **changes))
        if "cross_modal_enabled" in changes:
            self._validate_cross_modal(changes["cross_modal_enabled"])
        if self.sensory_graph is not None:
            # Gaze selects hard-clamped ports; it is not a soft precision in the PC energy.
            model_changes = {
                key: value for key, value in changes.items() if key != "spatial_precision"
            }
            if model_changes:
                self.sensory_graph.update_config(**model_changes)
        self.visual_readout.update_config(**changes)
        if (self.shader_model is None or self.sensory_graph is not None) and self.audio_readout is not None:
            self.audio_readout.update_config(**changes)

    def _validate_cross_modal(self, enabled: bool) -> None:
        if self.shader_model is not None:
            if enabled and self.shader_model.audio is None:
                raise ValueError("Cross-modal inference requires camera/audio readouts")
            return
        if enabled and (self.hidden_coupling is None or self.engine.use_shader):
            raise ValueError("Cross-modal inference requires camera/audio readouts and an array-backed canvas.")

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self.visual_readout.set_diagnostics_enabled(enabled)
        if self.audio_readout is not None:
            self.audio_readout.set_diagnostics_enabled(enabled)

    def reset_readouts(self) -> None:
        if self.sensory_graph is not None:
            self.sensory_graph.reset()
            self.graph_io.reset()
            self.audio_prediction = self.visual_observation = self.gaze_scores = None
            return
        self.visual_readout.set_spatial_preview_enabled(False)
        if self.visual_uncertainty is None:
            self.visual_readout.reset()
        if self.audio_readout is not None:
            self.audio_readout.set_spatial_preview_enabled(False)
            if self.shader_model is None and self.audio_uncertainty is None:
                self.audio_readout.reset()
        self.audio_prediction = None
        self.visual_observation = None
        if self.visual_uncertainty is not None:
            self.visual_uncertainty.reset()
        if self.audio_uncertainty is not None:
            self.audio_uncertainty.reset()
        self.gaze_scores = None

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
