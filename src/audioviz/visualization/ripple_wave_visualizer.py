import time
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from audioviz.engine import RippleEngine
from audioviz.physics import BoundaryCondition
from audioviz.sources import RippleSourceOrchestrator, RippleSourceOrchestratorConfig
from audioviz.sources.orchestrator import SharedGraphFactory
from audioviz.readouts.shared_predictive_coding import SharedPredictiveCodingGraph
from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.sources.gaze import ActiveVisionConfig, GazePolicy
from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty
from audioviz.readouts.spatial_precision import GaussianSpatialPrecision
from audioviz.ui.gaze_controls import GazeControls
from audioviz.visualization.ripple_renderers import (
    NumpyImageRenderer,
    OpenGLFieldRenderer,
)
from audioviz.visualization.prediction_diagnostics_view import PredictionDiagnosticsView
from audioviz.visualization.prediction_scene_window import PredictionSceneWindow
from audioviz.visualization.belief_diagnostics_view import BeliefDiagnosticsFrame, BeliefDiagnosticsView
from audioviz.visualization.attention_overlay import AttentionOverlay, fit_image_rect
from audioviz.utils.source_preview import SourceScenePreview
from audioviz.utils.continuous_graph_preview import continuous_graph_preview
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.ui.ripple_control_panel import (
    RippleControlPanel,
)
from audioviz.ui.ripple_source_controls import (
    RippleSourceControlBinding,
)
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.visualization.frame_rate_display import FrameRateDisplay
from audioviz.audio_processing.audio_processor import AudioProcessor

class RippleWaveVisualizer(VisualizerBase):
    @property
    def synthetic_source(self):
        return self.source_orchestrator.synthetic_source

    @property
    def audio_source(self):
        return self.source_orchestrator.audio_source

    @property
    def camera_source(self):
        return self.source_orchestrator.camera_source

    @property
    def prediction_error_transform(self):
        return self.source_orchestrator.prediction_error_transform

    @property
    def use_camera_source(self) -> bool:
        return self.camera_source.enabled

    @use_camera_source.setter
    def use_camera_source(self, enabled: bool) -> None:
        self.camera_source.enabled = bool(enabled)

    @property
    def use_synthetic(self) -> bool:
        return self.synthetic_source.enabled

    @use_synthetic.setter
    def use_synthetic(self, enabled: bool) -> None:
        self.synthetic_source.enabled = bool(enabled)

    @property
    def use_audio_source(self) -> bool:
        return self.audio_source.enabled

    @use_audio_source.setter
    def use_audio_source(self, enabled: bool) -> None:
        self.audio_source.enabled = bool(enabled)

    def __init__(self,
                 processor: Optional[AudioProcessor] = None,
                 n_sources: int = 1,
                 plane_size_m: Tuple[float, float] = (0.36, 0.62),
                 resolution: Tuple[int, int] = (400, 400),
                 amplitude: float = 1.0,
                 decay_alpha: float = 0.0,
                 speed: float = 340.0,
                 damping: float = 0.999,
                 apply_gaussian_smoothing: bool = False,
                 use_gpu: bool = False,
                 use_shader: bool = False,
                 boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
                 rgb_canvas_enabled: bool = False,
                 auto_color_activation_threshold: float = 0.1,
                 auto_color_floor: float = 0.1,
                 source_orchestrator_config: RippleSourceOrchestratorConfig | None = None,
                 camera_capture=None,
                 active_vision: ActiveVisionConfig | None = None,
                 forecast_dt: float | None = None,
                 wave_propagation_enabled: bool = True,
                 shared_graph_factory: SharedGraphFactory = SharedPredictiveCodingGraph,
                 **kwargs):
        if forecast_dt is not None and (
            isinstance(forecast_dt, bool) or not np.isfinite(forecast_dt) or forecast_dt <= 0
        ):
            raise ValueError("forecast_dt must be finite and positive")
        if source_orchestrator_config is None:
            source_orchestrator_config = RippleSourceOrchestratorConfig()
        if active_vision is not None:
            if not active_vision.shared_sensory_graph:
                GaussianSensoryUncertainty.validate_config(source_orchestrator_config.visual_pathway)
            if active_vision.audio_precision is None:
                if processor is not None or source_orchestrator_config.audio.enabled:
                    raise ValueError("Active vision requires a camera-only ripple window unless active audio is explicitly enabled")
            elif processor is None:
                raise ValueError("Active audio requires an audio processor")
            if not source_orchestrator_config.prediction_error.enabled or "camera_frame" not in source_orchestrator_config.prediction_error.inputs:
                raise ValueError("Active vision requires camera prediction-error inference")
        super().__init__(processor, **kwargs)
        self.active_vision_config = active_vision
        self.forecast_dt = forecast_dt
        self.gaze_controls = None
        self.gaze_status = None
        self._gaze_last_prediction_time = None

        self.processor = processor
        self.n_sources = n_sources
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.boundary_condition = boundary_condition
        self.rgb_canvas_enabled = bool(rgb_canvas_enabled)
        self.auto_color_activation_threshold = float(auto_color_activation_threshold)
        self.auto_color_floor = float(auto_color_floor)

        self.plane_size_m = plane_size_m
        self.resolution = resolution
        self.base_amplitude = float(amplitude)
        self.apply_gaussian_smoothing = apply_gaussian_smoothing
        self.amplitude = amplitude
        self.decay_alpha = decay_alpha
        self.speed = speed
        self.damping = damping
        self.time = 0.0
        self.control_panel: Optional[RippleControlPanel] = None
        self.prediction_diagnostics: PredictionDiagnosticsView | None = None
        self.prediction_scene: PredictionSceneWindow | None = None
        self.belief_diagnostics: BeliefDiagnosticsView | None = None
        self.attention_scores: np.ndarray | None = None
        self._attention_preview_time: float | None = None
        self._belief_frame: BeliefDiagnosticsFrame | None = None
        self.source_control_binding = RippleSourceControlBinding(self)
        self.auto_color_levels_enabled = True
        self.show_learning_overlay = False
        self.learning_overlay_stride = 8
        self.learning_overlay_threshold = 0.0
        self.learning_overlay_scale = 6.0
        self.engine = RippleEngine(
            resolution=self.resolution,
            plane_size_m=self.plane_size_m,
            n_sources=self.n_sources,
            speed=self.speed,
            damping=damping,
            amplitude=self.amplitude,
            decay_alpha=self.decay_alpha,
            use_gpu=self.use_gpu,
            use_shader=self.use_shader,
            boundary_condition=self.boundary_condition,
            use_external_opengl_context=self.use_shader,
            wave_propagation_enabled=wave_propagation_enabled,
        )
        self.dt = self.engine.dt
        self.source_orchestrator = RippleSourceOrchestrator(
            config=source_orchestrator_config,
            processor=processor,
            engine=self.engine,
            resolution=self.resolution,
            n_sources=self.n_sources,
            camera_capture=camera_capture,
            shared_graph_factory=shared_graph_factory,
            shared_sensory_graph=(
                active_vision if active_vision is not None and active_vision.shared_sensory_graph else None
            ),
        )
        self.source_orchestrator.set_base_amplitude(self.base_amplitude)
        if active_vision is not None:
            self.source_orchestrator.update_pathway_config(spatial_precision=active_vision.precision)
            if not active_vision.shared_sensory_graph:
                self.source_orchestrator.enable_visual_uncertainty(
                    dt=self.timer.interval()/1000,
                    initial_variance=active_vision.initial_variance,
                    process_variance=active_vision.process_variance,
                    memory_time_constant=active_vision.memory_time_constant,
                    audio_precision=active_vision.audio_precision,
                )
            self._gaze_precision = active_vision.precision
        self.visual_prediction = None if self.use_shader or active_vision is not None else self.source_orchestrator.visual_readout.predict(
            self.engine.get_field()
        )

        self.renderer = (
            OpenGLFieldRenderer(
                rgb_canvas_enabled=self.rgb_canvas_enabled,
                auto_level_floor=self.auto_color_floor,
                auto_level_activation_threshold=self.auto_color_activation_threshold,
            ) if self.use_shader else NumpyImageRenderer(
                rgb_canvas_enabled=self.rgb_canvas_enabled,
                auto_level_floor=self.auto_color_floor,
                auto_level_activation_threshold=self.auto_color_activation_threshold,
            )
        )
        self.attention_overlay = None
        if active_vision is not None:
            if isinstance(self.renderer, NumpyImageRenderer):
                self.attention_overlay = AttentionOverlay.on_image(
                    self.renderer.widget, self.renderer.image_item, self.resolution,
                )
            else:
                self.attention_overlay = AttentionOverlay(
                    self.renderer.widget,
                    lambda: fit_image_rect(
                        self.renderer.widget.width(), self.renderer.widget.height(), self.resolution,
                    ),
                    self.resolution,
                )

        if self.use_camera_source:
            self.camera_source.set_enabled(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.renderer.widget)

        if active_vision is not None:
            self.gaze_controls = GazeControls(active_vision.precision, mode=active_vision.mode, compact=True, parent=self)
            if active_vision.shared_sensory_graph:
                self.gaze_controls.use_hard_clamps(active_vision.foveal_radius)
            if not self.source_orchestrator.has_epistemic_scoring:
                self.gaze_controls.autonomous_gaze.setEnabled(False)
                self.gaze_controls.autonomous_gaze.setToolTip(
                    "Autonomous gaze awaits a matching uncertainty model and epistemic objective."
                )
            self.gaze_controls.hide()
            self.gaze_status = QtWidgets.QLabel()
            self.gaze_status.setWordWrap(True)
            senses = "camera only" if active_vision.audio_precision is None else "camera + global audio spectrum"
            self.gaze_description = (
                f"Shared latent canvas: {senses}; memory {active_vision.memory_time_constant:g} s. "
                "Mouse gaze follows the wave image; previews are under Predictions. "
                "Assigned noise and fixed weights; not calibrated confidence, sound localization or dot pursuit."
            )
            if active_vision.shared_sensory_graph:
                graph = self.source_orchestrator.sensory_graph
                self.gaze_description = (
                    f"Generative predictive coding: radius {active_vision.foveal_radius}-pixel circular fovea, "
                    f"{graph.visual_nodes} routed canvas values -> "
                    f"{graph.hidden_nodes} dense recurrent hidden -> {graph.visual_nodes} RGB / {graph.audio_nodes} audio terminals. "
                    "Observe clamps camera pixels strictly inside the disk, uniformly; outside it no camera evidence enters. "
                    "Foveal weights follow gaze; full-image predictions persist outside the current patch. "
                    "Missing or disabled inputs stay free, not zero. "
                    "The same residual energy drives inference, weight learning and canvas error feedback. "
                    "Information gain is local Gauss-Newton about shared hidden nodes with assigned unit residual noise, not calibrated confidence."
                )
            if isinstance(self.source_orchestrator.sensory_graph, ContinuousScenePredictiveCodingGraph):
                graph = self.source_orchestrator.sensory_graph
                self.gaze_description = (
                    f"Continuous scene: {graph.visual_shape[0]} x {graph.visual_shape[1]} world pixels, "
                    f"{graph.hidden_nodes} shared recurrent hidden units; leak time constant "
                    f"{graph.memory_time_constant:g} s. "
                    "The camera frame is the represented world; the circular fovea gates evidence only. "
                    "World-coordinate neural states and learned connections stay in place when gaze moves. "
                    "Every state evolves and the full forecast is decoded each interval. "
                    "No spikes, learned bias drives or frozen peripheral image cache. "
                    "Unforced state decays toward neutral RGB 0.5; a nonzero root can still drive it. "
                    "Autonomous gaze maximizes one-step joint-state information gain from the forecast "
                    "and learned observation precision; scores are local Gaussian estimates, not calibrated confidence. "
                    "No preferred outcomes or temporal covariance history; legacy Gaussian diagnostics stay disabled."
                )
            self.gaze_status.setToolTip(self.gaze_description)
            layout.setStretchFactor(self.renderer.widget, 1)
            self.gaze_controls.precision_changed.connect(self._update_gaze_precision)
            self.gaze_controls.mode_changed.connect(self._change_gaze_mode)
            self._change_gaze_mode()
            self.renderer.widget.setMouseTracking(True)
            if isinstance(self.renderer, NumpyImageRenderer):
                self.renderer.widget.viewport().setMouseTracking(True)
                self.renderer.widget.scene().sigMouseMoved.connect(self._move_image_gaze)
            else:
                self.renderer.widget.installEventFilter(self)

        toolbar = QtWidgets.QHBoxLayout()
        self.controls_button = QtWidgets.QPushButton("Controls")
        self.controls_button.setToolTip("Open observation, sensory inference, physics and display settings.")
        self.controls_button.clicked.connect(self.toggle_controls)
        toolbar.addWidget(self.controls_button)
        self.scene_button = QtWidgets.QPushButton("Predictions")
        self.scene_button.setToolTip("Open camera/prediction comparison, prediction image or 3D graph.")
        self.scene_button.setCheckable(True)
        self.scene_button.toggled.connect(self.set_prediction_scene_visible)
        toolbar.addWidget(self.scene_button)
        self.diagnostics_button = QtWidgets.QPushButton("Diagnostics")
        self.diagnostics_button.setToolTip("Open sensory inference and learning plots.")
        self.diagnostics_button.setCheckable(True)
        self.diagnostics_button.toggled.connect(self.set_model_diagnostics_visible)
        if self.source_orchestrator.has_temporal_beliefs and not self.source_orchestrator.has_gaussian_uncertainty:
            self.diagnostics_button.setEnabled(False)
            self.diagnostics_button.setToolTip(
                "Gaussian diagnostics require a matching uncertainty model. Use Predictions for the live scene."
            )
        toolbar.addWidget(self.diagnostics_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.frame_rate = FrameRateDisplay(self)
        self.renderer.frame_presented.connect(self.frame_rate.record_frame)
        footer = QtWidgets.QHBoxLayout()
        if self.gaze_status is not None:
            footer.addWidget(self.gaze_status, 1)
        footer.addWidget(self.frame_rate)
        layout.addLayout(footer)

    def _change_gaze_mode(self):
        if self.gaze_controls.autonomous_gaze.isChecked() and not self.source_orchestrator.has_epistemic_scoring:
            raise ValueError("Autonomous gaze requires a matching uncertainty model")
        self.source_orchestrator.gaze_policy = (
            GazePolicy() if self.gaze_controls.autonomous_gaze.isChecked() else None
        )
        self._update_gaze_precision()

    def _update_gaze_precision(self):
        self._gaze_precision = self.gaze_controls.gaze_precision(self._gaze_precision)
        self.source_orchestrator.update_pathway_config(
            spatial_precision=self.gaze_controls.observation_precision(self._gaze_precision),
        )
        self._update_gaze_status()
        self._update_attention_overlay()

    def _update_attention_overlay(self):
        if self.attention_overlay is not None:
            precision = self.source_orchestrator.visual_readout.config.spatial_precision
            available = self.camera_source.enabled and self.camera_source.status == "receiving"
            sample_points = ()
            io = self.source_orchestrator.graph_io
            if io is not None and precision is not None:
                _, pixel = io.nearest_pixel(precision.gaze)
                precision = replace(
                    precision,
                    gaze=(pixel[1]/max(1, self.resolution[1]-1), pixel[0]/max(1, self.resolution[0]-1)),
                    sigma=io.radius/min(self.resolution), baseline=0, peak=1,
                )
            self.attention_overlay.set_attention(
                precision, self.attention_scores, available=available, sample_points=sample_points,
                hard_fovea=io is not None,
            )
            if self.prediction_scene is not None:
                self.prediction_scene.image_view.set_attention(
                    self.attention_overlay.precision, self.attention_scores,
                    available=available, sample_points=sample_points,
                    hard_fovea=io is not None,
                    world_anchored=io is not None and io.world_anchored,
                )

    def _update_gaze_status(self):
        mode = "mouse" if self.gaze_controls.follow_mouse.isChecked() else (
            "autonomous" if self.gaze_controls.autonomous_gaze.isChecked() else "full canvas"
        )
        evidence = "observing" if self.gaze_controls.observe.isChecked() else "prediction only"
        camera = self.camera_source.status
        if self.camera_source.error is None or not self.camera_source.enabled:
            camera = f"Camera: {camera}"
        audio = ""
        audio_model = self.source_orchestrator.audio_uncertainty
        if audio_model is not None:
            state = "off" if not self.audio_source.enabled else (
                "prediction only" if audio_model.precision == 0 else (
                    "receiving" if self.audio_source.has_recent_observation() else "waiting"
                )
            )
            audio = f" | Sound: {state} (global spectrum)"
        if self.source_orchestrator.sensory_graph is not None:
            state = "off / unclamped" if not self.audio_source.enabled else (
                "receiving" if self.audio_source.has_recent_observation() else "waiting / unclamped"
            )
            model = (
                "continuous scene beliefs"
                if isinstance(self.source_orchestrator.sensory_graph, ContinuousScenePredictiveCodingGraph)
                else "shared predictive coding"
            )
            audio = f" | Sound: {state} | {model}"
        self.gaze_status.setText(f"Vision: {mode} | {evidence} | {camera}{audio}")

    def _set_mouse_gaze(self, x, y):
        if self.gaze_controls is None or not self.gaze_controls.follow_mouse.isChecked():
            return
        self.gaze_controls.display_gaze((float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))))
        self._update_gaze_precision()

    def _move_image_gaze(self, scene_position):
        image = self.renderer.image_item
        point = image.mapFromScene(scene_position)
        bounds = image.boundingRect()
        if bounds.contains(point):
            self._set_mouse_gaze(
                (point.x()-.5)/max(bounds.width()-1, 1),
                (point.y()-.5)/max(bounds.height()-1, 1),
            )

    def eventFilter(self, watched, event):
        if watched is self.renderer.widget and event.type() == QtCore.QEvent.MouseMove:
            width, height = watched.width(), watched.height()
            rows, cols = self.resolution
            rect = fit_image_rect(width, height, self.resolution)
            scale = rect.width()/cols
            point = event.localPos()
            if scale > 0 and rect.contains(point):
                self._set_mouse_gaze(
                    ((point.x()-rect.left())/scale-.5)/max(cols-1, 1),
                    ((point.y()-rect.top())/scale-.5)/max(rows-1, 1),
                )
        return super().eventFilter(watched, event)

    def _update_speed(self, val: float):
        self.speed = val
        self.dt = self.engine.dt

    def _update_amplitude(self, val: float):
        self.base_amplitude = val
        self.amplitude = val
        self.source_orchestrator.set_base_amplitude(val)

    def _update_decay_alpha(self, val: float):
        self.decay_alpha = val
    
    def _update_damping(self, val: float):
        self.damping = val

    def toggle_controls(self):
        if self.control_panel is None:
            self.control_panel = RippleControlPanel(
                self.engine,
                on_speed_changed=self._update_speed,
                on_amplitude_changed=self._update_amplitude,
                on_decay_alpha_changed=self._update_decay_alpha,
                on_damping_changed=self._update_damping,
                rgb_canvas_enabled=self.rgb_canvas_enabled,
                on_rgb_canvas_changed=self._update_rgb_canvas,
                rgb_canvas_available=callable(
                    getattr(self.renderer, "set_rgb_canvas_enabled", None)
                ),
                auto_color_levels_enabled=self.auto_color_levels_enabled,
                on_auto_color_levels_changed=self._update_auto_color_levels,
                auto_color_activation_threshold=self.auto_color_activation_threshold,
                on_auto_color_activation_threshold_changed=self._update_auto_color_activation_threshold,
                auto_color_floor=self.auto_color_floor,
                on_auto_color_floor_changed=self._update_auto_color_floor,
                source_toggles=self.source_control_binding.build_toggles(),
                source_sections=self.source_control_binding.build_sections(),
                on_source_control_changed=self.source_control_binding.update_control,
                on_source_toggle_changed=self.source_control_binding.update_toggle,
                before_reset=self.renderer.prepare_frame,
                on_reset=self._sync_after_reset,
                gaze_controls=self.gaze_controls,
                gaze_description=self.gaze_description if self.gaze_controls is not None else "",
                parent=self,
                on_belief_diagnostics=(
                    lambda: self.set_model_diagnostics_visible(True)
                ) if self.source_orchestrator.has_temporal_beliefs else None,
            )
            self.control_panel.resize(500, 760)
            if self.source_orchestrator.has_temporal_beliefs and not self.source_orchestrator.has_gaussian_uncertainty:
                self.control_panel.beliefs_button.setEnabled(False)
                self.control_panel.beliefs_button.setToolTip(self.diagnostics_button.toolTip())
        self.control_panel.show()
        self.control_panel.raise_()
        self.control_panel.activateWindow()

    def _sync_after_reset(self) -> None:
        self.time = self.engine.time
        self._gaze_last_prediction_time = None
        self._attention_preview_time = None
        self.attention_scores = None
        self._belief_frame = None
        self.visual_prediction = None
        self._update_attention_overlay()
        if self.belief_diagnostics is not None:
            self.belief_diagnostics.clear()
        self.source_orchestrator.reset_readouts()
        if self.prediction_scene is not None:
            self.prediction_scene.clear()
        if self.prediction_diagnostics is not None:
            self.prediction_diagnostics.clear()
            if self.prediction_diagnostics.audio_view is not None:
                self.prediction_diagnostics.audio_view.clear()
        reset_view = getattr(self.renderer, "reset_view", None)
        if callable(reset_view):
            reset_view()
        self.visual_prediction = (
            None if self.use_shader or self.source_orchestrator.has_temporal_beliefs
            else self.source_orchestrator.visual_readout.predict(self.engine.get_field())
        )
        if self.renderer.prepare_frame():
            self._render_canvas()

    def set_model_diagnostics_visible(self, visible: bool) -> None:
        if visible and self.source_orchestrator.has_temporal_beliefs and not self.source_orchestrator.has_gaussian_uncertainty:
            raise RuntimeError("Gaussian diagnostics require a matching uncertainty model")
        if not visible:
            if self.belief_diagnostics is not None:
                self.belief_diagnostics.hide()
            self.set_prediction_diagnostics_visible(False)
        elif not self.source_orchestrator.has_temporal_beliefs:
            self.set_prediction_diagnostics_visible(True)
        else:
            if self.belief_diagnostics is None:
                self.belief_diagnostics = BeliefDiagnosticsView(self._get_belief_frame, self)
                self.belief_diagnostics.visibility_changed.connect(self._belief_visibility_changed)
                self.belief_diagnostics.open_energy_requested.connect(
                    lambda: self.set_prediction_diagnostics_visible(True)
                )
            self.belief_diagnostics.show()
            self.belief_diagnostics.raise_()

    def _belief_visibility_changed(self, visible: bool) -> None:
        if not visible:
            self._belief_frame = None
        self._set_diagnostics_collection(visible)

    def _get_belief_frame(self) -> BeliefDiagnosticsFrame | None:
        self._refresh_attention_preview()
        return self._belief_frame

    def _refresh_attention_preview(self) -> None:
        orchestrator = self.source_orchestrator
        if isinstance(orchestrator.sensory_graph, ContinuousScenePredictiveCodingGraph):
            # Display only scores computed from the pre-evidence forecast; never rescore a posterior.
            self.attention_scores = orchestrator.gaze_scores
            self._update_attention_overlay()
            return
        if not orchestrator.has_gaussian_uncertainty or self.visual_prediction is None:
            return
        beliefs_visible = (
            self.belief_diagnostics is not None
            and self.belief_diagnostics.isVisible()
            and not self.belief_diagnostics.isMinimized()
        )
        image_visible = (
            self.prediction_scene is not None and self.prediction_scene.preview_active()
            and not self.prediction_scene.needs_hidden_preview
        )
        if not (beliefs_visible or image_visible or (self.isVisible() and not self.isMinimized())):
            return
        now = time.monotonic()
        if self._attention_preview_time is not None and now-self._attention_preview_time < .1:
            return
        if not self.renderer.prepare_frame():
            return
        precision = orchestrator.visual_readout.config.spatial_precision
        if precision is None:
            precision = GaussianSpatialPrecision(baseline=1, peak=0)
        self.attention_scores = (
            orchestrator.gaze_scores
            if orchestrator.gaze_scores is not None else
            orchestrator.visual_information_gain(replace(precision, sample=True))
        )
        self._attention_preview_time = now
        self._update_attention_overlay()
        if beliefs_visible:
            if orchestrator.sensory_graph is not None:
                self._belief_frame = self._shared_belief_frame(precision)
                return
            model = orchestrator.visual_uncertainty
            fields = ("mean", "variance", "sensory_mean", "sensory_covariance")
            visual = model.belief_snapshot(precision.gaze)
            audio = (
                orchestrator.audio_uncertainty.belief_snapshot()
                if orchestrator.audio_uncertainty is not None else None
            )
            self._belief_frame = BeliefDiagnosticsFrame(
                scores=self.attention_scores, gaze=precision.gaze, dt=orchestrator._uncertainty_dt,
                sampling=precision.sample,
                visual={key: np.asarray(visual[key]) for key in fields},
                audio={key: np.asarray(audio[key]) for key in fields} if audio is not None else None,
                visual_location=f"Vision pixel {visual['pixel']} ({visual['stage']})",
                audio_gain=(
                    orchestrator.audio_uncertainty.expected_information_gain(
                        prior=self.engine.get_prior(), dt=orchestrator._uncertainty_dt,
                    ) if orchestrator.audio_uncertainty is not None else None
                ),
            )

    def _shared_belief_frame(self, precision):
        orchestrator = self.source_orchestrator
        graph = orchestrator.sensory_graph
        joint = graph.belief_snapshot()
        hidden = slice(*joint["hidden_slice"])
        mean, covariance = joint["mean"], joint["hidden_covariance"]
        common = dict(
            mean=mean[hidden], variance=joint["variance"][hidden],
            covariance=covariance,
        )
        (row, col), pixel = orchestrator.graph_io.nearest_pixel(precision.gaze)
        start = (row*graph.visual_shape[1]+col)*3
        visual_nodes = slice(start, start+3)
        return BeliefDiagnosticsFrame(
            scores=self.attention_scores, gaze=precision.gaze,
            dt=orchestrator._uncertainty_dt, sampling=precision.sample,
            visual=dict(common, sensory_mean=mean[visual_nodes], sensory_covariance=graph.sensory_covariance(visual_nodes)),
            audio=dict(common, sensory_mean=mean[graph.audio_slice], sensory_covariance=graph.sensory_covariance(graph.audio_slice)),
            visual_location=f"Foveal center pixel {pixel} ({joint['stage']})",
            audio_gain=graph.information_gain(np.arange(graph.audio_slice.start, graph.audio_slice.stop)),
            joint=joint,
        )

    def set_prediction_diagnostics_visible(self, visible: bool) -> None:
        if visible and self.source_orchestrator.sensory_graph is not None:
            self.set_model_diagnostics_visible(True)
            return
        if visible:
            if self.prediction_diagnostics is None:
                self.prediction_diagnostics = PredictionDiagnosticsView(self)
                self.prediction_diagnostics.visibility_changed.connect(
                    self._set_diagnostics_collection
                )
                self.prediction_diagnostics.open_scene_requested.connect(
                    lambda: self.set_prediction_scene_visible(True)
                )
                if self.source_orchestrator.audio_readout is not None:
                    self.prediction_diagnostics.add_audio_tab()
            self.prediction_diagnostics.show()
            self.prediction_diagnostics.raise_()
        elif self.prediction_diagnostics is not None:
            self.prediction_diagnostics.hide()

    def set_prediction_scene_visible(self, visible: bool) -> None:
        if visible:
            if self.prediction_scene is None:
                self.prediction_scene = PredictionSceneWindow(self)
                if self.source_orchestrator.sensory_graph is not None:
                    selector = self.prediction_scene.layout_selector
                    selector.setCurrentIndex(selector.findData("comparison"))
                    continuous = isinstance(
                        self.source_orchestrator.sensory_graph, ContinuousScenePredictiveCodingGraph,
                    )
                    if continuous:
                        self.prediction_scene.configure_continuous_graph()
                    else:
                        selector.model().item(selector.findData("3d")).setEnabled(False)
                        selector.setToolTip("Shared graph nodes and weights are shown in Diagnostics / Joint Graph.")
                    if not self.source_orchestrator.has_gaussian_uncertainty:
                        self.prediction_scene.catalogue_button.setEnabled(False)
                        self.prediction_scene.catalogue_button.setToolTip(self.diagnostics_button.toolTip())
                self.prediction_scene.visibility_changed.connect(self._set_diagnostics_collection)
                self.prediction_scene.open_catalogue_requested.connect(
                    lambda: self.set_prediction_diagnostics_visible(True)
                )
            self.prediction_scene.show()
            self.prediction_scene.raise_()
        elif self.prediction_scene is not None:
            self.prediction_scene.hide()

    def _set_diagnostics_collection(self, _enabled: bool) -> None:
        plots_visible = self.prediction_diagnostics is not None and self.prediction_diagnostics.isVisible()
        beliefs_visible = self.belief_diagnostics is not None and self.belief_diagnostics.isVisible()
        scene_active = self.prediction_scene is not None and self.prediction_scene.preview_active()
        self.source_orchestrator.set_diagnostics_enabled(
            plots_visible or (
                scene_active and self.prediction_scene.needs_hidden_preview
                and not self.prediction_scene.continuous_graph
            )
        )
        with QtCore.QSignalBlocker(self.diagnostics_button):
            self.diagnostics_button.setChecked(plots_visible or beliefs_visible)
        with QtCore.QSignalBlocker(self.scene_button):
            self.scene_button.setChecked(self.prediction_scene is not None and self.prediction_scene.isVisible())

    def _update_auto_color_levels(self, enabled: bool) -> None:
        self.auto_color_levels_enabled = bool(enabled)
        set_auto_levels = getattr(self.renderer, "set_auto_percentile_levels", None)
        if callable(set_auto_levels):
            set_auto_levels(self.auto_color_levels_enabled)

    def _update_rgb_canvas(self, enabled: bool) -> None:
        self.rgb_canvas_enabled = bool(enabled)
        setter = getattr(self.renderer, "set_rgb_canvas_enabled", None)
        if callable(setter):
            setter(self.rgb_canvas_enabled)

    def _update_auto_color_activation_threshold(self, val: float) -> None:
        self.auto_color_activation_threshold = float(val)
        setter = getattr(self.renderer, "set_auto_level_activation_threshold", None)
        if callable(setter):
            setter(self.auto_color_activation_threshold)

    def _update_auto_color_floor(self, val: float) -> None:
        self.auto_color_floor = float(val)
        set_auto_level_floor = getattr(self.renderer, "set_auto_level_floor", None)
        if callable(set_auto_level_floor):
            set_auto_level_floor(self.auto_color_floor)

    def update_visualization(self):
        source_frame = self.source_orchestrator.resolve()
        self.engine.amplitude = source_frame.amplitude

        if not self.renderer.prepare_frame():
            return

        self._advance_canvas(source_frame.drive_grid)
        self.time = self.engine.time
        self._render_scene()
        self.source_control_binding.sync_audio_panel(
            self.control_panel,
            source_frame.audio_frequencies,
        )

    def _advance_canvas(self, drive_grid: np.ndarray | None) -> None:
        readout = self.source_orchestrator.visual_readout
        scene_requested = self.prediction_scene is not None and self.prediction_scene.take_spatial_preview_request()
        plots_visible = self.prediction_diagnostics is not None and self.prediction_diagnostics.isVisible()
        self.source_orchestrator.set_diagnostics_enabled(
            plots_visible or (
                scene_requested and self.prediction_scene.needs_hidden_preview
                and not self.prediction_scene.continuous_graph
            )
        )
        audio_view = (
            self.prediction_diagnostics.audio_view
            if self.prediction_diagnostics is not None else None
        )
        audio_readout = self.source_orchestrator.audio_readout
        self.engine.propagate(drive_grid)
        if self.active_vision_config is not None:
            now = time.monotonic()
            elapsed = (
                self.forecast_dt if self.forecast_dt is not None else
                self.timer.interval()/1000 if self._gaze_last_prediction_time is None
                else now-self._gaze_last_prediction_time
            )
            self.visual_prediction = self.source_orchestrator.predict_visual(dt=elapsed)
            self._gaze_last_prediction_time = now
        else:
            self.visual_prediction = self.source_orchestrator.predict_visual()
        self.source_orchestrator.correct_from_observations()
        self.source_control_binding.sync_camera_panel(self.control_panel)
        if self.gaze_controls is not None:
            self._update_gaze_status()
        if self.gaze_controls is not None and self.gaze_controls.autonomous_gaze.isChecked():
            self._gaze_precision = self.source_orchestrator.visual_readout.config.spatial_precision
            self.gaze_controls.display_gaze(self._gaze_precision.gaze)
        if self.attention_overlay is not None:
            self._refresh_attention_preview()
            self._update_attention_overlay()
        if self.prediction_scene is not None:
            if self.source_orchestrator.visual_observation is None:
                self.prediction_scene.invalidate_evidence("camera")
            if not self.audio_source.has_recent_observation():
                self.prediction_scene.invalidate_evidence("audio")
            if scene_requested:
                canvas_preview = (
                    self.engine.get_field_preview()
                    if self.prediction_scene.needs_hidden_preview
                    else self.engine.get_field_preview(max_side=self.prediction_scene.preview_max_side)
                )
                self.prediction_scene.record(
                    canvas_preview, self._capture_scene_sources(drive_grid),
                    graph_snapshot=(
                        self._capture_continuous_graph_preview()
                        if self.prediction_scene.continuous_graph and self.prediction_scene.needs_hidden_preview
                        else None
                    ),
                )
        if (
            self.prediction_diagnostics is not None
            and self.prediction_diagnostics.isVisible()
        ):
            snapshot = readout.diagnostics
            self.prediction_diagnostics.record(
                self.engine.time,
                snapshot,
            )
            if audio_view is not None and audio_readout is not None:
                audio_snapshot = audio_readout.diagnostics
                if audio_snapshot is not None:
                    audio_view.record(
                        self.engine.time, audio_snapshot,
                    )
                elif not self.source_orchestrator.audio_source.has_recent_observation():
                    audio_view.record(self.engine.time, None)

    def _capture_continuous_graph_preview(self):
        source = self.source_orchestrator
        return continuous_graph_preview(
            source.sensory_graph, source.visual_observation,
            foveal_mask=source.graph_io.observation_mask,
        )

    def _capture_scene_sources(self, drive_grid: np.ndarray | None) -> tuple[SourceScenePreview, ...]:
        orchestrator = self.source_orchestrator
        max_side = self.prediction_scene.preview_max_side
        sources = []
        for source, readout, prediction, observation, enabled in (
            (self.camera_source, orchestrator.visual_readout, self.visual_prediction,
             orchestrator.visual_observation,
             self.camera_source.enabled),
            (self.audio_source, orchestrator.audio_readout, orchestrator.audio_prediction,
             self.audio_source.recent_observation(), self.audio_source.enabled),
        ):
            mapping = source.scene_mapping
            has_stream_state_factor = readout is not None and not orchestrator.has_temporal_beliefs
            sources.append(SourceScenePreview(
                mapping=mapping, enabled=enabled,
                observation=mapping.sample(observation, max_side=max_side) if observation is not None else None,
                prediction=mapping.sample(prediction, max_side=max_side) if prediction is not None else None,
                hidden=(
                    spatial_preview(readout.hidden)
                    if readout is not None and self.prediction_scene.needs_hidden_preview
                    and not self.prediction_scene.continuous_graph else None
                ),
                hidden_energies=(
                    readout.diagnostics.hidden.channel_means
                    if readout is not None and readout.diagnostics is not None else ()
                ),
                recurrent_parent=(
                    "audio" if mapping.key == "camera" else "camera"
                ) if orchestrator.visual_readout.config.cross_modal_enabled else None,
                stream_state=source.enabled if has_stream_state_factor else None,
                stream_state_prediction=readout.stream_state_prediction if has_stream_state_factor else None,
            ))
        sources.append(SourceScenePreview(
            mapping=self.synthetic_source.scene_mapping,
            enabled=self.synthetic_source.enabled,
            observation=(
                self.synthetic_source.scene_mapping.sample(drive_grid)
                if drive_grid is not None and self.synthetic_source.enabled else None
            ),
        ))
        return tuple(sources)

    def _render_scene(self) -> None:
        self._render_canvas()
        render_learning_overlay = getattr(
            self.renderer,
            "render_prediction_coupling_overlay",
            None,
        )
        if callable(render_learning_overlay):
            render_learning_overlay(
                self.engine,
                enabled=self.show_learning_overlay,
                stride=self.learning_overlay_stride,
                threshold=self.learning_overlay_threshold,
                scale=self.learning_overlay_scale,
            )

    def _render_canvas(self) -> None:
        if self.use_shader:
            self.renderer.render(self.engine)
            return
        self.renderer.render(self.engine.get_field())

    def closeEvent(self, event):
        if self.belief_diagnostics is not None:
            self.belief_diagnostics.close()
        if self.control_panel is not None:
            self.control_panel.close()
        if self.prediction_scene is not None:
            self.prediction_scene.close()
        if self.prediction_diagnostics is not None:
            self.prediction_diagnostics.close()
        self.close_camera_source()
        if self.use_shader and self.renderer.prepare_frame():
            self.engine.propagator.close()
            self.renderer.widget.cleanup()
        super().closeEvent(event)

    def set_learning_overlay_enabled(self, enabled: bool) -> None:
        self.show_learning_overlay = bool(enabled)

    def set_learning_overlay_stride(self, stride: int) -> None:
        self.learning_overlay_stride = max(int(stride), 1)

    def set_learning_overlay_threshold(self, threshold: float) -> None:
        self.learning_overlay_threshold = max(float(threshold), 0.0)

    def set_learning_overlay_scale(self, scale: float) -> None:
        self.learning_overlay_scale = max(float(scale), 0.1)

    def close_camera_source(self) -> None:
        self.camera_source.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
