import time
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from audioviz.engine import RippleEngine
from audioviz.physics import BoundaryCondition
from audioviz.sources import RippleSourceOrchestrator, RippleSourceOrchestratorConfig
from audioviz.sources.pose import (
    PoseFrameSource,
    PoseGraphExtractor,
    PoseGraphState,
    RipplePoseConfig,
    build_pose_graph_segmentation_mask,
    centered_field_rect,
    map_pose_coords_to_field_positions,
    map_pose_segmentation_to_field_mask,
    pose_coords_in_image_support,
)
from audioviz.visualization.ripple_renderers import (
    NumpyImageRenderer,
    OpenGLFieldRenderer,
)
from audioviz.visualization.pose_debug_view import PoseDebugView
from audioviz.visualization.prediction_diagnostics_view import PredictionDiagnosticsView
from audioviz.visualization.prediction_scene_window import PredictionSceneWindow
from audioviz.utils.source_preview import SourceScenePreview
from audioviz.utils.spatial_preview import PREVIEW_MAX_SIDE, spatial_preview
from audioviz.visualization.standing_body_renderer import (
    StandingBodyRenderer,
    lookup_table_from_renderer,
)
from audioviz.ui.ripple_control_panel import (
    RippleControlPanel,
)
from audioviz.ui.ripple_source_controls import (
    RippleSourceControlBinding,
)
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor

POSE_RENDER_MODE_OVERLAY = "overlay"
POSE_RENDER_MODE_STANDING_BODY = "standing-body"
SUPPORTED_POSE_RENDER_MODES = (
    POSE_RENDER_MODE_OVERLAY,
    POSE_RENDER_MODE_STANDING_BODY,
)


def normalize_pose_render_mode(mode: str | None) -> str:
    resolved = (
        POSE_RENDER_MODE_OVERLAY
        if mode is None
        else str(mode).strip().lower()
    )
    if resolved not in SUPPORTED_POSE_RENDER_MODES:
        raise ValueError(
            "Unsupported pose_render_mode "
            f"{mode!r}. Expected one of {SUPPORTED_POSE_RENDER_MODES}."
        )
    return resolved


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
    def use_pose_sources(self) -> bool:
        return self.pose_source.enabled

    @use_pose_sources.setter
    def use_pose_sources(self, enabled: bool) -> None:
        self.pose_source.enabled = bool(enabled)

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
                 pose_config: RipplePoseConfig | None = None,
                 pose_extractor: PoseGraphExtractor | None = None,
                 pose_capture=None,
                 camera_capture=None,
                 **kwargs):
        if source_orchestrator_config is None:
            source_orchestrator_config = RippleSourceOrchestratorConfig()
        if pose_config is None:
            pose_config = RipplePoseConfig()
        super().__init__(processor, **kwargs)

        self.processor = processor
        self.n_sources = n_sources
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.boundary_condition = boundary_condition
        self.pose_source = PoseFrameSource(
            model_path=pose_config.model_path,
            camera_index=pose_config.camera_index,
            enabled=bool(pose_config.enabled),
            extractor=pose_extractor,
            capture=pose_capture,
        )
        self.standing_body_renderer = StandingBodyRenderer()
        self.rgb_canvas_enabled = bool(rgb_canvas_enabled)
        self.auto_color_activation_threshold = float(auto_color_activation_threshold)
        self.auto_color_floor = float(auto_color_floor)
        if self.use_pose_sources and (self.use_gpu or self.use_shader):
            raise NotImplementedError(
                "Pose-medium coupling currently requires the CPU ripple backend."
            )
        if source_orchestrator_config.camera_frame.enabled and self.use_shader:
            raise NotImplementedError(
                "Camera-frame source currently requires the CPU/GPU ripple backend."
            )

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
        self.source_control_binding = RippleSourceControlBinding(self)
        self.pose_graph_stiffness = pose_config.graph_stiffness
        self.pose_render_mode = normalize_pose_render_mode(pose_config.render_mode)
        self.pose_debug_view = pose_config.debug_view
        self.pose_debug_frame_count = 0
        self.auto_color_levels_enabled = True
        self.show_learning_overlay = False
        self.learning_overlay_stride = 8
        self.learning_overlay_threshold = 0.0
        self.learning_overlay_scale = 6.0
        self.body_boundary_transmission = float(pose_config.body_boundary_transmission)
        self.body_boundary_dissipation = float(pose_config.body_boundary_dissipation)
        self.pose_field_rect = centered_field_rect(
            self.resolution,
            width_fraction=pose_config.field_width_fraction,
            height_fraction=pose_config.field_height_fraction,
        )
        self.pose_state: PoseGraphState | None = None
        self.pose_last_update_time = time.monotonic()
        self._latest_pose_coords = np.zeros((0, 2), dtype=np.float32)
        self._latest_pose_adjacency = np.zeros((0, 0), dtype=np.float32)
        self._latest_pose_segmentation_mask: np.ndarray | None = None
        self._latest_pose_render_positions = np.zeros((0, 2), dtype=np.float32)
        self._latest_pose_valid = np.zeros((0,), dtype=bool)
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
            body_boundary_transmission=self.body_boundary_transmission,
            body_boundary_dissipation=self.body_boundary_dissipation,
            use_external_opengl_context=self.use_shader,
        )
        self.dt = self.engine.dt
        self.source_orchestrator = RippleSourceOrchestrator(
            config=source_orchestrator_config,
            processor=processor,
            engine=self.engine,
            resolution=self.resolution,
            n_sources=self.n_sources,
            camera_capture=camera_capture,
        )
        self.source_orchestrator.set_base_amplitude(self.base_amplitude)
        self.visual_prediction = self.source_orchestrator.visual_readout.predict(
            self.engine.get_field_numpy()
        )

        self.renderer = (
            OpenGLFieldRenderer() if self.use_shader else NumpyImageRenderer(
                rgb_canvas_enabled=self.rgb_canvas_enabled,
                auto_level_floor=self.auto_color_floor,
                auto_level_activation_threshold=self.auto_color_activation_threshold,
            )
        )
        self.pose_debug_widget = None
        self.pose_debug_viewer: PoseDebugView | None = None

        if self.use_pose_sources:
            self._set_pose_sources_enabled(True)
        if self.use_camera_source:
            self.camera_source.set_enabled(True)

        layout = QtWidgets.QVBoxLayout(self)
        if self.pose_debug_view:
            content = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            content.addWidget(self.renderer.widget)
            content.addWidget(self._create_pose_debug_widget())
            content.setStretchFactor(0, 3)
            content.setStretchFactor(1, 2)
            layout.addWidget(content)
        else:
            layout.addWidget(self.renderer.widget)

        controls_button = QtWidgets.QPushButton("Show Controls")
        controls_button.clicked.connect(self.toggle_controls)
        layout.addWidget(controls_button)
        self.diagnostics_button = QtWidgets.QPushButton("Show Inference / Learning Plots")
        self.diagnostics_button.setCheckable(True)
        self.diagnostics_button.setEnabled(not self.use_shader)
        self.diagnostics_button.toggled.connect(self.set_prediction_diagnostics_visible)
        layout.addWidget(self.diagnostics_button)
        self.scene_button = QtWidgets.QPushButton("Show Multimodal 3D Scene")
        self.scene_button.setCheckable(True)
        self.scene_button.setEnabled(not self.use_shader)
        self.scene_button.toggled.connect(self.set_prediction_scene_visible)
        layout.addWidget(self.scene_button)

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
                on_boundary_transmission_changed=self._update_boundary_transmission,
                on_boundary_dissipation_changed=self._update_boundary_dissipation,
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
            )
            self.control_panel.resize(360, 520)
        self.control_panel.show()
        self.control_panel.raise_()
        self.control_panel.activateWindow()

    def _sync_after_reset(self) -> None:
        self.time = self.engine.time
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
        self.visual_prediction = self.source_orchestrator.visual_readout.predict(
            self.engine.get_field_numpy()
        )
        if self.renderer.prepare_frame():
            self._render_canvas()

    def set_prediction_diagnostics_visible(self, visible: bool) -> None:
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
            if self.use_shader:
                raise NotImplementedError("The multimodal 3D scene requires an array-backed CPU/GPU canvas.")
            if self.prediction_scene is None:
                self.prediction_scene = PredictionSceneWindow(self)
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
        scene_active = self.prediction_scene is not None and self.prediction_scene.preview_active()
        self.source_orchestrator.set_diagnostics_enabled(plots_visible or scene_active)
        with QtCore.QSignalBlocker(self.diagnostics_button):
            self.diagnostics_button.setChecked(plots_visible)
        with QtCore.QSignalBlocker(self.scene_button):
            self.scene_button.setChecked(self.prediction_scene is not None and self.prediction_scene.isVisible())

    def _update_boundary_transmission(self, val: float) -> None:
        self.body_boundary_transmission = float(val)
        self.engine.set_body_boundary_transmission(val)

    def _update_boundary_dissipation(self, val: float) -> None:
        self.body_boundary_dissipation = float(val)
        self.engine.set_body_boundary_dissipation(val)

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

        if self.use_pose_sources:
            self._update_pose_visualization(source_frame.drive_grid)
            return

        if not self.renderer.prepare_frame():
            return

        self._advance_canvas(source_frame.drive_grid)
        self.time = self.engine.time
        self._render_scene()
        self.source_control_binding.sync_audio_panel(
            self.control_panel,
            source_frame.audio_frequencies,
        )

    def _set_pose_sources_enabled(self, enabled: bool) -> None:
        if enabled:
            if self.use_gpu or self.use_shader:
                raise NotImplementedError(
                    "Pose-medium coupling currently requires the CPU ripple backend."
                )
            self.pose_source.set_enabled(True)
            self.pose_last_update_time = time.monotonic()
        else:
            self.pose_source.set_enabled(False)
        self.pose_state = None
        self._clear_pose_runtime_state()
        self.engine.set_body_boundary_mask(None)

    def _clear_pose_runtime_state(self) -> None:
        self._latest_pose_coords = np.zeros((0, 2), dtype=np.float32)
        self._latest_pose_adjacency = np.zeros((0, 0), dtype=np.float32)
        self._latest_pose_segmentation_mask = None
        self._latest_pose_render_positions = np.zeros((0, 2), dtype=np.float32)
        self._latest_pose_valid = np.zeros((0,), dtype=bool)
        if self.pose_state is not None:
            self.pose_state.set_ripple_states(
                np.zeros(self.pose_state.num_nodes, dtype=np.float32)
            )

    def _map_pose_segmentation_to_render_mask(
        self,
        segmentation_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        if segmentation_mask is None:
            return None
        if self.pose_render_mode in (
            POSE_RENDER_MODE_OVERLAY,
            POSE_RENDER_MODE_STANDING_BODY,
        ):
            return map_pose_segmentation_to_field_mask(
                segmentation_mask,
                self.resolution,
                field_rect=self.pose_field_rect,
            )
        raise AssertionError(f"Unhandled pose_render_mode {self.pose_render_mode!r}")

    def _map_pose_positions_to_render_positions(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        if self.pose_render_mode in (
            POSE_RENDER_MODE_OVERLAY,
            POSE_RENDER_MODE_STANDING_BODY,
        ):
            return map_pose_coords_to_field_positions(
                positions,
                self.resolution,
                field_rect=self.pose_field_rect,
            )
        raise AssertionError(f"Unhandled pose_render_mode {self.pose_render_mode!r}")

    def _update_pose_visualization(
        self,
        drive_grid: np.ndarray | None,
    ) -> None:
        sample = self.pose_source.read()
        if sample is None:
            if self.prediction_scene is not None:
                self.prediction_scene.invalidate_evidence("pose")
            return
        frame, pose = sample
        segmentation_mask = self._resolve_pose_segmentation_mask(frame, pose)
        self._latest_pose_coords = np.asarray(pose.coords, dtype=np.float32).copy()
        self._latest_pose_adjacency = np.asarray(pose.adjacency, dtype=np.float32).copy()
        self._latest_pose_segmentation_mask = (
            None
            if segmentation_mask is None
            else np.asarray(segmentation_mask, dtype=np.float32).copy()
        )
        self.engine.set_body_boundary_mask(
            self._map_pose_segmentation_to_render_mask(segmentation_mask)
        )
        if self.pose_debug_view:
            self._update_pose_debug_view(frame, pose, segmentation_mask=segmentation_mask)
        if not pose.coords.size:
            self._latest_pose_render_positions = np.zeros((0, 2), dtype=np.float32)
            self._latest_pose_valid = np.zeros((0,), dtype=bool)
            self._step_engine_for_pose_mode(drive_grid)
            if self.pose_state is not None:
                self.pose_state.set_ripple_states(
                    np.zeros(self.pose_state.num_nodes, dtype=np.float32)
                )
                self.time = self.engine.time
                self._render_scene()
            return

        now = time.monotonic()
        dt = max(now - self.pose_last_update_time, 1e-6)
        self.pose_last_update_time = now

        if self.pose_state is None or self.pose_state.num_nodes != len(pose.coords):
            self.pose_state = PoseGraphState(
                len(pose.coords),
                pose.adjacency,
                velocity_smoothing_alpha=0.8,
            )
        self.pose_state.update(pose.coords, dt)

        positions = self.pose_state.get_positions()
        valid = pose_coords_in_image_support(positions)
        mapped_positions = self._map_pose_positions_to_render_positions(positions)
        self._latest_pose_render_positions = mapped_positions.copy()
        self._latest_pose_valid = valid.copy()
        self._step_engine_for_pose_mode(drive_grid)
        self._update_pose_ripple_states(mapped_positions, valid)
        self.time = self.engine.time
        self._render_scene()

    def _step_engine_for_pose_mode(
        self,
        drive_grid: np.ndarray | None,
    ) -> None:
        if not self.renderer.prepare_frame():
            return
        self._advance_canvas(drive_grid)

    def _advance_canvas(self, drive_grid: np.ndarray | None) -> None:
        readout = self.source_orchestrator.visual_readout
        scene_requested = self.prediction_scene is not None and self.prediction_scene.take_spatial_preview_request()
        audio_view = (
            self.prediction_diagnostics.audio_view
            if self.prediction_diagnostics is not None else None
        )
        audio_readout = self.source_orchestrator.audio_readout
        self.engine.propagate(drive_grid)
        self.visual_prediction = self.source_orchestrator.predict_visual()
        self.source_orchestrator.correct_from_observations()
        if self.prediction_scene is not None:
            if self.source_orchestrator.visual_observation is None:
                self.prediction_scene.invalidate_evidence("camera")
            if not self.audio_source.has_recent_observation():
                self.prediction_scene.invalidate_evidence("audio")
            if scene_requested:
                self.prediction_scene.record(
                    self.engine.get_field_preview(), self._capture_scene_sources(drive_grid)
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

    def _capture_scene_sources(self, drive_grid: np.ndarray | None) -> tuple[SourceScenePreview, ...]:
        orchestrator = self.source_orchestrator
        sources = []
        for source, readout, prediction, observation, enabled in (
            (self.camera_source, orchestrator.visual_readout, self.visual_prediction,
             orchestrator.visual_observation,
             self.camera_source.enabled and self.prediction_error_transform.applies_to("camera_frame")),
            (self.audio_source, orchestrator.audio_readout, orchestrator.audio_prediction,
             self.audio_source.recent_observation(), self.audio_source.enabled),
        ):
            mapping = source.scene_mapping
            sources.append(SourceScenePreview(
                mapping=mapping, enabled=enabled,
                observation=mapping.sample(observation) if observation is not None else None,
                prediction=mapping.sample(prediction) if prediction is not None else None,
                hidden=spatial_preview(readout.hidden) if readout is not None else None,
                hidden_energies=(
                    readout.diagnostics.hidden.channel_means
                    if readout is not None and readout.diagnostics is not None else ()
                ),
                recurrent_parent=(
                    "audio" if mapping.key == "camera" else "camera"
                ) if orchestrator.visual_readout.config.cross_modal_enabled else None,
            ))
        sources.append(SourceScenePreview(
            mapping=self.synthetic_source.scene_mapping,
            enabled=self.synthetic_source.enabled,
            observation=(
                self.synthetic_source.scene_mapping.sample(drive_grid)
                if drive_grid is not None and self.synthetic_source.enabled else None
            ),
        ))
        positions, edges = None, None
        if self.pose_source.enabled and np.any(self._latest_pose_valid):
            indices = np.flatnonzero(self._latest_pose_valid)[:PREVIEW_MAX_SIDE]
            positions = self._latest_pose_render_positions[indices] / np.array(
                [max(1, self.resolution[1] - 1), max(1, self.resolution[0] - 1)]
            )
            positions = self.pose_source.scene_mapping.sample(positions)
            edges = np.argwhere(np.triu(self._latest_pose_adjacency[np.ix_(indices, indices)], 1) > 0)
            edges.setflags(write=False)
        sources.append(SourceScenePreview(
            mapping=self.pose_source.scene_mapping, enabled=self.pose_source.enabled,
            observation=positions, edges=edges,
        ))
        return tuple(sources)

    def _update_pose_ripple_states(
        self,
        mapped_positions: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if self.pose_state is None:
            return
        ripple_states = np.zeros(self.pose_state.num_nodes, dtype=np.float32)
        if mapped_positions.size and np.any(valid):
            field = self.engine.get_field_numpy()
            cols = field.shape[1]
            rows = field.shape[0]
            valid_indices = np.flatnonzero(valid)
            sampled_positions = mapped_positions[valid]
            sample_cols = np.clip(
                np.rint(sampled_positions[:, 0]).astype(np.int32),
                0,
                cols - 1,
            )
            sample_rows = np.clip(
                np.rint(sampled_positions[:, 1]).astype(np.int32),
                0,
                rows - 1,
            )
            ripple_states[valid_indices] = np.mean(
                field[sample_rows, sample_cols],
                axis=1,
            ).astype(
                np.float32,
                copy=False,
            )
        self.pose_state.set_ripple_states(ripple_states)

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
        render_rgb_frame = getattr(self.renderer, "render_rgb_frame", None)
        if (
            self.use_pose_sources
            and self.pose_render_mode == POSE_RENDER_MODE_STANDING_BODY
            and callable(render_rgb_frame)
        ):
            render_rgb_frame(self._render_standing_body_rgb_frame())

    def _render_standing_body_rgb_frame(self) -> np.ndarray:
        return self.standing_body_renderer.render(
            field=self.engine.get_field_numpy(),
            lookup_table=lookup_table_from_renderer(self.renderer),
            pose_coords=self._latest_pose_coords,
            pose_adjacency=self._latest_pose_adjacency,
            segmentation_mask=self._latest_pose_segmentation_mask,
        )

    def _render_canvas(self) -> None:
        if self.use_shader:
            self.renderer.render(self.engine)
            return
        self.renderer.render(self.engine.get_field_numpy())

    def closeEvent(self, event):
        if self.prediction_scene is not None:
            self.prediction_scene.close()
        if self.prediction_diagnostics is not None:
            self.prediction_diagnostics.close()
        self.close_pose_sources()
        self.close_camera_source()
        super().closeEvent(event)

    def set_learning_overlay_enabled(self, enabled: bool) -> None:
        self.show_learning_overlay = bool(enabled)

    def set_learning_overlay_stride(self, stride: int) -> None:
        self.learning_overlay_stride = max(int(stride), 1)

    def set_learning_overlay_threshold(self, threshold: float) -> None:
        self.learning_overlay_threshold = max(float(threshold), 0.0)

    def set_learning_overlay_scale(self, scale: float) -> None:
        self.learning_overlay_scale = max(float(scale), 0.1)

    def close_pose_sources(self) -> None:
        self.pose_source.close()

    def close_camera_source(self) -> None:
        self.camera_source.close()

    def _create_pose_debug_widget(self):
        self.pose_debug_viewer = PoseDebugView()
        self.pose_debug_widget = self.pose_debug_viewer.widget
        return self.pose_debug_widget

    def _update_pose_debug_view(
        self,
        frame: np.ndarray,
        pose,
        *,
        segmentation_mask: np.ndarray | None = None,
    ) -> None:
        if self.pose_debug_viewer is None:
            return
        self.pose_debug_viewer.update(
            frame,
            pose,
            segmentation_mask=segmentation_mask,
        )
        self.pose_debug_frame_count = self.pose_debug_viewer.frame_count

    def _resolve_pose_segmentation_mask(
        self,
        frame: np.ndarray,
        pose,
    ) -> np.ndarray | None:
        if self._has_usable_segmentation_mask(pose.segmentation_mask):
            return pose.segmentation_mask
        if pose.coords.size == 0:
            return None
        return build_pose_graph_segmentation_mask(
            pose.coords,
            pose.adjacency,
            frame.shape[:2],
        )

    @staticmethod
    def _has_usable_segmentation_mask(segmentation_mask: np.ndarray | None) -> bool:
        if segmentation_mask is None:
            return False
        mask = np.asarray(segmentation_mask, dtype=np.float32)
        if mask.ndim != 2:
            return False
        return bool(np.any(mask >= np.float32(0.5)))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
