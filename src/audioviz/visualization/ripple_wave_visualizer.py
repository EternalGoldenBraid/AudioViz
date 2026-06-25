import time
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from audioviz.engine import RippleEngine
from audioviz.physics import BoundaryCondition
from audioviz.sources.pose import (
    PoseFrameSource,
    PoseGraphExtractor,
    PoseGraphState,
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
from audioviz.visualization.standing_body_renderer import (
    StandingBodyRenderer,
    lookup_table_from_renderer,
)
from audioviz.visualization.ripple_control_panel import (
    RippleControlPanel,
)
from audioviz.visualization.ripple_source_controls import (
    RippleSourceControlBinding,
)
from audioviz.visualization.ripple_source_orchestrator import (
    RippleSourceOrchestrator,
)
from audioviz.visualization.ripple_visualizer_config import (
    RipplePoseConfig,
    RippleSourceOrchestratorConfig,
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
        self.source_control_binding = RippleSourceControlBinding(self)
        self.pose_graph_stiffness = pose_config.graph_stiffness
        self.pose_render_mode = normalize_pose_render_mode(pose_config.render_mode)
        self.pose_debug_view = pose_config.debug_view
        self.pose_debug_frame_count = 0
        self.auto_color_levels_enabled = True
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
            pose_graph_stiffness=self.pose_graph_stiffness,
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

        self.renderer = (
            OpenGLFieldRenderer() if self.use_shader else NumpyImageRenderer(
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
        reset_view = getattr(self.renderer, "reset_view", None)
        if callable(reset_view):
            reset_view()
        if self.renderer.prepare_frame():
            self.renderer.render(self.engine)

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
        self._refresh_current_render()

    def _update_auto_color_activation_threshold(self, val: float) -> None:
        self.auto_color_activation_threshold = float(val)
        setter = getattr(self.renderer, "set_auto_level_activation_threshold", None)
        if callable(setter):
            setter(self.auto_color_activation_threshold)
        self._refresh_current_render()

    def _update_auto_color_floor(self, val: float) -> None:
        self.auto_color_floor = float(val)
        set_auto_level_floor = getattr(self.renderer, "set_auto_level_floor", None)
        if callable(set_auto_level_floor):
            set_auto_level_floor(self.auto_color_floor)
        self._refresh_current_render()

    def _refresh_current_render(self) -> None:
        if self.renderer.prepare_frame():
            self.renderer.render(self.engine)

    def update_visualization(self):
        source_frame = self.source_orchestrator.resolve()
        self.engine.amplitude = source_frame.amplitude

        if self.use_pose_sources:
            self._update_pose_visualization(source_frame.grid_excitation)
            return

        if source_frame.grid_excitation is None:
            if not self.renderer.prepare_frame():
                return
            self.engine.step_without_excitation()
            self.time = self.engine.time
            self.renderer.render(self.engine)
            self.source_control_binding.sync_audio_panel(
                self.control_panel,
                source_frame.audio_frequencies,
            )
            return

        if not self.renderer.prepare_frame():
            return

        self.engine.step_grid_excitation(source_frame.grid_excitation)
        self.time = self.engine.time
        self.renderer.render(self.engine)
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
        self._clear_pose_medium_state()
        self.engine.set_body_boundary_mask(None)

    def _clear_pose_medium_state(self) -> None:
        if self.engine.pose_values is not None:
            self.engine.pose_values[:] = 0
        if self.engine.pose_values_old is not None:
            self.engine.pose_values_old[:] = 0
        if self.engine.pose_valid is not None:
            self.engine.pose_valid[:] = False
        self._latest_pose_coords = np.zeros((0, 2), dtype=np.float32)
        self._latest_pose_adjacency = np.zeros((0, 0), dtype=np.float32)
        self._latest_pose_segmentation_mask = None

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
        source_excitation: np.ndarray | None,
    ) -> None:
        sample = self.pose_source.read()
        if sample is None:
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
            self._render_pose_field_without_detection(
                source_excitation,
            )
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

        self.engine.update_pose_medium(
            positions=mapped_positions,
            valid=valid,
            adjacency=pose.adjacency,
        )
        self._render_pose_medium(source_excitation)

    def _render_pose_field_without_detection(
        self,
        source_excitation: np.ndarray | None,
    ) -> None:
        if self.pose_state is None:
            if source_excitation is not None:
                if not self.renderer.prepare_frame():
                    return
                self.engine.step_grid_excitation(source_excitation)
                self.time = self.engine.time
                self._render_scene()
                return
            if not self.renderer.prepare_frame():
                return
            self._render_scene()
            return

        self.engine.update_pose_medium(
            positions=np.zeros((self.pose_state.num_nodes, 2), dtype=np.float32),
            valid=np.zeros(self.pose_state.num_nodes, dtype=bool),
            adjacency=self.pose_state.adjacency,
        )
        self._render_pose_medium(source_excitation)

    def _render_pose_medium(
        self,
        source_excitation: np.ndarray | None,
    ) -> None:
        if not self.renderer.prepare_frame():
            return

        self.engine.step_pose_medium(grid_excitation=source_excitation)
        if self.pose_state is not None:
            self.pose_state.set_ripple_states(self.engine.get_pose_medium_state())
        self.time = self.engine.time
        self._render_scene()

    def _render_scene(self) -> None:
        self.renderer.render(self.engine)
        render_rgb_frame = getattr(self.renderer, "render_rgb_frame", None)
        if (
            self.pose_render_mode == POSE_RENDER_MODE_STANDING_BODY
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

    def closeEvent(self, event):
        self.close_pose_sources()
        self.close_camera_source()
        super().closeEvent(event)

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
