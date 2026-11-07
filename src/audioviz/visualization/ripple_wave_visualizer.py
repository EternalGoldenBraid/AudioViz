from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from audioviz.engine import RippleEngine
from audioviz.physics import BoundaryCondition
from audioviz.sources import RippleSourceOrchestrator, RippleSourceOrchestratorConfig
from audioviz.visualization.ripple_renderers import (
    NumpyImageRenderer,
    OpenGLFieldRenderer,
)
from audioviz.visualization.prediction_diagnostics_view import PredictionDiagnosticsView
from audioviz.visualization.prediction_scene_window import PredictionSceneWindow
from audioviz.utils.source_preview import SourceScenePreview
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.ui.ripple_control_panel import (
    RippleControlPanel,
)
from audioviz.ui.ripple_source_controls import (
    RippleSourceControlBinding,
)
from audioviz.visualization.visualizer_base import VisualizerBase
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
                 wave_propagation_enabled: bool = True,
                 apply_gaussian_smoothing: bool = False,
                 use_gpu: bool = False,
                 use_shader: bool = False,
                 boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
                 rgb_canvas_enabled: bool = False,
                 auto_color_activation_threshold: float = 0.1,
                 auto_color_floor: float = 0.1,
                 source_orchestrator_config: RippleSourceOrchestratorConfig | None = None,
                 camera_capture=None,
                 **kwargs):
        if source_orchestrator_config is None:
            source_orchestrator_config = RippleSourceOrchestratorConfig()
        super().__init__(processor, **kwargs)

        self.processor = processor
        self.n_sources = n_sources
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.boundary_condition = boundary_condition
        self.rgb_canvas_enabled = bool(rgb_canvas_enabled)
        self.auto_color_activation_threshold = float(auto_color_activation_threshold)
        self.auto_color_floor = float(auto_color_floor)
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
            wave_propagation_enabled=wave_propagation_enabled,
            amplitude=self.amplitude,
            decay_alpha=self.decay_alpha,
            use_gpu=self.use_gpu,
            use_shader=self.use_shader,
            boundary_condition=self.boundary_condition,
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
        if self.use_camera_source:
            self.camera_source.set_enabled(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.renderer.widget)

        controls_button = QtWidgets.QPushButton("Display Settings")
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
                source_sections=tuple(
                    section for section in self.source_control_binding.build_sections()
                    if section.key == "learning-overlay"
                ),
                on_source_control_changed=self.source_control_binding.update_control,
                display_only=True,
            )
            self.control_panel.resize(360, 520)
        self._sync_display_controls()
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

    def _update_auto_color_levels(self, enabled: bool) -> None:
        self.auto_color_levels_enabled = bool(enabled)
        set_auto_levels = getattr(self.renderer, "set_auto_percentile_levels", None)
        if callable(set_auto_levels):
            set_auto_levels(self.auto_color_levels_enabled)
        self._sync_display_controls()

    def _update_rgb_canvas(self, enabled: bool) -> None:
        self.rgb_canvas_enabled = bool(enabled)
        setter = getattr(self.renderer, "set_rgb_canvas_enabled", None)
        if callable(setter):
            setter(self.rgb_canvas_enabled)
        self._sync_display_controls()

    def _update_auto_color_activation_threshold(self, val: float) -> None:
        self.auto_color_activation_threshold = float(val)
        setter = getattr(self.renderer, "set_auto_level_activation_threshold", None)
        if callable(setter):
            setter(self.auto_color_activation_threshold)
        self._sync_display_controls()

    def _update_auto_color_floor(self, val: float) -> None:
        self.auto_color_floor = float(val)
        set_auto_level_floor = getattr(self.renderer, "set_auto_level_floor", None)
        if callable(set_auto_level_floor):
            set_auto_level_floor(self.auto_color_floor)
        self._sync_display_controls()

    def _sync_display_controls(self) -> None:
        if self.control_panel is not None:
            self.control_panel.sync_display_values(
                rgb_canvas_enabled=self.rgb_canvas_enabled,
                auto_color_levels_enabled=self.auto_color_levels_enabled,
                auto_color_activation_threshold=self.auto_color_activation_threshold,
                auto_color_floor=self.auto_color_floor,
                overlay_values={
                    "show_learning_overlay": self.show_learning_overlay,
                    "overlay_stride": self.learning_overlay_stride,
                    "overlay_threshold": self.learning_overlay_threshold,
                    "overlay_scale": self.learning_overlay_scale,
                },
            )

    def update_visualization(self):
        source_frame = self.source_orchestrator.resolve()
        self.engine.amplitude = source_frame.amplitude

        if not self.renderer.prepare_frame():
            return

        self._advance_canvas(source_frame.drive_grid)
        self.time = self.engine.time
        self._render_scene()

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
            if self.source_orchestrator.secondary_observation is None:
                self.prediction_scene.invalidate_evidence("camera_secondary")
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
             self.camera_source.enabled),
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
                stream_state=source.enabled if readout is not None else None,
                stream_state_prediction=readout.stream_state_prediction if readout is not None else None,
            ))
        sources.append(SourceScenePreview(
            mapping=orchestrator.secondary_camera_source.scene_mapping,
            enabled=orchestrator.secondary_camera_source.enabled,
            observation=(
                orchestrator.secondary_camera_source.scene_mapping.sample(orchestrator.secondary_observation)
                if orchestrator.secondary_observation is not None else None
            ),
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
        self.renderer.render(self.engine.get_field_numpy())

    def closeEvent(self, event):
        if self.prediction_scene is not None:
            self.prediction_scene.close()
        if self.prediction_diagnostics is not None:
            self.prediction_diagnostics.close()
        self.close_camera_source()
        super().closeEvent(event)

    def set_learning_overlay_enabled(self, enabled: bool) -> None:
        self.show_learning_overlay = bool(enabled)
        self._sync_display_controls()

    def set_learning_overlay_stride(self, stride: int) -> None:
        self.learning_overlay_stride = max(int(stride), 1)
        self._sync_display_controls()

    def set_learning_overlay_threshold(self, threshold: float) -> None:
        self.learning_overlay_threshold = max(float(threshold), 0.0)
        self._sync_display_controls()

    def set_learning_overlay_scale(self, scale: float) -> None:
        self.learning_overlay_scale = max(float(scale), 0.1)
        self._sync_display_controls()

    def close_camera_source(self) -> None:
        self.camera_source.close()
        self.source_orchestrator.secondary_camera_source.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
