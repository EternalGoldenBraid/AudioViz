import time
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from audioviz.source_controls import SyntheticFrequencySource
from audioviz.engine import RippleEngine
from audioviz.physics import BoundaryCondition
from audioviz.sources.pose import (
    MediaPipePoseExtractor,
    PoseGraphExtractor,
    PoseGraphState,
    build_pose_graph_segmentation_mask,
    centered_field_rect,
    map_pose_coords_to_field_positions,
    map_pose_segmentation_to_field_mask,
    pose_coords_in_image_support,
    pose_graph_state_to_ripple_sources,
)
from audioviz.visualization.ripple_renderers import (
    NumpyImageRenderer,
    OpenGLFieldRenderer,
)
from audioviz.visualization.ripple_control_panel import (
    ControlPanelSection,
    RippleControlPanel,
    SourceToggle,
)
from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.signal_processing import (
    map_audio_freq_to_visual_freq,
    normalize_audio_visual_mapping_mode,
)

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
    POSE_DEBUG_MASK_COLOR = (255.0, 64.0, 208.0)
    POSE_DEBUG_MASK_ALPHA = 0.35
    POSE_DEBUG_MASK_OUTLINE_COLOR = (255, 255, 255)
    STANDING_BODY_SILHOUETTE_COLOR = np.array([220, 220, 230], dtype=np.uint8)
    STANDING_BODY_GRAPH_COLOR = np.array([255, 255, 255], dtype=np.uint8)

    def __init__(self,
                 processor: Optional[AudioProcessor] = None,
                 n_sources: int = 1,
                 plane_size_m: Tuple[float, float] = (0.36, 0.62),
                 resolution: Tuple[int, int] = (400, 400),
                 frequency: float = 440.0,
                 amplitude: float = 1.0,
                 decay_alpha: float = 0.0,
                 speed: float = 340.0,
                 damping: float = 0.999,
                 use_synthetic: bool = True,
                 apply_gaussian_smoothing: bool = False,
                 use_gpu: bool = False,
                 use_shader: bool = False,
                 boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
                 use_pose_sources: bool = False,
                 audio_visual_mapping_mode: str = "legacy",
                 audio_visual_mapping_alpha: float = 50.0,
                 audio_visual_mapping_f0: float = 50.0,
                 audio_visual_mapping_fc: float = 2000.0,
                 audio_visual_linear_scale: float = 0.05,
                 audio_visual_linear_offset: float = 0.0,
                 pose_model_path: str | None = None,
                 pose_camera_index: int = 0,
                 pose_acceleration_scale: float = 1.0,
                 pose_max_excitation: float | None = None,
                 pose_graph_stiffness: float = 0.25,
                 body_boundary_transmission: float = 0.0,
                 body_boundary_dissipation: float = 1.0,
                 pose_render_mode: str = POSE_RENDER_MODE_OVERLAY,
                 pose_drive_scale: float = 0.1,
                 pose_field_width_fraction: float = 1.0,
                 pose_field_height_fraction: float = 1.0,
                 pose_debug_view: bool = False,
                 pose_extractor: PoseGraphExtractor | None = None,
                 pose_capture=None,
                 **kwargs):

        super().__init__(processor, **kwargs)

        self.processor = processor
        self.use_synthetic = use_synthetic
        self.use_audio_source = processor is not None and not use_synthetic
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.pose_model_path = pose_model_path
        self.pose_camera_index = pose_camera_index
        self.boundary_condition = boundary_condition
        self.use_pose_sources = use_pose_sources
        self.audio_visual_mapping_mode = normalize_audio_visual_mapping_mode(
            audio_visual_mapping_mode
        )
        self.audio_visual_mapping_alpha = float(audio_visual_mapping_alpha)
        self.audio_visual_mapping_f0 = float(audio_visual_mapping_f0)
        self.audio_visual_mapping_fc = float(audio_visual_mapping_fc)
        self.audio_visual_linear_scale = float(audio_visual_linear_scale)
        self.audio_visual_linear_offset = float(audio_visual_linear_offset)
        if self.use_pose_sources and (self.use_gpu or self.use_shader):
            raise NotImplementedError(
                "Pose-medium coupling currently requires the CPU ripple backend."
            )

        self.n_sources = n_sources
        self.plane_size_m = plane_size_m
        self.resolution = resolution
        self.synthetic_frequencies = self._coerce_synthetic_frequencies(
            frequency,
            n_sources=self.n_sources,
        )
        self.frequency = float(self.synthetic_frequencies[0, 0])
        self.base_amplitude = float(amplitude)
        self.apply_gaussian_smoothing = apply_gaussian_smoothing
        self.amplitude = amplitude
        self.decay_alpha = decay_alpha
        self.speed = speed
        self.damping = damping
        self.time = 0.0
        self.control_panel: Optional[RippleControlPanel] = None
        self.pose_graph_stiffness = pose_graph_stiffness
        _ = pose_acceleration_scale, pose_max_excitation, pose_drive_scale
        self.pose_render_mode = normalize_pose_render_mode(pose_render_mode)
        self.pose_debug_view = pose_debug_view
        self.pose_debug_frame_count = 0
        self.auto_color_levels_enabled = True
        self.body_boundary_transmission = float(body_boundary_transmission)
        self.body_boundary_dissipation = float(body_boundary_dissipation)
        self.pose_field_rect = centered_field_rect(
            self.resolution,
            width_fraction=pose_field_width_fraction,
            height_fraction=pose_field_height_fraction,
        )
        self.pose_state: PoseGraphState | None = None
        self.pose_last_update_time = time.monotonic()
        self.pose_extractor = pose_extractor
        self.pose_capture = pose_capture
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

        self.renderer = (
            OpenGLFieldRenderer() if self.use_shader else NumpyImageRenderer()
        )
        self.pose_debug_widget = None
        self.pose_debug_image = None
        self.pose_debug_edges = None
        self.pose_debug_points = None

        if self.use_pose_sources:
            self._set_pose_sources_enabled(True)

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
                source_toggles=self._build_source_toggles(),
                source_sections=self._build_source_control_sections(),
                on_source_control_changed=self._update_source_control,
                on_source_toggle_changed=self._update_source_toggle,
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

    def update_visualization(self):
        freqs = self._resolve_ripple_frequencies()
        self.engine.amplitude = self._current_excitation_amplitude(freqs)

        if self.use_pose_sources:
            self._update_pose_visualization(freqs)
            return

        if freqs is None:
            if not self.renderer.prepare_frame():
                return
            self.engine.step_without_excitation()
            self.time = self.engine.time
            self.renderer.render(self.engine)
            return

        if not self.renderer.prepare_frame():
            return

        self.engine.step(freqs)
        self.time = self.engine.time
        self.renderer.render(self.engine)

    def _current_excitation_amplitude(self, freqs: np.ndarray | None) -> float:
        if (
            freqs is None
            or self.processor is None
            or not self.use_audio_source
            or self.use_synthetic
        ):
            return self.base_amplitude
        signal_level = float(getattr(self.processor, "current_signal_level", 0.0))
        if not np.isfinite(signal_level):
            return self.base_amplitude
        return self.base_amplitude * max(signal_level, 0.0)

    def _resolve_ripple_frequencies(self) -> np.ndarray | None:
        frequency_groups: list[np.ndarray] = []
        if self.use_synthetic:
            frequency_groups.append(self.synthetic_frequencies.copy())
        audio_frequencies = self._resolve_audio_frequencies()
        if audio_frequencies is not None:
            frequency_groups.append(audio_frequencies)
        if not frequency_groups:
            return None
        return np.concatenate(frequency_groups, axis=1)

    def _resolve_audio_frequencies(self) -> np.ndarray | None:
        if not self.use_audio_source or self.processor is None:
            return None
        top_k = self.processor.current_top_k_frequencies
        top_k = [f for f in top_k if f is not None and np.isfinite(f)]
        if len(top_k) == 0:
            return None
        visual_frequencies = map_audio_freq_to_visual_freq(
            np.asarray(top_k, dtype=np.float32),
            mode=self.audio_visual_mapping_mode,
            alpha=self.audio_visual_mapping_alpha,
            f0=self.audio_visual_mapping_f0,
            fc=self.audio_visual_mapping_fc,
            linear_scale=self.audio_visual_linear_scale,
            linear_offset=self.audio_visual_linear_offset,
        ).astype(np.float32, copy=False)
        return np.tile(visual_frequencies, (self.n_sources, 1))

    @staticmethod
    def _coerce_synthetic_frequencies(
        frequency: float | list[float] | tuple[float, ...] | np.ndarray,
        *,
        n_sources: int,
    ) -> np.ndarray:
        values = np.asarray(frequency, dtype=np.float32)
        if values.ndim == 0:
            return np.full((n_sources, 1), float(values), dtype=np.float32)
        if values.ndim == 1 and values.shape[0] == n_sources:
            return values.reshape(n_sources, 1).astype(np.float32, copy=False)
        if values.ndim == 2 and values.shape == (n_sources, 1):
            return values.astype(np.float32, copy=False)
        raise ValueError(
            "frequency must be a scalar, a length-n_sources vector, "
            "or an (n_sources, 1) matrix"
        )

    def _build_source_control_sections(self) -> tuple[ControlPanelSection, ...]:
        sections: list[ControlPanelSection] = []
        for index, frequency_hz in enumerate(self.synthetic_frequencies[:, 0]):
            sections.append(
                ControlPanelSection(
                    key=f"synthetic-source-{index}",
                    title=f"Synthetic Source {index + 1}",
                    controls=SyntheticFrequencySource(
                        frequency_hz=float(frequency_hz),
                        n_sources=1,
                    ).get_controls(),
                )
            )
        return tuple(sections)

    def _build_source_toggles(self) -> tuple[SourceToggle, ...]:
        return (
            SourceToggle(
                key="synthetic",
                label="Synthetic",
                enabled=self.use_synthetic,
                available=True,
            ),
            SourceToggle(
                key="audio",
                label="Audio",
                enabled=self.use_audio_source,
                available=self.processor is not None,
            ),
            SourceToggle(
                key="pose",
                label="Pose Graph",
                enabled=self.use_pose_sources,
                available=not (self.use_gpu or self.use_shader),
            ),
        )

    def _update_source_control(
        self,
        section_key: str,
        control_key: str,
        value: float | bool | int | str,
    ) -> None:
        if control_key != "frequency_hz" or not section_key.startswith("synthetic-source-"):
            raise KeyError(f"Unknown source control: {section_key}.{control_key}")
        index = int(section_key.removeprefix("synthetic-source-"))
        self.synthetic_frequencies[index, 0] = float(value)
        self.frequency = float(self.synthetic_frequencies[0, 0])

    def _update_source_toggle(self, source_key: str, enabled: bool) -> None:
        if source_key == "synthetic":
            self.use_synthetic = enabled
            return
        if source_key == "audio":
            if self.processor is None:
                raise RuntimeError("Audio source toggles require an audio processor.")
            self.use_audio_source = enabled
            return
        if source_key == "pose":
            self._set_pose_sources_enabled(enabled)
            return
        raise KeyError(f"Unknown source toggle: {source_key}")

    def _set_pose_sources_enabled(self, enabled: bool) -> None:
        if enabled:
            if self.use_gpu or self.use_shader:
                raise NotImplementedError(
                    "Pose-medium coupling currently requires the CPU ripple backend."
                )
            self._ensure_pose_source(
                model_path=self.pose_model_path,
                camera_index=self.pose_camera_index,
            )
            self.pose_last_update_time = time.monotonic()
        self.use_pose_sources = enabled
        self.pose_state = None
        self._clear_pose_medium_state()
        self.engine.set_body_boundary_mask(None)

    def _ensure_pose_source(
        self,
        *,
        model_path: str | None,
        camera_index: int,
    ) -> None:
        if self.pose_extractor is None:
            self.pose_extractor = MediaPipePoseExtractor(model_path=model_path)
        if self.pose_capture is None:
            cv2 = self._load_cv2()
            self.pose_capture = cv2.VideoCapture(camera_index)
            if not self.pose_capture.isOpened():
                self.pose_extractor.close()
                self.pose_extractor = None
                raise RuntimeError(f"Failed to open pose camera index {camera_index}")

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

    def _update_pose_visualization(self, freqs: np.ndarray | None) -> None:
        if self.pose_capture is None or self.pose_extractor is None:
            return

        ok, frame = self.pose_capture.read()
        if not ok:
            return

        pose = self.pose_extractor.extract(frame)
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
            self._render_pose_field_without_detection(freqs)
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
        self._render_pose_medium(freqs)

    def _render_pose_field_without_detection(self, freqs: np.ndarray | None) -> None:
        if self.pose_state is None:
            if freqs is not None:
                if not self.renderer.prepare_frame():
                    return
                self.engine.step(freqs)
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
        self._render_pose_medium(freqs)

    def _render_pose_field(self, source_excitations: np.ndarray) -> None:
        if not self.renderer.prepare_frame():
            return

        self.engine.step_source_excitations(source_excitations)
        self.time = self.engine.time
        self._render_scene()

    def _render_pose_medium(self, freqs: np.ndarray | None) -> None:
        if not self.renderer.prepare_frame():
            return

        self.engine.step_pose_medium(freqs)
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
        field = self.engine.get_field_numpy()
        floor_rgb = self._field_to_floor_rgb(field)
        frame = self._project_floor_to_perspective(floor_rgb)
        if self._latest_pose_coords.size == 0:
            return frame
        silhouette_mask, bbox = self._standing_body_silhouette_mask_and_bbox()
        if silhouette_mask is None or bbox is None:
            return frame
        return self._overlay_standing_body(
            frame,
            silhouette_mask=silhouette_mask,
            bbox=bbox,
            coords=self._latest_pose_coords,
            adjacency=self._latest_pose_adjacency,
        )

    def _field_to_floor_rgb(self, field: np.ndarray) -> np.ndarray:
        values = np.asarray(field, dtype=np.float32)
        limit = np.max(np.abs(values))
        limit = max(float(limit), 1e-6)
        normalized = np.clip((values / limit + 1.0) * 0.5, 0.0, 1.0)
        lookup = self._renderer_lookup_table()
        if lookup is None:
            gray = np.rint(normalized * 255.0).astype(np.uint8)
            return np.repeat(gray[..., None], 3, axis=2)
        index = np.rint(normalized * (len(lookup) - 1)).astype(np.int32)
        return np.ascontiguousarray(lookup[index])

    def _renderer_lookup_table(self) -> np.ndarray | None:
        renderer_lookup = getattr(self.renderer, "lookup_table", None)
        if renderer_lookup is not None:
            lookup = np.asarray(renderer_lookup, dtype=np.uint8)
            if lookup.ndim == 2 and lookup.shape[0] > 0:
                return lookup
        image_item = getattr(self.renderer, "image_item", None)
        if image_item is None:
            return None
        lookup = getattr(image_item, "lut", None)
        if callable(lookup):
            try:
                lookup = lookup()
            except AttributeError:
                return None
        if lookup is None:
            return None
        lookup = np.asarray(lookup, dtype=np.uint8)
        if lookup.ndim != 2 or lookup.shape[0] == 0:
            return None
        return lookup

    def _project_floor_to_perspective(self, floor_rgb: np.ndarray) -> np.ndarray:
        rows, cols = floor_rgb.shape[:2]
        output = np.zeros_like(floor_rgb)
        horizon = max(1, int(round(rows * 0.18)))
        center_x = (cols - 1) * 0.5
        height = max(rows - 1 - horizon, 1)
        ys, xs = np.indices((rows, cols), dtype=np.float32)
        depth = np.clip((ys - horizon) / height, 0.0, 1.0)
        src_y = np.rint((depth ** 1.2) * (rows - 1)).astype(np.int32)
        width_scale = 0.35 + 0.65 * (depth ** 1.2)
        src_x = ((xs - center_x) / np.maximum(width_scale, 1e-6)) + center_x
        src_x = np.rint(src_x).astype(np.int32)
        valid = (ys >= horizon) & (src_x >= 0) & (src_x < cols)
        output[valid] = floor_rgb[src_y[valid], src_x[valid]]
        return output

    def _standing_body_silhouette_mask_and_bbox(
        self,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        if self._latest_pose_segmentation_mask is None:
            return None, None
        mask = np.asarray(self._latest_pose_segmentation_mask, dtype=np.float32) >= 0.5
        if not np.any(mask):
            return None, None
        mask = mask[:, ::-1]
        ys, xs = np.nonzero(mask)
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        left = int(xs.min())
        right = int(xs.max()) + 1
        cropped = mask[top:bottom, left:right]
        if cropped.size == 0:
            return None, None
        return cropped, (top, left, bottom, right)

    def _overlay_standing_body(
        self,
        frame: np.ndarray,
        *,
        silhouette_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        coords: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        output = frame.copy()
        rows, cols = output.shape[:2]
        top, left, bottom, right = bbox
        mask_height = max(bottom - top, 1)
        mask_width = max(right - left, 1)
        mirrored_coords = np.asarray(coords, dtype=np.float32).copy()
        mirrored_coords[:, 0] = 1.0 - mirrored_coords[:, 0]
        base_y_norm = float(np.clip(np.max(mirrored_coords[:, 1]), 0.0, 1.0))
        anchor_x_norm = float(np.clip(np.mean(mirrored_coords[:, 0]), 0.0, 1.0))
        anchor_screen_x, anchor_screen_y = self._project_floor_point(
            anchor_x_norm * max(cols - 1, 1),
            base_y_norm * max(rows - 1, 1),
            rows=rows,
            cols=cols,
        )
        scale = 0.35 + 0.65 * base_y_norm
        body_height = max(16, int(round(mask_height * scale * 1.2)))
        body_width = max(8, int(round(body_height * (mask_width / max(mask_height, 1)))))
        body_height = min(body_height, rows)
        body_width = min(body_width, cols)
        top_screen = max(0, min(anchor_screen_y - body_height, rows - body_height))
        left_screen = int(np.clip(anchor_screen_x - body_width // 2, 0, max(cols - body_width, 0)))
        silhouette = self._resize_bool_mask(silhouette_mask, body_height, body_width)
        body_slice = output[top_screen:top_screen + body_height, left_screen:left_screen + body_width]
        body_slice[silhouette] = self._blend_rgb(
            body_slice[silhouette],
            self.STANDING_BODY_SILHOUETTE_COLOR,
            alpha=0.55,
        )

        bbox_width_norm = max((right - left) / max(self._latest_pose_segmentation_mask.shape[1], 1), 1e-6)
        bbox_height_norm = max((bottom - top) / max(self._latest_pose_segmentation_mask.shape[0], 1), 1e-6)
        left_norm = left / max(self._latest_pose_segmentation_mask.shape[1], 1)
        top_norm = top / max(self._latest_pose_segmentation_mask.shape[0], 1)
        projected_points: list[tuple[int, int] | None] = []
        for x_norm, y_norm in mirrored_coords:
            if not np.isfinite(x_norm) or not np.isfinite(y_norm):
                projected_points.append(None)
                continue
            local_x = (x_norm - left_norm) / bbox_width_norm
            local_y = (y_norm - top_norm) / bbox_height_norm
            if not (0.0 <= local_x <= 1.0 and 0.0 <= local_y <= 1.0):
                projected_points.append(None)
                continue
            point_x = left_screen + int(round(local_x * max(body_width - 1, 0)))
            point_y = top_screen + int(round(local_y * max(body_height - 1, 0)))
            projected_points.append((point_x, point_y))

        for start, end in np.argwhere(np.triu(np.asarray(adjacency, dtype=np.float32), k=1) > 0):
            point_a = projected_points[int(start)]
            point_b = projected_points[int(end)]
            if point_a is None or point_b is None:
                continue
            self._draw_line(output, point_a, point_b, self.STANDING_BODY_GRAPH_COLOR)
        for point in projected_points:
            if point is None:
                continue
            self._draw_disk(output, point, radius=2, color=self.STANDING_BODY_GRAPH_COLOR)
        return output

    def _project_floor_point(
        self,
        source_x: float,
        source_y: float,
        *,
        rows: int,
        cols: int,
    ) -> tuple[int, int]:
        horizon = max(1, int(round(rows * 0.18)))
        depth = np.clip(float(source_y) / max(rows - 1, 1), 0.0, 1.0)
        screen_y = horizon + int(round((depth ** (1.0 / 1.2)) * max(rows - 1 - horizon, 1)))
        width_scale = 0.35 + 0.65 * depth
        center_x = (cols - 1) * 0.5
        screen_x = center_x + (float(source_x) - center_x) * width_scale
        return int(round(screen_x)), int(round(screen_y))

    @staticmethod
    def _resize_bool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        row_index = np.rint(
            np.linspace(0, mask.shape[0] - 1, height, dtype=np.float32)
        ).astype(np.int32)
        col_index = np.rint(
            np.linspace(0, mask.shape[1] - 1, width, dtype=np.float32)
        ).astype(np.int32)
        return mask[row_index][:, col_index]

    @staticmethod
    def _blend_rgb(base: np.ndarray, color: np.ndarray, *, alpha: float) -> np.ndarray:
        blended = (1.0 - alpha) * base.astype(np.float32) + alpha * color.astype(np.float32)
        return np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)

    @classmethod
    def _draw_line(
        cls,
        frame: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: np.ndarray,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
        ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)
        rows, cols = frame.shape[:2]
        valid = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
        frame[ys[valid], xs[valid]] = color

    @classmethod
    def _draw_disk(
        cls,
        frame: np.ndarray,
        center: tuple[int, int],
        *,
        radius: int,
        color: np.ndarray,
    ) -> None:
        cx, cy = center
        rows, cols = frame.shape[:2]
        y0 = max(cy - radius, 0)
        y1 = min(cy + radius + 1, rows)
        x0 = max(cx - radius, 0)
        x1 = min(cx + radius + 1, cols)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        frame[y0:y1, x0:x1][mask] = color

    def closeEvent(self, event):
        self.close_pose_sources()
        super().closeEvent(event)

    def close_pose_sources(self) -> None:
        if self.pose_extractor is not None:
            self.pose_extractor.close()
            self.pose_extractor = None
        if self.pose_capture is not None:
            self.pose_capture.release()
            self.pose_capture = None

    def _create_pose_debug_widget(self):
        widget = pg.GraphicsLayoutWidget()
        plot = widget.addPlot(row=0, col=0)
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.invertY(True)

        self.pose_debug_image = pg.ImageItem(axisOrder="row-major")
        self.pose_debug_edges = pg.PlotDataItem(
            pen=pg.mkPen((255, 180, 40), width=2),
            connect="finite",
        )
        self.pose_debug_points = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(40, 180, 255),
            pen=pg.mkPen(255, 255, 255, width=1),
        )
        plot.addItem(self.pose_debug_image)
        plot.addItem(self.pose_debug_edges)
        plot.addItem(self.pose_debug_points)
        self.pose_debug_widget = widget
        return widget

    def _update_pose_debug_view(
        self,
        frame: np.ndarray,
        pose,
        *,
        segmentation_mask: np.ndarray | None = None,
    ) -> None:
        if self.pose_debug_image is None:
            return

        rgb_frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
        if segmentation_mask is None:
            segmentation_mask = pose.segmentation_mask
        if segmentation_mask is not None:
            rgb_frame = self._overlay_pose_debug_mask(
                rgb_frame,
                segmentation_mask,
            )
        self.pose_debug_image.setImage(rgb_frame, autoLevels=False)
        self.pose_debug_frame_count += 1

        height, width = frame.shape[:2]
        if not pose.coords.size:
            self.pose_debug_edges.setData([], [])
            self.pose_debug_points.setData([], [])
            return

        valid = pose_coords_in_image_support(pose.coords)
        if not np.any(valid):
            self.pose_debug_edges.setData([], [])
            self.pose_debug_points.setData([], [])
            return

        coords_px = pose.coords * np.array([width - 1, height - 1], dtype=np.float32)
        coords_px[:, 0] = (width - 1) - coords_px[:, 0]
        edge_xs = []
        edge_ys = []
        for i, j in np.argwhere(np.triu(pose.adjacency, k=1) > 0):
            if not (valid[i] and valid[j]):
                continue
            edge_xs.extend([coords_px[i, 0], coords_px[j, 0], np.nan])
            edge_ys.extend([coords_px[i, 1], coords_px[j, 1], np.nan])

        self.pose_debug_edges.setData(edge_xs, edge_ys)
        self.pose_debug_points.setData(coords_px[valid, 0], coords_px[valid, 1])

    def _overlay_pose_debug_mask(
        self,
        rgb_frame: np.ndarray,
        segmentation_mask: np.ndarray,
    ) -> np.ndarray:
        height, width = rgb_frame.shape[:2]
        mirrored_mask = self._mirrored_pose_debug_mask(
            segmentation_mask,
            height=height,
            width=width,
        )
        if not np.any(mirrored_mask):
            return rgb_frame

        overlay = rgb_frame.astype(np.float32, copy=True)
        tint = np.asarray(self.POSE_DEBUG_MASK_COLOR, dtype=np.float32)
        alpha = np.float32(self.POSE_DEBUG_MASK_ALPHA)
        overlay[mirrored_mask] = (
            overlay[mirrored_mask] * (np.float32(1.0) - alpha)
            + tint * alpha
        )
        outline = self._pose_debug_mask_outline(mirrored_mask)
        overlay[outline] = np.asarray(self.POSE_DEBUG_MASK_OUTLINE_COLOR, dtype=np.float32)
        return np.ascontiguousarray(np.rint(overlay).astype(np.uint8))

    @staticmethod
    def _mirrored_pose_debug_mask(
        segmentation_mask: np.ndarray,
        *,
        height: int,
        width: int,
    ) -> np.ndarray:
        mask = np.asarray(segmentation_mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError("segmentation_mask must have shape (rows, cols)")

        row_index = np.rint(
            np.linspace(0, mask.shape[0] - 1, height, dtype=np.float32)
        ).astype(np.int32)
        col_index = np.rint(
            np.linspace(0, mask.shape[1] - 1, width, dtype=np.float32)
        ).astype(np.int32)
        return mask[row_index][:, col_index][:, ::-1] >= np.float32(0.5)

    @staticmethod
    def _pose_debug_mask_outline(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        center = padded[1:-1, 1:-1]
        neighbors_same = (
            padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        return center & ~neighbors_same

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

    @staticmethod
    def _load_cv2():
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for pose-driven ripple sources. "
                "Install the pose-demo dependency group first."
            ) from exc
        return cv2


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
