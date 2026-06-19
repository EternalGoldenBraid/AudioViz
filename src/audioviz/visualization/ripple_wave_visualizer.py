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


class RippleWaveVisualizer(VisualizerBase):
    POSE_DEBUG_MASK_COLOR = (255.0, 64.0, 208.0)
    POSE_DEBUG_MASK_ALPHA = 0.35
    POSE_DEBUG_MASK_OUTLINE_COLOR = (255, 255, 255)

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
                 pose_model_path: str | None = None,
                 pose_camera_index: int = 0,
                 pose_acceleration_scale: float = 1.0,
                 pose_max_excitation: float | None = None,
                 pose_graph_stiffness: float = 0.25,
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
        self.apply_gaussian_smoothing = apply_gaussian_smoothing
        self.amplitude = amplitude
        self.decay_alpha = decay_alpha
        self.speed = speed
        self.damping = damping
        self.time = 0.0
        self.control_panel: Optional[RippleControlPanel] = None
        self.pose_graph_stiffness = pose_graph_stiffness
        _ = pose_acceleration_scale, pose_max_excitation, pose_drive_scale
        self.pose_debug_view = pose_debug_view
        self.pose_debug_frame_count = 0
        self.auto_color_levels_enabled = True
        self.pose_field_rect = centered_field_rect(
            self.resolution,
            width_fraction=pose_field_width_fraction,
            height_fraction=pose_field_height_fraction,
        )
        self.pose_state: PoseGraphState | None = None
        self.pose_last_update_time = time.monotonic()
        self.pose_extractor = pose_extractor
        self.pose_capture = pose_capture

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

    def _update_auto_color_levels(self, enabled: bool) -> None:
        self.auto_color_levels_enabled = bool(enabled)
        set_auto_levels = getattr(self.renderer, "set_auto_percentile_levels", None)
        if callable(set_auto_levels):
            set_auto_levels(self.auto_color_levels_enabled)

    def update_visualization(self):
        freqs = self._resolve_ripple_frequencies()

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
        return np.tile(np.asarray(top_k, dtype=np.float32), (self.n_sources, 1))

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

    def _update_pose_visualization(self, freqs: np.ndarray | None) -> None:
        if self.pose_capture is None or self.pose_extractor is None:
            return

        ok, frame = self.pose_capture.read()
        if not ok:
            return

        pose = self.pose_extractor.extract(frame)
        segmentation_mask = self._resolve_pose_segmentation_mask(frame, pose)
        self.engine.set_body_boundary_mask(
            map_pose_segmentation_to_field_mask(
                segmentation_mask,
                self.resolution,
                field_rect=self.pose_field_rect,
            )
            if segmentation_mask is not None
            else None
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
        mapped_positions = map_pose_coords_to_field_positions(
            positions,
            self.resolution,
            field_rect=self.pose_field_rect,
        )

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
                self.renderer.render(self.engine)
                return
            if not self.renderer.prepare_frame():
                return
            self.renderer.render(self.engine)
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
        self.renderer.render(self.engine)

    def _render_pose_medium(self, freqs: np.ndarray | None) -> None:
        if not self.renderer.prepare_frame():
            return

        self.engine.step_pose_medium(freqs)
        if self.pose_state is not None:
            self.pose_state.set_ripple_states(self.engine.get_pose_medium_state())
        self.time = self.engine.time
        self.renderer.render(self.engine)

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
