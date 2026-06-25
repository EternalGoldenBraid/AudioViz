import time
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from audioviz.sources import CameraFrameSource
from audioviz.source_controls import (
    AudioSourceControls,
    CameraFrameSourceControls,
    SyntheticFrequencySource,
)
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
from audioviz.visualization.prediction_error_transform import (
    PredictionErrorTransform,
    PredictionErrorTransformConfig,
)
from audioviz.visualization.pose_debug_view import PoseDebugView
from audioviz.visualization.standing_body_renderer import (
    StandingBodyRenderer,
    lookup_table_from_renderer,
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


def _format_frequency_readout(frequencies: tuple[float, ...]) -> str:
    if not frequencies:
        return "—"
    return ", ".join(f"{value:.1f}" for value in frequencies)


class RippleWaveVisualizer(VisualizerBase):
    @property
    def use_camera_source(self) -> bool:
        return self.camera_source.enabled

    @use_camera_source.setter
    def use_camera_source(self, enabled: bool) -> None:
        self.camera_source.enabled = bool(enabled)

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
                 use_audio_source: bool | None = None,
                 apply_gaussian_smoothing: bool = False,
                 use_gpu: bool = False,
                 use_shader: bool = False,
                 boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
                 use_pose_sources: bool = False,
                 use_camera_source: bool = False,
                 camera_source_index: int = 0,
                 camera_source_gain: float = 1.0,
                 prediction_error_transform_enabled: bool = False,
                 prediction_error_inputs: tuple[str, ...] = ("camera_frame",),
                 prediction_error_predictor_source: str = "ripple_state",
                 prediction_error_sigma: float = 0.1,
                 prediction_error_output_mode: str = "bits",
                 prediction_error_activation_function: str = "softsign",
                 prediction_error_activation_scale: float = 1.0,
                 prediction_error_gain: float = 1.0,
                 prediction_error_prediction_clip: float = 10.0,
                 prediction_error_max_output: float = 10.0,
                 prediction_error_learning_enabled: bool = False,
                 prediction_error_learning_rate: float = 1e-4,
                 prediction_error_learning_weight_decay: float = 1e-4,
                 prediction_error_learning_weight_clip: float = 1.0,
                 prediction_error_learning_gradient_clip: float = 1.0,
                 audio_visual_mapping_mode: str = "legacy",
                 audio_visual_mapping_alpha: float = 50.0,
                 audio_visual_mapping_f0: float = 50.0,
                 audio_visual_mapping_fc: float = 2000.0,
                 audio_visual_linear_scale: float = 0.05,
                 audio_visual_linear_offset: float = 0.0,
                 audio_signal_gate_threshold: float = 0.05,
                 audio_drive_amplitude: float = 1.0,
                 auto_color_activation_threshold: float = 0.1,
                 auto_color_floor: float = 0.1,
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
                 camera_capture=None,
                 **kwargs):

        super().__init__(processor, **kwargs)

        self.processor = processor
        self.use_synthetic = use_synthetic
        self.use_audio_source = (
            processor is not None and not use_synthetic
            if use_audio_source is None
            else processor is not None and bool(use_audio_source)
        )
        self.use_gpu = use_gpu
        self.use_shader = use_shader
        self.pose_model_path = pose_model_path
        self.pose_camera_index = pose_camera_index
        self.boundary_condition = boundary_condition
        self.use_pose_sources = use_pose_sources
        self.camera_source = CameraFrameSource(
            resolution=resolution,
            camera_index=camera_source_index,
            gain=camera_source_gain,
            enabled=bool(use_camera_source),
            capture=camera_capture,
        )
        self.prediction_error_transform_enabled = bool(
            prediction_error_transform_enabled
        )
        self.prediction_error_inputs = tuple(prediction_error_inputs)
        self.prediction_error_predictor_source = str(prediction_error_predictor_source)
        self.prediction_error_sigma = float(prediction_error_sigma)
        self.prediction_error_output_mode = str(prediction_error_output_mode)
        self.prediction_error_activation_function = str(
            prediction_error_activation_function
        )
        self.prediction_error_activation_scale = float(
            prediction_error_activation_scale
        )
        self.prediction_error_gain = float(prediction_error_gain)
        self.prediction_error_prediction_clip = float(
            prediction_error_prediction_clip
        )
        self.prediction_error_max_output = float(prediction_error_max_output)
        self.prediction_error_learning_enabled = bool(
            prediction_error_learning_enabled
        )
        self.prediction_error_learning_rate = float(prediction_error_learning_rate)
        self.prediction_error_learning_weight_decay = float(
            prediction_error_learning_weight_decay
        )
        self.prediction_error_learning_weight_clip = float(
            prediction_error_learning_weight_clip
        )
        self.prediction_error_learning_gradient_clip = float(
            prediction_error_learning_gradient_clip
        )
        self.prediction_error_transform = PredictionErrorTransform(
            config=PredictionErrorTransformConfig(
                enabled=self.prediction_error_transform_enabled,
                inputs=self.prediction_error_inputs,
                predictor_source=self.prediction_error_predictor_source,
                sigma=self.prediction_error_sigma,
                output_mode=self.prediction_error_output_mode,
                activation_function=self.prediction_error_activation_function,
                activation_scale=self.prediction_error_activation_scale,
                gain=self.prediction_error_gain,
                prediction_clip=self.prediction_error_prediction_clip,
                max_output=self.prediction_error_max_output,
                learning_enabled=self.prediction_error_learning_enabled,
                learning_rate=self.prediction_error_learning_rate,
                learning_weight_decay=self.prediction_error_learning_weight_decay,
                learning_weight_clip=self.prediction_error_learning_weight_clip,
                learning_gradient_clip=self.prediction_error_learning_gradient_clip,
            ),
            resolution=resolution,
        )
        self.standing_body_renderer = StandingBodyRenderer()
        self.audio_visual_mapping_mode = normalize_audio_visual_mapping_mode(
            audio_visual_mapping_mode
        )
        self.audio_visual_mapping_alpha = float(audio_visual_mapping_alpha)
        self.audio_visual_mapping_f0 = float(audio_visual_mapping_f0)
        self.audio_visual_mapping_fc = float(audio_visual_mapping_fc)
        self.audio_visual_linear_scale = float(audio_visual_linear_scale)
        self.audio_visual_linear_offset = float(audio_visual_linear_offset)
        self.audio_signal_gate_threshold = float(audio_signal_gate_threshold)
        self.audio_drive_amplitude = float(audio_drive_amplitude)
        self.auto_color_activation_threshold = float(auto_color_activation_threshold)
        self.auto_color_floor = float(auto_color_floor)
        if self.use_pose_sources and (self.use_gpu or self.use_shader):
            raise NotImplementedError(
                "Pose-medium coupling currently requires the CPU ripple backend."
            )
        if self.use_camera_source and self.use_shader:
            raise NotImplementedError(
                "Camera-frame source currently requires the CPU/GPU ripple backend."
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
        if self.processor is not None:
            self.processor.minimum_signal_level = self.audio_signal_gate_threshold

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
        freqs = self._resolve_ripple_frequencies()
        camera_excitation = self._resolve_camera_frame_excitation()
        self.engine.amplitude = self._current_excitation_amplitude(freqs)

        if self.use_pose_sources:
            self._update_pose_visualization(freqs, camera_excitation=camera_excitation)
            return

        if freqs is None and camera_excitation is None:
            if not self.renderer.prepare_frame():
                return
            self.engine.step_without_excitation()
            self.time = self.engine.time
            self.renderer.render(self.engine)
            self._sync_audio_control_panel(freqs)
            return

        if not self.renderer.prepare_frame():
            return

        if camera_excitation is not None:
            self.engine.step_grid_excitation(camera_excitation, frequencies=freqs)
        else:
            self.engine.step(freqs)
        self.time = self.engine.time
        self.renderer.render(self.engine)
        self._sync_audio_control_panel(freqs)

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
        return self.base_amplitude * self.audio_drive_amplitude * max(signal_level, 0.0)

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
        signal_level = float(getattr(self.processor, "current_signal_level", 0.0))
        if signal_level < self.audio_signal_gate_threshold:
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
        if self.processor is not None:
            controls = AudioSourceControls(
                signal_gate_threshold=self.audio_signal_gate_threshold,
                drive_amplitude=self.audio_drive_amplitude,
                minimum_peak_magnitude=float(
                    getattr(
                        self.processor,
                        "minimum_frequency_peak_magnitude",
                        0.1,
                    )
                ),
                peak_prominence_ratio=float(
                    getattr(
                        self.processor,
                        "minimum_frequency_peak_to_median_ratio",
                        5.0,
                    )
                ),
                top_k_count=int(getattr(self.processor, "num_top_frequencies", 3)),
                mapping_mode=self.audio_visual_mapping_mode,
                mapping_alpha=self.audio_visual_mapping_alpha,
                mapping_f0=self.audio_visual_mapping_f0,
                mapping_fc=self.audio_visual_mapping_fc,
                linear_scale=self.audio_visual_linear_scale,
                linear_offset=self.audio_visual_linear_offset,
            ).get_controls()
            sections.append(
                ControlPanelSection(
                    key="audio-source",
                    title="Audio Source",
                    controls=controls,
                )
            )
        sections.append(
            ControlPanelSection(
                key="camera-source",
                title="Camera Source",
                controls=CameraFrameSourceControls(
                    gain=self.camera_source.gain,
                ).get_controls(),
            )
        )
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
                key="camera",
                label="Camera Frame",
                enabled=self.use_camera_source,
                available=not self.use_shader,
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
        if section_key == "audio-source":
            self._update_audio_source_control(control_key, value)
            return
        if section_key == "camera-source":
            self._update_camera_source_control(control_key, value)
            return
        if control_key != "frequency_hz" or not section_key.startswith("synthetic-source-"):
            raise KeyError(f"Unknown source control: {section_key}.{control_key}")
        index = int(section_key.removeprefix("synthetic-source-"))
        self.synthetic_frequencies[index, 0] = float(value)
        self.frequency = float(self.synthetic_frequencies[0, 0])

    def _update_audio_source_control(
        self,
        control_key: str,
        value: float | bool | int | str,
    ) -> None:
        if self.processor is None:
            raise RuntimeError("Audio source controls require an audio processor.")
        if control_key == "signal_gate_threshold":
            self.audio_signal_gate_threshold = float(value)
            self.processor.minimum_signal_level = float(value)
            return
        if control_key == "drive_amplitude":
            self.audio_drive_amplitude = float(value)
            return
        if control_key == "minimum_peak_magnitude":
            self.processor.minimum_frequency_peak_magnitude = float(value)
            return
        if control_key == "peak_prominence_ratio":
            self.processor.minimum_frequency_peak_to_median_ratio = float(value)
            return
        if control_key == "top_k_count":
            self.processor.set_num_top_frequencies(int(round(float(value))))
            return
        if control_key == "mapping_mode":
            self.audio_visual_mapping_mode = normalize_audio_visual_mapping_mode(str(value))
            return
        if control_key == "mapping_alpha":
            self.audio_visual_mapping_alpha = float(value)
            return
        if control_key == "mapping_f0":
            self.audio_visual_mapping_f0 = float(value)
            return
        if control_key == "mapping_fc":
            self.audio_visual_mapping_fc = float(value)
            return
        if control_key == "linear_scale":
            self.audio_visual_linear_scale = float(value)
            return
        if control_key == "linear_offset":
            self.audio_visual_linear_offset = float(value)
            return
        if control_key in {
            "signal_level",
            "gate_open",
            "detected_frequencies",
            "mapped_frequencies",
        }:
            return
        raise KeyError(f"Unknown audio source control: {control_key}")

    def _update_camera_source_control(
        self,
        control_key: str,
        value: float | bool | int | str,
    ) -> None:
        if control_key == "gain":
            self.camera_source.set_gain(float(value))
            return
        raise KeyError(f"Unknown camera source control: {control_key}")

    def _sync_audio_control_panel(self, freqs: np.ndarray | None) -> None:
        if self.control_panel is None or self.processor is None:
            return
        gate_open = (
            self.use_audio_source
            and float(getattr(self.processor, "current_signal_level", 0.0))
            >= self.audio_signal_gate_threshold
            and freqs is not None
        )
        detected = tuple(
            float(freq)
            for freq in getattr(self.processor, "current_top_k_frequencies", [])
            if freq is not None and np.isfinite(freq)
        )
        mapped = () if freqs is None else tuple(float(freq) for freq in freqs[0])
        self.control_panel.set_source_control_value(
            "audio-source",
            "signal_level",
            f"{float(getattr(self.processor, 'current_signal_level', 0.0)):.2f}",
        )
        self.control_panel.set_source_control_value(
            "audio-source",
            "gate_open",
            "open" if gate_open else "closed",
        )
        self.control_panel.set_source_control_value(
            "audio-source",
            "detected_frequencies",
            _format_frequency_readout(detected),
        )
        self.control_panel.set_source_control_value(
            "audio-source",
            "mapped_frequencies",
            _format_frequency_readout(mapped),
        )

    def _update_source_toggle(self, source_key: str, enabled: bool) -> None:
        if source_key == "synthetic":
            self.use_synthetic = enabled
            return
        if source_key == "audio":
            if self.processor is None:
                raise RuntimeError("Audio source toggles require an audio processor.")
            self.use_audio_source = enabled
            return
        if source_key == "camera":
            if enabled:
                if self.use_shader:
                    raise NotImplementedError(
                        "Camera-frame source currently requires the CPU/GPU ripple backend."
                    )
            self.camera_source.set_enabled(enabled)
            return
        if source_key == "pose":
            self._set_pose_sources_enabled(enabled)
            return
        raise KeyError(f"Unknown source toggle: {source_key}")

    def _resolve_camera_frame_excitation(self) -> np.ndarray | None:
        excitation = self.camera_source.excitation()
        if excitation is None:
            return None
        return self._apply_source_transforms("camera_frame", excitation)

    def _apply_source_transforms(
        self,
        source_key: str,
        excitation: np.ndarray,
    ) -> np.ndarray:
        return self.prediction_error_transform.apply(
            source_key=source_key,
            observation=excitation,
            ripple_state=self.engine.get_field_numpy(),
        )

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

    def _update_pose_visualization(
        self,
        freqs: np.ndarray | None,
        *,
        camera_excitation: np.ndarray | None = None,
    ) -> None:
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
            self._render_pose_field_without_detection(
                freqs,
                camera_excitation=camera_excitation,
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
        self._render_pose_medium(freqs, camera_excitation=camera_excitation)

    def _render_pose_field_without_detection(
        self,
        freqs: np.ndarray | None,
        *,
        camera_excitation: np.ndarray | None = None,
    ) -> None:
        if self.pose_state is None:
            if freqs is not None or camera_excitation is not None:
                if not self.renderer.prepare_frame():
                    return
                if camera_excitation is not None:
                    self.engine.step_grid_excitation(camera_excitation, frequencies=freqs)
                else:
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
        self._render_pose_medium(freqs, camera_excitation=camera_excitation)

    def _render_pose_field(self, source_excitations: np.ndarray) -> None:
        if not self.renderer.prepare_frame():
            return

        self.engine.step_source_excitations(source_excitations)
        self.time = self.engine.time
        self._render_scene()

    def _render_pose_medium(
        self,
        freqs: np.ndarray | None,
        *,
        camera_excitation: np.ndarray | None = None,
    ) -> None:
        if not self.renderer.prepare_frame():
            return

        self.engine.step_pose_medium(freqs, grid_excitation=camera_excitation)
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
        if self.pose_extractor is not None:
            self.pose_extractor.close()
            self.pose_extractor = None
        if self.pose_capture is not None:
            self.pose_capture.release()
            self.pose_capture = None

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
