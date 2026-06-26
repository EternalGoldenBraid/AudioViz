import numpy as np

from audioviz.sources import (
    AudioSourceConfig,
    CameraFrameSourceConfig,
    RippleSourceOrchestratorConfig,
    SyntheticSourceConfig,
)
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig


class _FakeProcessor:
    def __init__(self, current_top_k_frequencies, current_signal_level=1.0):
        self.current_top_k_frequencies = current_top_k_frequencies
        self.current_signal_level = current_signal_level
        self.minimum_frequency_peak_magnitude = 0.1
        self.minimum_frequency_peak_to_median_ratio = 5.0
        self.minimum_signal_level = 0.05
        self.num_top_frequencies = len(current_top_k_frequencies)

    def set_num_top_frequencies(self, count):
        count = int(count)
        self.num_top_frequencies = count
        values = [value for value in self.current_top_k_frequencies if value is not None][:count]
        values.extend([None] * (count - len(values)))
        self.current_top_k_frequencies = values


def _source_config(
    *,
    use_synthetic: bool = False,
    use_audio_source: bool | None = None,
    prediction_error: PredictionErrorTransformConfig | None = None,
) -> RippleSourceOrchestratorConfig:
    return RippleSourceOrchestratorConfig(
        synthetic=SyntheticSourceConfig(enabled=use_synthetic, frequency=440.0),
        audio=AudioSourceConfig(enabled=use_audio_source),
        camera_frame=CameraFrameSourceConfig(enabled=False),
        prediction_error=(
            PredictionErrorTransformConfig()
            if prediction_error is None
            else prediction_error
        ),
    )


def test_ripple_source_binding_updates_audio_processor_and_mapping(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    processor = _FakeProcessor([440.0, 880.0, None], current_signal_level=0.25)
    visualizer = RippleWaveVisualizer(
        processor=processor,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=2.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
    )
    visualizer.timer.stop()

    visualizer.source_control_binding.update_control(
        "audio-source",
        "signal_gate_threshold",
        0.2,
    )
    visualizer.source_control_binding.update_control(
        "audio-source",
        "drive_amplitude",
        1.5,
    )
    visualizer.source_control_binding.update_control(
        "audio-source",
        "minimum_peak_magnitude",
        0.3,
    )
    visualizer.source_control_binding.update_control(
        "audio-source",
        "peak_prominence_ratio",
        7.0,
    )
    visualizer.source_control_binding.update_control("audio-source", "top_k_count", 2)
    visualizer.source_control_binding.update_control(
        "audio-source",
        "mapping_mode",
        "linear",
    )
    visualizer.source_control_binding.update_control(
        "audio-source",
        "linear_scale",
        0.1,
    )

    assert visualizer.audio_source.signal_gate_threshold == 0.2
    assert processor.minimum_signal_level == 0.2
    assert visualizer.audio_source.drive_amplitude == 1.5
    assert processor.minimum_frequency_peak_magnitude == 0.3
    assert processor.minimum_frequency_peak_to_median_ratio == 7.0
    assert processor.num_top_frequencies == 2
    assert visualizer.audio_source.mapping_mode == "linear"
    assert visualizer.audio_source.linear_scale == 0.1


def test_ripple_source_binding_updates_camera_source_gain(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(2, 2),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(use_synthetic=False),
    )
    visualizer.timer.stop()

    visualizer.source_control_binding.update_control("camera-source", "gain", 3.0)

    assert visualizer.camera_source.gain == 3.0


def test_ripple_source_binding_updates_learning_dynamics_config(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(
            use_synthetic=False,
            prediction_error=PredictionErrorTransformConfig(
                enabled=True,
                learning_enabled=False,
                learning_rate=1e-4,
                learning_weight_decay=1e-4,
                learning_weight_clip=1.0,
                learning_gradient_clip=1.0,
            ),
        ),
    )
    visualizer.timer.stop()

    visualizer.source_control_binding.update_control(
        "learning-dynamics",
        "learning_enabled",
        True,
    )
    visualizer.source_control_binding.update_control(
        "learning-dynamics",
        "learning_rate",
        0.05,
    )
    visualizer.source_control_binding.update_control(
        "learning-dynamics",
        "learning_weight_decay",
        0.02,
    )
    visualizer.source_control_binding.update_control(
        "learning-dynamics",
        "learning_weight_clip",
        3.0,
    )
    visualizer.source_control_binding.update_control(
        "learning-dynamics",
        "learning_gradient_clip",
        4.0,
    )

    config = visualizer.prediction_error_transform.config
    assert config.learning_enabled is True
    assert config.learning_rate == 0.05
    assert config.learning_weight_decay == 0.02
    assert config.learning_weight_clip == 3.0
    assert config.learning_gradient_clip == 4.0


def test_ripple_source_binding_updates_learning_overlay_state(qapp):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    visualizer = RippleWaveVisualizer(
        processor=None,
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        source_orchestrator_config=_source_config(
            use_synthetic=False,
            prediction_error=PredictionErrorTransformConfig(
                enabled=True,
            ),
        ),
    )
    visualizer.timer.stop()

    visualizer.source_control_binding.update_control(
        "learning-overlay",
        "show_learning_overlay",
        True,
    )
    visualizer.source_control_binding.update_control(
        "learning-overlay",
        "overlay_stride",
        6,
    )
    visualizer.source_control_binding.update_control(
        "learning-overlay",
        "overlay_threshold",
        0.2,
    )
    visualizer.source_control_binding.update_control(
        "learning-overlay",
        "overlay_scale",
        8.0,
    )

    assert visualizer.show_learning_overlay is True
    assert visualizer.learning_overlay_stride == 6
    assert visualizer.learning_overlay_threshold == 0.2
    assert visualizer.learning_overlay_scale == 8.0
