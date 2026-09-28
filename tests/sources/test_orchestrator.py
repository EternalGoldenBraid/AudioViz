import numpy as np

from audioviz.engine import RippleEngine
from audioviz.sources import (
    AudioSourceConfig,
    CameraFrameSourceConfig,
    RippleSourceOrchestrator,
    RippleSourceOrchestratorConfig,
    SyntheticSourceConfig,
)
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.utils.signal_processing import map_audio_freq_to_visual_freq


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


class _OffsetVisualReadout:
    def predict(self, prior):
        return np.asarray(prior, dtype=np.float32) + np.float32(0.25)

    def backproject(self, visual_correction):
        return np.asarray(visual_correction, dtype=np.float32) * np.float32(2.0)


def _build_engine(*, amplitude: float = 1.0) -> RippleEngine:
    return RippleEngine(
        resolution=(24, 32),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=amplitude,
        use_gpu=False,
    )


def _source_config(
    *,
    use_synthetic: bool = False,
    use_audio_source: bool | None = None,
    audio_signal_gate_threshold: float = 0.05,
    audio_drive_amplitude: float = 1.0,
    audio_visual_mapping_mode: str = "legacy",
    audio_visual_mapping_alpha: float = 50.0,
    audio_visual_mapping_f0: float = 50.0,
    audio_visual_mapping_fc: float = 2000.0,
    audio_visual_linear_scale: float = 0.05,
    audio_visual_linear_offset: float = 0.0,
) -> RippleSourceOrchestratorConfig:
    return RippleSourceOrchestratorConfig(
        synthetic=SyntheticSourceConfig(enabled=use_synthetic, frequency=440.0),
        audio=AudioSourceConfig(
            enabled=use_audio_source,
            signal_gate_threshold=audio_signal_gate_threshold,
            drive_amplitude=audio_drive_amplitude,
            mapping_mode=audio_visual_mapping_mode,
            mapping_alpha=audio_visual_mapping_alpha,
            mapping_f0=audio_visual_mapping_f0,
            mapping_fc=audio_visual_mapping_fc,
            linear_scale=audio_visual_linear_scale,
            linear_offset=audio_visual_linear_offset,
        ),
        camera_frame=CameraFrameSourceConfig(enabled=False),
        prediction_error=PredictionErrorTransformConfig(),
    )


def test_ripple_source_orchestrator_respects_explicit_audio_source_enabled_state():
    engine = _build_engine()
    processor = _FakeProcessor([440.0])
    orchestrator = RippleSourceOrchestrator(
        config=_source_config(use_synthetic=False, use_audio_source=False),
        processor=processor,
        engine=engine,
        resolution=engine.resolution,
        n_sources=1,
    )

    resolved = orchestrator.resolve()

    assert resolved.audio_frequencies is None
    assert resolved.drive_grid is None
    assert resolved.amplitude == 1.0


def test_ripple_source_orchestrator_uses_configured_audio_frequency_mapping():
    engine = _build_engine()
    processor = _FakeProcessor([440.0, 880.0, None])
    orchestrator = RippleSourceOrchestrator(
        config=_source_config(
            use_synthetic=False,
            audio_visual_mapping_alpha=10.0,
            audio_visual_mapping_f0=100.0,
            audio_visual_mapping_fc=10_000.0,
        ),
        processor=processor,
        engine=engine,
        resolution=engine.resolution,
        n_sources=1,
    )

    resolved = orchestrator.resolve()

    expected = map_audio_freq_to_visual_freq(
        np.asarray([440.0, 880.0], dtype=np.float32),
        alpha=10.0,
        f0=100.0,
        fc=10_000.0,
    ).astype(np.float32)
    assert resolved.audio_frequencies is not None
    np.testing.assert_allclose(resolved.audio_frequencies, expected.reshape(1, -1))


def test_ripple_source_orchestrator_scales_audio_only_excitation_by_signal_level():
    engine = _build_engine(amplitude=2.0)
    processor = _FakeProcessor([440.0], current_signal_level=0.25)
    orchestrator = RippleSourceOrchestrator(
        config=_source_config(use_synthetic=False),
        processor=processor,
        engine=engine,
        resolution=engine.resolution,
        n_sources=1,
    )
    orchestrator.set_base_amplitude(engine.amplitude)

    resolved = orchestrator.resolve()

    assert resolved.amplitude == 0.5


def test_ripple_source_orchestrator_skips_correction_when_camera_feed_is_off():
    engine = _build_engine()
    scalar_state = np.array(
        [
            [0.5, -0.25] + [0.0] * 30,
            [0.75, -0.5] + [0.0] * 30,
        ]
        + [[0.0] * 32 for _ in range(22)],
        dtype=np.float32,
    )
    field = np.broadcast_to(
        scalar_state[..., None],
        engine.canvas_shape,
    ).copy()
    engine.Z[:] = field
    engine.Z_old[:] = field
    engine.propagator.Z[:] = field
    engine.propagator.Z_old[:] = field
    orchestrator = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False, frequency=440.0),
            audio=AudioSourceConfig(enabled=False),
            camera_frame=CameraFrameSourceConfig(enabled=False),
            prediction_error=PredictionErrorTransformConfig(
                enabled=True,
                inputs=("camera_frame",),
                activation_function="linear_clipped",
                gain=1.0,
                learning_enabled=False,
                max_output=10.0,
            ),
        ),
        processor=None,
        engine=engine,
        resolution=engine.resolution,
        n_sources=1,
    )

    source_frame = orchestrator.resolve()
    engine.propagate(source_frame.drive_grid)
    prior = engine.get_prior_numpy().copy()
    visual_prediction = orchestrator.predict_visual()
    correction = orchestrator.resolve_visual_observation_correction(
        visual_prediction=visual_prediction,
    )

    assert source_frame.drive_grid is None
    assert correction is None
    np.testing.assert_array_equal(engine.get_field_numpy(), prior)


def test_ripple_source_orchestrator_routes_visual_error_through_readout():
    engine = _build_engine()
    orchestrator = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False),
            audio=AudioSourceConfig(enabled=False),
            camera_frame=CameraFrameSourceConfig(enabled=False),
            prediction_error=PredictionErrorTransformConfig(
                enabled=True,
                activation_function="linear_clipped",
                gain=1.0,
            ),
        ),
        processor=None,
        engine=engine,
        resolution=engine.resolution,
        n_sources=1,
        visual_readout=_OffsetVisualReadout(),
    )
    observation = np.ones(engine.canvas_shape, dtype=np.float32)
    orchestrator.camera_source.excitation = lambda: observation
    engine.propagate()

    prediction = orchestrator.predict_visual()
    correction = orchestrator.correct_from_observations(
        visual_prediction=prediction,
    )

    np.testing.assert_allclose(prediction, 0.25)
    np.testing.assert_allclose(correction, 1.5)
    np.testing.assert_allclose(engine.get_field_numpy(), 1.5)
