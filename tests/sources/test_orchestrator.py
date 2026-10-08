import numpy as np

from audioviz.engine import RippleEngine
from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
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


class _RecordingVisualReadout(PredictiveCodingRGBReadout):
    def infer(self, prior, observation, *, stream_state=None):
        self.received_prior = prior.copy()
        self.received_observation = None if observation is None else observation.copy()
        self.received_stream_state = stream_state
        return super().infer(prior, observation, stream_state=stream_state)


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


def test_captured_audio_does_not_drive_waves_or_scale_external_stimulation():
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

    assert resolved.amplitude == 2.0
    assert resolved.drive_grid is None


def test_ripple_source_orchestrator_infers_off_context_without_a_camera_image():
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
    orchestrator.predict_visual()
    correction = orchestrator.resolve_visual_observation_correction()

    assert source_frame.drive_grid is None
    assert correction is not None
    assert np.any(correction != 0)
    assert orchestrator.visual_observation is None
    np.testing.assert_array_equal(engine.get_field_numpy(), prior)


def test_ripple_source_orchestrator_routes_inference_through_readout():
    engine = _build_engine()
    readout = _RecordingVisualReadout(canvas_shape=engine.canvas_shape)
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
        visual_readout=readout,
    )
    observation = np.ones(engine.canvas_shape, dtype=np.float32)
    orchestrator.camera_source.excitation = lambda: observation
    engine.propagate()

    prediction = orchestrator.predict_visual()
    correction = orchestrator.correct_from_observations()

    np.testing.assert_allclose(prediction, 0.0)
    np.testing.assert_array_equal(readout.received_prior, engine.get_prior_numpy())
    np.testing.assert_array_equal(readout.received_observation, observation)
    assert readout.received_stream_state is orchestrator.camera_source.enabled
    assert np.any(correction != 0.0)
    np.testing.assert_allclose(engine.get_field_numpy(), correction)


def test_visual_pathway_learning_does_not_change_conductances_or_wave_velocity():
    engine = _build_engine()
    orchestrator = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(
                learning_enabled=True, learning_rate=1.0
            ),
        ),
        processor=None,
        engine=engine,
        resolution=engine.resolution,
        n_sources=1,
    )
    reads = []
    observation = np.full(engine.canvas_shape, 0.7, dtype=np.float32)

    def observe():
        reads.append(True)
        return observation

    orchestrator.camera_source.excitation = observe
    weights_before = orchestrator.visual_readout.visual_weights.copy()
    for _ in range(10):
        engine.propagate()
        prior = engine.get_prior_numpy().copy()
        velocity_before = engine.Z - engine.Z_old
        orchestrator.predict_visual()
        orchestrator.correct_from_observations()
        np.testing.assert_array_equal(engine.get_prior_numpy(), prior)
        np.testing.assert_allclose(engine.Z - engine.Z_old, velocity_before, atol=1e-6)
        np.testing.assert_array_equal(engine.prediction_horizontal_edge_weights, 1.0)
        np.testing.assert_array_equal(engine.prediction_vertical_edge_weights, 1.0)

    assert len(reads) == 10
    assert np.any(orchestrator.visual_readout.visual_weights != weights_before)


def test_missing_camera_frame_infers_context_without_learning_from_black():
    engine = _build_engine()
    orchestrator = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(learning_enabled=True),
        ),
        processor=None,
        engine=engine,
        resolution=engine.resolution,
        n_sources=1,
    )
    orchestrator.set_diagnostics_enabled(True)
    orchestrator.camera_source.excitation = lambda: None
    engine.propagate(np.ones(engine.canvas_shape))
    orchestrator.predict_visual()
    hidden_before = orchestrator.visual_readout.hidden.copy()
    weights_before = orchestrator.visual_readout.visual_weights.copy()

    assert orchestrator.correct_from_observations() is not None

    assert np.any(orchestrator.visual_readout.hidden != hidden_before)
    np.testing.assert_array_equal(orchestrator.visual_readout.visual_weights, weights_before)
    assert not orchestrator.visual_readout.diagnostics.observation_present
    assert orchestrator.visual_readout.diagnostics.stream_state.observed is False
    assert orchestrator.visual_readout.diagnostics.weight_update_norm > 0


def test_audio_and_camera_joint_inference_use_one_observation_and_preserve_wave_velocity(monkeypatch):
    import audioviz.sources.audio as audio_module

    processor = _FakeProcessor([440], current_signal_level=0)
    processor.frame_counter = 1
    processor.spectrogram_buffers = [np.full((32, 2), 0.8)]
    processor.stft_window = np.ones(16)
    processor.n_mels = None
    engine = _build_engine()
    orchestrator = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            audio=AudioSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(learning_enabled=True),
        ),
        processor=processor, engine=engine, resolution=engine.resolution, n_sources=1,
    )
    clock = [0.0]
    monkeypatch.setattr(audio_module, "monotonic", lambda: clock[0])
    reads = []

    def camera():
        reads.append(True)
        return np.full(engine.canvas_shape, 0.6)

    orchestrator.camera_source.excitation = camera
    orchestrator.set_diagnostics_enabled(True)
    engine.propagate()
    orchestrator.predict_visual()
    velocity = engine.Z - engine.Z_old
    orchestrator.correct_from_observations()
    assert reads == [True]
    np.testing.assert_array_equal(orchestrator.visual_observation, 0.6)
    np.testing.assert_allclose(orchestrator.audio_source.recent_observation(), 0.1)
    assert not orchestrator.audio_source.has_new_observation()
    assert orchestrator.audio_readout.diagnostics is not None
    assert orchestrator.visual_readout.diagnostics is not None
    assert orchestrator.audio_readout.diagnostics.inference_energy == orchestrator.visual_readout.diagnostics.inference_energy
    np.testing.assert_allclose(engine.Z - engine.Z_old, velocity, atol=1e-6)
    np.testing.assert_array_equal(engine.prediction_horizontal_edge_weights, 1)
    audio_weights = orchestrator.audio_readout.visual_weights.copy()
    orchestrator.camera_source.excitation = lambda: None
    engine.propagate()
    orchestrator.predict_visual()
    assert orchestrator.correct_from_observations() is not None
    assert orchestrator.visual_observation is None
    np.testing.assert_array_equal(orchestrator.audio_readout.visual_weights, audio_weights)

    # A new silent audio block is evidence; an unchanged generation is not.
    processor.frame_counter += 1
    processor.spectrogram_buffers[0].fill(0)
    orchestrator.predict_visual()
    assert orchestrator.correct_from_observations() is not None
    np.testing.assert_array_equal(orchestrator.audio_readout.diagnostics.observation_means, 0)
    assert orchestrator.audio_source.has_recent_observation()
    clock[0] = 0.6
    assert not orchestrator.audio_source.has_recent_observation()
    assert orchestrator.audio_source.recent_observation() is None
    orchestrator.audio_source.set_enabled(False)
    processor.frame_counter += 1
    assert orchestrator.audio_source.observation() is None
    orchestrator.reset_readouts()
    assert np.all(orchestrator.audio_readout.hidden == 0)
    assert orchestrator.audio_readout.diagnostics is None
