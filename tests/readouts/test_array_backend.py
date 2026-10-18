import builtins
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.predictive_coding import PredictiveCodingHiddenCoupling, infer_joint
from audioviz.sources import (
    AudioSourceConfig,
    CameraFrameSourceConfig,
    RippleSourceOrchestrator,
    RippleSourceOrchestratorConfig,
)
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS
from audioviz.utils.source_preview import SourceSceneMapping
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.utils.array_backend import project_channels


def test_numpy_inference_does_not_require_cupy(monkeypatch):
    original_import = builtins.__import__

    def forbid_cupy(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            pytest.fail("NumPy inference must not import CuPy.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_cupy)
    engine = RippleEngine(resolution=(4, 6), plane_size_m=(1, 1))
    source = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(),
        processor=None, engine=engine, resolution=engine.resolution, n_sources=1,
    )
    engine.propagate()
    assert source.visual_readout.backend is np
    assert isinstance(source.predict_visual(), np.ndarray)
    assert isinstance(source.correct_from_observations(), np.ndarray)
    source.set_diagnostics_enabled(True)
    source.predict_visual()
    source.correct_from_observations()
    assert source.visual_readout.diagnostics is not None


def _assert_snapshot_equal(expected, actual, *, rtol=1e-9, atol=1e-10):
    if is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_snapshot_equal(getattr(expected, field.name), getattr(actual, field.name), rtol=rtol, atol=atol)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for left, right in zip(expected, actual):
            _assert_snapshot_equal(left, right, rtol=rtol, atol=atol)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert not actual.flags.writeable
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    elif expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("coupled", (False, True))
@pytest.mark.parametrize("learning", (False, True))
@pytest.mark.parametrize("diagnostics", (False, True))
@pytest.mark.parametrize("evidence", ("both", "camera", "audio", "neither"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_gpu_matches_cpu_prediction_inference_learning_and_snapshots(
    cupy_backend, coupled, learning, diagnostics, evidence, dtype,
):
    cp = cupy_backend
    config = PredictiveCodingConfig(
        inference_steps=3, learning_enabled=learning,
        learning_rate=.1, weight_decay=.002, cross_modal_enabled=coupled, dtype=dtype,
    )
    shape = (5, 7, 3)
    pairs = []
    for xp in (np, cp):
        camera = PredictiveCodingRGBReadout(canvas_shape=shape, config=config, backend=xp)
        audio = PredictiveCodingAudioReadout(canvas_shape=shape, config=config, backend=xp)
        coupling = PredictiveCodingHiddenCoupling(camera, audio) if coupled else None
        rng = np.random.default_rng(4)
        for branch in (camera, audio):
            branch.hidden[:] = xp.asarray(rng.normal(scale=.2, size=branch.hidden.shape))
            branch.set_diagnostics_enabled(diagnostics)
            branch.set_spatial_preview_enabled(diagnostics)
            if coupling is not None:
                coupling.weights[branch][:] = xp.asarray(
                    rng.normal(scale=.1, size=coupling.weights[branch].shape)
                )
        pairs.append((camera, audio, coupling))
    rng = np.random.default_rng(7)
    rtol, atol = (2e-5, 2e-7) if dtype == "float32" else (1e-9, 1e-10)
    for frame in range(3):
        prior = rng.normal(scale=.1, size=shape).astype(np.float32)
        image = rng.random(shape).astype(np.float32) if evidence in ("both", "camera") else None
        spectrum = rng.random(AUDIO_SPECTRAL_BANDS).astype(np.float32) if evidence in ("both", "audio") else None
        predictions, corrections = [], []
        for xp, (camera, audio, coupling) in zip((np, cp), pairs):
            canvas = xp.asarray(prior)
            predictions.append(
                coupling.predict(canvas) if coupling is not None
                else (camera.predict(canvas), audio.predict(canvas))
            )
            corrections.append(infer_joint(
                canvas, ((camera, image), (audio, spectrum)), coupling=coupling,
                stream_states={camera: bool(frame % 2), audio: not bool(frame % 2)},
            ))
            np.testing.assert_array_equal(cp.asnumpy(canvas), prior)
        for expected, actual in zip(predictions[0], predictions[1]):
            assert isinstance(actual, cp.ndarray)
            assert actual.dtype == cp.float32
            np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=max(rtol, 1e-6), atol=max(atol, 1e-8))
        assert isinstance(corrections[1], cp.ndarray)
        assert corrections[1].dtype == cp.float32
        np.testing.assert_allclose(cp.asnumpy(corrections[1]), corrections[0], rtol=max(rtol, 1e-6), atol=max(atol, 1e-8))
        for expected, actual in zip(pairs[0][:2], pairs[1][:2]):
            assert actual.backend is cp
            for name in (
                "hidden", "hidden_error", "visual_error", "canvas_weights",
                "visual_weights", "stream_state_weights", "stream_state_bias",
            ):
                array = getattr(actual, name)
                assert isinstance(array, cp.ndarray)
                assert array.dtype == cp.dtype(dtype)
                np.testing.assert_allclose(cp.asnumpy(array), getattr(expected, name), rtol=rtol, atol=atol)
            _assert_snapshot_equal(expected.diagnostics, actual.diagnostics, rtol=rtol, atol=atol)
            np.testing.assert_allclose(actual.stream_state_prediction, expected.stream_state_prediction, rtol=rtol, atol=atol)
        if coupled:
            for expected_branch, actual_branch in zip(pairs[0][:2], pairs[1][:2]):
                np.testing.assert_allclose(
                    cp.asnumpy(pairs[1][2].weights[actual_branch]),
                    pairs[0][2].weights[expected_branch], rtol=rtol, atol=atol,
                )


def test_gpu_orchestrator_keeps_the_graph_on_device_without_host_downloads(cupy_backend, monkeypatch):
    cp = cupy_backend
    engine = RippleEngine(resolution=(8, 12), plane_size_m=(1, 1), use_gpu=True)
    processor = SimpleNamespace(
        frame_counter=0, n_mels=None, stft_window=np.ones(16),
        spectrogram_buffers=[np.ones((16, 1), dtype=np.float32)],
    )
    source = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            audio=AudioSourceConfig(enabled=True),
            camera_frame=CameraFrameSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(
                cross_modal_enabled=True, learning_enabled=True,
            ),
        ),
        processor=processor, engine=engine, resolution=engine.resolution, n_sources=1,
    )
    source.camera_source.excitation = lambda: np.full(engine.canvas_shape, .5, dtype=np.float32)

    def forbid_download(*args, **kwargs):
        pytest.fail("The numerical graph must not download full arrays to the host.")

    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", forbid_download)
        patch.setattr(engine, "get_prior_numpy", forbid_download)
        patch.setattr(engine, "get_field_numpy", forbid_download)
        for _ in range(3):
            processor.frame_counter += 1
            engine.propagate()
            velocity = engine.Z - engine.Z_old
            prediction = source.predict_visual()
            correction = source.correct_from_observations()
            assert isinstance(prediction, cp.ndarray)
            assert isinstance(source.audio_prediction, cp.ndarray)
            assert isinstance(correction, cp.ndarray)
            assert engine.get_prior() is engine.prior_Z
            assert engine.get_field() is engine.Z
            assert bool(cp.allclose(engine.Z - engine.Z_old, velocity, atol=1e-7))
            assert bool(cp.isfinite(engine.Z).all())
        weights = source.visual_readout.visual_weights.copy()
        source.reset_readouts()
        assert bool(cp.all(source.visual_readout.hidden == 0))
        assert bool(cp.all(source.audio_readout.hidden == 0))
        assert bool(cp.all(source.visual_readout.visual_weights == weights))
        assert source.visual_readout.stream_state_prediction is None


def test_mixed_backends_are_rejected_explicitly(cupy_backend):
    cp = cupy_backend
    cpu = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3))
    gpu = PredictiveCodingAudioReadout(canvas_shape=cpu.canvas_shape, backend=cp)
    with pytest.raises(ValueError, match="same backend"):
        PredictiveCodingHiddenCoupling(cpu, gpu)
    with pytest.raises(ValueError, match="array backend"):
        infer_joint(np.zeros(cpu.canvas_shape), ((cpu, None), (gpu, None)), stream_states={cpu: True, gpu: False})
    engine = RippleEngine(resolution=(2, 3), plane_size_m=(1, 1), use_gpu=True)
    with pytest.raises(ValueError, match="engine's array backend"):
        RippleSourceOrchestrator(
            config=RippleSourceOrchestratorConfig(), processor=None, engine=engine,
            resolution=engine.resolution, n_sources=1, visual_readout=cpu,
        )


def test_joint_inference_rejects_mixed_precision():
    camera = PredictiveCodingRGBReadout(
        canvas_shape=(2, 3, 3), config=PredictiveCodingConfig(dtype="float32"),
    )
    audio = PredictiveCodingAudioReadout(canvas_shape=camera.canvas_shape)
    with pytest.raises(ValueError, match="same dtype"):
        PredictiveCodingHiddenCoupling(camera, audio)
    with pytest.raises(ValueError, match="share a dtype"):
        infer_joint(
            np.zeros(camera.canvas_shape), ((camera, None), (audio, None)),
            stream_states={camera: True, audio: True},
        )


def test_gpu_validation_rejects_bad_inputs_and_divergence_before_learning(cupy_backend):
    cp = cupy_backend
    readout = PredictiveCodingRGBReadout(
        canvas_shape=(2, 3, 3), backend=cp,
        config=PredictiveCodingConfig(learning_enabled=True),
    )
    prior = cp.zeros(readout.canvas_shape)
    with pytest.raises(ValueError, match="shape"):
        readout.predict(cp.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        readout.infer(prior, cp.full(readout.canvas_shape, cp.nan))
    audio = PredictiveCodingAudioReadout(canvas_shape=readout.canvas_shape, backend=cp)
    with pytest.raises(ValueError, match="band count"):
        audio.infer(prior, cp.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        audio.infer(prior, cp.full(AUDIO_SPECTRAL_BANDS, cp.inf))
    weights_before = readout.canvas_weights.copy()
    readout.hidden.fill(cp.inf)
    with pytest.raises(FloatingPointError, match="diverged"):
        readout.infer(prior, cp.zeros(readout.canvas_shape))
    np.testing.assert_array_equal(cp.asnumpy(readout.canvas_weights), cp.asnumpy(weights_before))
    readout.reset()
    readout.stream_state_bias.fill(cp.inf)
    with pytest.raises(FloatingPointError, match="diverged"):
        readout.infer(prior, None, stream_state=True)


def test_gpu_previews_download_only_bounded_samples(cupy_backend, monkeypatch):
    cp = cupy_backend
    field = cp.arange(100 * 200 * 6, dtype=cp.float64).reshape(100, 200, 6)
    transfers = []
    asnumpy = cp.asnumpy

    def record_download(values, *args, **kwargs):
        transfers.append(values.shape)
        return asnumpy(values, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", record_download)
        preview = spatial_preview(field)
        spectrum = SourceSceneMapping("audio", "Audio", "spectrum", "sensory").sample(cp.arange(16))
    assert transfers == [(32, 64, 6), (16,)]
    assert isinstance(preview, np.ndarray)
    assert preview.shape == (32, 64, 6)
    assert not preview.flags.writeable
    assert not spectrum.flags.writeable
    saved = preview.copy()
    field.fill(0)
    np.testing.assert_array_equal(preview, saved)


@pytest.mark.parametrize("inputs,outputs", ((3, 6), (6, 3), (6, 6), (33, 3), (3, 33)))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_gpu_channel_projection_handles_strided_inputs_and_large_channel_fallback(
    cupy_backend, inputs, outputs, dtype,
):
    cp = cupy_backend
    rng = np.random.default_rng(18)
    base = rng.normal(size=(4, 7, inputs)).astype(dtype)
    activity = base[:, ::-1]
    weights = rng.normal(size=(outputs, inputs)).astype(dtype).T
    actual = project_channels(cp.asarray(base)[:, ::-1], cp.asarray(weights))
    np.testing.assert_allclose(
        cp.asnumpy(actual), activity @ weights,
        rtol=1e-5 if dtype == "float32" else 1e-12,
        atol=1e-6 if dtype == "float32" else 1e-12,
    )


def test_float32_learning_stays_close_to_float64_over_repeated_frames(cupy_backend):
    cp = cupy_backend
    pairs = []
    for xp, dtype in ((np, "float64"), (cp, "float32")):
        config = PredictiveCodingConfig(
            dtype=dtype, inference_steps=3, learning_enabled=True, cross_modal_enabled=True,
        )
        camera = PredictiveCodingRGBReadout(canvas_shape=(3, 5, 3), backend=xp, config=config)
        audio = PredictiveCodingAudioReadout(canvas_shape=camera.canvas_shape, backend=xp, config=config)
        pairs.append((camera, audio, PredictiveCodingHiddenCoupling(camera, audio)))
    rng = np.random.default_rng(9)
    for _ in range(60):
        prior = rng.normal(scale=.2, size=(3, 5, 3)).astype(np.float32)
        image = rng.random(prior.shape).astype(np.float32)
        spectrum = rng.random(AUDIO_SPECTRAL_BANDS).astype(np.float32)
        for camera, audio, coupling in pairs:
            canvas = camera.backend.asarray(prior)
            coupling.predict(canvas)
            infer_joint(
                canvas, ((camera, image), (audio, spectrum)), coupling=coupling,
                stream_states={camera: True, audio: True},
            )
    for reference, actual in zip(pairs[0][:2], pairs[1][:2]):
        for name in ("hidden", "canvas_weights", "visual_weights", "stream_state_weights", "stream_state_bias"):
            np.testing.assert_allclose(cp.asnumpy(getattr(actual, name)), getattr(reference, name), rtol=1e-4, atol=1e-5)
