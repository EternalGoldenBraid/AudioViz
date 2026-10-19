from dataclasses import replace

import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.opengl_predictive import OpenGLPredictiveModel
from audioviz.readouts.predictive_coding import (
    PredictiveCodingConfig, PredictiveCodingHiddenCoupling, PredictiveCodingRGBReadout, infer_joint,
)
from audioviz.readouts.spatial_precision import GaussianSpatialPrecision
from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty


@pytest.fixture
def shader_engine(qapp):
    from PyQt5 import QtGui

    context = QtGui.QOpenGLContext()
    surface_format = QtGui.QSurfaceFormat()
    surface_format.setVersion(4, 3)
    surface_format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    context.setFormat(surface_format)
    if not context.create() or (context.format().majorVersion(), context.format().minorVersion()) < (4, 3):
        pytest.skip("Real OpenGL 4.3 context unavailable; run with QT_QPA_PLATFORM=xcb")
    engine = RippleEngine(
        resolution=(19, 23), plane_size_m=(10, 10), use_shader=True, boundary_condition="neumann",
    )
    yield engine
    engine.propagator.close()


@pytest.mark.parametrize("coupled,learning", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("precision", (
    None, GaussianSpatialPrecision(gaze=(0.2, 0.7), sigma=0.18),
    GaussianSpatialPrecision(sample=False),
))
def test_shader_joint_frames_match_numpy_with_missing_evidence_and_learning(shader_engine, coupled, learning, precision):
    engine = shader_engine
    config = PredictiveCodingConfig(
        dtype="float32", inference_steps=3, cross_modal_enabled=coupled,
        learning_enabled=learning, learning_rate=.07, weight_decay=.01,
        gradient_clip=.08, weight_clip=.95,
        spatial_precision=precision,
    )
    model = OpenGLPredictiveModel(engine, config, audio_enabled=True)
    camera = PredictiveCodingRGBReadout(canvas_shape=engine.canvas_shape, config=config)
    audio = PredictiveCodingAudioReadout(canvas_shape=engine.canvas_shape, config=config)
    coupling = PredictiveCodingHiddenCoupling(camera, audio) if coupled else None
    if precision is not None:
        for branch in (model.camera, model.audio, camera, audio):
            branch.set_diagnostics_enabled(True)
    rng = np.random.default_rng(42)
    for frame_index, (image_present, audio_present, states) in enumerate((
        (True, True, (True, True)),
        (False, True, (True, True)),
        (True, False, (True, False)),
        (False, False, (False, False)),
    )):
        if precision is not None:
            moved = replace(precision, gaze=(0.2 + 0.15 * frame_index, 0.7 - 0.1 * frame_index))
            for branch in (model.camera, camera, audio):
                branch.update_config(spatial_precision=moved)
        engine.propagate(rng.normal(0, .1, engine.canvas_shape).astype(np.float32))
        prior = engine.get_prior()
        host_prior = prior.download()
        prediction, audio_prediction = model.predict(prior)
        if coupling:
            expected_prediction, expected_audio = coupling.predict(host_prior)
        else:
            expected_prediction, expected_audio = camera.predict(host_prior), audio.predict(host_prior)
        np.testing.assert_allclose(prediction.download(), expected_prediction, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(audio_prediction.download(), expected_audio, atol=2e-6, rtol=2e-5)
        frozen_prediction = prediction.download()
        image = rng.random(engine.canvas_shape, dtype=np.float32) if image_present else None
        spectrum = rng.random(16, dtype=np.float32) if audio_present else None
        delta = model.infer(prior, image, spectrum, states=states)
        expected_delta = infer_joint(
            host_prior, ((camera, image), (audio, spectrum)), coupling=coupling,
            stream_states={camera: states[0], audio: states[1]},
        )
        np.testing.assert_allclose(delta.download(), expected_delta, atol=3e-6, rtol=4e-4)
        for branch, reference in ((model.camera, camera), (model.audio, audio)):
            if precision is not None:
                np.testing.assert_allclose(
                    branch.diagnostics.inference_energy, reference.diagnostics.inference_energy,
                    atol=2e-6, rtol=2e-5,
                )
                if reference.diagnostics.observation_present:
                    assert branch.diagnostics.visual.mean == pytest.approx(
                        reference.diagnostics.visual.mean, abs=2e-6,
                    )
                    assert branch.diagnostics.visual.variance == pytest.approx(
                        reference.diagnostics.visual.variance, abs=2e-6,
                    )
            np.testing.assert_allclose(branch.hidden.download(), reference.hidden, atol=3e-6, rtol=3e-5)
            np.testing.assert_allclose(branch.hidden_error.download(), reference.hidden_error, atol=3e-6)
            for actual, expected in zip(branch.weights()[:4], (
                reference.canvas_weights, reference.visual_weights, reference.stream_state_weights, reference.stream_state_bias,
            )):
                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
            if coupled:
                np.testing.assert_allclose(branch.weights()[4], coupling.weights[reference], atol=3e-6)
        np.testing.assert_array_equal(prediction.download(), frozen_prediction)
        engine.apply_observation_correction(delta)
        np.testing.assert_allclose(engine.get_field_numpy(), host_prior + expected_delta, atol=3e-6)
    learned = model.weights.download()
    model.reset()
    np.testing.assert_array_equal(model.camera.hidden.download(), 0)
    np.testing.assert_array_equal(model.weights.download(), learned)
    model.update_config(inference_steps=2, learning_enabled=False)
    assert model.audio.config.inference_steps == 2
    with pytest.raises(ValueError, match="rebuilding"):
        model.update_config(hidden_channels=7)


@pytest.mark.parametrize("boundary", ("neumann", "cyclic"))
def test_shader_native_wave_rgb_parity_and_both_buffer_correction(shader_engine, monkeypatch, boundary):
    engine = shader_engine
    cpu = RippleEngine(
        resolution=engine.resolution, plane_size_m=engine.plane_size_m, boundary_condition=boundary,
    )
    engine.boundary_condition = engine.propagator.boundary_condition = cpu.boundary_condition
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="native storage"):
        engine.propagator.apply_correction(np.zeros(engine.canvas_shape, dtype=np.float32))
    for driven in (True, False, True, False):
        drive = rng.random(engine.canvas_shape, dtype=np.float32) if driven else None
        engine.propagate(drive)
        cpu.propagate(drive)
        # Native propagation and preview must not invoke texture readback.
        with monkeypatch.context() as patch:
            patch.setattr(engine.propagator, "get_state", lambda: pytest.fail("Full-field readback"))
            preview = engine.get_field_preview(max_side=11)
            assert max(preview.shape[:2]) <= 11 and not preview.flags.writeable
            correction = engine.get_field()
        change = rng.normal(0, .02, engine.canvas_shape).astype(np.float32)
        correction.upload(change)
        engine.apply_observation_correction(correction)
        cpu.apply_observation_correction(change)
        np.testing.assert_allclose(engine.get_field_numpy(), cpu.get_field_numpy(), atol=5e-6)


def test_shader_diagnostics_and_divergence(shader_engine):
    config = PredictiveCodingConfig(dtype="float32", inference_steps=2, learning_enabled=True)
    model = OpenGLPredictiveModel(shader_engine, config, audio_enabled=False)
    camera = PredictiveCodingRGBReadout(canvas_shape=shader_engine.canvas_shape, config=config)
    model.camera.set_diagnostics_enabled(True)
    model.camera.set_spatial_preview_enabled(True)
    camera.set_diagnostics_enabled(True)
    shader_engine.propagate(np.ones(shader_engine.canvas_shape, dtype=np.float32) * .1)
    prior = shader_engine.get_prior()
    model.predict(prior)
    camera.predict(prior.download())
    image = np.full(shader_engine.canvas_shape, .3, dtype=np.float32)
    model.infer(prior, image, None, states=(True, False))
    infer_joint(prior.download(), ((camera, image),), stream_states={camera: True})
    actual, expected = model.camera.diagnostics, camera.diagnostics
    np.testing.assert_allclose(actual.inference_energy, expected.inference_energy, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(actual.hidden.channel_means, expected.hidden.channel_means, atol=1e-6)
    assert actual.spatial is not None and not actual.spatial.hidden.flags.writeable
    assert actual.observation_present and actual.stream_state.observed
    with pytest.raises(ValueError, match="finite"):
        model.infer(prior, np.full(shader_engine.canvas_shape, np.nan), None, states=(True, False))
    model.camera.hidden.upload(np.full(model.camera.hidden.shape, np.nan, dtype=np.float32))
    with pytest.raises(FloatingPointError, match="diverged"):
        model.predict(prior)
    with pytest.raises(ValueError, match="float32"):
        OpenGLPredictiveModel(shader_engine, replace(config, dtype="float64"), audio_enabled=False)


def test_shader_precision_stays_native_without_full_field_downloads(shader_engine, monkeypatch):
    from audioviz.utils.opengl_compute import GLArray

    model = OpenGLPredictiveModel(
        shader_engine, PredictiveCodingConfig(
            dtype="float32", spatial_precision=GaussianSpatialPrecision(),
            learning_enabled=True, inference_steps=2,
        ), audio_enabled=False,
    )
    shader_engine.propagate()
    image = model.preprocess_camera(np.ones((19, 23, 3), dtype=np.uint8) * 200)
    download = GLArray.download

    def bounded_download(array):
        assert array.size <= 32, "Inference must not download images, hidden fields or precision maps"
        return download(array)

    with monkeypatch.context() as patch:
        patch.setattr(GLArray, "download", bounded_download)
        model.predict(shader_engine.get_prior())
        model.infer(shader_engine.get_prior(), image, None, states=(True, False))


@pytest.mark.parametrize("precision", (
    GaussianSpatialPrecision(peak=1),
    GaussianSpatialPrecision(gaze=(0, 1), sigma=1, baseline=0, peak=1),
    GaussianSpatialPrecision(
        gaze=(np.int32(0), np.int32(1)), sigma=np.int32(1),
        baseline=np.int32(0), peak=np.int32(1),
    ),
))
def test_shader_float_precision_uniforms_accept_integral_parameters(shader_engine, precision):
    config = PredictiveCodingConfig(dtype="float32", inference_steps=2, spatial_precision=precision)
    model = OpenGLPredictiveModel(shader_engine, config, audio_enabled=False)
    reference = PredictiveCodingRGBReadout(canvas_shape=shader_engine.canvas_shape, config=config)
    shader_engine.propagate()
    prior = shader_engine.get_prior()
    prediction, _ = model.predict(prior)
    np.testing.assert_allclose(prediction.download(), reference.predict(prior.download()), atol=2e-6)
    observation = np.full(shader_engine.canvas_shape, 0.3, dtype=np.float32)
    correction = model.infer(prior, observation, None, states=(True, False))
    expected = infer_joint(prior.download(), ((reference, observation),), stream_states={reference: True})
    np.testing.assert_allclose(correction.download(), expected, atol=3e-6)


def test_shader_uncertainty_matches_gaussian_reference_and_downloads_only_gain_grid(shader_engine, monkeypatch):
    from audioviz.utils.opengl_compute import GLArray

    shape = shader_engine.resolution
    config = PredictiveCodingConfig(dtype="float32", inference_steps=2)
    shader = OpenGLPredictiveModel(shader_engine, config, audio_enabled=False)
    readout = PredictiveCodingRGBReadout(canvas_shape=(*shape, 3), config=config)
    native = GaussianSensoryUncertainty(shader.camera, device=shader_engine.propagator.compute)
    reference = GaussianSensoryUncertainty(readout)
    rng = np.random.default_rng(74)
    shader_engine.propagate(rng.normal(0, .1, (*shape, 3)).astype(np.float32))
    prior = shader_engine.get_prior()
    host_prior = prior.download()
    for observed, sampling, gaze in (
        (True, True, (0.2, 0.7)), (False, True, (0.5, 0.5)),
        (True, False, (0, 1)), (True, True, (0.8, 0.3)),
    ):
        precision = GaussianSpatialPrecision(gaze=gaze, sample=sampling, baseline=0, peak=1)
        shader.camera.update_config(spatial_precision=precision)
        readout.update_config(spatial_precision=precision)
        shader.predict(prior)
        readout.predict(host_prior)
        native.predict(dt=.1)
        reference.predict(dt=.1)
        image = rng.random((*shape, 3), dtype=np.float32) if observed else None
        shader.infer(prior, image, None, states=(True, False))
        readout.infer(host_prior, image, stream_state=True)
        native.condition(precision, observed=observed)
        reference.condition(precision, observed=observed)
        np.testing.assert_allclose(native.variance.download(), reference.variance, atol=2e-7)
        download = GLArray.download

        def only_grid(array):
            assert array.size == 35, "Advisory scoring must only download the candidate grid"
            return download(array)

        for max_side in (11, 1, 64, 11):
            expected = reference.expected_information_gain(
                precision, prior=host_prior, dt=.1, grid_shape=(5, 7), max_side=max_side,
            )
            with monkeypatch.context() as patch:
                patch.setattr(GLArray, "download", only_grid)
                actual = native.expected_information_gain(
                    precision, prior=prior, dt=.1, grid_shape=(5, 7), max_side=max_side,
                )
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    tiny_precision = GaussianSpatialPrecision(baseline=1e-8, peak=0)
    np.testing.assert_allclose(
        native.expected_information_gain(tiny_precision, prior=prior, dt=.1, grid_shape=(3, 3)),
        reference.expected_information_gain(tiny_precision, prior=host_prior, dt=.1, grid_shape=(3, 3)),
        rtol=1e-5, atol=1e-12,
    )


@pytest.mark.parametrize("kind", ("rgb", "rgba", "gray", "float"))
def test_shader_camera_preprocessing_and_exact_levels(shader_engine, kind):
    from audioviz.sources.camera import CameraFrameSource
    from audioviz.visualization.ripple_renderers import OpenGLFieldRenderer, _percentile_abs_limit

    model = OpenGLPredictiveModel(
        shader_engine, PredictiveCodingConfig(dtype="float32", inference_steps=1), audio_enabled=False,
    )
    rng = np.random.default_rng(31)
    shape = (11, 17) if kind == "gray" else (11, 17, 4 if kind == "rgba" else 3)
    frame = rng.integers(0, 256, shape, dtype=np.uint8)
    if kind == "float":
        frame = frame.astype(np.float32)
        frame[0, 0] = (np.nan, np.inf, -np.inf)
    image = model.preprocess_camera(frame, gain=.7)
    expected = CameraFrameSource.frame_to_excitation_grid(frame, resolution=shader_engine.resolution, gain=.7)
    np.testing.assert_allclose(image.download(), expected, atol=1e-7)
    renderer = OpenGLFieldRenderer(rgb_canvas_enabled=True)
    try:
        assert renderer._native_percentile(image, 98.) == pytest.approx(_percentile_abs_limit(expected, percentile=98.), abs=1e-7)
        renderer.set_rgb_canvas_enabled(False)
        assert renderer._native_percentile(image, 98.) == pytest.approx(
            _percentile_abs_limit(expected.mean(axis=2), percentile=98.), abs=5e-7,
        )
        # Native camera evidence is copied on-device into the settling graph.
        shader_engine.propagate()
        model.predict(shader_engine.get_prior())
        model.infer(shader_engine.get_prior(), image, None, states=(True, False))
    finally:
        renderer.widget.close()
