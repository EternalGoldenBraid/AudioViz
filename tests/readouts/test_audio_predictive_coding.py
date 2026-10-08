from copy import deepcopy

import numpy as np

from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.predictive_coding import infer_joint
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS, spectral_observation


def test_joint_updates_follow_nonlinear_energy_gradients_and_do_not_depend_on_branch_order():
    config = PredictiveCodingConfig(hidden_channels=2, inference_steps=1, learning_enabled=False)
    shape = (2, 3, 3)
    rng = np.random.default_rng(2)
    canvas = rng.normal(scale=0.1, size=shape)
    camera = PredictiveCodingRGBReadout(canvas_shape=shape, config=config)
    audio = PredictiveCodingAudioReadout(canvas_shape=shape, config=config)
    camera.hidden[:] = rng.normal(scale=0.1, size=camera.hidden.shape)
    audio.hidden[:] = rng.normal(scale=0.1, size=audio.hidden.shape)
    image = rng.random(shape)
    spectrum = rng.random(AUDIO_SPECTRAL_BANDS)
    reverse_camera, reverse_audio = deepcopy(camera), deepcopy(audio)
    before_camera, before_audio = camera.hidden.copy(), audio.hidden.copy()
    epsilon = 1e-6

    def energy():
        return camera.energy(canvas, image) + audio.energy(canvas, spectrum)

    def derivative(values, index):
        original = values[index]
        values[index] = original + epsilon
        plus = energy()
        values[index] = original - epsilon
        minus = energy()
        values[index] = original
        return (plus - minus) / (2 * epsilon)

    canvas_gradient = derivative(canvas, (0, 1, 2))
    camera_gradient = derivative(camera.hidden, (1, 1, 0))
    audio_gradient = derivative(audio.hidden, (0, 0, 1))
    correction = infer_joint(canvas, ((camera, image), (audio, spectrum)))
    reverse = infer_joint(canvas, ((reverse_audio, spectrum), (reverse_camera, image)))
    pixels = shape[0] * shape[1]

    np.testing.assert_allclose(correction[0, 1, 2], -config.canvas_rate * pixels * canvas_gradient, rtol=1e-5)
    np.testing.assert_allclose(camera.hidden[1, 1, 0] - before_camera[1, 1, 0],
                               -config.inference_rate * pixels * camera_gradient, rtol=1e-5)
    np.testing.assert_allclose(audio.hidden[0, 0, 1] - before_audio[0, 0, 1],
                               -config.inference_rate * pixels * audio_gradient, rtol=1e-5)
    np.testing.assert_array_equal(correction, reverse)
    np.testing.assert_array_equal(camera.hidden, reverse_camera.hidden)
    np.testing.assert_array_equal(audio.hidden, reverse_audio.hidden)


def test_audio_learning_is_local_and_preview_collection_does_not_change_it():
    config = PredictiveCodingConfig(learning_enabled=True, learning_rate=0.1)
    quiet = PredictiveCodingAudioReadout(canvas_shape=(2, 3, 3), config=config)
    visible = deepcopy(quiet)
    visible.set_diagnostics_enabled(True)
    visible.set_spatial_preview_enabled(True)
    canvas = np.full((2, 3, 3), 0.2)
    evidence = np.linspace(0.0, 0.8, AUDIO_SPECTRAL_BANDS)
    before = visible.visual_weights.copy()
    prediction = visible.predict(canvas)
    np.testing.assert_array_equal(prediction, quiet.predict(canvas))
    actual = visible.infer(canvas, evidence)
    expected = quiet.infer(canvas, evidence)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(visible.hidden, quiet.hidden)
    np.testing.assert_array_equal(visible.canvas_weights, quiet.canvas_weights)
    np.testing.assert_array_equal(visible.visual_weights, quiet.visual_weights)
    expected_gradient = np.outer(visible.visual_error, np.tanh(visible.hidden).mean(axis=(0, 1)))
    np.testing.assert_allclose(visible.visual_weights - before, config.learning_rate * expected_gradient)
    snapshot = visible.diagnostics
    assert snapshot.inference_energy[-1] < snapshot.inference_energy[0]
    assert len(snapshot.visual.channel_means) == AUDIO_SPECTRAL_BANDS
    np.testing.assert_array_equal(snapshot.spatial.prediction, prediction)
    np.testing.assert_array_equal(snapshot.spatial.observation, evidence.astype(np.float32))
    assert not snapshot.spatial.observation.flags.writeable
    visible.reset()
    assert np.all(visible.hidden == 0)
    assert visible.diagnostics is None


def test_audio_features_retain_level_and_silence_without_peak_gating_or_channel_cancellation():
    spectra = [np.full((32, 2), value) for value in (2.0, 4.0)]
    bands = spectral_observation(spectra, window_sum=6, mel_power=False, gain=1)
    np.testing.assert_allclose(bands, 1)
    assert bands.shape == (AUDIO_SPECTRAL_BANDS,)
    quiet = spectral_observation([np.zeros((4, 2))], window_sum=6, mel_power=False, gain=1)
    np.testing.assert_array_equal(quiet, 0)
    power = spectral_observation([np.full((32, 2), 9)], window_sum=6, mel_power=True, gain=1)
    np.testing.assert_array_equal(power, bands)
