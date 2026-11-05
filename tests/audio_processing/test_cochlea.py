import numpy as np
import pytest

from audioviz.audio_processing.cochlea import Cochlea, CochleaConfig


def test_prediction_precedes_evidence_and_error_corrects_only_position():
    bank = Cochlea(22050, CochleaConfig(nodes=3))
    block = bank.process(np.array([0.6]))
    np.testing.assert_array_equal(block.prediction, [0])
    np.testing.assert_array_equal(block.error, [0.6])
    np.testing.assert_allclose(bank.state[:, 0], 0.1)
    np.testing.assert_array_equal(bank.state[:, 1], 0)
    old = bank.state.copy()
    block = bank.process(np.array([-0.2]))
    expected_x = bank._cosine * old[:, 0] + bank._sine * old[:, 1]
    np.testing.assert_allclose(block.prediction, [expected_x.sum()])
    np.testing.assert_allclose(block.error + block.prediction, [-0.2])


def test_prefix_predictions_and_state_do_not_depend_on_block_partition_or_future():
    samples = np.random.default_rng(1).normal(scale=0.2, size=1200)
    whole, split = Cochlea(22050), Cochlea(22050)
    reference = whole.process(samples)
    blocks = [split.process(chunk) for chunk in np.split(samples, [1, 73, 511, 1024])]
    np.testing.assert_array_equal(np.concatenate([b.prediction for b in blocks]), reference.prediction)
    np.testing.assert_array_equal(split.state, whole.state)
    changed = samples.copy()
    changed[600:] *= -2
    future = Cochlea(22050).process(changed)
    np.testing.assert_array_equal(future.prediction[:601], reference.prediction[:601])


def test_exact_free_dynamics_decay_without_evidence_correction():
    bank = Cochlea(22050, CochleaConfig(correction_rate=0))
    bank.state[:, 0] = 1
    frames = 4000
    bank.process(np.zeros(frames))
    expected = np.exp(-bank.damping * frames / bank.sample_rate)
    np.testing.assert_allclose(np.linalg.norm(bank.state, axis=1), expected, rtol=1e-11, atol=1e-14)


def test_graph_feedback_preserves_observation_and_changes_next_prediction():
    bank = Cochlea(22050, CochleaConfig(graph_rate=0.25))
    observation = bank.process(np.array([0.5])).state.copy()
    bank.correct_from_graph(np.ones(bank.observation_size))
    np.testing.assert_array_equal(bank.last_block.state, observation)
    np.testing.assert_allclose(bank.state.ravel(), 0.75 * observation + 0.25)
    block = bank.process(np.zeros(1))
    assert block.prediction[0] > 1
    bank.reset()
    assert bank.last_block is None
    np.testing.assert_array_equal(bank.state, 0)


def test_tuned_tone_is_predicted_and_silence_settles():
    bank = Cochlea(22050)
    frequency = bank.damped_omega[7] / (2 * np.pi)
    time = np.arange(22050) / 22050
    samples = 0.3 * np.sin(2 * np.pi * frequency * time)
    block = bank.process(samples)
    assert np.mean(block.error[-11025:] ** 2) < 0.1 * np.mean(samples[-11025:] ** 2)
    assert np.linalg.norm(bank.state, axis=1).argmax() == 7
    bank.process(np.zeros(22050))
    assert np.linalg.norm(bank.state) < 1e-7


def test_mixture_activates_distinct_nodes_and_broadband_input_stays_finite():
    bank = Cochlea(22050)
    time = np.arange(22050) / 22050
    frequencies = bank.damped_omega[[5, 9]] / (2 * np.pi)
    raw = sum(0.2 * np.sin(2 * np.pi * frequency * time) for frequency in frequencies)
    block = bank.process(raw)
    assert set(np.argsort(np.linalg.norm(bank.state, axis=1))[-2:]) == {5, 9}
    assert np.mean(block.error[-11025:] ** 2) < 0.15 * np.mean(raw[-11025:] ** 2)
    noise = np.random.default_rng(8).normal(size=22050)
    noisy = bank.process(noise)
    assert np.isfinite(noisy.prediction).all()
    assert np.max(np.abs(noisy.prediction)) < 4


@pytest.mark.parametrize("change", [
    {"nodes": 0}, {"nodes": 1.5}, {"quality_factor": 0.5},
    {"minimum_frequency": -1}, {"maximum_frequency": 10},
    {"correction_rate": 1.1}, {"graph_rate": np.nan},
])
def test_invalid_config_is_rejected(change):
    with pytest.raises(ValueError):
        CochleaConfig(**change)


def test_invalid_audio_or_graph_state_cannot_mutate_the_bank():
    bank = Cochlea(22050)
    bank.process(np.array([0.1]))
    before = bank.state.copy()
    for samples in (np.array([]), np.zeros((2, 1)), np.array([np.nan])):
        with pytest.raises(ValueError):
            bank.process(samples)
    for values in (np.zeros(3), np.full(bank.observation_size, np.inf)):
        with pytest.raises(ValueError):
            bank.correct_from_graph(values)
    np.testing.assert_array_equal(bank.state, before)
    with pytest.raises(ValueError, match="Nyquist"):
        Cochlea(16000)
