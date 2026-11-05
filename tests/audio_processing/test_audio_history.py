import numpy as np
import pytest

from audioviz.audio_processing.cochlea import Cochlea
from audioviz.audio_processing.history import TimedHistory, history_grid
from audioviz.tui.graph_runtime import GraphRuntime


def test_oscillator_history_is_bounded_timed_and_partition_independent():
    whole = Cochlea(22050, history_hop=8, history_columns=5, sample_offset=100)
    split = Cochlea(22050, history_hop=8, history_columns=5, sample_offset=100)
    samples = np.sin(np.arange(73) * 0.2)
    whole.process(samples)
    for block in np.split(samples, [1, 17, 54, 67]):
        split.process(block)
    history = whole.activity_history()
    np.testing.assert_array_equal(history.samples, [140, 148, 156, 164, 172])
    np.testing.assert_array_equal(history.values, split.activity_history().values)
    np.testing.assert_array_equal(whole.state, split.state)
    with pytest.raises(ValueError):
        history.values[:] = 0
    before = history.values.copy()
    whole.process(np.zeros(20))
    np.testing.assert_array_equal(history.values, before)
    whole.reset(sample_offset=1000)
    assert whole.activity_history().values.shape == (16, 0)


def test_history_records_rms_radius_not_a_phase_sample_or_raw_error():
    bank = Cochlea(22050, history_hop=8)
    expected = []
    for value in np.linspace(-0.4, 0.3, 8):
        bank.process(np.array([value]))
        expected.append(np.sum(bank.state**2, axis=1))
    np.testing.assert_allclose(bank.activity_history().values[:, 0], np.sqrt(np.mean(expected, axis=0)))
    captured = bank.activity_history()
    bank.correct_from_graph(np.ones(bank.observation_size))
    np.testing.assert_array_equal(bank.activity_history().values, captured.values)


def test_shared_snapshot_uses_one_sample_clock_across_model_switches():
    runtime = GraphRuntime(resolution=8, camera=False)
    try:
        runtime.step()
        runtime.set_audio_model("cochlea")
        runtime.step()
        history = runtime.processor.get_audio_history()
        assert history.sample_end == 2048
        assert history.waveform.shape == (88200,)
        np.testing.assert_array_equal(history.cochlea.samples, [1280, 1536, 1792, 2048])
        assert history.frequencies[0] > 0  # native mel centres, not FFT-bin labels
        with pytest.raises(ValueError):
            history.waveform[0] = 1
        before = runtime.processor.cochlea.state.copy()
        runtime.processor.get_audio_history()
        runtime.processor.get_audio_history()
        np.testing.assert_array_equal(runtime.processor.cochlea.state, before)
    finally:
        runtime.stop()


def test_common_time_grid_keeps_missing_data_and_uses_actual_spectral_timestamps():
    history = TimedHistory(np.array([[1, 2, 3, 4]]), np.array([2, 4, 4, 6]))
    grid = history_grid(history, sample_end=8, sample_span=8, columns=8)
    np.testing.assert_array_equal(grid, [[np.nan, np.nan, 1, 1, 3, 3, 4, 4]])
    activity = TimedHistory(np.array([[5, 6]]), np.array([4, 6]), window_samples=2)
    np.testing.assert_array_equal(
        history_grid(activity, sample_end=8, sample_span=8, columns=8),
        [[np.nan, np.nan, 5, 5, 6, 6, np.nan, np.nan]],
    )
