import numpy as np

from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.diagnostics import EnergyStatistics


def test_energy_statistics_measure_squared_error_population_not_signed_error():
    error = np.array([[[-2.0, 0.0], [2.0, 4.0]]])

    stats = EnergyStatistics.from_error(error)

    assert stats.mean == 3.0
    assert stats.variance == 9.0
    assert stats.channel_means == (2.0, 4.0)


def test_diagnostics_capture_inference_and_actual_learning_without_changing_updates():
    config = PredictiveCodingConfig(learning_enabled=True, learning_rate=0.1)
    quiet = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3), config=config)
    visible = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3), config=config)
    visible.set_diagnostics_enabled(True)
    canvas = np.full((2, 3, 3), 0.2, dtype=np.float32)
    observation = np.broadcast_to([0.8, 0.3, 0.6], canvas.shape)
    quiet_prediction = quiet.predict(canvas)
    visible_prediction = visible.predict(canvas)
    energy_before = visible.energy(canvas, observation)
    weights_before = visible.canvas_weights.copy(), visible.visual_weights.copy()

    quiet_correction = quiet.infer(canvas, observation)
    visible_correction = visible.infer(canvas, observation)
    snapshot = visible.diagnostics

    assert quiet.diagnostics is None
    assert snapshot is not None
    np.testing.assert_array_equal(quiet_prediction, visible_prediction)
    np.testing.assert_array_equal(quiet_correction, visible_correction)
    np.testing.assert_array_equal(quiet.hidden, visible.hidden)
    np.testing.assert_array_equal(quiet.canvas_weights, visible.canvas_weights)
    np.testing.assert_array_equal(quiet.visual_weights, visible.visual_weights)
    assert len(snapshot.inference_energy) == config.inference_steps + 1
    assert snapshot.inference_energy[0] == energy_before
    assert snapshot.inference_energy[-1] < energy_before
    assert snapshot.hidden == EnergyStatistics.from_error(visible.hidden_error)
    assert snapshot.visual == EnergyStatistics.from_error(visible.visual_error)
    np.testing.assert_allclose(
        snapshot.inference_energy[-1],
        config.hidden_channels * snapshot.hidden.mean + 3 * snapshot.visual.mean,
    )
    expected_norm = np.sqrt(
        np.sum((visible.canvas_weights - weights_before[0]) ** 2)
        + np.sum((visible.visual_weights - weights_before[1]) ** 2)
    )
    assert snapshot.weight_update_norm == expected_norm
    assert snapshot.weight_update_norm > 0.0
    assert not snapshot.visual_weights.flags.writeable
    visible.visual_weights.fill(0.0)
    assert np.any(snapshot.visual_weights != 0.0)

    visible.predict(canvas)
    assert visible.diagnostics is None
    visible.infer(canvas, observation)
    visible.reset()
    assert visible.diagnostics is None
    visible.set_diagnostics_enabled(False)
    visible.infer(canvas, observation)
    assert visible.diagnostics is None


def test_inference_without_learning_reports_zero_weight_update():
    readout = PredictiveCodingRGBReadout(canvas_shape=(1, 1, 3))
    readout.set_diagnostics_enabled(True)
    readout.infer(np.zeros((1, 1, 3)), np.ones((1, 1, 3)))

    assert not readout.diagnostics.learning_enabled
    assert readout.diagnostics.weight_update_norm == 0.0
    assert readout.diagnostics.inference_energy[-1] < readout.diagnostics.inference_energy[0]
