import numpy as np

from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.diagnostics import EnergyStatistics
from audioviz.utils.spatial_preview import spatial_preview


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


def test_spatial_preview_preserves_frame_alignment_and_numerical_updates():
    shape = (100, 200, 3)
    config = PredictiveCodingConfig(inference_steps=2, learning_enabled=True)
    quiet = PredictiveCodingRGBReadout(canvas_shape=shape, config=config)
    visible = PredictiveCodingRGBReadout(canvas_shape=shape, config=config)
    visible.set_diagnostics_enabled(True)
    visible.set_spatial_preview_enabled(True)
    canvas = np.random.default_rng(4).normal(scale=0.2, size=shape).astype(np.float32)
    observation = np.random.default_rng(5).random(shape).astype(np.float32)

    quiet_prediction = quiet.predict(canvas)
    visible_prediction = visible.predict(canvas)
    quiet_correction = quiet.infer(canvas, observation)
    visible_correction = visible.infer(canvas, observation)
    preview = visible.diagnostics.spatial

    np.testing.assert_array_equal(quiet_prediction, visible_prediction)
    np.testing.assert_array_equal(quiet_correction, visible_correction)
    np.testing.assert_array_equal(quiet.hidden, visible.hidden)
    np.testing.assert_array_equal(quiet.canvas_weights, visible.canvas_weights)
    np.testing.assert_array_equal(quiet.visual_weights, visible.visual_weights)
    assert preview.hidden.shape == (32, 64, config.hidden_channels)
    assert preview.prediction.shape == preview.observation.shape == (32, 64, 3)
    np.testing.assert_array_equal(preview.prediction, spatial_preview(visible_prediction))
    np.testing.assert_array_equal(preview.observation, spatial_preview(observation))
    np.testing.assert_array_equal(preview.hidden, spatial_preview(visible.hidden))
    assert not np.allclose(
        preview.prediction, spatial_preview(np.tanh(visible.hidden) @ visible.visual_weights.T)
    )
    for field in (preview.prediction, preview.observation, preview.hidden):
        assert field.dtype == np.float32
        assert not field.flags.writeable
    visible.hidden.fill(0)
    observation.fill(0)
    assert np.any(preview.hidden != 0)
    assert np.any(preview.observation != 0)

    visible.infer(canvas, observation)
    assert visible.diagnostics.spatial is None
    visible.predict(canvas)
    visible.reset()
    assert visible._preview_prediction is None
    assert visible.diagnostics is None


def test_spatial_preview_requires_both_opt_ins_and_stops_sampling_when_disabled(monkeypatch):
    import audioviz.readouts.predictive_coding as pc

    readout = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3))
    canvas = np.zeros((2, 3, 3))
    readout.set_spatial_preview_enabled(True)
    assert not readout.spatial_preview_enabled
    readout.set_diagnostics_enabled(True)
    readout.set_spatial_preview_enabled(True)
    readout.predict(canvas)
    readout.infer(canvas, np.ones_like(canvas))
    assert readout.diagnostics.spatial is not None
    readout.set_spatial_preview_enabled(False)
    assert readout.diagnostics.spatial is None

    def unexpected_sample(_field):
        raise AssertionError("disabled previews must not sample spatial fields")

    monkeypatch.setattr(pc, "spatial_preview", unexpected_sample)
    readout.predict(canvas)
    readout.infer(canvas, np.ones_like(canvas))
    assert readout.diagnostics is not None
    assert readout.diagnostics.spatial is None
    readout.set_diagnostics_enabled(False)
    assert not readout.spatial_preview_enabled
