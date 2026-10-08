import numpy as np

from audioviz.readouts import PredictiveCodingRGBReadout
from audioviz.visualization.prediction_diagnostics_view import PredictionDiagnosticsView


def _snapshot(*, spatial=False):
    readout = PredictiveCodingRGBReadout(canvas_shape=(2, 3, 3))
    readout.set_diagnostics_enabled(True)
    readout.set_spatial_preview_enabled(spatial)
    readout.predict(np.zeros((2, 3, 3)))
    readout.infer(np.zeros((2, 3, 3)), np.full((2, 3, 3), 0.7))
    return readout.diagnostics


def test_diagnostics_plot_bounded_actual_samples_graph_and_missing_evidence(qapp):
    view = PredictionDiagnosticsView()
    snapshot = _snapshot()
    for index in range(view.HISTORY_LIMIT + 5):
        view.record(index * 0.01, snapshot)
    view.graph_toggle.setChecked(True)
    view.refresh()

    assert len(view.history) == view.HISTORY_LIMIT
    times, hidden_means = view.mean_curves[0].getData()
    _, camera_variances = view.variance_curves[1].getData()
    np.testing.assert_allclose(times[0], 0.05)
    np.testing.assert_allclose(hidden_means, snapshot.hidden.mean)
    np.testing.assert_allclose(camera_variances, snapshot.visual.variance)
    steps, energies = view.inference_curve.getData()
    np.testing.assert_array_equal(steps, np.arange(len(snapshot.inference_energy)))
    np.testing.assert_array_equal(energies, snapshot.inference_energy)
    assert len(view._graph_edges) == 36
    assert len(view._graph_nodes.data) == 12
    assert view._graph_edges[0][0].opts["pen"].widthF() == (
        0.5 + abs(snapshot.canvas_weights[0, 0])
    )
    assert "Weight learning off" in view.status.text()

    view.record(4.0, None)
    view.refresh()
    assert np.isnan(view.mean_curves[0].getData()[1][-1])
    assert view.inference_curve.getData()[0] is None
    assert view._graph_channels == 0
    assert "No fresh camera evidence" in view.status.text()
    view.record(4.1, None)
    assert len(view.history) == view.HISTORY_LIMIT
    view.clear()
    assert len(view.history) == 0
    assert view.mean_curves[0].getData()[0] is None
    view.close()


def test_diagnostics_visibility_controls_refresh_timer(qapp):
    view = PredictionDiagnosticsView()
    events = []
    view.visibility_changed.connect(events.append)

    assert not view.refresh_timer.isActive()
    view.show()
    qapp.processEvents()
    assert view.refresh_timer.isActive()
    view.close()
    assert not view.refresh_timer.isActive()
    assert events == [True, False]


def test_catalogue_selects_energy_and_2d_weights_without_holding_spatial_previews(qapp):
    view = PredictionDiagnosticsView()
    view.show()
    assert view.charts.isVisible()
    assert not view.graph_container.isVisible()
    view.graph_toggle.setChecked(True)
    view.energy_toggle.setChecked(False)
    view.record(1.0, _snapshot(spatial=True))
    view.refresh()
    assert not view.charts.isVisible()
    assert view.graph_container.isVisible()
    assert view.latest.spatial is None
    assert len(view._graph_edges) == 36
    view.graph_toggle.setChecked(False)
    assert view.empty_catalogue.isVisible()
    view.energy_toggle.setChecked(True)
    assert view.charts.isVisible()
    assert not view.empty_catalogue.isVisible()
    view.close()


def test_catalogue_forwards_scene_requests_from_both_modalities(qapp):
    view = PredictionDiagnosticsView()
    view.add_audio_tab()
    requests = []
    view.open_scene_requested.connect(lambda: requests.append(True))
    view.show()
    view.open_scene_requested.emit()
    view.audio_view.open_scene_requested.emit()
    assert requests == [True, True]
    view.close()
