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


def test_layer_mode_is_lazy_throttled_and_clears_missing_or_hidden_previews(
    qapp, monkeypatch, layer_view_stub
):
    import audioviz.visualization.prediction_diagnostics_view as diagnostics_view

    clock = [0.0]
    monkeypatch.setattr(diagnostics_view, "monotonic", lambda: clock[0])
    view = PredictionDiagnosticsView()
    events = []
    view.spatial_preview_changed.connect(events.append)
    view.show()
    view.graph_toggle.setChecked(True)
    assert view.layer_view is None
    assert not view.take_spatial_preview_request()
    view.layer_toggle.setChecked(True)
    assert isinstance(view.layer_view, layer_view_stub)
    assert view.graph_stack.currentWidget() is view.layer_view
    assert view.take_spatial_preview_request()
    assert not view.take_spatial_preview_request()

    snapshot = _snapshot(spatial=True)
    canvas = np.full((2, 3, 3), 0.4, dtype=np.float32)
    view.record(1.0, snapshot, canvas_preview=canvas)
    assert view.layer_view.updates == 0
    view.refresh()
    assert view.layer_view.preview == (canvas, snapshot)
    assert view.layer_view.updates == 1
    view.refresh()
    assert view.layer_view.updates == 1
    view.reset_camera.click()
    assert view.layer_view.resets == 1

    clock[0] = 0.05
    view.record(1.05, None)
    assert view.layer_view.preview is None
    assert view.layer_view.clears == 1
    view.record(1.06, None)
    assert view.layer_view.clears == 1
    assert not view.take_spatial_preview_request()
    clock[0] = 0.1
    assert view.take_spatial_preview_request()
    assert not view.take_spatial_preview_request()

    view.graph_toggle.setChecked(False)
    assert not view.take_spatial_preview_request()
    assert events[-1] is False
    view.graph_toggle.setChecked(True)
    assert view.take_spatial_preview_request()
    view.record(2.0, snapshot, canvas_preview=canvas)
    view.layer_toggle.setChecked(False)
    assert view.graph_stack.currentWidget() is view.graph_plot
    assert view._spatial_sample is None
    assert view.latest.spatial is None
    assert not view.take_spatial_preview_request()
    view.layer_toggle.setChecked(True)
    view.record(3.0, snapshot, canvas_preview=canvas)
    view.hide()
    assert view._spatial_sample is None
    assert not view.take_spatial_preview_request()
    assert events[-1] is False
    view.show()
    assert view.take_spatial_preview_request()
    view.clear()
    assert view._spatial_sample is None
    assert view.take_spatial_preview_request()
    qapp.processEvents()
    assert view.graph_legend.height() >= view.graph_legend.heightForWidth(
        view.graph_legend.width()
    )
    view.close()


def test_layer_context_failure_is_explicit_and_restores_usable_2d_graph(qapp, layer_view_stub):
    view = PredictionDiagnosticsView()
    view.show()
    view.graph_toggle.setChecked(True)
    view.record(1.0, _snapshot())
    view.layer_toggle.setChecked(True)

    view.layer_view.rendering_failed.emit("No compatible OpenGL context")

    assert not view.layer_toggle.isChecked()
    assert view.layer_error.isVisible()
    assert "No compatible OpenGL context" in view.layer_error.text()
    assert view.graph_stack.currentWidget() is view.graph_plot
    assert len(view._graph_edges) == 36
    assert not view.wants_spatial_preview()
    view.close()
