import numpy as np
from PyQt5 import QtCore

from audioviz.sources.camera import CameraFrameSource
from audioviz.utils.source_preview import SourceScenePreview
from audioviz.sources.audio import AudioRippleSource
from audioviz.visualization.ripple_renderers import _canvas_to_rgb
from audioviz.visualization.prediction_scene_window import PredictionSceneWindow


def test_scene_throttles_latest_samples_and_releases_hidden_minimized_reset_and_failed_views(
    qapp, monkeypatch, layer_view_stub
):
    import audioviz.visualization.prediction_scene_window as scene_module

    clock = [0.0]
    monkeypatch.setattr(scene_module, "monotonic", lambda: clock[0])
    view = PredictionSceneWindow()
    assert view.layer_view is None
    assert not view.take_spatial_preview_request()
    view.show()
    assert isinstance(view.layer_view, layer_view_stub)
    assert view.take_spatial_preview_request()
    assert not view.take_spatial_preview_request()
    canvas = np.ones((2, 3, 3), dtype=np.float32)
    source = SourceScenePreview(CameraFrameSource.scene_mapping, True, observation=canvas)
    view.record(canvas, (source,))
    assert view.layer_view.updates == 1
    view.invalidate_evidence("camera")
    assert view.sources[0].observation is None
    assert view.layer_view.preview[1][0].observation is None
    assert "no current evidence" in view.status.text()
    clock[0] = 0.1
    assert view.take_spatial_preview_request()
    view.setWindowState(QtCore.Qt.WindowMinimized)
    assert not view.take_spatial_preview_request()
    assert view.canvas is None
    view.setWindowState(QtCore.Qt.WindowNoState)
    assert view.take_spatial_preview_request()
    view.record(canvas, (source,))
    view.hide()
    assert view.canvas is None
    assert view.layer_view.preview is None
    view.show()
    assert view.take_spatial_preview_request()
    view.clear()
    assert view.take_spatial_preview_request()
    view.layer_view.rendering_failed.emit("No compatible OpenGL context")
    assert view.error.isVisible()
    assert "No compatible OpenGL context" in view.error.text()
    assert not view.take_spatial_preview_request()
    assert view.canvas is None
    view.close()


def test_2d_layouts_show_pre_evidence_prediction_and_clear_missing_evidence_without_opengl(qapp):
    view = PredictionSceneWindow()
    view.layout_selector.setCurrentIndex(view.layout_selector.findData("comparison"))
    view.show()
    try:
        assert view.layer_view is None
        assert view.preview_active()
        assert view.preview_max_side == 512
        assert not view.needs_hidden_preview
        canvas = np.full((4, 6, 3), .2, dtype=np.float32)
        predicted = np.full_like(canvas, .7)
        observed = np.full_like(canvas, .4)
        sources = (
            SourceScenePreview(
                CameraFrameSource.scene_mapping, True, observation=observed, prediction=predicted,
            ),
            SourceScenePreview(
                AudioRippleSource.scene_mapping, True,
                observation=np.arange(16), prediction=np.linspace(0, 1, 16),
            ),
        )
        view.record(canvas, sources)
        np.testing.assert_array_equal(view.image_view.images["canvas"].image, _canvas_to_rgb(canvas, limit=1))
        np.testing.assert_array_equal(view.image_view.images["prediction"].image, _canvas_to_rgb(predicted, limit=1))
        assert view.image_view.panels["audio"].isVisible()
        view.invalidate_evidence("camera")
        assert view.image_view.images["observation"].image is None
        assert view.image_view.images["prediction"].image is not None
        view.invalidate_evidence("audio")
        assert view.image_view.audio_observation.xData is None
        assert view.image_view.audio_prediction.xData is not None
        view.layout_selector.setCurrentIndex(view.layout_selector.findData("prediction"))
        assert view.take_spatial_preview_request()
        view.record(canvas, sources)
        assert view.image_view.panels["prediction"].isVisible()
        for key in ("canvas", "observation", "audio"):
            assert not view.image_view.panels[key].isVisible()
        view.setWindowState(QtCore.Qt.WindowMinimized)
        assert not view.take_spatial_preview_request()
        assert view.image_view.images["prediction"].image is None
        view.setWindowState(QtCore.Qt.WindowNoState)
        assert view.take_spatial_preview_request()
        view.record(canvas, sources)
        view.hide()
        assert view.canvas is None
        assert view.image_view.images["prediction"].image is None
    finally:
        view.close()


def test_failed_3d_scene_can_switch_to_working_2d_layout(qapp, layer_view_stub):
    view = PredictionSceneWindow()
    view.show()
    try:
        view.layer_view.rendering_failed.emit("No OpenGL")
        assert not view.preview_active()
        view.layout_selector.setCurrentIndex(view.layout_selector.findData("prediction"))
        assert view.preview_active()
        assert not view.error.isVisible()
        assert view.take_spatial_preview_request()
        image = np.full((3, 4, 3), .5, dtype=np.float32)
        view.record(image, (SourceScenePreview(CameraFrameSource.scene_mapping, False, prediction=image),))
        assert view.image_view.images["prediction"].image is not None
        view.layout_selector.setCurrentIndex(view.layout_selector.findData("3d"))
        assert not view.preview_active()
        assert view.error.isVisible()
    finally:
        view.close()


def test_continuous_graph_is_frozen_until_explicit_refresh(qapp, layer_view_stub):
    from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
    from audioviz.utils.continuous_graph_preview import continuous_graph_preview

    graph = ContinuousScenePredictiveCodingGraph(visual_shape=(4, 6), audio_nodes=0)
    image = np.full((4, 6, 3), .6)
    graph.forecast(image, 1 / 30)
    graph.condition(image)
    snapshot = continuous_graph_preview(graph, image, foveal_mask=np.ones((4, 6), dtype=bool))
    window = PredictionSceneWindow()
    window.configure_continuous_graph()
    window.live_graph.setChecked(False)
    window.show()
    try:
        assert window.take_spatial_preview_request()
        assert window.snapshot_button.isVisible()
        window.record(image, (), graph_snapshot=snapshot)
        assert window.layer_view.preview is snapshot
        assert not window.take_spatial_preview_request()
        window.invalidate_evidence("camera")
        assert window.graph_snapshot is snapshot
        assert "Frozen snapshot" in window.status.text()
        window.snapshot_button.click()
        assert window.take_spatial_preview_request()
        window.record(image, (), graph_snapshot=snapshot)
        assert window.layer_view.updates == 2
        window.layout_selector.setCurrentIndex(window.layout_selector.findData("comparison"))
        assert not window.snapshot_button.isVisible()
        assert window.take_spatial_preview_request()
        window.record(image, ())
        window.layout_selector.setCurrentIndex(window.layout_selector.findData("3d"))
        assert window.take_spatial_preview_request()
        window.record(image, (), graph_snapshot=snapshot)
        window.hide()
        assert window.graph_snapshot is None
        assert window.layer_view.preview is None
    finally:
        window.close()


def test_live_graph_throttles_updates_and_freezes_only_the_view(qapp, monkeypatch, layer_view_stub):
    import audioviz.visualization.prediction_scene_window as scene_module
    from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
    from audioviz.utils.continuous_graph_preview import continuous_graph_preview

    clock = [0.]
    monkeypatch.setattr(scene_module, "monotonic", lambda: clock[0])
    graph = ContinuousScenePredictiveCodingGraph(visual_shape=(4, 6), audio_nodes=0)
    image = np.full((4, 6, 3), .6)
    graph.forecast(image, 1 / 30)
    graph.condition(image)
    snapshot = continuous_graph_preview(graph, image, foveal_mask=np.ones((4, 6), dtype=bool))
    window = PredictionSceneWindow()
    window.configure_continuous_graph()
    window.show()
    try:
        assert window.live_graph.isChecked() and window.live_graph.isVisible()
        assert not window.snapshot_button.isEnabled()
        assert window.take_spatial_preview_request()
        window.record(image, (), graph_snapshot=snapshot)
        assert "Live graph" in window.status.text()
        clock[0] = .09
        assert not window.take_spatial_preview_request()
        clock[0] = .1
        assert window.take_spatial_preview_request()
        window.record(image, (), graph_snapshot=snapshot)
        assert window.layer_view.updates == 2
        window.invalidate_evidence("camera")
        assert window.graph_snapshot.reference is None
        assert not window.graph_snapshot.observed.any()
        assert window.graph_snapshot.error is None
        window.live_graph.setChecked(False)
        assert window.snapshot_button.isEnabled()
        assert "Frozen snapshot" in window.status.text()
        clock[0] = 10.
        assert not window.take_spatial_preview_request()
        window.snapshot_button.click()
        assert window.take_spatial_preview_request()
        window.record(image, (), graph_snapshot=snapshot)
        window.invalidate_evidence("camera")
        assert window.graph_snapshot is snapshot
        window.live_graph.setChecked(True)
        assert window.take_spatial_preview_request()
        window.record(image, (), graph_snapshot=snapshot)
        window.hide()
        clock[0] = 20.
        assert not window.take_spatial_preview_request()
        window.show()
        assert window.take_spatial_preview_request()
        window.layout_selector.setCurrentIndex(window.layout_selector.findData("comparison"))
        assert not window.live_graph.isVisible()
    finally:
        window.close()
