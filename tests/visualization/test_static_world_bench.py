import json
import sys
import traceback

import numpy as np
import pytest
from PyQt5 import QtCore, QtTest

from audioviz.visualization.static_world_bench import StaticWorldBench
from audioviz.visualization import ripple_wave_visualizer as visualizer_module
from audioviz.utils.array_backend import to_numpy


@pytest.fixture(autouse=True)
def fail_on_qt_callback_errors(monkeypatch):
    errors = []
    monkeypatch.setattr(
        sys, "excepthook",
        lambda *error: errors.append("".join(traceback.format_exception(*error))),
    )
    yield
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_bench_uses_actual_model_fixed_clock_and_missing_evidence_gaps(qapp, request, monkeypatch, backend, tmp_path):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    now = [10.]
    monkeypatch.setattr(visualizer_module.time, "monotonic", lambda: now[0])
    bench = StaticWorldBench(
        resolution=(20, 28), foveal_radius=4, backend=backend, dt=.025, motion_enabled=False,
    )
    bench.lock_gaze.setChecked(True)
    bench.show()
    qapp.processEvents()
    try:
        assert not bench.timer.isActive() and not bench.visualizer.timer.isActive()
        assert bench.capture.read_count == 0 and not bench.history
        source = bench.visualizer.source_orchestrator
        graph = source.sensory_graph
        initial = [to_numpy(getattr(graph, key)).copy() for key in graph._PARAMETERS]
        first = bench.step_once()
        assert not first.wave_propagation and not bench.visualizer.engine.wave_propagation_enabled
        assert first.time == .025 and first.observed_rgb == 49*3
        assert first.energy_final <= first.energy_initial+1e-10
        assert not first.learning_enabled and first.canvas_rate == 0
        assert source._uncertainty_dt == .025
        assert bench.capture.read_count == 1
        assert bench.visualizer.attention_scores is None
        before = source.visual_readout.config.spatial_precision.gaze
        bench.visualizer._set_mouse_gaze(0, 0)
        assert source.visual_readout.config.spatial_precision.gaze == before
        np.testing.assert_array_equal(
            spatial_field(bench.visualizer.engine.get_field()), 0,
        )

        bench.controls_button.click()
        controls = bench.visualizer.control_panel.source_control_widgets
        assert "residual precisions" in controls["learning-dynamics", "learning_enabled"].toolTip()
        assert "log precisions" in controls["learning-dynamics", "learning_rate"].toolTip()
        controls["learning-dynamics", "inference_steps"].setValue(3)
        controls["learning-dynamics", "inference_rate"].setValue(0)
        controls["learning-dynamics", "learning_enabled"].setChecked(True)
        now[0] += 7
        second = bench.step_once()
        assert source._uncertainty_dt == .025 and second.time == .05
        assert second.forecast_mse == pytest.approx(second.settled_mse, abs=1e-14)
        assert second.inference_steps == 3 and second.inference_rate == 0
        assert second.learning_updates == 1 and second.learning_enabled
        assert any(
            not np.array_equal(previous, to_numpy(getattr(graph, key)))
            for previous, key in zip(initial, graph._PARAMETERS)
        )
        bench.visualizer.gaze_controls.observe.setChecked(False)
        missing = bench.step_once()
        assert missing.observed_rgb == 0
        assert missing.forecast_mse is None and missing.settled_mse is None
        assert missing.learning_updates == 1 and bench.capture.read_count == 2
        assert "MSE unavailable" in bench.status.text()
        assert np.isnan(bench.forecast_curve.getData()[1][-1])
        rmse = np.sqrt(np.mean(
            (bench.scene.astype(np.float32) / 255 - bench.forecast_image.image)**2, axis=-1,
        ))
        np.testing.assert_array_equal(bench.error_image.image, bench.error_colormap.map(rmse, mode="byte"))
        path = tmp_path / "trace.json"
        bench.save_trace(path)
        trace = json.loads(path.read_text())
        assert trace["total_steps"] == 3 and len(trace["frames"]) == 3
        assert trace["frames"][-1]["forecast_mse"] is None
        assert trace["frames"][1]["inference_rate"] == 0
        assert not trace["wave_propagation"] and trace["seed"] == 0
        assert not any(frame["wave_propagation"] for frame in trace["frames"])
        assert trace["forecast_dt"] == .025
        bench.visualizer.gaze_controls.observe.setChecked(True)
        controls["learning-dynamics", "canvas_rate"].setValue(.05)
        feedback = bench.step_once()
        assert feedback.canvas_rate == .05 and feedback.canvas_gradient_norm > 0
        assert np.any(spatial_field(bench.visualizer.engine.get_field()) != 0)
        panel = bench.visualizer.control_panel
        assert not panel.wave_dynamics_button.isChecked()
        assert not panel.wave_dynamics_button.isEnabled()
        panel.wave_dynamics_button.click()
        assert not bench.step_once().wave_propagation
        bench.new_trial()
        qapp.processEvents()
        assert not bench.visualizer.engine.wave_propagation_enabled
        assert not bench.step_once().wave_propagation
        bench.save_trace(path)
        assert not json.loads(path.read_text())["wave_propagation"]
        playback = QtTest.QSignalSpy(bench.timer.timeout)
        bench.run_button.setChecked(True)
        assert bench.timer.isActive() and not bench.step_button.isEnabled()
        assert playback.wait(2000)
        assert bench.steps > 1
        bench.run_button.setChecked(False)
        assert not bench.timer.isActive() and bench.step_button.isEnabled()
    finally:
        bench.close()


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_learned_whole_scene_prediction_decays_without_erasing_connections(qapp, request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    bench = StaticWorldBench(
        resolution=(40, 60), foveal_radius=4, backend=backend,
        motion_enabled=False,
    )
    bench.lock_gaze.setChecked(True)
    bench.show()
    qapp.processEvents()
    try:
        source = bench.visualizer.source_orchestrator
        graph = source.sensory_graph
        initial_weights = to_numpy(graph.sensory_weights).copy()
        source.update_pathway_config(learning_enabled=True)
        for _ in range(160):
            observed = bench.step_once()
        assert observed.learning_updates == 160
        assert observed.visual_potential_norm > .01
        np.testing.assert_array_equal(to_numpy(graph.hidden_bias), 0.)
        np.testing.assert_array_equal(to_numpy(graph.sensory_bias), 0.)
        assert not np.array_equal(to_numpy(graph.sensory_weights), initial_weights)
        learned = {
            name: to_numpy(getattr(graph, name)).copy()
            for name in (*graph._PARAMETERS, *graph._PRECISIONS)
        }
        bench.visualizer.gaze_controls.observe.setChecked(False)
        first_free = bench.step_once()
        for _ in range(89):
            resting = bench.step_once()
        assert bench.capture.read_count == 160
        assert resting.forecast_mse is None and resting.learning_updates == 160
        assert resting.visual_potential_norm < observed.visual_potential_norm * .01
        assert resting.neutral_visual_mae < first_free.neutral_visual_mae * .01
        for name, weights in learned.items():
            np.testing.assert_array_equal(to_numpy(getattr(graph, name)), weights)
    finally:
        bench.close()


def spatial_field(value):
    if hasattr(value, "download"):
        return value.download()
    return to_numpy(value)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_new_trial_restores_seeded_parameters_and_keeps_selected_inference_settings(
    qapp, request, monkeypatch, tmp_path, backend,
):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    assert StaticWorldBench.HISTORY_LIMIT == 512
    monkeypatch.setattr(StaticWorldBench, "HISTORY_LIMIT", 2)
    bench = StaticWorldBench(resolution=(20, 28), foveal_radius=4, backend=backend, motion_enabled=False)
    bench.lock_gaze.setChecked(True)
    bench.show()
    qapp.processEvents()
    try:
        graph = bench.visualizer.source_orchestrator.sensory_graph
        before = {
            key: to_numpy(getattr(graph, key)).copy()
            for key in (*graph._PARAMETERS, *graph._PRECISIONS)
        }
        first = bench.step_once()
        bench.visualizer.source_orchestrator.update_pathway_config(
            inference_steps=4, inference_rate=.05, learning_enabled=True,
        )
        bench.step_once()
        old_capture = bench.capture
        bench.new_trial()
        qapp.processEvents()
        graph = bench.visualizer.source_orchestrator.sensory_graph
        assert bench.steps == 0 and not bench.history
        assert old_capture.read() == (False, None)
        assert bench.capture.read_count == 0 and graph.updates == 0
        assert graph.config.inference_steps == 4 and graph.config.inference_rate == .05
        assert graph.config.learning_enabled
        for key, initial in before.items():
            np.testing.assert_array_equal(to_numpy(getattr(graph, key)), initial)
        bench.visualizer.source_orchestrator.update_pathway_config(
            inference_steps=8, inference_rate=.1, learning_enabled=False,
        )
        repeated = bench.step_once()
        assert repeated.forecast_mse == first.forecast_mse
        assert repeated.settled_mse == first.settled_mse
        assert repeated.inference_energy == first.inference_energy
        bench.lock_gaze.setChecked(False)
        bench.visualizer._set_mouse_gaze(0, 0)
        bench.step_once()
        assert bench.history[-1].gaze == (0., 0.)
        assert bench.history[-1].observed_rgb < 49*3
        bench.step_once()
        assert bench.steps == 3 and [frame.step for frame in bench.history] == [2, 3]
        path = tmp_path / "bounded.json"
        bench.save_trace(path)
        trace = json.loads(path.read_text())
        assert trace["total_steps"] == 3 and trace["history_limit"] == 2
        assert [frame["step"] for frame in trace["frames"]] == [2, 3]
    finally:
        bench.close()


def test_fixed_forecast_interval_rejects_nonpositive_or_boolean_values(qapp):
    for dt in (True, 0, -1, np.nan):
        with pytest.raises(ValueError, match="dt"):
            StaticWorldBench(backend="numpy", dt=dt)


def test_full_scene_state_does_not_receive_camera_truth_outside_fovea(qapp, monkeypatch):
    bench = StaticWorldBench(resolution=(20, 28), foveal_radius=4, backend="numpy")
    bench.lock_gaze.setChecked(True)
    bench.show()
    qapp.processEvents()
    try:
        bench.step_once()
        graph = bench.visualizer.source_orchestrator.sensory_graph
        expected_state = graph.visual_potential.copy()
        expected_hidden = graph.mean[graph.hidden_slice].copy()
        bench.new_trial()
        qapp.processEvents()
        source = bench.visualizer.source_orchestrator
        read = bench.capture.read

        def changed_periphery():
            ok, bgr = read()
            changed = bgr.copy()
            changed[~source.graph_io.observation_mask] = 255
            return ok, changed

        monkeypatch.setattr(bench.capture, "read", changed_periphery)
        bench.step_once()
        np.testing.assert_array_equal(source.sensory_graph.visual_potential, expected_state)
        np.testing.assert_array_equal(
            source.sensory_graph.mean[source.sensory_graph.hidden_slice], expected_hidden,
        )
    finally:
        bench.close()


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_image_mouse_gaze_selects_the_next_sample_without_consuming_evidence(
    qapp, request, monkeypatch, backend,
):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    monkeypatch.setattr(StaticWorldBench, "PREVIEW_MAX_SIDE", 16)
    bench = StaticWorldBench(resolution=(40, 60), foveal_radius=4, backend=backend)
    bench.show()
    qapp.processEvents()
    try:
        assert not bench.lock_gaze.isChecked()
        source = bench.visualizer.source_orchestrator
        for image, gaze in (
            (bench.world_image, (.25, .75)),
            (bench.forecast_image, (.75, .25)),
            (bench.error_image, (0., 0.)),
        ):
            reads_before = bench.capture.read_count
            position = image.getViewBox().mapViewToScene(
                QtCore.QPointF(.5+59*gaze[0], .5+39*gaze[1]),
            )
            bench.images.scene().sigMouseMoved.emit(position)
            assert source.visual_readout.config.spatial_precision.gaze == pytest.approx(gaze)
            assert bench.capture.read_count == reads_before
            x, y = bench.outlines[0].getData()
            assert (x.min()+x.max())/2 == round(59*gaze[0])+.5
            assert (y.min()+y.max())/2 == round(39*gaze[1])+.5
            frame = bench.step_once()
            assert frame.gaze == pytest.approx(gaze)
            assert bench.capture.read_count == reads_before+1
            io, graph = source.graph_io, source.sensory_graph
            observed = io.observation_mask.reshape(-1)
            expected = bench.scene[tuple(io.positions[observed].T)].astype(np.float32)/255
            actual = to_numpy(graph.evidence[graph.visual_slice]).reshape(-1, 3)[observed]
            np.testing.assert_array_equal(actual, expected)
            assert frame.observed_rgb == int(observed.sum())*3
            assert bench.visualizer.attention_scores is None
        assert bench.forecast_image.image.shape[0] <= 16
        bench.lock_gaze.setChecked(True)
        bench.images.scene().sigMouseMoved.emit(
            bench.world_image.getViewBox().mapViewToScene(QtCore.QPointF(30, 20)),
        )
        assert source.visual_readout.config.spatial_precision.gaze == (0., 0.)
        assert bench.step_once().gaze == (0., 0.)
    finally:
        bench.close()


def test_scene_bench_defaults_to_mouse_without_fabricating_scores(qapp):
    bench = StaticWorldBench(resolution=(40, 60), foveal_radius=4, backend="numpy")
    bench.show()
    qapp.processEvents()
    try:
        source = bench.visualizer.source_orchestrator
        controls = bench.visualizer.gaze_controls
        assert controls.follow_mouse.isChecked()
        assert controls.autonomous_gaze.isEnabled()
        assert not controls.full_canvas.isEnabled()
        assert source.gaze_policy is None
        assert not hasattr(bench, "model_selector")
        assert not bench.motion_enabled.isChecked()
        bench.visualizer._set_mouse_gaze(.75, .25)
        assert bench.step_once().gaze == (.75, .25)
        controls.observe.setChecked(False)
        frame = bench.step_once()
        assert frame.gaze == (.75, .25) and frame.observed_rgb == 0
        assert bench.capture.read_count == 1 and source.gaze_scores is None
        bench.new_trial()
        qapp.processEvents()
        assert bench.visualizer.gaze_controls.follow_mouse.isChecked()
        assert bench.visualizer.source_orchestrator.gaze_policy is None
    finally:
        bench.close()


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_brownian_world_moves_without_evidence_and_freezes_and_restarts_reproducibly(
    qapp, request, backend, tmp_path,
):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    bench = StaticWorldBench(
        resolution=(40, 60), foveal_radius=4, backend=backend, brownian_scale=20.,
        motion_enabled=True,
    )
    bench.lock_gaze.setChecked(True)
    bench.show()
    qapp.processEvents()
    try:
        assert bench.motion_enabled.isChecked()
        initial_scene = bench.scene.copy()
        controls = bench.visualizer.gaze_controls
        controls.observe.setChecked(False)
        first_offsets = []
        for _ in range(3):
            frame = bench.step_once()
            first_offsets.append(frame.world_offset)
            assert frame.forecast_mse is None and frame.settled_mse is None
            assert frame.learning_updates == 0 and frame.gaze == (.5, .5)
        assert bench.world.time == pytest.approx(3*bench.dt) and bench.capture.read_count == 0
        assert not np.array_equal(bench.scene, initial_scene)
        np.testing.assert_array_equal(bench.world_image.image, bench.scene)
        controls.observe.setChecked(True)
        observed = bench.step_once()
        assert bench.capture.read_count == 1 and observed.observed_rgb == 49*3
        source = bench.visualizer.source_orchestrator
        io, graph = source.graph_io, source.sensory_graph
        observed_mask = io.observation_mask.reshape(-1)
        expected = bench.scene[tuple(io.positions[observed_mask].T)].astype(np.float32)/255
        np.testing.assert_array_equal(
            to_numpy(graph.evidence[graph.visual_slice]).reshape(-1, 3)[observed_mask], expected,
        )
        bench.motion_enabled.setChecked(False)
        assert not bench.motion_scale.isEnabled()
        frozen_scene = bench.scene.copy()
        frozen = bench.step_once()
        assert frozen.world_offset == observed.world_offset and not frozen.brownian_enabled
        np.testing.assert_array_equal(bench.scene, frozen_scene)
        path = tmp_path / "moving-world.json"
        bench.save_trace(path)
        trace = json.loads(path.read_text())
        assert trace["world_seed"] == 0 and trace["world_time"] == pytest.approx(5*bench.dt)
        assert trace["motion_enabled"] is False
        assert trace["frames"][0]["brownian_enabled"] is True
        assert trace["frames"][0]["brownian_scale"] == 20.
        assert trace["frames"][-1]["world_pixel_offset"] == list(frozen.world_pixel_offset)
        bench.new_trial()
        qapp.processEvents()
        assert bench.world.time == 0 and bench.world.pixel_offset == (0, 0)
        np.testing.assert_array_equal(bench.scene, initial_scene)
        assert not bench.motion_enabled.isChecked() and bench.motion_scale.value() == 20.
        bench.motion_enabled.setChecked(True)
        for expected_offset in first_offsets:
            assert bench.step_once().world_offset == expected_offset
        bench.motion_scale.setValue(0.)
        zero_scale = bench.step_once()
        assert zero_scale.brownian_enabled and zero_scale.brownian_scale == 0.
        assert zero_scale.world_offset == first_offsets[-1]
    finally:
        bench.close()


def test_cli_static_control_reaches_the_same_bench(qapp, monkeypatch):
    from audioviz.visualization import static_world_bench as module
    created = []

    class Window:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def show(self):
            pass

    monkeypatch.setattr(module, "StaticWorldBench", Window)
    monkeypatch.setattr(module.QtWidgets.QApplication, "exec", lambda self: 0)
    old_style = qapp.styleSheet()
    try:
        assert module.main(["--static-scene", "--brownian-scale", "6", "--backend", "numpy"]) == 0
        assert created[0]["motion_enabled"] is False
        assert created[0]["brownian_scale"] == 6 and created[0]["backend"] == "numpy"
        assert module.main([
            "--memory-time-constant", ".7",
        ]) == 0
        assert created[1]["memory_time_constant"] == .7
        assert created[1]["motion_enabled"] is False
        assert "model" not in created[1]
        assert module.main(["--moving-scene"]) == 0
        assert created[2]["motion_enabled"] is True
    finally:
        qapp.setStyleSheet(old_style)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_world_slots_keep_evolving_after_gaze_moves_and_render_every_pixel(
    qapp, request, backend, tmp_path, layer_view_stub,
):
    from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph

    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    bench = StaticWorldBench(
        resolution=(40, 60), foveal_radius=4, backend=backend,
        memory_time_constant=.2, motion_enabled=False,
    )
    bench.lock_gaze.setChecked(True)
    bench.show()
    qapp.processEvents()
    try:
        source = bench.visualizer.source_orchestrator
        graph, io = source.sensory_graph, source.graph_io
        assert isinstance(graph, ContinuousScenePredictiveCodingGraph)
        assert graph.visual_shape == (40, 60) and io.world_anchored
        assert io.positions.shape == (40*60, 2) and io.valid.all()
        assert not bench.visualizer.gaze_controls.autonomous_gaze.isEnabled()
        assert not bench.visualizer.diagnostics_button.isEnabled()
        assert bench.visualizer.scene_button.isEnabled()
        assert bench.state_plots.isVisible()
        bench.lock_gaze.setChecked(False)
        assert bench.visualizer.gaze_controls.autonomous_gaze.isEnabled()
        bench.lock_gaze.setChecked(True)
        for _ in range(10):
            first = bench.step_once()
        assert first.observed_rgb == 49*3
        observed = io.observation_mask.reshape(-1)
        np.testing.assert_array_equal(
            to_numpy(graph.evidence[graph.visual_slice]).reshape(-1, 3)[observed],
            bench.scene[tuple(io.positions[observed].T)].astype(np.float32)/255,
        )
        assert np.isnan(to_numpy(graph.evidence[graph.visual_slice]).reshape(-1, 3)[~observed]).all()
        bench.visualizer.scene_button.click()
        popup = bench.visualizer.prediction_scene
        popup.layout_selector.setCurrentIndex(popup.layout_selector.findData("3d"))
        popup.live_graph.setChecked(False)
        reads = bench.capture.read_count
        bench.step_once()
        snapshot = popup.graph_snapshot
        assert snapshot is not None
        assert bench.capture.read_count == reads + 1
        np.testing.assert_array_equal(snapshot.fovea, io.observation_mask)
        bench.controls_button.click()
        panel = bench.visualizer.control_panel
        assert not panel.beliefs_button.isEnabled()
        assert "continuous" in panel.source_control_widgets["learning-dynamics", "graph_backend"].text()
        assert "curvature" not in panel.source_control_widgets["learning-dynamics", "graph_backend"].text()
        bench.memory_tau.setValue(.8)
        assert graph.memory_time_constant == .8
        source.update_pathway_config(inference_rate=0.)
        before = to_numpy(graph.visual_potential).reshape(40, 60, 3).copy()
        assert np.linalg.norm(before[io.observation_mask]) > 0
        bench.lock_gaze.setChecked(False)
        bench.visualizer._set_mouse_gaze(0., 0.)
        moved = bench.step_once()
        assert moved.gaze == (0., 0.) and moved.observed_rgb < 49*3
        assert popup.graph_snapshot is snapshot
        np.testing.assert_allclose(
            to_numpy(graph.visual_potential).reshape(40, 60, 3),
            before*np.exp(-bench.dt/.8), atol=1e-12,
        )
        expected_prediction = 1/(1+np.exp(-before*np.exp(-bench.dt/.8)))
        np.testing.assert_allclose(
            spatial_field(bench.visualizer.visual_prediction), expected_prediction, atol=1e-7,
        )
        np.testing.assert_allclose(
            bench.forecast_image.image, expected_prediction, atol=1e-7,
        )
        popup.snapshot_button.click()
        bench.visualizer.gaze_controls.observe.setChecked(False)
        reads = bench.capture.read_count
        missing = bench.step_once()
        assert missing.forecast_mse is None and missing.settled_mse is None
        assert bench.capture.read_count == reads and graph.updates == 0
        refreshed = popup.graph_snapshot
        assert refreshed is not snapshot and refreshed.reference is not None
        assert refreshed.independent_reference and refreshed.error is not None
        assert not refreshed.observed.any()
        np.testing.assert_array_equal(refreshed.visual_indices, snapshot.visual_indices)
        np.testing.assert_array_equal(refreshed.fovea, io.observation_mask)
        assert bench.neutral_curve.getData()[1][-1] == missing.neutral_visual_mae
        assert bench.hidden_norm_curve.getData()[1][-1] == missing.decoded_hidden_norm
        assert bench.visual_norm_curve.getData()[1][-1] == missing.visual_potential_norm
        path = tmp_path / "continuous-trace.json"
        bench.save_trace(path)
        trace = json.loads(path.read_text())
        assert trace["model"] == "continuous-scene" and trace["memory_time_constant"] == .8
        assert "spike_threshold" not in trace and "membrane_capacitance" not in trace
        assert trace["frames"][-1]["neutral_visual_mae"] == missing.neutral_visual_mae
        bench.new_trial()
        qapp.processEvents()
        graph = bench.visualizer.source_orchestrator.sensory_graph
        assert isinstance(graph, ContinuousScenePredictiveCodingGraph)
        assert graph.memory_time_constant == .8
        assert not bench.history and bench.capture.read_count == 0
        np.testing.assert_array_equal(to_numpy(graph.visual_potential), 0)
        np.testing.assert_array_equal(to_numpy(graph.mean[graph.hidden_slice]), 0)
        bench.visualizer.gaze_controls.observe.setChecked(True)
        assert not bench.history and bench.step_once().observed_rgb > 0
    finally:
        bench.close()


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
def test_epistemic_gaze_scores_after_one_forecast_and_selects_before_camera_acquisition(
    qapp, request, monkeypatch, backend,
):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    bench = StaticWorldBench(resolution=(20, 28), foveal_radius=3, backend=backend, motion_enabled=False)
    bench.show()
    qapp.processEvents()
    try:
        source = bench.visualizer.source_orchestrator
        graph = source.sensory_graph
        xp = graph.backend
        precision = graph.observation_precision[graph.visual_slice].reshape(20, 28, 3)
        precision[:, 14:] *= 10
        controls = bench.visualizer.gaze_controls
        controls.autonomous_gaze.setChecked(True)
        assert source.gaze_policy is not None and source.has_epistemic_scoring
        assert not source.has_gaussian_uncertainty
        events, scores = [], []
        forecast, scoring, read, condition = (
            graph.forecast, graph.candidate_information_gains, bench.capture.read, graph.condition,
        )

        def forecast_once(*args, **kwargs):
            events.append("forecast")
            return forecast(*args, **kwargs)

        def score(*args, **kwargs):
            assert graph.stage == "forecast" and bench.capture.read_count == 0
            assert not bool(xp.any(graph.clamped))
            before = graph.belief_snapshot()
            result = scoring(*args, **kwargs)
            after = graph.belief_snapshot()
            for key in ("mean", "sensory_potential", *graph._PARAMETERS, *graph._PRECISIONS):
                np.testing.assert_array_equal(after[key], before[key])
            events.append("score")
            scores.append(result.copy())
            return result

        def acquire():
            events.append("capture")
            assert source.graph_io.gaze[0] >= .5
            return read()

        def settle(*args, **kwargs):
            events.append("condition")
            return condition(*args, **kwargs)

        monkeypatch.setattr(graph, "forecast", forecast_once)
        monkeypatch.setattr(graph, "candidate_information_gains", score)
        monkeypatch.setattr(bench.capture, "read", acquire)
        monkeypatch.setattr(graph, "condition", settle)
        frame = bench.step_once()
        assert events == ["forecast", "score", "capture", "condition"]
        assert frame.gaze[0] >= .5 and bench.capture.read_count == 1
        assert source.gaze_scores.shape == (9, 13)
        np.testing.assert_array_equal(source.gaze_scores, scores[0])
        np.testing.assert_array_equal(bench.visualizer.attention_scores, scores[0])
        gx, gy = frame.gaze
        assert scores[0][round(gy*8), round(gx*12)] == pytest.approx(scores[0].max())
        np.testing.assert_array_equal(
            to_numpy(graph.clamped[graph.visual_slice]).reshape(20, 28, 3).all(axis=-1),
            source.graph_io.observation_mask,
        )
        before = {name: to_numpy(getattr(graph, name)).copy() for name in (*graph._PARAMETERS, *graph._PRECISIONS)}
        controls.observe.setChecked(False)
        events.clear()
        unobserved = bench.step_once()
        assert events == ["forecast", "condition"]
        assert unobserved.gaze == frame.gaze and unobserved.observed_rgb == 0
        assert source.gaze_scores is None and bench.visualizer.attention_scores is None
        assert bench.capture.read_count == 1
        for name, values in before.items():
            np.testing.assert_array_equal(to_numpy(getattr(graph, name)), values)
        bench.lock_gaze.setChecked(True)
        assert source.gaze_policy is None
        assert not controls.autonomous_gaze.isEnabled()
    finally:
        bench.close()
