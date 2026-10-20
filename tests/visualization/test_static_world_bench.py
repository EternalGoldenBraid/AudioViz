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
        np.testing.assert_array_equal(bench.error_image.image, np.full((20, 28, 4), (65, 65, 65, 255), dtype=np.uint8))
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
        panel.wave_dynamics_button.click()
        assert "ON" in panel.wave_dynamics_button.text()
        assert bench.step_once().wave_propagation
        bench.new_trial()
        qapp.processEvents()
        assert bench.visualizer.engine.wave_propagation_enabled
        assert bench.step_once().wave_propagation
        bench.save_trace(path)
        assert json.loads(path.read_text())["wave_propagation"]
        bench.run_button.setChecked(True)
        assert bench.timer.isActive() and not bench.step_button.isEnabled()
        QtTest.QTest.qWait(80)
        assert bench.steps > 1
        bench.run_button.setChecked(False)
        assert not bench.timer.isActive() and bench.step_button.isEnabled()
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
        before = {key: to_numpy(getattr(graph, key)).copy() for key in graph._PARAMETERS}
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
            expected = bench.scene[tuple(io.positions[io.valid].T)].astype(np.float32)/255
            actual = to_numpy(graph.evidence[graph.visual_slice]).reshape(-1, 3)[io.valid]
            np.testing.assert_array_equal(actual, expected)
            assert frame.observed_rgb == int(io.valid.sum())*3
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


def test_autonomous_gaze_routes_the_selected_patch_before_capture_and_holds_without_evidence(qapp, monkeypatch):
    bench = StaticWorldBench(resolution=(40, 60), foveal_radius=4, backend="numpy")
    bench.show()
    qapp.processEvents()
    try:
        source = bench.visualizer.source_orchestrator
        controls = bench.visualizer.gaze_controls
        controls.autonomous_gaze.setChecked(True)
        assert source.gaze_policy is not None
        scores = np.ones((9, 13))
        scores[2, 9] = 2
        monkeypatch.setattr(source, "visual_information_gain", lambda precision: scores)
        selected_before_reads = []
        select = source.gaze_policy.select

        def record_selection(scores, current):
            selected_before_reads.append(bench.capture.read_count)
            return select(scores, current)

        monkeypatch.setattr(source.gaze_policy, "select", record_selection)
        captured_gazes = []
        read = bench.capture.read

        def record_capture():
            captured_gazes.append(source.graph_io.gaze)
            return read()

        monkeypatch.setattr(bench.capture, "read", record_capture)
        assert bench.step_once().gaze == (.75, .25)
        assert selected_before_reads == [0]
        assert captured_gazes == [(.75, .25)]
        controls.observe.setChecked(False)
        frame = bench.step_once()
        assert frame.gaze == (.75, .25) and frame.observed_rgb == 0
        assert selected_before_reads == [0] and captured_gazes == [(.75, .25)]
        bench.new_trial()
        qapp.processEvents()
        assert bench.visualizer.gaze_controls.autonomous_gaze.isChecked()
        assert bench.visualizer.source_orchestrator.gaze_policy is not None
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
        expected = bench.scene[tuple(io.positions[io.valid].T)].astype(np.float32)/255
        np.testing.assert_array_equal(
            to_numpy(graph.evidence[graph.visual_slice]).reshape(-1, 3)[io.valid], expected,
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
    finally:
        qapp.setStyleSheet(old_style)
