import importlib.util
from pathlib import Path

import numpy as np
import pytest

from audioviz.physics.wave_propagator import ForceDrivenCanvasCPU
from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout


spec = importlib.util.spec_from_file_location(
    "continuous_inference_review",
    Path(__file__).resolve().parents[2] / "scripts/continuous_inference_review.py",
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_legacy_capture_records_applied_updates_including_clipping():
    model = PredictiveCodingRGBReadout(
        canvas_shape=(1, 1, 3),
        config=PredictiveCodingConfig(
            learning_enabled=True, learning_rate=.2, gradient_clip=1e-6,
        ),
    )
    canvas = np.zeros((1, 1, 3))
    model.predict(canvas)
    before = {
        name: getattr(model, name).copy()
        for name in ("canvas_weights", "visual_weights")
    }
    model.infer(canvas, np.full_like(canvas, .9))
    record = review.record_legacy(model, before, 7, "held")
    assert record["frame"] == 7
    assert record["time"] == 7/30
    for name, key, update_name in (
        ("canvas_weights", "Wc", "canvas_weights"),
        ("visual_weights", "Ws", "sensory_weights"),
    ):
        expected = getattr(model, name) - before[name]
        np.testing.assert_array_equal(record[key], before[name])
        np.testing.assert_array_equal(record["weight_updates"][update_name], expected)
        assert record["weight_update_norms"][update_name] == np.linalg.norm(expected)
    actual = np.asarray(record["weight_updates"]["sensory_weights"])
    assert np.max(np.abs(actual)) > 0.
    assert not np.allclose(actual, .2 * model._observation_gradient(), atol=1e-10)
    model.visual_weights.fill(0.)
    np.testing.assert_array_equal(record["Ws"], before["visual_weights"])


def test_continuous_capture_samples_actual_updates_and_keeps_full_matrix_norms():
    graph = ContinuousScenePredictiveCodingGraph(
        visual_shape=(1, 2), audio_nodes=0, hidden_nodes=2,
        config=PredictiveCodingConfig(
            hidden_channels=2, inference_rate=2., learning_enabled=True,
        ),
    )
    canvas = np.full((1, 2, 3), .2)
    image = np.ones_like(canvas)
    graph.forecast(canvas, .1)
    before = {name: getattr(graph, name).copy() for name in graph._PARAMETERS}
    graph.condition(image)
    indices = np.array([0, 5])
    record = review.record_graph(graph, 0, "held", image, canvas, indices, .5, .4)
    for name, key in (
        ("canvas_weights", "Wc"),
        ("sensory_weights", "Ws"),
        ("recurrent_weights", "Wh"),
    ):
        expected = getattr(graph, name) - before[name]
        assert record["weight_update_norms"][name] == np.linalg.norm(expected)
        initial = before[name]
        if name == "canvas_weights":
            expected, initial = expected[:, indices], initial[:, indices]
        elif name == "sensory_weights":
            expected, initial = expected[indices], initial[indices]
        np.testing.assert_array_equal(record[key], initial)
        np.testing.assert_array_equal(record["weight_updates"][name], expected)
    assert record["weight_update_norms"]["sensory_weights"] > 0.


@pytest.mark.parametrize("strength", (0., 1.))
def test_force_capture_matches_gradient_and_elapsed_inference_time(strength):
    graph = review.plain_graph(
        (2, 2), hidden=2, inference_rate=2., learning_enabled=True,
    )
    canvas = ForceDrivenCanvasCPU((2, 2, 3), dx=1., speed=1., damping_rate=.8)
    canvas.position[0, 0] = .2
    graph.forecast(canvas.position, .05)
    graph.condition(np.ones_like(canvas.position))
    snapshot = graph.belief_snapshot()
    control = ForceDrivenCanvasCPU((2, 2, 3), dx=1., speed=1., damping_rate=.8)
    control.position[:] = canvas.position
    result = review.advance_canvas_force(graph, canvas, .05, strength, .02)
    np.testing.assert_allclose(
        result["force_start"].ravel(), -strength * snapshot["canvas_gradient"],
        rtol=1e-12, atol=1e-15,
    )
    assert result["simulation_time"] == pytest.approx(.05)
    assert result["wave_substeps"] == 3
    assert result["wave_dt"] * result["wave_substeps"] == pytest.approx(graph._last_forecast_dt)
    assert graph.updates == 1
    assert np.linalg.norm(graph.sensory_weights - snapshot["sensory_weights"]) > 0.
    if strength == 0.:
        for _ in range(result["wave_substeps"]):
            control.step(np.zeros_like(control.position), result["wave_dt"], force_curvature=0.)
        np.testing.assert_array_equal(canvas.position, control.position)
        np.testing.assert_array_equal(canvas.velocity, control.velocity)


@pytest.fixture
def dual_review(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/continuous_dual_review.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("continuous_dual_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dual_messages_cover_every_coordinate_and_match_full_gradients(dual_review):
    graph = review.plain_graph((2, 3), hidden=2, inference_rate=2., learning_enabled=True)
    canvas = np.linspace(-.2, .3, 18).reshape(2, 3, 3)
    image = np.linspace(.1, .9, 18).reshape(2, 3, 3)
    graph.forecast(canvas, .05)
    graph.condition(image)
    groups = dual_review.spatial_groups((2, 3))
    nodes, edges = dual_review.topology(groups, 2)
    boundary = dual_review.sample_boundary(graph, canvas, image, groups)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([group["indices"] for group in groups])), np.arange(18),
    )
    assert len(boundary["messages"]) == len(edges)
    h = graph.mean[graph.hidden_slice]
    a = np.tanh(h)
    eh = h-graph.canvas_weights @ canvas.ravel()-graph.recurrent_weights @ a
    ez = graph.sensory_potential-graph.sensory_weights @ a
    for index in range(2):
        hidden = 3*len(groups)+index
        predictions = sum(
            float(value[0]) for edge, value in zip(edges, boundary["messages"])
            if edge["target"] == hidden and edge["kind"] == "prediction"
        )
        feedback = sum(
            float(value[0]) for edge, value in zip(edges, boundary["messages"])
            if edge["target"] == hidden and edge["kind"] == "feedback"
        )
        assert predictions == pytest.approx(
            (graph.canvas_weights @ canvas.ravel()+graph.recurrent_weights @ a)[index],
        )
        assert feedback == pytest.approx(
            ((graph.recurrent_weights.T @ eh+graph.sensory_weights.T @ ez)*(1-a*a))[index],
        )
    for region, group in enumerate(groups):
        feedback = sum(
            value for edge, value in zip(edges, boundary["messages"])
            if edge["target"] == region and edge["kind"] == "feedback"
        )
        np.testing.assert_allclose(feedback, (graph.canvas_weights.T @ eh)[group["indices"]])
    assert sum(node["coordinates"] for node in nodes if node["layer"] == "sensory") == 18
    for region, group in enumerate(groups):
        np.testing.assert_allclose(
            boundary["gradients"][2*len(groups)+region],
            (image.ravel()-graph.sensory_potential)[group["indices"]],
        )


def test_dual_message_change_can_exist_without_node_motion(dual_review):
    graph = review.plain_graph((1, 1), hidden=2)
    graph.mean[graph.hidden_slice] = [.2, -.1]
    graph.sensory_potential[:] = [.1, -.2, .3]
    canvas = np.ones((1, 1, 3))
    groups = dual_review.spatial_groups((1, 1))
    old = dual_review.sample_boundary(graph, canvas, np.zeros_like(canvas), groups)
    graph.sensory_weights *= -2.
    new = dual_review.sample_boundary(graph, canvas, np.zeros_like(canvas), groups)
    record = dual_review.record_boundary(new, old, .1)
    assert all(node[3] == 0. for node in record["nodes"])
    assert any(message[3] > 0. for message in record["messages"])
    graph.canvas_weights.fill(0.)
    graph.recurrent_weights.fill(0.)
    graph.sensory_weights.fill(0.)
    zero = dual_review.sample_boundary(graph, canvas, np.zeros_like(canvas), groups)
    _, edges = dual_review.topology(groups, 2)
    assert all(
        np.all(value == 0.) for edge, value in zip(edges, zero["messages"])
        if edge["family"] != "observation"
    )


def test_dual_input_atlas_matches_the_observed_record_boundaries(dual_review, tmp_path):
    import json
    from PIL import Image

    dual_review.capture(tmp_path, 2)
    data = json.loads((tmp_path/"dual-capture.json").read_text())
    columns = data["settings"]["input_atlas"]["columns"]
    assert data["settings"]["input_atlas"]["frames"] == len(data["records"]) == 3
    height, width = data["settings"]["shape"]
    groups = dual_review.spatial_groups((height, width))
    with Image.open(tmp_path/"dual-input.png") as atlas:
        for frame in range(3):
            expected = np.zeros((height, width, 3)) if frame == 0 else review.camera_frame(frame-1)
            x, y = (frame%columns)*width, (frame//columns)*height
            np.testing.assert_array_equal(
                np.asarray(atlas.crop((x, y, x+width, y+height))),
                np.rint(expected*255).astype(np.uint8),
            )
            for region, group in enumerate(groups):
                assert data["records"][frame]["nodes"][2*len(groups)+region][0] == pytest.approx(
                    float(np.mean(expected.ravel()[group["indices"]])),
                )
