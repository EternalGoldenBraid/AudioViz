import numpy as np

from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.utils.continuous_graph_preview import continuous_graph_preview


def test_preview_samples_real_prelearning_weights_and_world_anchored_state_without_mutation():
    graph = ContinuousScenePredictiveCodingGraph(visual_shape=(80, 120), hidden_nodes=16)
    graph.update_config(learning_enabled=True)
    aperture = np.zeros(graph.visual_shape, dtype=bool)
    aperture[10:30, 20:40] = True
    observed = np.full((*graph.visual_shape, 3), .8)
    graph.forecast(np.full_like(observed, .2), 1 / 30, aperture_mask=aperture)
    graph.condition(observed, visual_mask=aperture)
    before = graph.belief_snapshot()
    weights = {name: getattr(graph, name).copy() for name in graph._PARAMETERS}
    preview = continuous_graph_preview(graph, observed, foveal_mask=aperture)
    assert max(preview.prediction.shape[:2]) <= 64
    assert len(preview.visual_indices) <= 25
    assert len(preview.hidden_indices) == 12
    assert len(preview.audio_indices) == 4
    sensory = np.concatenate((preview.visual_indices, graph.visual_nodes + preview.audio_indices))
    np.testing.assert_array_equal(
        preview.sensory_weights,
        graph._snapshot_weights["sensory_weights"][np.ix_(sensory, preview.hidden_indices)],
    )
    np.testing.assert_array_equal(
        preview.canvas_weights,
        graph._snapshot_weights["canvas_weights"][np.ix_(preview.hidden_indices, preview.visual_indices)],
    )
    np.testing.assert_array_equal(preview.hidden, graph.mean[graph.hidden_slice][preview.hidden_indices])
    assert np.any(preview.observed) and not np.all(preview.observed)
    np.testing.assert_allclose(preview.error, preview.reference - preview.prediction)
    assert np.all(preview.error[~preview.observed.astype(bool)] > 0)
    assert not graph.sensory_error[graph.visual_slice].reshape(*graph.visual_shape, 3)[~aperture].any()
    assert not preview.sensory_weights.flags.writeable
    changed_reference = observed.copy()
    changed_reference[~aperture] = .1
    changed_preview = continuous_graph_preview(graph, changed_reference, foveal_mask=aperture)
    np.testing.assert_array_equal(changed_preview.prediction, preview.prediction)
    np.testing.assert_array_equal(changed_preview.observed, preview.observed)
    sampled_mask = preview.observed.astype(bool)
    np.testing.assert_array_equal(changed_preview.error[sampled_mask], preview.error[sampled_mask])
    assert np.all(changed_preview.error[~sampled_mask] != preview.error[~sampled_mask])
    for name, values in weights.items():
        np.testing.assert_array_equal(getattr(graph, name), values)
    np.testing.assert_array_equal(graph.mean, before["mean"])
    np.testing.assert_array_equal(graph.clamped, before["clamped"])
    np.testing.assert_array_equal(graph.sensory_error, before["sensory_error"])
    frozen = preview.prediction.copy()
    graph.forecast(np.zeros_like(observed), 1 / 30, aperture_mask=~aperture)
    graph.condition()
    next_preview = continuous_graph_preview(graph, None, foveal_mask=~aperture)
    np.testing.assert_array_equal(preview.prediction, frozen)
    np.testing.assert_array_equal(next_preview.visual_indices, preview.visual_indices)
    np.testing.assert_array_equal(next_preview.hidden_indices, preview.hidden_indices)
    assert next_preview.reference is None
    assert not next_preview.observed.any()
    assert next_preview.error is None
