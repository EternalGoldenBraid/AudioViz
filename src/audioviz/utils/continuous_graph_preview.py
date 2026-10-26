from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.utils.array_backend import to_numpy
from audioviz.utils.spatial_preview import preview_indices, spatial_preview


@dataclass(frozen=True)
class ContinuousGraphPreview:
    visual_shape: tuple[int, int]
    hidden_count: int
    audio_count: int
    prediction: np.ndarray
    reference: np.ndarray | None
    independent_reference: bool
    observed: np.ndarray
    error: np.ndarray | None
    canvas: np.ndarray
    fovea: np.ndarray
    visual_indices: np.ndarray
    hidden_indices: np.ndarray
    audio_indices: np.ndarray
    hidden: np.ndarray
    audio: np.ndarray
    canvas_weights: np.ndarray
    sensory_weights: np.ndarray
    recurrent_weights: np.ndarray


def continuous_graph_preview(
    graph: ContinuousScenePredictiveCodingGraph,
    reference,
    *,
    foveal_mask: np.ndarray,
    independent_reference: bool = False,
) -> ContinuousGraphPreview:
    """Copy bounded settled state/topology; full-reference error is display-only."""
    xp = graph.backend
    rows, columns = preview_indices(graph.visual_shape, max_side=5)
    pixels = (rows[:, None] * graph.visual_shape[1] + columns[None, :]).reshape(-1)
    visual = pixels * 3 + np.arange(len(pixels)) % 3
    hidden = np.linspace(0, graph.hidden_nodes - 1, min(12, graph.hidden_nodes), dtype=np.intp)
    audio = np.linspace(0, graph.audio_nodes - 1, min(4, graph.audio_nodes), dtype=np.intp)
    sensory = np.concatenate((visual, graph.visual_nodes + audio))

    def host(values):
        result = np.array(to_numpy(values), copy=True)
        result.setflags(write=False)
        return result

    def block(values, row_indices, column_indices):
        return host(values[xp.asarray(row_indices)[:, None], xp.asarray(column_indices)[None, :]])

    shape = (*graph.visual_shape, 3)
    weights = graph._snapshot_weights
    mask = graph.clamped[graph.visual_slice].reshape(shape).any(axis=-1)
    prediction = spatial_preview(graph.visual_prediction)
    reference = spatial_preview(reference) if reference is not None else None
    error = host(reference - prediction) if reference is not None else None
    return ContinuousGraphPreview(
        visual_shape=graph.visual_shape, hidden_count=graph.hidden_nodes,
        audio_count=graph.audio_nodes,
        prediction=prediction, reference=reference, independent_reference=independent_reference,
        observed=spatial_preview(mask),
        error=error,
        canvas=spatial_preview(graph.canvas.reshape(shape)),
        fovea=spatial_preview(foveal_mask),
        visual_indices=host(visual), hidden_indices=host(hidden), audio_indices=host(audio),
        hidden=host(graph.mean[graph.hidden_slice][xp.asarray(hidden)]),
        audio=host(graph.audio_prediction[xp.asarray(audio)]),
        canvas_weights=block(weights["canvas_weights"], hidden, visual),
        sensory_weights=block(weights["sensory_weights"], sensory, hidden),
        recurrent_weights=block(weights["recurrent_weights"], hidden, hidden),
    )
