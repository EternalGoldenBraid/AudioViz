"""Step metrics for ContinuousScenePredictiveCodingGraph."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audioviz.offline.scene_metrics import StaticWorldMetrics
from audioviz.readouts.continuous_scene_predictive_coding import (
    ContinuousScenePredictiveCodingGraph,
    ContinuousSceneSnapshot,
)

__all__ = [
    "ContinuousSceneWorldMetrics",
    "measure_continuous_scene_step",
    "posterior_visual_prediction",
]


@dataclass(frozen=True)
class ContinuousSceneWorldMetrics(StaticWorldMetrics):
    decoded_hidden_norm: float
    visual_potential_norm: float
    neutral_visual_mae: float
    unobserved_neutral_visual_mae: float | None


def _posterior_snapshot(
    graph_or_snapshot: ContinuousScenePredictiveCodingGraph | ContinuousSceneSnapshot,
    caller: str,
) -> ContinuousSceneSnapshot:
    if isinstance(graph_or_snapshot, ContinuousScenePredictiveCodingGraph):
        snapshot = graph_or_snapshot.belief_snapshot()
    elif isinstance(graph_or_snapshot, dict):
        snapshot = graph_or_snapshot
    else:
        raise ValueError(f"{caller} requires a ContinuousScenePredictiveCodingGraph or snapshot")
    if snapshot.get("stage") != "posterior" or not snapshot.get("inference_energy"):
        raise RuntimeError(f"{caller} requires a posterior graph after condition()")
    return snapshot


def posterior_visual_prediction(
    graph: ContinuousScenePredictiveCodingGraph | ContinuousSceneSnapshot,
) -> np.ndarray:
    """Return host (H_w, W_w, 3) pre-learning posterior visual prediction without clamping."""
    snapshot = _posterior_snapshot(graph, "posterior_visual_prediction")
    v0, v1 = snapshot["visual_slice"]
    clamped = np.asarray(snapshot["clamped"][v0:v1], dtype=bool)
    evidence = np.asarray(snapshot["evidence"][v0:v1], dtype=np.float64)
    sensory_error = np.asarray(snapshot["sensory_error"][v0:v1], dtype=np.float64)
    mean = np.asarray(snapshot["mean"][v0:v1], dtype=np.float64)
    prediction = np.where(clamped, evidence - sensory_error, mean)
    return prediction.reshape(*snapshot["visual_shape"], 3).copy()


def measure_continuous_scene_step(
    graph: ContinuousScenePredictiveCodingGraph,
    *,
    step: int,
    dt: float,
    gaze: tuple[float, float],
) -> ContinuousSceneWorldMetrics:
    """Measure pre/post visual MSE, residual energies, and whole-scene state telemetry after condition()."""
    if not isinstance(graph, ContinuousScenePredictiveCodingGraph):
        raise ValueError("graph must be a ContinuousScenePredictiveCodingGraph")
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) < 0:
        raise ValueError("step must be a non-negative integer")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float, np.number))
        or not np.isfinite(dt)
        or float(dt) <= 0
    ):
        raise ValueError("dt must be finite and positive")
    try:
        gaze_arr = np.asarray(gaze, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("gaze must contain two normalized coordinates") from exc
    if (
        isinstance(gaze, (str, bytes))
        or gaze_arr.shape != (2,)
        or any(isinstance(coord, bool) for coord in gaze)
        or not np.isfinite(gaze_arr).all()
        or np.any((gaze_arr < 0.0) | (gaze_arr > 1.0))
    ):
        raise ValueError("gaze must contain two normalized coordinates in [0, 1]")

    snapshot = _posterior_snapshot(graph, "measure_continuous_scene_step")
    trajectory = tuple(float(energy) for energy in snapshot["inference_energy"])
    v0, v1 = snapshot["visual_slice"]
    h0, h1 = snapshot["hidden_slice"]
    visual_clamped = np.asarray(snapshot["clamped"][v0:v1], dtype=bool)
    observed_rgb = int(np.count_nonzero(visual_clamped))

    prior_visual = np.asarray(snapshot["prior_mean"], dtype=np.float64)[v0:v1]
    evidence_visual = np.asarray(snapshot["evidence"][v0:v1], dtype=np.float64)
    sensory_error = np.asarray(snapshot["sensory_error"], dtype=np.float64)
    hidden_error = np.asarray(snapshot["hidden_error"], dtype=np.float64)

    if observed_rgb > 0:
        forecast_residual = evidence_visual[visual_clamped] - prior_visual[visual_clamped]
        settled_residual = sensory_error[v0:v1][visual_clamped]
        forecast_mse: float | None = float(np.mean(forecast_residual * forecast_residual))
        settled_mse: float | None = float(np.mean(settled_residual * settled_residual))
    else:
        forecast_mse = None
        settled_mse = None

    hidden_energy = float(snapshot["hidden_energy"])
    sensory_energy = float(snapshot["sensory_energy"])
    canvas_gradient = np.asarray(snapshot["canvas_gradient"], dtype=np.float64)
    decoded_hidden = np.asarray(snapshot["mean"][h0:h1], dtype=np.float64)
    visual_potential = np.asarray(snapshot["visual_potential"], dtype=np.float64)

    aperture_visual = np.asarray(snapshot["aperture_mask"][v0:v1], dtype=bool)
    if np.any(aperture_visual):
        neutral_visual_mae = float(np.mean(np.abs(prior_visual[aperture_visual] - 0.5)))
    else:
        neutral_visual_mae = 0.0

    unobserved_mask = aperture_visual & (~visual_clamped)
    if np.any(unobserved_mask):
        unobserved_neutral_visual_mae: float | None = float(
            np.mean(np.abs(prior_visual[unobserved_mask] - 0.5))
        )
    else:
        unobserved_neutral_visual_mae = None

    return ContinuousSceneWorldMetrics(
        step=int(step),
        time=float(int(step) * float(dt)),
        gaze=(float(gaze_arr[0]), float(gaze_arr[1])),
        observed_rgb=observed_rgb,
        forecast_mse=forecast_mse,
        settled_mse=settled_mse,
        energy_initial=float(trajectory[0]),
        energy_final=float(snapshot["energy"]),
        hidden_energy=hidden_energy,
        sensory_energy=sensory_energy,
        canvas_gradient_norm=float(np.linalg.norm(canvas_gradient)),
        learning_updates=int(snapshot["updates"]),
        inference_steps=int(snapshot["inference_steps"]),
        inference_rate=float(snapshot["inference_rate"]),
        learning_enabled=bool(snapshot["learning_enabled"]),
        learning_rate=float(snapshot["learning_rate"]),
        canvas_rate=float(graph.config.canvas_rate),
        memory_time_constant=float(graph.memory_time_constant),
        inference_energy=trajectory,
        decoded_hidden_norm=float(np.linalg.norm(decoded_hidden)),
        visual_potential_norm=float(np.linalg.norm(visual_potential)),
        neutral_visual_mae=neutral_visual_mae,
        unobserved_neutral_visual_mae=unobserved_neutral_visual_mae,
    )
