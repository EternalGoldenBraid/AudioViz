"""Controlled static-world step metrics for SharedPredictiveCodingGraph."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audioviz.readouts.shared_predictive_coding import (
    SharedPredictiveCodingGraph,
    SharedPredictiveCodingSnapshot,
)
from audioviz.utils.array_backend import to_numpy

__all__ = [
    "StaticWorldMetrics",
    "measure_static_world_step",
    "posterior_visual_prediction",
]


@dataclass(frozen=True)
class StaticWorldMetrics:
    step: int
    time: float
    gaze: tuple[float, float]
    observed_rgb: int
    forecast_mse: float | None
    settled_mse: float | None
    energy_initial: float
    energy_final: float
    hidden_energy: float
    sensory_energy: float
    canvas_gradient_norm: float
    learning_updates: int
    inference_steps: int
    inference_rate: float
    learning_enabled: bool
    learning_rate: float
    canvas_rate: float
    memory_time_constant: float
    inference_energy: tuple[float, ...]


def _posterior_snapshot(
    graph_or_snapshot: SharedPredictiveCodingGraph | SharedPredictiveCodingSnapshot,
    caller: str,
) -> SharedPredictiveCodingSnapshot:
    if isinstance(graph_or_snapshot, SharedPredictiveCodingGraph):
        snapshot = graph_or_snapshot.belief_snapshot()
    elif isinstance(graph_or_snapshot, dict):
        snapshot = graph_or_snapshot
    else:
        raise ValueError(f"{caller} requires a SharedPredictiveCodingGraph or snapshot")
    if snapshot.get("stage") != "posterior" or not snapshot.get("inference_energy"):
        raise RuntimeError(f"{caller} requires a posterior graph after condition()")
    return snapshot


def posterior_visual_prediction(
    graph: SharedPredictiveCodingGraph | SharedPredictiveCodingSnapshot,
) -> np.ndarray:
    """Return host (K, 1, 3) pre-learning posterior visual prediction without clamping."""
    snapshot = _posterior_snapshot(graph, "posterior_visual_prediction")
    v0, v1 = snapshot["visual_slice"]
    clamped = np.asarray(snapshot["clamped"][v0:v1], dtype=bool)
    evidence = np.asarray(snapshot["evidence"][v0:v1], dtype=np.float64)
    sensory_error = np.asarray(snapshot["sensory_error"][v0:v1], dtype=np.float64)
    mean = np.asarray(snapshot["mean"][v0:v1], dtype=np.float64)
    prediction = np.where(clamped, evidence - sensory_error, mean)
    return prediction.reshape(*snapshot["visual_shape"], 3).copy()


def measure_static_world_step(
    graph: SharedPredictiveCodingGraph,
    *,
    step: int,
    dt: float,
    gaze: tuple[float, float],
) -> StaticWorldMetrics:
    """Measure pre/post visual MSE and pre-learning residual energies after condition()."""
    if not isinstance(graph, SharedPredictiveCodingGraph):
        raise ValueError("graph must be a SharedPredictiveCodingGraph")
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

    snapshot = _posterior_snapshot(graph, "measure_static_world_step")
    trajectory = tuple(float(energy) for energy in snapshot["inference_energy"])
    v0, v1 = snapshot["visual_slice"]
    visual_clamped = np.asarray(snapshot["clamped"][v0:v1], dtype=bool)
    observed_rgb = int(np.count_nonzero(visual_clamped))

    prior_visual = np.asarray(to_numpy(graph.prior_mean), dtype=np.float64)[v0:v1]
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

    hidden_energy = 0.5 * float(hidden_error @ hidden_error)
    sensory_energy = 0.5 * float(sensory_error @ sensory_error)
    canvas_gradient = np.asarray(snapshot["canvas_gradient"], dtype=np.float64)

    return StaticWorldMetrics(
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
    )
