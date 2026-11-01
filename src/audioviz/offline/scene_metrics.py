"""Model-neutral measurements shared by scene experiments."""

from __future__ import annotations

from dataclasses import dataclass


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
