from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyStatistics:
    mean: float
    variance: float
    channel_means: tuple[float, ...]

    @classmethod
    def from_error(cls, error: np.ndarray) -> EnergyStatistics:
        energy = 0.5 * error**2
        return cls(
            mean=float(energy.mean()),
            variance=float(energy.var()),
            channel_means=tuple(
                float(value) for value in energy.mean(axis=tuple(range(error.ndim - 1)))
            ),
        )


@dataclass(frozen=True)
class SensoryPreview:
    hidden: np.ndarray
    prediction: np.ndarray
    observation: np.ndarray


@dataclass(frozen=True)
class PredictiveCodingDiagnostics:
    hidden: EnergyStatistics
    visual: EnergyStatistics
    inference_energy: tuple[float, ...]
    canvas_means: tuple[float, ...]
    hidden_means: tuple[float, ...]
    observation_means: tuple[float, ...]
    canvas_weights: np.ndarray
    visual_weights: np.ndarray
    learning_enabled: bool
    learning_rate: float
    weight_update_norm: float
    spatial: SensoryPreview | None = None
    observation_present: bool = True
    cross_modal_weights: np.ndarray | None = None
    cross_parent_means: tuple[float, ...] = ()


def mean_energy(hidden_error: np.ndarray, visual_error: np.ndarray) -> float:
    """Spatially averaged field energy, with global sensory energy counted once."""
    if visual_error.ndim == 1:
        return float(0.5 * (np.sum(hidden_error**2, axis=-1).mean() + np.sum(visual_error**2)))
    return float(
        0.5
        * (
            np.sum(hidden_error**2, axis=-1)
            + np.sum(visual_error**2, axis=-1)
        ).mean()
    )


def copy_weights(weights: np.ndarray) -> np.ndarray:
    snapshot = weights.copy()
    snapshot.setflags(write=False)
    return snapshot
