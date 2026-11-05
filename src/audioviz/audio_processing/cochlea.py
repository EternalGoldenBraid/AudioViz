"""Causal raw-audio inference in a fixed bank of damped oscillators."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CochleaConfig:
    nodes: int = 16
    minimum_frequency: float = 80.0
    maximum_frequency: float = 8000.0
    quality_factor: float = 16.0
    correction_rate: float = 0.5
    graph_rate: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.nodes, bool) or not isinstance(self.nodes, int) or self.nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        for name in ("minimum_frequency", "maximum_frequency", "quality_factor"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_frequency < self.minimum_frequency:
            raise ValueError("maximum_frequency must be at least minimum_frequency")
        if self.quality_factor <= 0.5:
            raise ValueError("quality_factor must exceed 0.5 for underdamped oscillators")
        for name in ("correction_rate", "graph_rate"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class CochleaBlock:
    prediction: np.ndarray
    error: np.ndarray
    state: np.ndarray


class Cochlea:
    """Each sample is predicted before its residual corrects the position states.

    State rows are (position, quadrature), both in audio units. Quadrature is
    (velocity + damping * position) / damped_angular_frequency.
    """

    def __init__(self, sample_rate: float, config: CochleaConfig | None = None) -> None:
        self.config = config or CochleaConfig()
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if self.config.maximum_frequency >= sample_rate / 2:
            raise ValueError("cochlea frequencies must be below Nyquist")
        self.sample_rate = float(sample_rate)
        self.frequencies = np.geomspace(
            self.config.minimum_frequency, self.config.maximum_frequency, self.config.nodes
        )
        omega = 2 * np.pi * self.frequencies
        self.damping = omega / (2 * self.config.quality_factor)
        self.damped_omega = np.sqrt(omega**2 - self.damping**2)
        radius = np.exp(-self.damping / self.sample_rate)
        angle = self.damped_omega / self.sample_rate
        self._cosine = radius * np.cos(angle)
        self._sine = radius * np.sin(angle)
        self.state = np.zeros((self.config.nodes, 2), dtype=np.float64)
        self.last_block: CochleaBlock | None = None

    @property
    def observation_size(self) -> int:
        return 2 * self.config.nodes

    def reset(self) -> None:
        self.state.fill(0.0)
        self.last_block = None

    def process(self, samples: np.ndarray) -> CochleaBlock:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)):
            raise ValueError("cochlea input must be a nonempty finite mono waveform")
        prediction = np.empty(samples.size, dtype=np.float64)
        error = np.empty_like(prediction)
        x, q = self.state.T.copy()
        rate = self.config.correction_rate / self.config.nodes
        for index, sample in enumerate(samples):
            x, q = self._cosine * x + self._sine * q, self._cosine * q - self._sine * x
            prediction[index] = x.sum()
            error[index] = sample - prediction[index]
            x += rate * error[index]
        state = np.column_stack((x, q))
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(error)):
            raise FloatingPointError("Cochlea inference produced non-finite values")
        self.state[:] = state
        self.last_block = CochleaBlock(prediction, error, state.reshape(-1).copy())
        return self.last_block

    def correct_from_graph(self, prediction: np.ndarray) -> None:
        """One boundary correction; the captured raw-inferred target stays fixed."""
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != (self.observation_size,) or not np.all(np.isfinite(prediction)):
            raise ValueError("graph prediction must be a finite flattened cochlea state")
        self.state += self.config.graph_rate * (prediction.reshape(self.state.shape) - self.state)
