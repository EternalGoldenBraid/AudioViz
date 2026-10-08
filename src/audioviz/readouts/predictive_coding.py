from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np

from audioviz.readouts.diagnostics import (
    EnergyStatistics,
    PredictiveCodingDiagnostics,
    SensoryPreview,
    StreamStateDiagnostics,
    copy_weights,
    mean_energy,
)
from audioviz.utils.spatial_preview import spatial_preview

@dataclass(frozen=True)
class PredictiveCodingConfig:
    hidden_channels: int = 6
    inference_steps: int = 8
    inference_rate: float = 0.1
    canvas_rate: float = 0.01
    learning_enabled: bool = False
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    weight_clip: float = 10.0
    gradient_clip: float = 1.0
    cross_modal_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.cross_modal_enabled, bool):
            raise ValueError("cross_modal_enabled must be a boolean")
        for name in ("hidden_channels", "inference_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("inference_rate", "canvas_rate", "learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("weight_clip", "gradient_clip"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class PredictiveCodingRGBReadout:
    """Local PC inference with persistent per-pixel states and shared weights."""

    def __init__(
        self,
        *,
        canvas_shape: tuple[int, ...],
        config: PredictiveCodingConfig | None = None,
    ) -> None:
        if (
            len(canvas_shape) != 3
            or canvas_shape[2] != 3
            or any(size <= 0 for size in canvas_shape)
        ):
            raise ValueError("predictive RGB readout requires a nonempty RGB canvas")
        self.canvas_shape = tuple(canvas_shape)
        self.config = PredictiveCodingConfig() if config is None else config
        self.hidden = np.zeros(
            (*self.canvas_shape[:2], self.config.hidden_channels),
            dtype=np.float64,
        )
        rng = np.random.default_rng(0)
        self.canvas_weights = rng.normal(
            scale=0.1, size=(self.config.hidden_channels, 3)
        )
        self.visual_weights = rng.normal(
            scale=0.1, size=(3, self.config.hidden_channels)
        )
        for channel in range(min(3, self.config.hidden_channels)):
            self.canvas_weights[channel, channel] = 1.0
            self.visual_weights[channel, channel] = 1.0
        self.stream_state_weights = rng.normal(scale=.1, size=(1, self.config.hidden_channels))
        self.stream_state_bias = np.zeros(1)
        self.stream_state_prediction: float | None = None
        self.hidden_error = np.zeros_like(self.hidden)
        self.visual_error = np.zeros(self.canvas_shape, dtype=np.float64)
        self.diagnostics_enabled = False
        self.diagnostics: PredictiveCodingDiagnostics | None = None
        self.spatial_preview_enabled = False
        self._preview_prediction: np.ndarray | None = None

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self.diagnostics_enabled = bool(enabled)
        self.diagnostics = None
        if not enabled:
            self.set_spatial_preview_enabled(False)

    def set_spatial_preview_enabled(self, enabled: bool) -> None:
        self.spatial_preview_enabled = bool(enabled) and self.diagnostics_enabled
        if not self.spatial_preview_enabled:
            self._preview_prediction = None
            if self.diagnostics is not None and self.diagnostics.spatial is not None:
                self.diagnostics = replace(self.diagnostics, spatial=None)

    def update_config(self, **changes) -> None:
        next_config = replace(self.config, **changes)
        if next_config.hidden_channels != self.config.hidden_channels:
            raise ValueError("hidden_channels cannot change without rebuilding the readout")
        self.config = next_config

    def reset(self) -> None:
        """Clear belief states and errors, retaining learned weights."""
        self.hidden.fill(0.0)
        self.hidden_error.fill(0.0)
        self.visual_error.fill(0.0)
        self.diagnostics = None
        self._preview_prediction = None
        self.stream_state_prediction = None

    def predict(self, prior: np.ndarray) -> np.ndarray:
        canvas = self._coerce(prior, name="canvas prior")
        predicted_hidden = np.tanh(canvas) @ self.canvas_weights.T
        # Advance memory from the wave prior before any new sensory evidence.
        self.hidden += self.config.inference_rate * (predicted_hidden - self.hidden)
        return self._capture_prediction()

    def _capture_prediction(self) -> np.ndarray:
        self.diagnostics = None
        self._preview_prediction = None
        prediction = self._predict_observation(np.tanh(self.hidden))
        self.stream_state_prediction = self.predict_stream_state()
        self._require_finite(self.hidden, prediction)
        if self.spatial_preview_enabled:
            self._preview_prediction = self._preview_observation(prediction)
        return prediction.astype(np.float32)

    def infer(
        self, prior: np.ndarray, observation: np.ndarray | None,
        *, stream_state: bool | None = None,
    ) -> np.ndarray:
        return infer_joint(
            prior, ((self, observation),),
            stream_states={self: stream_state} if stream_state is not None else None,
        )

    def predict_stream_state(self, hidden_activity: np.ndarray | None = None) -> float:
        activity = np.tanh(self.hidden) if hidden_activity is None else hidden_activity
        logit = float((activity.mean(axis=(0, 1)) @ self.stream_state_weights.T + self.stream_state_bias)[0])
        self._require_finite(logit)
        return float(.5 * (1 + np.tanh(.5 * logit)))

    def _stream_state_terms(
        self, observed: bool, hidden_activity: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        prediction = self.predict_stream_state(hidden_activity)
        error = float(observed) - prediction
        return prediction, error, error * prediction * (1 - prediction)

    def _finish_inference(
        self, canvas: np.ndarray, observed: np.ndarray | None, inference_energy: list[float],
        *, hidden_error: np.ndarray | None = None, learn: bool = True,
        cross_weights: np.ndarray | None = None,
        cross_parent_means: tuple[float, ...] = (),
        cross_update_norm: float = 0.0,
        stream_state: bool | None = None,
    ) -> None:
        self.hidden_error = (
            self.hidden - np.tanh(canvas) @ self.canvas_weights.T
            if hidden_error is None else hidden_error
        )
        self.visual_error = (
            np.zeros_like(self.visual_error) if observed is None
            else observed - self._predict_observation(np.tanh(self.hidden))
        )
        weights_before = None
        parameters = (
            self.canvas_weights, self.visual_weights,
            self.stream_state_weights, self.stream_state_bias,
        )
        if self.diagnostics_enabled:
            weights_before = tuple(weights.copy() for weights in parameters)
        stream_terms = self._stream_state_terms(stream_state) if stream_state is not None else None
        if learn and self.config.learning_enabled and self.config.learning_rate:
            self._learn(canvas, learn_sensory=observed is not None)
            if stream_terms is not None:
                direction = stream_terms[2]
                self._update_weights(
                    self.stream_state_weights,
                    direction * np.tanh(self.hidden).mean(axis=(0, 1))[None, :],
                )
                self._update_weights(self.stream_state_bias, np.array([direction]))
        if weights_before is not None:
            self.diagnostics = PredictiveCodingDiagnostics(
                hidden=EnergyStatistics.from_error(self.hidden_error),
                visual=(
                    EnergyStatistics.from_error(self.visual_error) if observed is not None
                    else EnergyStatistics(
                        np.nan, np.nan, (np.nan,) * self.visual_weights.shape[0]
                    )
                ),
                inference_energy=tuple(inference_energy),
                canvas_means=tuple(float(value) for value in canvas.mean(axis=(0, 1))),
                hidden_means=tuple(float(value) for value in self.hidden.mean(axis=(0, 1))),
                observation_means=tuple(
                    float(value)
                    for value in (
                        observed if observed is not None
                        else self._predict_observation(np.tanh(self.hidden))
                    ).mean(axis=tuple(range(self.visual_error.ndim - 1)))
                ),
                canvas_weights=copy_weights(self.canvas_weights),
                visual_weights=copy_weights(self.visual_weights),
                learning_enabled=self.config.learning_enabled,
                learning_rate=self.config.learning_rate,
                weight_update_norm=float(
                    np.sqrt(
                        sum(np.sum((weights - before)**2) for weights, before in zip(parameters, weights_before))
                        + cross_update_norm**2
                    )
                ),
                spatial=(
                    SensoryPreview(
                        hidden=spatial_preview(self.hidden),
                        prediction=self._preview_prediction,
                        observation=self._preview_observation(observed),
                    )
                    if observed is not None and self.spatial_preview_enabled and self._preview_prediction is not None
                    else None
                ),
                observation_present=observed is not None,
                cross_modal_weights=copy_weights(cross_weights) if cross_weights is not None else None,
                cross_parent_means=cross_parent_means,
                stream_state=(
                    StreamStateDiagnostics(
                        observed=stream_state, prediction=stream_terms[0], energy=.5 * stream_terms[1]**2,
                        weights=copy_weights(self.stream_state_weights),
                        bias=float(self.stream_state_bias[0]),
                    )
                    if stream_state is not None and stream_terms is not None else None
                ),
            )
        self._preview_prediction = None

    def energy(self, canvas: np.ndarray, observation: np.ndarray) -> float:
        hidden_error, visual_error = self._errors(
            self._coerce(canvas, name="canvas state"),
            self._coerce_observation(observation),
        )
        return mean_energy(hidden_error, visual_error)

    def _errors(
        self, canvas: np.ndarray, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.hidden - np.tanh(canvas) @ self.canvas_weights.T,
            observation - self._predict_observation(np.tanh(self.hidden)),
        )

    def _learn(self, canvas: np.ndarray, *, learn_sensory: bool = True) -> None:
        pixel_count = self.canvas_shape[0] * self.canvas_shape[1]
        canvas_gradient = (
            self.hidden_error.reshape(-1, self.config.hidden_channels).T
            @ np.tanh(canvas).reshape(-1, 3)
        ) / pixel_count
        self._update_weights(self.canvas_weights, canvas_gradient)
        if learn_sensory:
            self._update_weights(self.visual_weights, self._observation_gradient())

    def _update_weights(self, weights: np.ndarray, gradient: np.ndarray) -> None:
        gradient = gradient - self.config.weight_decay * weights
        weights += self.config.learning_rate * np.clip(
            gradient, -self.config.gradient_clip, self.config.gradient_clip
        )
        np.clip(weights, -self.config.weight_clip, self.config.weight_clip, out=weights)
        self._require_finite(weights)

    def _observation_gradient(self) -> np.ndarray:
        pixel_count = self.canvas_shape[0] * self.canvas_shape[1]
        return (
            self.visual_error.reshape(-1, 3).T
            @ np.tanh(self.hidden).reshape(-1, self.config.hidden_channels)
        ) / pixel_count

    def _predict_observation(self, hidden_activity: np.ndarray) -> np.ndarray:
        return hidden_activity @ self.visual_weights.T

    def _coerce_observation(self, observation: np.ndarray) -> np.ndarray:
        return self._coerce(observation, name="camera observation")

    def _preview_observation(self, observation: np.ndarray) -> np.ndarray:
        return spatial_preview(observation)

    def _coerce(self, values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.canvas_shape:
            raise ValueError(f"{name} must match the readout canvas shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _require_finite(*values: np.ndarray | float) -> None:
        if any(not np.all(np.isfinite(value)) for value in values):
            raise FloatingPointError(
                "Predictive coding diverged; reduce inference or learning rates."
            )


class PredictiveCodingHiddenCoupling:
    """Two zero-initialized, shared per-pixel hidden-to-hidden projections."""

    def __init__(
        self, camera: PredictiveCodingRGBReadout, audio: PredictiveCodingRGBReadout
    ) -> None:
        if camera is audio or camera.canvas_shape != audio.canvas_shape:
            raise ValueError("Hidden coupling requires distinct branches on the same canvas.")
        self.branches = (camera, audio)
        self.weights = {
            target: np.zeros((target.hidden.shape[-1], parent.hidden.shape[-1]))
            for target, parent in ((camera, audio), (audio, camera))
        }

    def parent(self, branch: PredictiveCodingRGBReadout) -> PredictiveCodingRGBReadout:
        if branch is self.branches[0]:
            return self.branches[1]
        if branch is self.branches[1]:
            return self.branches[0]
        raise ValueError("Readout does not belong to this hidden coupling.")

    def hidden_errors(self, canvas: np.ndarray) -> dict[PredictiveCodingRGBReadout, np.ndarray]:
        activity = np.tanh(canvas)
        return {
            branch: branch.hidden - activity @ branch.canvas_weights.T
            - np.tanh(self.parent(branch).hidden) @ self.weights[branch].T
            for branch in self.branches
        }

    def feedback(
        self, branch: PredictiveCodingRGBReadout,
        errors: dict[PredictiveCodingRGBReadout, np.ndarray],
    ) -> np.ndarray:
        child = self.parent(branch)
        return errors[child] @ self.weights[child]

    def predict(self, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        canvas = self.branches[0]._coerce(prior, name="canvas prior")
        errors = self.hidden_errors(canvas)
        updates = [
            branch.config.inference_rate * (
                -errors[branch] + (1 - np.tanh(branch.hidden)**2) * self.feedback(branch, errors)
            )
            for branch in self.branches
        ]
        for branch, update in zip(self.branches, updates):
            branch.hidden += update
        camera, audio = self.branches
        return camera._capture_prediction(), audio._capture_prediction()


def infer_joint(
    prior: np.ndarray,
    branches: tuple[tuple[PredictiveCodingRGBReadout, np.ndarray | None], ...],
    *,
    coupling: PredictiveCodingHiddenCoupling | None = None,
    stream_states: Mapping[PredictiveCodingRGBReadout, bool] | None = None,
) -> np.ndarray:
    """Infer active sensory branches synchronously, with weights fixed until settling."""
    if not branches:
        raise ValueError("Joint inference requires at least one observed branch.")
    if coupling is not None and (
        len(branches) != 2 or {branch for branch, _ in branches} != set(coupling.branches)
    ):
        raise ValueError("Coupled inference requires both hidden branches.")
    first = branches[0][0]
    states = {} if stream_states is None else dict(stream_states)
    if not set(states).issubset({branch for branch, _ in branches}):
        raise ValueError("Stream-state context must belong to an inferred branch.")
    for state in states.values():
        if not isinstance(state, bool):
            raise ValueError("Stream-state context requires a boolean observation.")
    canvas = first._coerce(prior, name="canvas prior").copy()
    steps = first.config.inference_steps
    observations = []
    for branch, observed in branches:
        if branch.canvas_shape != first.canvas_shape or branch.config.inference_steps != steps:
            raise ValueError("Joint branches must share canvas shape and inference step count.")
        if observed is None and coupling is None and branch not in states:
            raise ValueError("Uncoupled inference requires sensory or stream-state evidence.")
        observations.append((branch, branch._coerce_observation(observed) if observed is not None else None))
    collect = any(branch.diagnostics_enabled for branch, _ in observations)
    trajectory: list[float] = []
    for _ in range(steps):
        canvas_activity = np.tanh(canvas)
        correction = np.zeros_like(canvas)
        hidden_updates = []
        energy = 0.0
        errors = coupling.hidden_errors(canvas) if coupling is not None else {
            branch: branch.hidden - canvas_activity @ branch.canvas_weights.T
            for branch, _ in observations
        }
        for branch, observed in observations:
            hidden_activity = np.tanh(branch.hidden)
            hidden_error = errors[branch]
            sensory_error = (
                np.zeros_like(branch.visual_error) if observed is None
                else observed - branch._predict_observation(hidden_activity)
            )
            correction += branch.config.canvas_rate * (
                (1.0 - canvas_activity**2) * (hidden_error @ branch.canvas_weights)
            )
            feedback = sensory_error @ branch.visual_weights
            if branch in states:
                _, state_error, state_direction = branch._stream_state_terms(states[branch], hidden_activity)
                feedback = feedback + state_direction * branch.stream_state_weights[0]
                if collect:
                    energy += .5 * state_error**2
            if coupling is not None:
                feedback = feedback + coupling.feedback(branch, errors)
            hidden_updates.append(
                branch.config.inference_rate
                * (-hidden_error + (1.0 - hidden_activity**2) * feedback)
            )
            if collect:
                energy += mean_energy(hidden_error, sensory_error)
        if collect:
            trajectory.append(energy)
        # Every direction is computed before any branch or the shared canvas moves.
        canvas += correction
        for (branch, _), update in zip(observations, hidden_updates):
            branch.hidden += update
            branch._require_finite(canvas, branch.hidden)
    final_errors = coupling.hidden_errors(canvas) if coupling is not None else {
        branch: branch.hidden - np.tanh(canvas) @ branch.canvas_weights.T
        for branch, _ in observations
    }
    if collect:
        trajectory.append(sum(
            mean_energy(final_errors[branch], np.zeros_like(branch.visual_error) if observed is None
                        else observed - branch._predict_observation(np.tanh(branch.hidden)))
            for branch, observed in observations
        ) + sum(.5 * branch._stream_state_terms(state)[1]**2 for branch, state in states.items()))
    learn = bool(states) or any(observed is not None for _, observed in observations)
    cross_updates = {}
    if coupling is not None and learn:
        for branch in coupling.branches:
            weights = coupling.weights[branch]
            before = weights.copy() if branch.diagnostics_enabled else None
            if branch.config.learning_enabled and branch.config.learning_rate:
                parent = coupling.parent(branch)
                gradient = (
                    final_errors[branch].reshape(-1, branch.hidden.shape[-1]).T
                    @ np.tanh(parent.hidden).reshape(-1, parent.hidden.shape[-1])
                ) / (canvas.shape[0] * canvas.shape[1])
                branch._update_weights(weights, gradient)
            cross_updates[branch] = float(np.linalg.norm(weights - before)) if before is not None else 0.0
    for branch, observed in observations:
        branch._finish_inference(
            canvas, observed, trajectory, hidden_error=final_errors[branch], learn=learn,
            cross_weights=coupling.weights[branch] if coupling is not None else None,
            cross_parent_means=(
                tuple(float(value) for value in coupling.parent(branch).hidden.mean(axis=(0, 1)))
                if coupling is not None else ()
            ),
            cross_update_norm=cross_updates.get(branch, 0.0),
            stream_state=states.get(branch),
        )
    correction = canvas - first._coerce(prior, name="canvas prior")
    first._require_finite(correction)
    return correction.astype(np.float32)
