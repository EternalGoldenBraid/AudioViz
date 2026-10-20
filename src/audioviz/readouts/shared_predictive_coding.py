"""Bounded generative predictive coding with fixed-root local GN uncertainty."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypedDict

import numpy as np

from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.utils.array_backend import to_numpy


class SharedPredictiveCodingSnapshot(TypedDict):
    model: Literal["predictive-coding"]
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    clamped: np.ndarray
    evidence: np.ndarray
    visual_prediction: np.ndarray
    audio_prediction: np.ndarray
    visual_shape: tuple[int, int]
    visual_slice: tuple[int, int]
    audio_slice: tuple[int, int]
    hidden_slice: tuple[int, int]
    learning_enabled: bool
    learning_rate: float
    updates: int
    stage: Literal["forecast", "posterior"]
    canvas: np.ndarray
    canvas_gradient: np.ndarray
    hidden_error: np.ndarray
    sensory_error: np.ndarray
    energy: float
    inference_energy: tuple[float, ...]
    canvas_weights: np.ndarray
    recurrent_weights: np.ndarray
    sensory_weights: np.ndarray
    hidden_bias: np.ndarray
    sensory_bias: np.ndarray
    inference_steps: int
    inference_rate: float


class SharedPredictiveCodingGraph:
    """Canvas -> recurrent hidden -> terminal sensory generative graph.

    All residuals have unit, uncalibrated variance. Inference, local learning,
    root feedback and GN curvature use the same squared residual energy.
    The recurrent matrix has zero diagonal and Frobenius norm <= 0.5, so
    I - R diag(tanh'(h)) is nonsingular and hidden GN precision is SPD.
    Covariance is a local linear approximation conditional on fixed root and
    weights, not a learned population or temporal covariance.

    Snapshots describe the pre-learning model that generated the returned root
    direction; public parameter arrays hold the weights for the next frame.
    Information gain retains the forecast linearization throughout the frame,
    scoring counterfactual fresh evidence against its hidden-only precision.
    Before a new forecast it therefore scores the previous forecast's root.
    A supplied config's hidden_channels must match hidden_nodes; without a
    config, hidden_nodes also initializes that structural config field.
    """

    MAX_NODES = 96
    _PARAMETERS = (
        "canvas_weights", "recurrent_weights", "sensory_weights",
        "hidden_bias", "sensory_bias",
    )

    def __init__(
        self, *, visual_shape=(4, 4), audio_nodes=16, hidden_nodes=8,
        config: PredictiveCodingConfig | None = None,
        memory_time_constant=1 / 3, backend=np, seed=0,
    ):
        def count(value, name, minimum=1):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            return int(value)

        if len(visual_shape) != 2:
            raise ValueError("visual_shape must contain rows and columns")
        self.visual_shape = tuple(count(v, "visual_shape") for v in visual_shape)
        if max(self.visual_shape) > 4:
            raise ValueError("visual_shape dimensions must be at most 4")
        self.visual_nodes = int(np.prod(self.visual_shape)) * 3
        self.audio_nodes = count(audio_nodes, "audio_nodes", 0)
        self.hidden_nodes = count(hidden_nodes, "hidden_nodes")
        self._sensory_nodes = self.visual_nodes + self.audio_nodes
        self.total_nodes = self._sensory_nodes + self.hidden_nodes
        if self.total_nodes > self.MAX_NODES:
            raise ValueError("total graph nodes must not exceed 96")
        if (
            isinstance(memory_time_constant, bool)
            or not np.isfinite(memory_time_constant) or memory_time_constant <= 0
        ):
            raise ValueError("memory_time_constant must be finite and positive")
        self.memory_time_constant = float(memory_time_constant)
        self.config = (
            PredictiveCodingConfig(hidden_channels=self.hidden_nodes) if config is None else config
        )
        self._validate_config(self.config)
        if self.config.hidden_channels != self.hidden_nodes:
            raise ValueError("config.hidden_channels must match hidden_nodes")
        self.backend = backend
        self.visual_slice = slice(0, self.visual_nodes)
        self.audio_slice = slice(self.visual_nodes, self._sensory_nodes)
        self.hidden_slice = slice(self._sensory_nodes, self.total_nodes)
        rng = np.random.default_rng(seed)
        xp = backend
        self.canvas_weights = xp.asarray(
            rng.normal(0, 0.3 / np.sqrt(self.visual_nodes), (self.hidden_nodes, self.visual_nodes)),
            dtype=xp.float64,
        )
        self.recurrent_weights = xp.asarray(
            rng.normal(0, 0.1 / np.sqrt(self.hidden_nodes), (self.hidden_nodes, self.hidden_nodes)),
            dtype=xp.float64,
        )
        self.sensory_weights = xp.asarray(
            rng.normal(0, 0.5 / np.sqrt(self.hidden_nodes), (self._sensory_nodes, self.hidden_nodes)),
            dtype=xp.float64,
        )
        self.hidden_bias = xp.zeros(self.hidden_nodes, dtype=xp.float64)
        self.sensory_bias = xp.zeros(self._sensory_nodes, dtype=xp.float64)
        for name in self._PARAMETERS:
            xp.clip(getattr(self, name), -self.config.weight_clip, self.config.weight_clip,
                    out=getattr(self, name))
        self._project_recurrent()
        self.updates = 0
        self.reset()

    @staticmethod
    def _validate_config(config):
        if not isinstance(config, PredictiveCodingConfig):
            raise ValueError("config must be PredictiveCodingConfig")
        if config.dtype != "float64":
            raise ValueError("shared predictive coding requires float64")
        if config.spatial_precision is not None or config.cross_modal_enabled:
            raise ValueError("spatial_precision and separate cross-modal coupling are unsupported")
        if not isinstance(config.learning_enabled, bool):
            raise ValueError("learning_enabled must be a boolean")

    @property
    def learning_enabled(self):
        return self.config.learning_enabled

    @property
    def learning_rate(self):
        return self.config.learning_rate

    @property
    def stage(self) -> Literal["forecast", "posterior"]:
        return "forecast" if self._pending else "posterior"

    @property
    def update_count(self):
        return self.updates

    def update_config(self, **changes):
        next_config = replace(self.config, **changes)
        self._validate_config(next_config)
        if next_config.hidden_channels != self.config.hidden_channels:
            raise ValueError("hidden_channels cannot change without rebuilding")
        if self._pending:
            raise RuntimeError("finish the pending forecast before updating config")
        self.config = next_config

    def set_learning(self, enabled=None, rate=None):
        changes = {}
        if enabled is not None:
            changes["learning_enabled"] = enabled
        if rate is not None:
            changes["learning_rate"] = rate
        self.update_config(**changes)

    def _project_recurrent(self):
        xp = self.backend
        xp.fill_diagonal(self.recurrent_weights, 0.0)
        norm = float(xp.linalg.norm(self.recurrent_weights))
        if norm > 0.5:
            self.recurrent_weights *= 0.5 / norm

    def _terms(self, hidden):
        """Energy and its exact hidden/root/parameter derivatives at fixed evidence."""
        xp = self.backend
        activity = xp.tanh(hidden)
        root_activity = xp.tanh(self.canvas)
        hidden_error = (
            hidden - self.canvas_weights @ root_activity
            - self.recurrent_weights @ activity - self.hidden_bias
        )
        logits = self.sensory_weights @ activity + self.sensory_bias
        prediction = 0.5 * (1 + xp.tanh(0.5 * logits))
        sensory_error = xp.where(
            self.clamped[:self._sensory_nodes],
            xp.nan_to_num(self.evidence[:self._sensory_nodes]) - prediction, 0.0,
        )
        hidden_jacobian = (
            xp.eye(self.hidden_nodes, dtype=xp.float64)
            - self.recurrent_weights * (1 - activity**2)[None, :]
        )
        decoder_jacobian = (
            (prediction * (1 - prediction))[:, None]
            * self.sensory_weights * (1 - activity**2)[None, :]
        )
        energy = 0.5 * float(hidden_error @ hidden_error + sensory_error @ sensory_error)
        hidden_gradient = hidden_jacobian.T @ hidden_error - decoder_jacobian.T @ sensory_error
        root_gradient = -(self.canvas_weights.T @ hidden_error) * (1 - root_activity**2)
        return (
            energy, hidden_gradient, root_gradient, hidden_error, sensory_error,
            prediction, hidden_jacobian, decoder_jacobian,
        )

    def _weight_gradients(self, hidden):
        xp = self.backend
        _, _, _, eh, es, prediction, _, _ = self._terms(hidden)
        recurrent = -xp.outer(eh, xp.tanh(hidden))
        xp.fill_diagonal(recurrent, 0.0)
        sensory_delta = es * prediction * (1 - prediction)
        return (
            -xp.outer(eh, xp.tanh(self.canvas)), recurrent,
            -xp.outer(sensory_delta, xp.tanh(hidden)), -eh, -sensory_delta,
        )

    def _infer(self):
        xp = self.backend
        hidden = self.mean[self.hidden_slice].copy()
        energy = self._terms(hidden)[0]
        trajectory = [energy]
        for _ in range(self.config.inference_steps):
            if self.config.inference_rate == 0:
                break
            gradient = self._terms(hidden)[1]
            direction = xp.clip(gradient, -self.config.gradient_clip, self.config.gradient_clip)
            slope = float(gradient @ direction)
            if slope <= 0:
                break
            step = self.config.inference_rate
            for _ in range(40):
                candidate = hidden - step * direction
                candidate_energy = self._terms(candidate)[0]
                if np.isfinite(candidate_energy) and candidate_energy <= energy - 1e-4 * step * slope:
                    hidden, energy = candidate, candidate_energy
                    trajectory.append(energy)
                    break
                step *= 0.5
            else:
                break
        self.mean[self.hidden_slice] = hidden
        self.inference_energy = tuple(trajectory)

    def _capture(self):
        xp = self.backend
        (
            self.energy, _, self.canvas_gradient, self.hidden_error, self.sensory_error,
            prediction, a, j,
        ) = self._terms(self.mean[self.hidden_slice])
        observed = self.clamped[:self._sensory_nodes]
        self.mean[:self._sensory_nodes] = xp.where(
            observed, self.evidence[:self._sensory_nodes], prediction,
        )
        jo = j[observed]
        precision = a.T @ a + jo.T @ jo
        p = xp.linalg.solve(precision, xp.eye(self.hidden_nodes, dtype=xp.float64))
        p = 0.5 * (p + p.T)
        free_j = xp.where(observed[:, None], 0.0, j)
        cross = free_j @ p
        self.covariance = xp.zeros((self.total_nodes, self.total_nodes), dtype=xp.float64)
        self.covariance[self.hidden_slice, self.hidden_slice] = p
        self.covariance[:self._sensory_nodes, self.hidden_slice] = cross
        self.covariance[self.hidden_slice, :self._sensory_nodes] = cross.T
        self.covariance[:self._sensory_nodes, :self._sensory_nodes] = (
            cross @ free_j.T + xp.diag((~observed).astype(xp.float64))
        )
        self._snapshot_weights = {
            name: getattr(self, name).copy() for name in self._PARAMETERS
        }
        return prediction, a, j

    def reset(self):
        """Clear state and pending evidence, retaining every parameter and update count."""
        xp = self.backend
        self._pending = False
        self.canvas = xp.zeros(self.visual_nodes, dtype=xp.float64)
        self.mean = xp.zeros(self.total_nodes, dtype=xp.float64)
        self.clamped = xp.zeros(self.total_nodes, dtype=bool)
        self.evidence = xp.full(self.total_nodes, xp.nan, dtype=xp.float64)
        self.inference_energy = ()
        prediction, a, j = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()
        self.prior_covariance = self.covariance.copy()
        self._prior_precision = a.T @ a
        self._prior_decoder = j.copy()

    def _set_predictions(self, prediction):
        self.visual_prediction = prediction[self.visual_slice].reshape(
            *self.visual_shape, 3,
        ).astype(self.backend.float32).copy()
        self.audio_prediction = prediction[self.audio_slice].astype(self.backend.float32).copy()

    def _vector(self, value, name, size, image=False):
        xp = self.backend
        arr = xp.asarray(value, dtype=xp.float64)
        shapes = ((size,), (*self.visual_shape, 3)) if image else ((size,),)
        if arr.shape not in shapes:
            raise ValueError(f"{name} shape must be one of {shapes}")
        return arr.reshape(-1).copy()

    def forecast(self, canvas, dt):
        """Relax hidden-only energy at a fixed, signed root and save the prior."""
        if self._pending:
            raise RuntimeError("condition the pending forecast before forecasting again")
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        root = self._vector(canvas, "canvas", self.visual_nodes, image=True)
        xp = self.backend
        if not bool(xp.all(xp.isfinite(root))):
            raise ValueError("canvas must be finite")
        self.canvas = root
        self.clamped.fill(False)
        self.evidence.fill(xp.nan)
        target = self.canvas_weights @ xp.tanh(root) + self.hidden_bias
        retention = np.exp(-float(dt) / self.memory_time_constant)
        self.mean[self.hidden_slice] = (
            retention * self.mean[self.hidden_slice] + (1 - retention) * target
        )
        self._infer()
        prediction, a, j = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()
        self.prior_covariance = self.covariance.copy()
        self._prior_precision = a.T @ a
        self._prior_decoder = j.copy()
        self._pending = True
        return self.visual_prediction.copy(), self.audio_prediction.copy()

    predict = forecast

    def condition(self, visual=None, audio=None, *, visual_mask=None):
        """Clamp fresh evidence and return unscaled -dE/dcanvas before learning."""
        if not self._pending:
            raise RuntimeError("forecast before conditioning")
        xp = self.backend
        observed = xp.zeros(self.total_nodes, dtype=bool)
        evidence = xp.full(self.total_nodes, xp.nan, dtype=xp.float64)
        mask = xp.ones(self.visual_nodes, dtype=bool)
        if visual_mask is not None:
            mask_arr = xp.asarray(visual_mask)
            if mask_arr.dtype != xp.dtype(bool):
                raise ValueError("visual_mask must be boolean")
            if mask_arr.shape == self.visual_shape:
                mask_arr = xp.repeat(mask_arr[:, :, None], 3, axis=2)
            elif mask_arr.shape != (*self.visual_shape, 3):
                raise ValueError("visual_mask must match visual pixels or RGB coordinates")
            mask = mask_arr.reshape(-1)
        if visual is not None:
            values = self._vector(visual, "visual", self.visual_nodes, image=True)
            if not bool(xp.all(xp.isfinite(values[mask]))):
                raise ValueError("observed visual evidence must be finite")
            observed[self.visual_slice] = mask
            evidence[self.visual_slice] = xp.where(mask, values, xp.nan)
        if audio is not None:
            if self.audio_nodes == 0:
                raise ValueError("audio evidence supplied with no audio nodes")
            values = self._vector(audio, "audio", self.audio_nodes)
            if not bool(xp.all(xp.isfinite(values))):
                raise ValueError("audio evidence must be finite")
            observed[self.audio_slice] = True
            evidence[self.audio_slice] = values
        self.clamped, self.evidence = observed, evidence
        self._infer()
        self._capture()
        direction = -self.canvas_gradient.copy()
        if self.learning_enabled and self.learning_rate > 0 and bool(xp.any(observed)):
            gradients = self._weight_gradients(self.mean[self.hidden_slice])
            for name, gradient in zip(self._PARAMETERS, gradients):
                parameter = getattr(self, name)
                delta = xp.clip(
                    gradient + self.config.weight_decay * parameter,
                    -self.config.gradient_clip, self.config.gradient_clip,
                )
                parameter -= self.learning_rate * delta
                xp.clip(parameter, -self.config.weight_clip, self.config.weight_clip, out=parameter)
            self._project_recurrent()
            self.updates += 1
        self._pending = False
        return direction.reshape(*self.visual_shape, 3)

    def information_gain(self, indices, *, dt=None):
        """Counterfactual fresh-evidence information in the stored prior Gaussian.

        ``dt`` is accepted for gaze-call compatibility; it cannot forecast a new
        root. Call forecast(new_canvas, dt) to linearize about new canvas samples.
        """
        if dt is not None and (isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0):
            raise ValueError("dt must be finite and positive")
        arr = np.asarray(to_numpy(indices))
        if arr.dtype == bool:
            if arr.shape == self.visual_shape:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.shape == (*self.visual_shape, 3):
                arr = arr.reshape(-1)
            if arr.ndim != 1 or arr.size not in (
                self.visual_nodes, self._sensory_nodes, self.total_nodes,
            ):
                raise ValueError("boolean indices must match visual, sensory or total nodes")
            selected = np.flatnonzero(arr)
        elif arr.size == 0:
            return 0.0
        elif arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("indices must be integers or a boolean mask")
        else:
            selected = np.unique(arr)
        if np.any(selected < 0) or np.any(selected >= self._sensory_nodes):
            raise ValueError("information gain indices must select sensory nodes")
        xp = self.backend
        selected = xp.asarray(selected, dtype=xp.int64)
        if selected.size == 0:
            return 0.0
        j = self._prior_decoder
        baseline = self._prior_precision
        candidate = j[selected]
        _, before = xp.linalg.slogdet(baseline)
        _, after = xp.linalg.slogdet(baseline + candidate.T @ candidate)
        return max(0.0, 0.5 * float(after - before))

    def belief_snapshot(self) -> SharedPredictiveCodingSnapshot:
        """Return independent host arrays, all evaluated before this frame's learning."""
        def host(value):
            return np.asarray(to_numpy(value)).copy()

        snapshot = {
            name: host(getattr(self, name)) for name in (
                "mean", "covariance", "prior_mean", "prior_covariance", "clamped",
                "evidence", "visual_prediction", "audio_prediction", "canvas",
                "canvas_gradient", "hidden_error", "sensory_error",
            )
        }
        snapshot.update({name: host(value) for name, value in self._snapshot_weights.items()})
        snapshot.update(
            model="predictive-coding", variance=np.diag(snapshot["covariance"]).copy(),
            visual_shape=self.visual_shape,
            visual_slice=(self.visual_slice.start, self.visual_slice.stop),
            audio_slice=(self.audio_slice.start, self.audio_slice.stop),
            hidden_slice=(self.hidden_slice.start, self.hidden_slice.stop),
            learning_enabled=self.learning_enabled, learning_rate=self.learning_rate,
            updates=self.updates, stage=self.stage, energy=self.energy,
            inference_energy=self.inference_energy,
            inference_steps=self.config.inference_steps, inference_rate=self.config.inference_rate,
        )
        return snapshot
