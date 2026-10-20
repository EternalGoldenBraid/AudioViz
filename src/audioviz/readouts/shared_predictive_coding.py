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
    prior_variance: np.ndarray
    covariance: np.ndarray
    hidden_covariance: np.ndarray
    prior_covariance: np.ndarray
    prior_hidden_covariance: np.ndarray
    clamped: np.ndarray
    evidence: np.ndarray
    aperture_mask: np.ndarray
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
    Covariance is a hidden-only local linear approximation conditional on fixed
    root and weights, with per-node marginal variances and optional small
    sensory submatrix queries rather than a dense all-sensory covariance matrix.

    Snapshots describe the pre-learning model that generated the returned root
    direction; public parameter arrays hold the weights for the next frame.
    Information gain retains the forecast linearization throughout the frame,
    scoring counterfactual fresh evidence against its hidden-only precision.
    Before a new forecast it therefore scores the previous forecast's root.
    A supplied config's hidden_channels must match hidden_nodes; without a
    config, hidden_nodes also initializes that structural config field.
    """

    MAX_VISUAL_SLOTS = 4096
    MAX_AUDIO_NODES = 256
    MAX_HIDDEN_NODES = 64
    MAX_CANDIDATE_BATCH = 4096
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
        self.visual_pixels = int(np.prod(self.visual_shape))
        if self.visual_pixels > self.MAX_VISUAL_SLOTS:
            raise ValueError(f"visual_shape slots must not exceed {self.MAX_VISUAL_SLOTS}")
        self.visual_nodes = self.visual_pixels * 3
        self.audio_nodes = count(audio_nodes, "audio_nodes", 0)
        if self.audio_nodes > self.MAX_AUDIO_NODES:
            raise ValueError(f"audio_nodes must not exceed {self.MAX_AUDIO_NODES}")
        self.hidden_nodes = count(hidden_nodes, "hidden_nodes")
        if self.hidden_nodes > self.MAX_HIDDEN_NODES:
            raise ValueError(f"hidden_nodes must not exceed {self.MAX_HIDDEN_NODES}")
        self._sensory_nodes = self.visual_nodes + self.audio_nodes
        self.total_nodes = self._sensory_nodes + self.hidden_nodes
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

    def _terms_for(self, hidden, canvas, aperture_mask, clamped, evidence):
        xp = self.backend
        activity = xp.tanh(hidden)
        raw_root_activity = xp.tanh(canvas)
        root_activity = xp.where(aperture_mask, raw_root_activity, 0.0)
        hidden_error = (
            hidden - self.canvas_weights @ root_activity
            - self.recurrent_weights @ activity - self.hidden_bias
        )
        logits = self.sensory_weights @ activity + self.sensory_bias
        prediction = 0.5 * (1 + xp.tanh(0.5 * logits))
        sensory_error = xp.where(
            clamped[:self._sensory_nodes],
            xp.nan_to_num(evidence[:self._sensory_nodes]) - prediction, 0.0,
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
        root_gradient = xp.where(
            aperture_mask,
            -(self.canvas_weights.T @ hidden_error) * (1 - raw_root_activity**2),
            0.0,
        )
        return (
            energy, hidden_gradient, root_gradient, hidden_error, sensory_error,
            prediction, hidden_jacobian, decoder_jacobian,
        )

    def _terms(self, hidden):
        """Energy and its exact hidden/root/parameter derivatives at fixed evidence."""
        return self._terms_for(
            hidden, self.canvas, self.aperture_mask, self.clamped, self.evidence,
        )

    def _weight_gradients(self, hidden):
        xp = self.backend
        _, _, _, eh, es, prediction, _, _ = self._terms(hidden)
        recurrent = -xp.outer(eh, xp.tanh(hidden))
        xp.fill_diagonal(recurrent, 0.0)
        sensory_delta = es * prediction * (1 - prediction)
        root_activity = xp.where(self.aperture_mask, xp.tanh(self.canvas), 0.0)
        return (
            -xp.outer(eh, root_activity), recurrent,
            -xp.outer(sensory_delta, xp.tanh(hidden)), -eh, -sensory_delta,
        )

    def _relax_unclamped(self, hidden, canvas, aperture_mask):
        xp = self.backend
        hidden = hidden.copy()
        root_activity = xp.where(aperture_mask, xp.tanh(canvas), 0.0)
        target = self.canvas_weights @ root_activity + self.hidden_bias
        eye = xp.eye(self.hidden_nodes, dtype=xp.float64)
        activity = xp.tanh(hidden)
        hidden_error = hidden - target - self.recurrent_weights @ activity
        energy = 0.5 * float(hidden_error @ hidden_error)
        trajectory = [energy]
        for _ in range(self.config.inference_steps):
            if self.config.inference_rate == 0:
                break
            hidden_jacobian = eye - self.recurrent_weights * (1 - activity**2)[None, :]
            gradient = hidden_jacobian.T @ hidden_error
            direction = xp.clip(gradient, -self.config.gradient_clip, self.config.gradient_clip)
            slope = float(gradient @ direction)
            if slope <= 0:
                break
            step = self.config.inference_rate
            for _ in range(40):
                candidate = hidden - step * direction
                cand_activity = xp.tanh(candidate)
                cand_error = candidate - target - self.recurrent_weights @ cand_activity
                candidate_energy = 0.5 * float(cand_error @ cand_error)
                if np.isfinite(candidate_energy) and candidate_energy <= energy - 1e-4 * step * slope:
                    hidden, activity, hidden_error, energy = (
                        candidate, cand_activity, cand_error, candidate_energy,
                    )
                    trajectory.append(energy)
                    break
                step *= 0.5
            else:
                break
        return hidden, tuple(trajectory)

    def _relax(self, hidden, canvas, aperture_mask, clamped, evidence):
        xp = self.backend
        if not bool(xp.any(clamped)):
            return self._relax_unclamped(hidden, canvas, aperture_mask)
        hidden = hidden.copy()
        energy = self._terms_for(hidden, canvas, aperture_mask, clamped, evidence)[0]
        trajectory = [energy]
        for _ in range(self.config.inference_steps):
            if self.config.inference_rate == 0:
                break
            gradient = self._terms_for(hidden, canvas, aperture_mask, clamped, evidence)[1]
            direction = xp.clip(gradient, -self.config.gradient_clip, self.config.gradient_clip)
            slope = float(gradient @ direction)
            if slope <= 0:
                break
            step = self.config.inference_rate
            for _ in range(40):
                candidate = hidden - step * direction
                candidate_energy = self._terms_for(
                    candidate, canvas, aperture_mask, clamped, evidence,
                )[0]
                if np.isfinite(candidate_energy) and candidate_energy <= energy - 1e-4 * step * slope:
                    hidden, energy = candidate, candidate_energy
                    trajectory.append(energy)
                    break
                step *= 0.5
            else:
                break
        return hidden, tuple(trajectory)

    def _infer(self):
        hidden, trajectory = self._relax(
            self.mean[self.hidden_slice], self.canvas, self.aperture_mask,
            self.clamped, self.evidence,
        )
        self.mean[self.hidden_slice] = hidden
        self.inference_energy = trajectory

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
        sensory_variance = xp.sum((free_j @ p) * free_j, axis=1) + (~observed).astype(xp.float64)
        self.covariance = p
        self.hidden_covariance = p
        self.variance = xp.concatenate((sensory_variance, xp.diag(p)))
        self._current_decoder = j.copy()
        self._snapshot_weights = {
            name: getattr(self, name).copy() for name in self._PARAMETERS
        }
        return prediction, a, j

    def reset(self):
        """Clear state and pending evidence, retaining every parameter and update count."""
        xp = self.backend
        self._pending = False
        self.canvas = xp.zeros(self.visual_nodes, dtype=xp.float64)
        self.aperture_mask = xp.ones(self.visual_nodes, dtype=bool)
        self.mean = xp.zeros(self.total_nodes, dtype=xp.float64)
        self._forecast_base_hidden = xp.zeros(self.hidden_nodes, dtype=xp.float64)
        self.clamped = xp.zeros(self.total_nodes, dtype=bool)
        self.evidence = xp.full(self.total_nodes, xp.nan, dtype=xp.float64)
        self.inference_energy = ()
        prediction, a, j = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()
        self.prior_covariance = self.covariance.copy()
        self.prior_hidden_covariance = self.prior_covariance
        self.prior_variance = self.variance.copy()
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
        shapes = (
            ((size,), (*self.visual_shape, 3), (self.visual_pixels, 3))
            if image else ((size,),)
        )
        if arr.shape not in shapes:
            raise ValueError(f"{name} shape must be one of {shapes}")
        return arr.reshape(-1).copy()

    def _visual_mask(self, mask, name, *, allow_flat=False):
        xp = self.backend
        if mask is None:
            return xp.ones(self.visual_nodes, dtype=bool)
        mask_arr = xp.asarray(mask)
        if mask_arr.dtype != xp.dtype(bool):
            raise ValueError(f"{name} must be boolean")
        pixel_shapes = (self.visual_shape, (self.visual_pixels,))
        rgb_shapes = ((*self.visual_shape, 3), (self.visual_pixels, 3))
        if allow_flat:
            rgb_shapes = (*rgb_shapes, (self.visual_nodes,))
        if mask_arr.shape in pixel_shapes:
            return xp.repeat(mask_arr.reshape(-1), 3)
        if mask_arr.shape in rgb_shapes:
            return mask_arr.reshape(-1).copy()
        raise ValueError(f"{name} must match visual pixels or RGB coordinates")

    def forecast(self, canvas, dt, *, aperture_mask=None):
        """Relax hidden-only energy at a fixed, signed root and save the prior."""
        if self._pending:
            raise RuntimeError("condition the pending forecast before forecasting again")
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        aperture = self._visual_mask(aperture_mask, "aperture_mask", allow_flat=True)
        root = self._vector(canvas, "canvas", self.visual_nodes, image=True)
        xp = self.backend
        if not bool(xp.all(xp.isfinite(root[aperture]))):
            raise ValueError("canvas must be finite")
        self.canvas = xp.where(
            aperture, root, xp.nan_to_num(root, nan=0.0, posinf=0.0, neginf=0.0),
        )
        self.aperture_mask = aperture
        self.clamped.fill(False)
        self.evidence.fill(xp.nan)
        self._forecast_base_hidden = self.mean[self.hidden_slice].copy()
        root_activity = xp.where(aperture, xp.tanh(self.canvas), 0.0)
        target = self.canvas_weights @ root_activity + self.hidden_bias
        retention = np.exp(-float(dt) / self.memory_time_constant)
        self.mean[self.hidden_slice] = (
            retention * self._forecast_base_hidden + (1 - retention) * target
        )
        self._infer()
        prediction, a, j = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()
        self.prior_covariance = self.covariance.copy()
        self.prior_hidden_covariance = self.prior_covariance
        self.prior_variance = self.variance.copy()
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
        mask = self._visual_mask(visual_mask, "visual_mask")
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
        self._forecast_base_hidden = self.mean[self.hidden_slice].copy()
        self._pending = False
        return direction.reshape(*self.visual_shape, 3)

    def _sensory_indices(self, indices, *, unique=True) -> np.ndarray:
        if isinstance(indices, slice):
            start, stop, step = indices.indices(self._sensory_nodes)
            return np.arange(start, stop, step, dtype=np.int64)
        arr = np.asarray(to_numpy(indices))
        if arr.dtype == bool:
            if arr.shape in (self.visual_shape, (self.visual_pixels,)):
                arr = np.repeat(arr.reshape(-1), 3)
            elif arr.shape in ((*self.visual_shape, 3), (self.visual_pixels, 3)):
                arr = arr.reshape(-1)
            if arr.ndim != 1 or arr.size not in (
                self.visual_nodes, self._sensory_nodes, self.total_nodes,
            ):
                raise ValueError("boolean indices must match visual, sensory or total nodes")
            selected = np.flatnonzero(arr)
        elif arr.size == 0:
            return np.empty(0, dtype=np.int64)
        elif arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("indices must be integers or a boolean mask")
        else:
            selected = arr.astype(np.int64, copy=False)
            if unique:
                selected = np.unique(selected)
            elif np.unique(selected).size != selected.size:
                raise ValueError("sensory_covariance indices must not contain duplicates")
        if np.any(selected < 0) or np.any(selected >= self._sensory_nodes):
            raise ValueError("indices must select sensory nodes")
        return selected

    def sensory_covariance(
        self, indices, *, stage: Literal["current", "prior"] = "current",
    ) -> np.ndarray:
        """Return the local Gauss-Newton covariance for a small sensory subset (<= 64 nodes)."""
        if stage not in ("current", "prior"):
            raise ValueError("stage must be 'current' or 'prior'")
        selected = self._sensory_indices(indices, unique=False)
        if selected.size > 64:
            raise ValueError("sensory_covariance subset must not exceed 64 nodes")
        if selected.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        xp = self.backend
        idx = xp.asarray(selected, dtype=xp.int64)
        if stage == "prior":
            j_sub = self._prior_decoder[idx]
            p = self.prior_covariance
            diag = xp.ones(selected.size, dtype=xp.float64)
        else:
            observed = self.clamped[:self._sensory_nodes][idx]
            j_sub = xp.where(observed[:, None], 0.0, self._current_decoder[idx])
            p = self.covariance
            diag = (~observed).astype(xp.float64)
        cov = j_sub @ p @ j_sub.T + xp.diag(diag)
        cov = 0.5 * (cov + cov.T)
        return np.asarray(to_numpy(cov), dtype=np.float64).copy()

    def _candidate_batch_inputs(self, canvases, aperture_masks):
        arr = np.asarray(to_numpy(canvases), dtype=np.float64)
        visual_rgb = (*self.visual_shape, 3)
        if arr.ndim >= len(visual_rgb) + 1 and arr.shape[-len(visual_rgb):] == visual_rgb:
            batch_shape = arr.shape[:-len(visual_rgb)]
        elif arr.ndim >= 3 and arr.shape[-2:] == (self.visual_pixels, 3):
            batch_shape = arr.shape[:-2]
        elif arr.ndim >= 2 and arr.shape[-1:] == (self.visual_nodes,):
            batch_shape = arr.shape[:-1]
        else:
            raise ValueError(
                "canvases must have a non-empty batch prefix followed by visual shape",
            )
        batch_size = int(np.prod(batch_shape))
        if batch_size < 1 or batch_size > self.MAX_CANDIDATE_BATCH:
            raise ValueError(
                f"candidate batch size must be between 1 and {self.MAX_CANDIDATE_BATCH}",
            )
        roots = arr.reshape(batch_size, self.visual_nodes)
        if aperture_masks is None:
            apertures = np.ones((batch_size, self.visual_nodes), dtype=bool)
        else:
            mask_arr = np.asarray(to_numpy(aperture_masks))
            if mask_arr.dtype != np.dtype(bool):
                raise ValueError("aperture_masks must be boolean")
            pixel_shapes = (
                (*batch_shape, *self.visual_shape),
                (*batch_shape, self.visual_pixels),
            )
            rgb_shapes = (
                (*batch_shape, *self.visual_shape, 3),
                (*batch_shape, self.visual_pixels, 3),
                (*batch_shape, self.visual_nodes),
            )
            if mask_arr.shape in pixel_shapes:
                apertures = np.repeat(
                    mask_arr.reshape(batch_size, self.visual_pixels), 3, axis=1,
                )
            elif mask_arr.shape in rgb_shapes:
                apertures = mask_arr.reshape(batch_size, self.visual_nodes).copy()
            else:
                raise ValueError(
                    "aperture_masks must match batch prefix and visual pixels or RGB coordinates",
                )
        if not bool(np.all(np.isfinite(roots[apertures]))):
            raise ValueError("canvas must be finite")
        return batch_shape, roots, apertures

    def candidate_information_gains(
        self, canvases, dt, *, aperture_masks=None,
    ) -> np.ndarray:
        """Non-mutating prospective hidden information gains for a batch of candidate patches."""
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        batch_shape, roots, apertures = self._candidate_batch_inputs(canvases, aperture_masks)
        batch_size = roots.shape[0]
        has_any = np.any(apertures, axis=1)
        if not bool(np.any(has_any)):
            return np.zeros(batch_shape, dtype=np.float64)
        wc = np.asarray(to_numpy(self.canvas_weights), dtype=np.float64)
        rw = np.asarray(to_numpy(self.recurrent_weights), dtype=np.float64)
        bh = np.asarray(to_numpy(self.hidden_bias), dtype=np.float64)
        ws = np.asarray(to_numpy(self.sensory_weights[:self.visual_nodes]), dtype=np.float64)
        bs = np.asarray(to_numpy(self.sensory_bias[:self.visual_nodes]), dtype=np.float64)
        base_hidden = np.asarray(
            to_numpy(self._forecast_base_hidden if self._pending else self.mean[self.hidden_slice]),
            dtype=np.float64,
        )
        clean_roots = np.where(apertures, roots, 0.0)
        root_activity = np.where(apertures, np.tanh(clean_roots), 0.0)
        targets = root_activity @ wc.T + bh[None, :]
        retention = np.exp(-float(dt) / self.memory_time_constant)
        hidden = retention * base_hidden[None, :] + (1.0 - retention) * targets
        eye = np.eye(self.hidden_nodes, dtype=np.float64)
        activity = np.tanh(hidden)
        hidden_error = hidden - targets - activity @ rw.T
        energy = 0.5 * np.einsum("bi,bi->b", hidden_error, hidden_error)
        active = has_any.copy()
        clip = self.config.gradient_clip
        rate = self.config.inference_rate
        for _ in range(self.config.inference_steps):
            if rate == 0 or not bool(np.any(active)):
                break
            gradient = hidden_error - (1.0 - activity**2) * (hidden_error @ rw)
            direction = np.clip(gradient, -clip, clip)
            slope = np.einsum("bi,bi->b", gradient, direction)
            active &= slope > 0
            if not bool(np.any(active)):
                break
            idx = np.flatnonzero(active)
            step = np.full(idx.size, rate, dtype=np.float64)
            for _ in range(40):
                candidate = hidden[idx] - step[:, None] * direction[idx]
                cand_activity = np.tanh(candidate)
                cand_error = candidate - targets[idx] - cand_activity @ rw.T
                cand_energy = 0.5 * np.einsum("bi,bi->b", cand_error, cand_error)
                accepted = np.isfinite(cand_energy) & (
                    cand_energy <= energy[idx] - 1e-4 * step * slope[idx]
                )
                if bool(np.any(accepted)):
                    ok_idx = idx[accepted]
                    hidden[ok_idx] = candidate[accepted]
                    activity[ok_idx] = cand_activity[accepted]
                    hidden_error[ok_idx] = cand_error[accepted]
                    energy[ok_idx] = cand_energy[accepted]
                if bool(np.all(accepted)):
                    break
                idx = idx[~accepted]
                step = step[~accepted] * 0.5
            else:
                active[idx] = False
        one_minus_a2 = 1.0 - activity**2
        a_jac = eye[None, :, :] - rw[None, :, :] * one_minus_a2[:, None, :]
        baselines = np.einsum("bij,bik->bjk", a_jac, a_jac)
        logits = np.clip(activity @ ws.T + bs[None, :], -60.0, 60.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        weights = np.where(apertures, pred * (1.0 - pred), 0.0)
        ws_outer = (ws[:, :, None] * ws[:, None, :]).reshape(self.visual_nodes, -1)
        jtj = ((weights * weights) @ ws_outer).reshape(
            batch_size, self.hidden_nodes, self.hidden_nodes,
        )
        jtj *= one_minus_a2[:, :, None] * one_minus_a2[:, None, :]
        _, before = np.linalg.slogdet(baselines)
        _, after = np.linalg.slogdet(baselines + jtj)
        scores = np.where(has_any, np.maximum(0.0, 0.5 * (after - before)), 0.0)
        return scores.reshape(batch_shape).astype(np.float64, copy=False)

    def candidate_information_gain(self, canvas, dt, *, aperture_mask=None) -> float:
        """Non-mutating prospective hidden information gain for a candidate root patch."""
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        arr = np.asarray(to_numpy(canvas), dtype=np.float64)
        shapes = ((self.visual_nodes,), (*self.visual_shape, 3), (self.visual_pixels, 3))
        if arr.shape not in shapes:
            raise ValueError(f"canvas shape must be one of {shapes}")
        root = arr.reshape(-1)
        wc = np.asarray(to_numpy(self.canvas_weights), dtype=np.float64)
        rw = np.asarray(to_numpy(self.recurrent_weights), dtype=np.float64)
        bh = np.asarray(to_numpy(self.hidden_bias), dtype=np.float64)
        ws = np.asarray(to_numpy(self.sensory_weights[:self.visual_nodes]), dtype=np.float64)
        bs = np.asarray(to_numpy(self.sensory_bias[:self.visual_nodes]), dtype=np.float64)
        base_hidden = np.asarray(
            to_numpy(self._forecast_base_hidden if self._pending else self.mean[self.hidden_slice]),
            dtype=np.float64,
        )
        if aperture_mask is None:
            if not bool(np.isfinite(root).all()):
                raise ValueError("canvas must be finite")
            target = wc @ np.tanh(root) + bh
            ws_act, bs_act = ws, bs
        else:
            mask_arr = np.asarray(to_numpy(aperture_mask))
            if mask_arr.dtype != np.dtype(bool):
                raise ValueError("aperture_mask must be boolean")
            pixel_shapes = (self.visual_shape, (self.visual_pixels,))
            rgb_shapes = (
                (*self.visual_shape, 3), (self.visual_pixels, 3), (self.visual_nodes,),
            )
            if mask_arr.shape in pixel_shapes:
                pix_mask = mask_arr.reshape(-1)
                if not bool(pix_mask.any()):
                    return 0.0
                if bool(pix_mask.all()):
                    if not bool(np.isfinite(root).all()):
                        raise ValueError("canvas must be finite")
                    target = wc @ np.tanh(root) + bh
                    ws_act, bs_act = ws, bs
                else:
                    aperture = np.repeat(pix_mask, 3)
                    active_root = root[aperture]
                    if not bool(np.isfinite(active_root).all()):
                        raise ValueError("canvas must be finite")
                    target = wc[:, aperture] @ np.tanh(active_root) + bh
                    ws_act, bs_act = ws[aperture], bs[aperture]
            elif mask_arr.shape in rgb_shapes:
                aperture = mask_arr.reshape(-1)
                active_root = root[aperture]
                if active_root.size == 0:
                    return 0.0
                if not bool(np.isfinite(active_root).all()):
                    raise ValueError("canvas must be finite")
                target = wc[:, aperture] @ np.tanh(active_root) + bh
                ws_act, bs_act = ws[aperture], bs[aperture]
            else:
                raise ValueError("aperture_mask must match visual pixels or RGB coordinates")
        retention = np.exp(-float(dt) / self.memory_time_constant)
        hidden = retention * base_hidden + (1.0 - retention) * target
        eye = np.eye(self.hidden_nodes, dtype=np.float64)
        activity = np.tanh(hidden)
        hidden_error = hidden - target - rw @ activity
        energy = 0.5 * float(hidden_error @ hidden_error)
        clip = self.config.gradient_clip
        rate = self.config.inference_rate
        for _ in range(self.config.inference_steps):
            if rate == 0:
                break
            gradient = hidden_error - (1.0 - activity**2) * (hidden_error @ rw)
            direction = np.clip(gradient, -clip, clip)
            slope = float(gradient @ direction)
            if slope <= 0:
                break
            step = rate
            for _ in range(40):
                candidate = hidden - step * direction
                cand_activity = np.tanh(candidate)
                cand_error = candidate - target - rw @ cand_activity
                cand_energy = 0.5 * float(cand_error @ cand_error)
                if np.isfinite(cand_energy) and cand_energy <= energy - 1e-4 * step * slope:
                    hidden, activity, hidden_error, energy = (
                        candidate, cand_activity, cand_error, cand_energy,
                    )
                    break
                step *= 0.5
            else:
                break
        one_minus_a2 = 1.0 - activity**2
        a_jac = eye - rw * one_minus_a2[None, :]
        baseline = a_jac.T @ a_jac
        logits = np.clip(ws_act @ activity + bs_act, -60.0, 60.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        scaled_ws = (pred * (1.0 - pred))[:, None] * ws_act
        jtj = (scaled_ws.T @ scaled_ws) * (one_minus_a2[:, None] * one_minus_a2[None, :])
        _, before = np.linalg.slogdet(baseline)
        _, after = np.linalg.slogdet(baseline + jtj)
        return max(0.0, 0.5 * float(after - before))

    def information_gain(self, indices, *, dt=None):
        """Counterfactual fresh-evidence information in the stored prior Gaussian.

        ``dt`` is accepted for gaze-call compatibility; it cannot forecast a new
        root. Call forecast(new_canvas, dt) or candidate_information_gain(new_canvas, dt)
        to linearize about new canvas samples.
        """
        if dt is not None and (isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0):
            raise ValueError("dt must be finite and positive")
        selected = self._sensory_indices(indices, unique=True)
        if selected.size == 0:
            return 0.0
        xp = self.backend
        selected = xp.asarray(selected, dtype=xp.int64)
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
                "mean", "prior_mean", "variance", "prior_variance",
                "covariance", "prior_covariance", "clamped", "evidence",
                "aperture_mask", "visual_prediction", "audio_prediction",
                "canvas", "canvas_gradient", "hidden_error", "sensory_error",
            )
        }
        snapshot["hidden_covariance"] = snapshot["covariance"].copy()
        snapshot["prior_hidden_covariance"] = snapshot["prior_covariance"].copy()
        snapshot.update({name: host(value) for name, value in self._snapshot_weights.items()})
        snapshot.update(
            model="predictive-coding",
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
