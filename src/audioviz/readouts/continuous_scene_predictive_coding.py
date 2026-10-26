"""Continuous whole-scene neural predictive coding with fovea-gated evidence."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypedDict

import numpy as np

from audioviz.readouts.config import PredictiveCodingConfig
from audioviz.readouts.continuous_state_covariance import ContinuousStateCovariance
from audioviz.utils.array_backend import to_numpy

__all__ = [
    "ContinuousScenePredictiveCodingGraph",
    "ContinuousSceneSnapshot",
]


class ContinuousSceneSnapshot(TypedDict):
    model: Literal["continuous-scene"]
    mean: np.ndarray
    prior_mean: np.ndarray
    sensory_potential: np.ndarray
    prior_sensory_potential: np.ndarray
    visual_potential: np.ndarray
    prior_visual_potential: np.ndarray
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
    sensory_potential_error: np.ndarray
    sensory_error: np.ndarray
    energy: float
    hidden_energy: float
    sensory_energy: float
    inference_energy: tuple[float, ...]
    memory_time_constant: float
    canvas_weights: np.ndarray
    recurrent_weights: np.ndarray
    sensory_weights: np.ndarray
    hidden_residual_precision: np.ndarray
    coupling_precision: np.ndarray
    observation_precision: np.ndarray
    hidden_bias: np.ndarray
    sensory_bias: np.ndarray
    inference_steps: int
    inference_rate: float


class ContinuousScenePredictiveCodingGraph:
    """Continuous whole-image sensory logit field coupled to one recurrent hidden population.

    Every sensory coordinate $i \\in \\{1, \\dots, S\\}$ is backed by a continuous
    neural potential $z_i(t) \\in \\mathbb{R}$, distinct from its decoded sensory
    prediction ($\\hat{y}_{v,i} = \\sigma(z_{v,i}) \\in (0, 1)$ for visual RGB slots
    and $\\hat{y}_{a,i} = z_{a,i}$ for audio slots). A single shared recurrent
    hidden state $h(t) \\in \\mathbb{R}^H$ couples across the scene through top-down
    generative weights $W_s \\in \\mathbb{R}^{S \\times H}$ and recurrent weights
    $W_h \\in \\mathbb{R}^{H \\times H}$ ($\\text{diag}(W_h) = 0$).

    Each predictive residual has a learned diagonal precision. Its Gaussian
    factor energy is ``L(e, p) = sum(p * e**2 - log(p)) / 2``, omitting only
    the fixed ``log(2*pi)`` constant; observation factors exist only at observed
    coordinates. See :meth:`_gaussian_residual_energy` for the literature source.
    The full energy is ``(||h||**2 + gamma_z*||z||**2)/2 + G*sum(L)``.
    Here ``gamma_z = sqrt(H/max(S,H))`` and ``G = 20*inference_rate``.
    G tempers entire Gaussian factors, including their normalization terms.
    This is a recurrent Gaussian-factor energy, not a claim that its product
    defines a normalized directed generative model.

    States follow the diagonal metric ``M = diag(I_H, gamma_z*I_S)``.
    With ``alpha = exp(-dt/(inference_steps*tau))``, each microstep is
    ``x_next = alpha*x - (1-alpha)*G*interaction_gradient``. Interaction currents
    are clipped; monotone descent is not guaranteed for arbitrary step sizes.
    Zero inference gain gives exact passive leak.

    Connection and log-precision learning use the untempered interaction
    objective at the settled point estimates. The fixed connection metric
    ``diag(c_h, c_h, c_cpl*gamma_z)`` preserves initial learning-rate scaling;
    learned precision still weights connection gradients. This implements the
    Laplace-encoded energy at the means, not posterior-expected likelihood
    learning. No precision hyperprior, temporal covariance propagation or
    survival preference is introduced. Zero residual has no finite precision
    optimum under this objective. Biases remain zero, and no full sensory
    covariance matrix is allocated.
    """

    MAX_VISUAL_SLOTS = 480 * 640
    MAX_AUDIO_NODES = 256
    MAX_HIDDEN_NODES = 64
    MAX_GAZE_CANDIDATES = 9 * 13
    _LEAK_ENERGY_WEIGHT = 1.0
    _HIDDEN_RESIDUAL_WEIGHT = 0.25
    _COUPLING_WEIGHT = 0.25
    _OBSERVATION_GAIN = 12.0
    _INTEGRATION_GAIN = 20.0
    has_gaussian_uncertainty = False
    _PARAMETERS = (
        "canvas_weights", "recurrent_weights", "sensory_weights",
    )
    _PRECISIONS = (
        "hidden_residual_precision", "coupling_precision", "observation_precision",
    )

    def __init__(
        self,
        *,
        visual_shape=(4, 4),
        audio_nodes=16,
        hidden_nodes=8,
        config: PredictiveCodingConfig | None = None,
        memory_time_constant: float = 1 / 3,
        backend=np,
        seed=0,
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

        self.memory_time_constant = self._validate_positive_float(
            memory_time_constant, "memory_time_constant",
        )
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
        self._gamma_z = float(np.sqrt(self.hidden_nodes / max(self._sensory_nodes, self.hidden_nodes)))
        self._max_sensory_weight_norm = float(0.4 * np.sqrt(self._sensory_nodes))

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
        self.hidden_residual_precision = xp.full(
            self.hidden_nodes, self._HIDDEN_RESIDUAL_WEIGHT, dtype=xp.float64,
        )
        self.coupling_precision = xp.full(
            self._sensory_nodes, self._COUPLING_WEIGHT * self._gamma_z, dtype=xp.float64,
        )
        self.observation_precision = xp.full(
            self._sensory_nodes, self._OBSERVATION_GAIN * self._gamma_z, dtype=xp.float64,
        )
        for name in self._PARAMETERS:
            xp.clip(
                getattr(self, name), -self.config.weight_clip, self.config.weight_clip,
                out=getattr(self, name),
            )
        self._project_weights()
        self.updates = 0
        self.reset()

    @staticmethod
    def _validate_positive_float(value, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
        return float(value)

    @staticmethod
    def _validate_config(config):
        if not isinstance(config, PredictiveCodingConfig):
            raise ValueError("config must be PredictiveCodingConfig")
        if config.dtype != "float64":
            raise ValueError("continuous scene predictive coding requires float64")
        if config.spatial_precision is not None or config.cross_modal_enabled:
            raise ValueError("spatial_precision and separate cross-modal coupling are unsupported")
        if not isinstance(config.learning_enabled, bool):
            raise ValueError("learning_enabled must be a boolean")

    @property
    def learning_enabled(self) -> bool:
        return self.config.learning_enabled

    @property
    def learning_rate(self) -> float:
        return self.config.learning_rate

    @property
    def stage(self) -> Literal["forecast", "posterior"]:
        return "forecast" if self._pending else "posterior"

    @property
    def update_count(self) -> int:
        return self.updates

    @property
    def visual_potential(self):
        return self.sensory_potential[self.visual_slice]

    @property
    def prior_visual_potential(self):
        return self.prior_sensory_potential[self.visual_slice]

    def set_memory_time_constant(self, value: float) -> None:
        if self._pending:
            raise RuntimeError("finish the pending forecast before updating memory_time_constant")
        self.memory_time_constant = self._validate_positive_float(value, "memory_time_constant")

    def update_config(self, **changes) -> None:
        if self._pending:
            raise RuntimeError("finish the pending forecast before updating config")
        tau = changes.pop("memory_time_constant", None)
        if changes:
            next_config = replace(self.config, **changes)
            self._validate_config(next_config)
            if next_config.hidden_channels != self.config.hidden_channels:
                raise ValueError("hidden_channels cannot change without rebuilding")
            self.config = next_config
        if tau is not None:
            self.memory_time_constant = self._validate_positive_float(tau, "memory_time_constant")

    def set_learning(self, enabled=None, rate=None) -> None:
        changes = {}
        if enabled is not None:
            changes["learning_enabled"] = enabled
        if rate is not None:
            changes["learning_rate"] = rate
        self.update_config(**changes)

    def _project_weights(self) -> None:
        xp = self.backend
        xp.fill_diagonal(self.recurrent_weights, 0.0)
        rec_norm = float(xp.linalg.norm(self.recurrent_weights))
        if rec_norm > 0.5:
            self.recurrent_weights *= 0.5 / rec_norm
        sens_norm = float(xp.linalg.norm(self.sensory_weights))
        if sens_norm > self._max_sensory_weight_norm:
            self.sensory_weights *= self._max_sensory_weight_norm / sens_norm

    def _decode_sensory(self, sensory_potential):
        xp = self.backend
        prediction = xp.empty_like(sensory_potential)
        prediction[self.visual_slice] = 0.5 * (
            1.0 + xp.tanh(0.5 * sensory_potential[self.visual_slice])
        )
        if self.audio_nodes:
            prediction[self.audio_slice] = sensory_potential[self.audio_slice]
        return prediction

    def _gaussian_residual_energy(self, error, precision, *, observed=None) -> float:
        """Diagonal Gaussian factor energy, omitting the fixed log(2*pi) constant.

        Buckley, Kim, McGregor & Seth (2017), "The free energy principle for
        action and perception: A mathematical review", Section 5.1, equations
        (25), (29), (32): e.T @ inv(Sigma) @ e / 2 + logdet(Sigma) / 2.
        https://arxiv.org/abs/1705.09156
        Here precision = 1/variance, so the normalization is -sum(log(precision))/2.
        Missing observations contribute neither residual nor normalization.
        This evaluates residuals at the neural-state means; it does not include
        posterior residual variance or the entropy of the neural-state belief.
        """
        xp = self.backend
        terms = precision * error**2 - xp.log(precision)
        if observed is not None:
            terms = xp.where(observed, terms, 0.0)
        return 0.5 * float(xp.sum(terms))

    def _terms_for(self, hidden, sensory_potential, canvas, aperture_mask, clamped, evidence):
        xp = self.backend
        activity = xp.tanh(hidden)
        dtanh = 1.0 - activity**2
        raw_root_activity = xp.tanh(canvas)
        root_activity = xp.where(aperture_mask, raw_root_activity, 0.0)
        hidden_error = (
            hidden - self.canvas_weights @ root_activity
            - self.recurrent_weights @ activity
        )
        sensory_potential_error = sensory_potential - self.sensory_weights @ activity
        prediction = self._decode_sensory(sensory_potential)
        sensory_error = xp.where(
            clamped[:self._sensory_nodes],
            xp.nan_to_num(evidence[:self._sensory_nodes]) - prediction, 0.0,
        )

        sensory_slope = xp.ones_like(prediction)
        sensory_slope[self.visual_slice] = (
            prediction[self.visual_slice] * (1.0 - prediction[self.visual_slice])
        )
        gamma_z = self._gamma_z
        weighted_hidden_error = self.hidden_residual_precision * hidden_error
        weighted_coupling_error = self.coupling_precision * sensory_potential_error
        observation_drive = self.observation_precision * sensory_error * sensory_slope
        gain = self._INTEGRATION_GAIN * float(self.config.inference_rate)

        hidden_leak_energy = 0.5 * self._LEAK_ENERGY_WEIGHT * float(hidden @ hidden)
        sensory_leak_energy = 0.5 * self._LEAK_ENERGY_WEIGHT * gamma_z * float(
            sensory_potential @ sensory_potential
        )
        hidden_residual_energy = gain * self._gaussian_residual_energy(
            hidden_error, self.hidden_residual_precision,
        )
        coupling_energy = gain * self._gaussian_residual_energy(
            sensory_potential_error, self.coupling_precision,
        )
        observation_energy = gain * self._gaussian_residual_energy(
            sensory_error, self.observation_precision, observed=clamped[:self._sensory_nodes],
        )
        hidden_energy = hidden_leak_energy + hidden_residual_energy
        sensory_energy = sensory_leak_energy + coupling_energy + observation_energy
        energy = hidden_energy + sensory_energy

        hidden_grad_interaction = (
            weighted_hidden_error - (self.recurrent_weights.T @ weighted_hidden_error) * dtanh
            - (self.sensory_weights.T @ weighted_coupling_error) * dtanh
        )
        sensory_grad_interaction = (weighted_coupling_error - observation_drive) / gamma_z
        root_gradient = xp.where(
            aperture_mask,
            -(self.canvas_weights.T @ weighted_hidden_error) * (1.0 - raw_root_activity**2),
            0.0,
        )
        return (
            energy, hidden_energy, sensory_energy,
            hidden_grad_interaction, sensory_grad_interaction, root_gradient,
            hidden_error, sensory_potential_error, sensory_error, prediction,
        )

    def _terms(self, hidden, sensory_potential):
        return self._terms_for(
            hidden, sensory_potential, self.canvas, self.aperture_mask,
            self.clamped, self.evidence,
        )

    def _interaction_energy(self, hidden, sensory_potential) -> float:
        """Untempered sum of the normalized Gaussian factors in _gaussian_residual_energy."""
        _, _, _, _, _, _, eh, ez, es, _ = self._terms(hidden, sensory_potential)
        return (
            self._gaussian_residual_energy(eh, self.hidden_residual_precision)
            + self._gaussian_residual_energy(ez, self.coupling_precision)
            + self._gaussian_residual_energy(
                es, self.observation_precision, observed=self.clamped[:self._sensory_nodes],
            )
        )

    def _raw_weight_gradients(self, hidden, sensory_potential):
        """Exact Euclidean partial derivatives (dE_int/dWc, dE_int/dWh, dE_int/dWs) of _interaction_energy."""
        xp = self.backend
        _, _, _, _, _, _, eh, ez, _, _ = self._terms(hidden, sensory_potential)
        activity = xp.tanh(hidden)
        weighted_hidden_error = self.hidden_residual_precision * eh
        recurrent = -xp.outer(weighted_hidden_error, activity)
        xp.fill_diagonal(recurrent, 0.0)
        root_activity = xp.where(self.aperture_mask, xp.tanh(self.canvas), 0.0)
        return (
            -xp.outer(weighted_hidden_error, root_activity),
            recurrent,
            -xp.outer(self.coupling_precision * ez, activity),
        )

    def _weight_gradients(self, hidden, sensory_potential):
        """Natural block-metric weight gradients M_W^{-1} dE_int/dW with M_W = diag(c_h, c_h, c_cpl*gamma_z)."""
        grad_wc, grad_wh, grad_ws = self._raw_weight_gradients(hidden, sensory_potential)
        c_h = self._HIDDEN_RESIDUAL_WEIGHT
        c_cpl_gz = self._COUPLING_WEIGHT * self._gamma_z
        return (
            grad_wc / c_h,
            grad_wh / c_h,
            grad_ws / c_cpl_gz,
        )

    def _precision_gradients(self):
        """Partial derivatives of E_int with respect to log precision at the captured state.

        Differentiating Buckley et al. (2017), equations (25), (32), gives
        dL/dlog(p) = (p*e**2 - 1)/2 for each active Gaussian factor.
        https://arxiv.org/abs/1705.09156
        The -1 is the likelihood-normalization derivative. Residuals are held
        at their captured means; posterior-expected precision learning would
        additionally require the residual variance under the state belief.
        """
        xp = self.backend
        gradients = (
            0.5 * (self.hidden_residual_precision * self.hidden_error**2 - 1.0),
            0.5 * (self.coupling_precision * self.sensory_potential_error**2 - 1.0),
            xp.where(
                self.clamped[:self._sensory_nodes],
                0.5 * (self.observation_precision * self.sensory_error**2 - 1.0), 0.0,
            ),
        )
        return gradients

    def _updated_precisions(self):
        """Propose a clipped log-space descent step without a precision floor or ceiling."""
        xp = self.backend
        proposed = {}
        for name, gradient in zip(self._PRECISIONS, self._precision_gradients()):
            step = self.learning_rate * xp.clip(
                gradient, -self.config.gradient_clip, self.config.gradient_clip,
            )
            precision = getattr(self, name)
            candidate = xp.exp(xp.log(precision) - step)
            if name == "observation_precision":
                # Missing evidence must not introduce even log/exp roundoff changes.
                candidate = xp.where(self.clamped[:self._sensory_nodes], candidate, precision)
            if not bool(xp.all(xp.isfinite(candidate) & (candidate > 0))):
                raise FloatingPointError(f"{name} learning produced non-finite or non-positive precision")
            proposed[name] = candidate
        return proposed

    def _substep_factors(self, dt: float) -> tuple[float, float]:
        """Return (alpha, beta) for exact zero-order-hold leak and interaction descent over dt / M."""
        steps = self.config.inference_steps
        sub_dt = float(dt) / float(steps)
        alpha = float(np.exp(-sub_dt / self.memory_time_constant))
        beta = float((1.0 - alpha) * self._INTEGRATION_GAIN * self.config.inference_rate)
        return alpha, beta

    def _capture(self):
        xp = self.backend
        (
            self.energy, self.hidden_energy, self.sensory_energy,
            _, _, self.canvas_gradient, self.hidden_error,
            self.sensory_potential_error, self.sensory_error, prediction,
        ) = self._terms(self.mean[self.hidden_slice], self.sensory_potential)
        observed = self.clamped[:self._sensory_nodes]
        self.mean[:self._sensory_nodes] = xp.where(
            observed, self.evidence[:self._sensory_nodes], prediction,
        )
        self._snapshot_weights = {
            name: getattr(self, name).copy()
            for name in (*self._PARAMETERS, "hidden_bias", "sensory_bias")
        }
        self._snapshot_precisions = {
            name: getattr(self, name).copy() for name in self._PRECISIONS
        }
        self._snapshot_inference_rate = float(self.config.inference_rate)
        return prediction

    def _set_predictions(self, prediction) -> None:
        self.visual_prediction = prediction[self.visual_slice].reshape(
            *self.visual_shape, 3,
        ).astype(self.backend.float32).copy()
        self.audio_prediction = prediction[self.audio_slice].astype(self.backend.float32).copy()

    def reset(self) -> None:
        """Clear neural states and evidence, retaining learned weights and residual precisions."""
        xp = self.backend
        self._pending = False
        self.hidden_bias.fill(0.0)
        self.sensory_bias.fill(0.0)
        self.canvas = xp.zeros(self.visual_nodes, dtype=xp.float64)
        self.aperture_mask = xp.ones(self.visual_nodes, dtype=bool)
        self.mean = xp.zeros(self.total_nodes, dtype=xp.float64)
        self.sensory_potential = xp.zeros(self._sensory_nodes, dtype=xp.float64)
        self.prior_sensory_potential = xp.zeros(self._sensory_nodes, dtype=xp.float64)
        self.clamped = xp.zeros(self.total_nodes, dtype=bool)
        self.evidence = xp.full(self.total_nodes, xp.nan, dtype=xp.float64)
        self.inference_energy = ()
        self._interval_start = (
            self.mean[self.hidden_slice].copy(),
            self.sensory_potential.copy(),
        )
        prediction = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()

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
        """Advance continuous (h, z) prior dynamics over [t, t + dt] before new evidence."""
        if self._pending:
            raise RuntimeError("condition the pending forecast before forecasting again")
        dt_val = self._validate_positive_float(dt, "dt")
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
        self._last_forecast_dt = dt_val

        self._interval_start = (
            self.mean[self.hidden_slice].copy(),
            self.sensory_potential.copy(),
        )

        steps = self.config.inference_steps
        alpha, beta = self._substep_factors(dt_val)
        clip = self.config.gradient_clip

        hidden, z = (arr.copy() for arr in self._interval_start)
        for _ in range(steps):
            if beta > 0.0:
                _, _, _, grad_h, grad_z, _, _, _, _, _ = self._terms_for(
                    hidden, z, self.canvas, self.aperture_mask, self.clamped, self.evidence,
                )
                hidden = alpha * hidden - beta * xp.clip(grad_h, -clip, clip)
                z = alpha * z - beta * xp.clip(grad_z, -clip, clip)
            else:
                hidden = alpha * hidden
                z = alpha * z

        self.mean[self.hidden_slice] = hidden
        self.sensory_potential = z
        self.prior_sensory_potential = z.copy()
        prediction = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()
        self._pending = True
        return self.visual_prediction.copy(), self.audio_prediction.copy()

    predict = forecast

    def condition(self, visual=None, audio=None, *, visual_mask=None):
        """Resolve [t, t + dt] with fovea-masked evidence (or keep prior forecast when unobserved)."""
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
        sensory_observed = observed[:self._sensory_nodes]
        has_observed = bool(xp.any(sensory_observed))

        steps = self.config.inference_steps
        dt_val = getattr(self, "_last_forecast_dt", 1.0 / 30.0)
        alpha, beta = self._substep_factors(dt_val)
        clip = self.config.gradient_clip

        initial_hidden, initial_z = (
            self._interval_start if has_observed and beta > 0.0
            else (self.prior_mean[self.hidden_slice], self.prior_sensory_potential)
        )
        initial_energy = self._terms_for(
            initial_hidden, initial_z, self.canvas, self.aperture_mask,
            self.clamped, self.evidence,
        )[0]
        energy_trajectory = [initial_energy]

        if has_observed and beta > 0.0:
            hidden, z = (arr.copy() for arr in self._interval_start)
            for _ in range(steps):
                _, _, _, grad_h, grad_z, _, _, _, _, _ = self._terms_for(
                    hidden, z, self.canvas, self.aperture_mask, self.clamped, self.evidence,
                )
                hidden = alpha * hidden - beta * xp.clip(grad_h, -clip, clip)
                z = alpha * z - beta * xp.clip(grad_z, -clip, clip)
                step_energy = self._terms_for(
                    hidden, z, self.canvas, self.aperture_mask, self.clamped, self.evidence,
                )[0]
                energy_trajectory.append(step_energy)
            self.mean[self.hidden_slice] = hidden
            self.sensory_potential = z

        self.inference_energy = tuple(energy_trajectory)
        prediction = self._capture()
        self._set_predictions(prediction)
        direction = -self.canvas_gradient.copy()

        if self.learning_enabled and self.learning_rate > 0 and has_observed:
            # Both parameter groups use the same captured state and pre-update precisions.
            precisions = self._updated_precisions()
            gradients = self._weight_gradients(self.mean[self.hidden_slice], self.sensory_potential)
            for name, gradient in zip(self._PARAMETERS, gradients):
                parameter = getattr(self, name)
                delta = xp.clip(
                    gradient + self.config.weight_decay * parameter,
                    -self.config.gradient_clip, self.config.gradient_clip,
                )
                parameter -= self.learning_rate * delta
                xp.clip(parameter, -self.config.weight_clip, self.config.weight_clip, out=parameter)
            self._project_weights()
            for name, values in precisions.items():
                getattr(self, name)[:] = values
            self.updates += 1
        self._pending = False
        return direction.reshape(*self.visual_shape, 3)

    def state_covariance(self) -> ContinuousStateCovariance:
        """Infer covariance at the captured, pre-learning state, weights and precisions.

        Buckley et al. (2017), Section 4, equation (17), gives the Laplace
        covariance as inverse energy curvature: https://arxiv.org/abs/1705.09156
        This implementation uses positive Gauss-Newton curvature, not the exact
        nonlinear Hessian. Factor log-normalizers have no state derivative
        because residual precisions are independent learned parameters.
        """
        xp = self.backend
        weights = self._snapshot_weights
        precisions = self._snapshot_precisions
        hidden_slope = 1 - xp.tanh(self.mean[self.hidden_slice])**2
        hidden_jacobian = xp.eye(self.hidden_nodes, dtype=xp.float64) - (
            weights["recurrent_weights"] * hidden_slope[None, :]
        )
        sensory_jacobian = weights["sensory_weights"] * hidden_slope[None, :]
        # Neural potentials stay inferred at observed pixels; clamped RGB means are not these states.
        prediction = self._decode_sensory(self.sensory_potential)
        sensory_slope = xp.ones_like(prediction)
        sensory_slope[self.visual_slice] = prediction[self.visual_slice] * (
            1 - prediction[self.visual_slice]
        )
        gain = self._INTEGRATION_GAIN * self._snapshot_inference_rate
        # Observation noise precision contributes curvature through the decoder's squared slope.
        observation_curvature = (
            gain * precisions["observation_precision"]
            * self.clamped[:self._sensory_nodes] * sensory_slope**2
        )
        # J.T Pi J omits residual-times-second-derivative terms of the exact nonlinear Hessian.
        # Positive leak precision anchors free directions, so covariance needs no added jitter.
        return ContinuousStateCovariance(
            hidden_jacobian=hidden_jacobian, sensory_jacobian=sensory_jacobian,
            observation_curvature=observation_curvature,
            hidden_prior_precision=self._LEAK_ENERGY_WEIGHT,
            sensory_prior_precision=self._LEAK_ENERGY_WEIGHT * self._gamma_z,
            hidden_residual_precision=gain * precisions["hidden_residual_precision"],
            coupling_precision=gain * precisions["coupling_precision"],
            backend=xp, stage=self.stage,
        )

    def candidate_information_gains(self, gazes, *, foveal_radius):
        """Score next visual observations at normalized (x,y) gazes, before new evidence.

        Returns a host score array with shape gazes.shape[:-1], in nats.
        Friston et al. (2015), "Active inference and epistemic value",
        https://doi.org/10.1080/17588928.2015.1020053, separates epistemic
        information gain from preferred outcomes. With no outcome preferences
        or action costs the epistemic EFE contribution is the negative of
        these scores. This is a one-observation local Gaussian approximation;
        it does not forecast learning, scene transitions or covariance history.

        Each gaze selects in-bounds camera RGB slots. The likelihood uses the
        forecast sigmoid slope and effective precision G*observation_precision,
        matching the model's inference curvature. Parameters/linearization are
        frozen during imagination; no camera values are read or fabricated.
        """
        from audioviz.readouts.shared_sensory_io import circular_foveal_offsets

        if not self._pending:
            raise RuntimeError("score candidate gaze after forecast and before condition")
        values = np.asarray(gazes, dtype=np.float64)
        if (
            values.ndim < 2 or values.shape[-1] != 2
            or not 1 <= values.size // 2 <= self.MAX_GAZE_CANDIDATES
            or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
        ):
            raise ValueError(
                f"gazes must contain 1..{self.MAX_GAZE_CANDIDATES} normalized (x,y) pairs",
            )
        offsets = circular_foveal_offsets(foveal_radius)
        xp = self.backend
        uncertainty = self.state_covariance()
        prediction = self._decode_sensory(self.sensory_potential)[self.visual_slice]
        added = (
            self._INTEGRATION_GAIN * self._snapshot_inference_rate
            * self._snapshot_precisions["observation_precision"][self.visual_slice]
            * (prediction * (1 - prediction))**2
        )
        scores = np.empty(values.shape[:-1], dtype=np.float64)
        centers = np.rint(
            values.reshape(-1, 2)[:, ::-1] * (np.asarray(self.visual_shape) - 1),
        ).astype(np.intp)
        for index, center in enumerate(centers):
            positions = offsets + center
            positions = positions[np.all((positions >= 0) & (positions < self.visual_shape), axis=1)]
            pixels = positions[:, 0] * self.visual_shape[1] + positions[:, 1]
            rgb = xp.asarray((pixels[:, None] * 3 + np.arange(3)).reshape(-1))
            scores.flat[index] = uncertainty.sensory_information_gain(rgb, added[rgb])
        return scores

    def candidate_information_gain(self, gaze, *, foveal_radius) -> float:
        """Single-gaze wrapper for candidate_information_gains; gaze is normalized (x,y)."""
        values = np.asarray(gaze, dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("gaze must contain one normalized (x,y) pair")
        return float(self.candidate_information_gains(values[None, :], foveal_radius=foveal_radius)[0])

    def belief_snapshot(self) -> ContinuousSceneSnapshot:
        """Return independent host arrays, all evaluated before this frame's learning."""
        def host(value):
            return np.asarray(to_numpy(value)).copy()

        snapshot = {
            name: host(getattr(self, name)) for name in (
                "mean", "prior_mean", "sensory_potential", "prior_sensory_potential",
                "visual_potential", "prior_visual_potential",
                "clamped", "evidence", "aperture_mask",
                "visual_prediction", "audio_prediction",
                "canvas", "canvas_gradient", "hidden_error",
                "sensory_potential_error", "sensory_error",
            )
        }
        snapshot.update({name: host(value) for name, value in self._snapshot_weights.items()})
        snapshot.update({name: host(value) for name, value in self._snapshot_precisions.items()})
        snapshot.update(
            model="continuous-scene",
            visual_shape=self.visual_shape,
            visual_slice=(self.visual_slice.start, self.visual_slice.stop),
            audio_slice=(self.audio_slice.start, self.audio_slice.stop),
            hidden_slice=(self.hidden_slice.start, self.hidden_slice.stop),
            learning_enabled=self.learning_enabled,
            learning_rate=self.learning_rate,
            updates=self.updates,
            stage=self.stage,
            energy=self.energy,
            hidden_energy=self.hidden_energy,
            sensory_energy=self.sensory_energy,
            inference_energy=self.inference_energy,
            memory_time_constant=self.memory_time_constant,
            inference_steps=self.config.inference_steps,
            inference_rate=self.config.inference_rate,
        )
        return snapshot
