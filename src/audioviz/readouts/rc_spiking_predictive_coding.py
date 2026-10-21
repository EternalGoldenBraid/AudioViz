"""Continuous single-timescale RC-spiking predictive coding generative graph."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypedDict

import numpy as np

from audioviz.readouts.predictive_coding import PredictiveCodingConfig
from audioviz.utils.array_backend import to_numpy


class RCSpikingSnapshot(TypedDict):
    model: Literal["rc-spiking"]
    mean: np.ndarray
    prior_mean: np.ndarray
    hidden_potential: np.ndarray
    prior_hidden_potential: np.ndarray
    error_potential: np.ndarray
    last_hidden_spikes: np.ndarray
    last_error_spikes: np.ndarray
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
    forecast_hidden_spikes: int
    condition_hidden_spikes: int
    condition_error_spikes: int
    committed_hidden_spikes: int
    hidden_spike_trajectory: tuple[int, ...]
    error_spike_trajectory: tuple[int, ...]
    spike_threshold: float
    memory_time_constant: float
    membrane_capacitance: float
    membrane_resistance: float
    canvas_weights: np.ndarray
    recurrent_weights: np.ndarray
    sensory_weights: np.ndarray
    hidden_bias: np.ndarray
    sensory_bias: np.ndarray
    inference_steps: int
    inference_rate: float


class RCSpikingPredictiveCodingGraph:
    """Continuous leaky RC-spiking predictive coding on one recurrent hidden population.

    Unlike DiffPC's two-phase dyadic schedule and non-leaky delta accumulators,
    this graph evolves three distinct physical quantities on a single continuous
    clock with RC time constant ``tau_m = memory_time_constant``:

    1. **Membrane potentials** ``hidden_potential`` ($v_h \\in \\mathbb{R}^H$) and
       ``error_potential`` ($v_e \\in \\mathbb{R}^S$), which leaky-integrate incoming
       synaptic/error currents and undergo subtractive reset $\\pm \\theta$ upon
       crossing ``spike_threshold`` ($\\theta > 0$).
    2. **Emitted ternary spikes** $s_h \\in \\{-1, 0, +1\\}^H$ and
       $s_e \\in \\{-1, 0, +1\\}^S$, representing discrete current pulses.
    3. **Decoded RC belief state** $h = \\text{mean}[\\text{hidden\\_slice}] \\in \\mathbb{R}^H$,
       a leaky postsynaptic RC capacitor ($h \\leftarrow \\alpha_{\\text{sub}} h + \\theta s_h$)
       decoded into bias-free sensory predictions $\\hat{y}_s = \\sigma(W_s \\tanh(h))$.

    Biases are disabled ($b_h = 0, b_s = 0$) because constant DC offsets do not
    decay across the $1/R$ membrane leak. Subtractive reset reduces $v_h$ while
    transferring charge $\\theta s_h$ into $h$, so spike emission advances the
    decoded belief rather than erasing it. When external forcing is zero
    ($r_{\\mathcal{A}} = 0$, no clamped evidence) and $|v_h| < \\theta$, no spikes
    fire and $h$ decays over each frame by the exact RC factor
    $\\exp(-\\Delta t / \\tau_m)$, driving $\\hat{y}_v \\to \\sigma(0) = 0.5$ even after
    synaptic weight learning.

    This prototype does not fabricate a Gaussian posterior covariance or expected
    information gain; ``candidate_information_gains`` is explicitly unavailable.
    """

    MAX_VISUAL_SLOTS = 4096
    MAX_AUDIO_NODES = 256
    MAX_HIDDEN_NODES = 64
    DEFAULT_SPIKE_THRESHOLD = 0.02
    DEFAULT_MEMBRANE_CAPACITANCE = 0.02
    has_gaussian_uncertainty = False
    _PARAMETERS = (
        "canvas_weights", "recurrent_weights", "sensory_weights",
    )

    def __init__(
        self,
        *,
        visual_shape=(4, 4),
        audio_nodes=16,
        hidden_nodes=8,
        config: PredictiveCodingConfig | None = None,
        memory_time_constant: float = 1 / 3,
        membrane_capacitance: float = DEFAULT_MEMBRANE_CAPACITANCE,
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
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
        self.membrane_capacitance = self._validate_positive_float(
            membrane_capacitance, "membrane_capacitance",
        )
        self.spike_threshold = self._validate_positive_float(
            spike_threshold, "spike_threshold",
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
            xp.clip(
                getattr(self, name), -self.config.weight_clip, self.config.weight_clip,
                out=getattr(self, name),
            )
        self._project_recurrent()
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
            raise ValueError("RC-spiking predictive coding requires float64")
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
    def membrane_resistance(self) -> float:
        return float(self.memory_time_constant / self.membrane_capacitance)

    def set_memory_time_constant(self, value: float) -> None:
        if self._pending:
            raise RuntimeError("finish the pending forecast before updating memory_time_constant")
        self.memory_time_constant = self._validate_positive_float(value, "memory_time_constant")

    def set_membrane_capacitance(self, value: float) -> None:
        if self._pending:
            raise RuntimeError("finish the pending forecast before updating membrane_capacitance")
        self.membrane_capacitance = self._validate_positive_float(value, "membrane_capacitance")

    def set_spike_threshold(self, value: float) -> None:
        if self._pending:
            raise RuntimeError("finish the pending forecast before updating spike_threshold")
        self.spike_threshold = self._validate_positive_float(value, "spike_threshold")

    def update_config(self, **changes) -> None:
        if self._pending:
            raise RuntimeError("finish the pending forecast before updating config")
        tau = changes.pop("memory_time_constant", None)
        cap = changes.pop("membrane_capacitance", None)
        threshold = changes.pop("spike_threshold", None)
        if changes:
            next_config = replace(self.config, **changes)
            self._validate_config(next_config)
            if next_config.hidden_channels != self.config.hidden_channels:
                raise ValueError("hidden_channels cannot change without rebuilding")
            self.config = next_config
        if tau is not None:
            self.memory_time_constant = self._validate_positive_float(tau, "memory_time_constant")
        if cap is not None:
            self.membrane_capacitance = self._validate_positive_float(cap, "membrane_capacitance")
        if threshold is not None:
            self.spike_threshold = self._validate_positive_float(threshold, "spike_threshold")

    def set_learning(self, enabled=None, rate=None) -> None:
        changes = {}
        if enabled is not None:
            changes["learning_enabled"] = enabled
        if rate is not None:
            changes["learning_rate"] = rate
        self.update_config(**changes)

    def _project_recurrent(self) -> None:
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
            - self.recurrent_weights @ activity
        )
        logits = self.sensory_weights @ activity
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
            -xp.outer(sensory_delta, xp.tanh(hidden)),
        )

    def _emit_ternary_spikes(self, potential):
        xp = self.backend
        threshold = self.spike_threshold
        spikes = xp.sign(potential) * (xp.abs(potential) >= threshold)
        next_potential = potential - threshold * spikes
        count = int(xp.count_nonzero(spikes))
        return spikes, next_potential, count

    def _capture(self):
        xp = self.backend
        (
            self.energy, _, self.canvas_gradient, self.hidden_error, self.sensory_error,
            prediction, _, _,
        ) = self._terms(self.mean[self.hidden_slice])
        observed = self.clamped[:self._sensory_nodes]
        self.mean[:self._sensory_nodes] = xp.where(
            observed, self.evidence[:self._sensory_nodes], prediction,
        )
        self._snapshot_weights = {
            name: getattr(self, name).copy()
            for name in (*self._PARAMETERS, "hidden_bias", "sensory_bias")
        }
        return prediction

    def _substep_factors(self, dt: float) -> tuple[float, float]:
        """Return (alpha, beta) for exact zero-order-hold RC integration over dt / M."""
        steps = self.config.inference_steps
        sub_dt = float(dt) / float(steps)
        alpha = float(np.exp(-sub_dt / self.memory_time_constant))
        zoh_integral = self.memory_time_constant * (1.0 - alpha) / self.membrane_capacitance
        beta = float(zoh_integral * self.config.inference_rate)
        return alpha, beta

    def reset(self) -> None:
        """Clear membrane potentials, decoded state, and spike histories, retaining weights."""
        xp = self.backend
        self._pending = False
        self.hidden_bias.fill(0.0)
        self.sensory_bias.fill(0.0)
        self.canvas = xp.zeros(self.visual_nodes, dtype=xp.float64)
        self.aperture_mask = xp.ones(self.visual_nodes, dtype=bool)
        self.mean = xp.zeros(self.total_nodes, dtype=xp.float64)
        self.hidden_potential = xp.zeros(self.hidden_nodes, dtype=xp.float64)
        self.prior_hidden_potential = xp.zeros(self.hidden_nodes, dtype=xp.float64)
        self.error_potential = xp.zeros(self._sensory_nodes, dtype=xp.float64)
        self.last_hidden_spikes = xp.zeros(self.hidden_nodes, dtype=xp.float64)
        self.last_error_spikes = xp.zeros(self._sensory_nodes, dtype=xp.float64)
        self.clamped = xp.zeros(self.total_nodes, dtype=bool)
        self.evidence = xp.full(self.total_nodes, xp.nan, dtype=xp.float64)
        self.inference_energy = ()
        self.forecast_hidden_spikes = 0
        self.condition_hidden_spikes = 0
        self.condition_error_spikes = 0
        self.committed_hidden_spikes = 0
        self.hidden_spike_trajectory = ()
        self.error_spike_trajectory = ()
        self._forecast_spike_trajectory: tuple[int, ...] = ()
        self._interval_start = (
            self.mean[self.hidden_slice].copy(),
            self.hidden_potential.copy(),
            self.error_potential.copy(),
            self.last_hidden_spikes.copy(),
        )
        prediction = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()

    def _set_predictions(self, prediction) -> None:
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
        """Advance continuous RC leak and tonic prior spiking over dt before new evidence."""
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

        # Save start-of-interval state at t so condition() can replay the exact
        # same physical interval [t, t + dt] when sensory evidence is present.
        self._interval_start = (
            self.mean[self.hidden_slice].copy(),
            self.hidden_potential.copy(),
            self.error_potential.copy(),
            self.last_hidden_spikes.copy(),
        )

        steps = self.config.inference_steps
        alpha, beta = self._substep_factors(dt_val)
        clip = self.config.gradient_clip

        root_activity = xp.where(aperture, xp.tanh(self.canvas), 0.0)
        external_drive = self.canvas_weights @ root_activity

        hidden, v_h, v_e, prev_spikes = (arr.copy() for arr in self._interval_start)
        forecast_spikes_per_step: list[int] = []

        for _ in range(steps):
            hidden = alpha * hidden
            v_h = alpha * v_h
            v_e = alpha * v_e
            if beta > 0.0:
                prior_current = xp.clip(
                    external_drive + self.recurrent_weights @ (self.spike_threshold * prev_spikes),
                    -clip,
                    clip,
                )
                v_h = v_h + beta * prior_current
                s_h, v_h, fired = self._emit_ternary_spikes(v_h)
                hidden = hidden + self.spike_threshold * s_h
                prev_spikes = s_h
                forecast_spikes_per_step.append(fired)
            else:
                prev_spikes = xp.zeros(self.hidden_nodes, dtype=xp.float64)
                forecast_spikes_per_step.append(0)

        self.mean[self.hidden_slice] = hidden
        self.hidden_potential = v_h
        self.prior_hidden_potential = v_h.copy()
        self.error_potential = v_e
        self.last_hidden_spikes = prev_spikes
        self.last_error_spikes = xp.zeros(self._sensory_nodes, dtype=xp.float64)
        self._forecast_spike_trajectory = tuple(forecast_spikes_per_step)
        self.forecast_hidden_spikes = int(sum(forecast_spikes_per_step))

        prediction = self._capture()
        self._set_predictions(prediction)
        self.prior_mean = self.mean.copy()
        self._pending = True
        return self.visual_prediction.copy(), self.audio_prediction.copy()

    predict = forecast

    def condition(self, visual=None, audio=None, *, visual_mask=None):
        """Resolve [t, t + dt] with sensory evidence (or keep forecast when unforced/unobserved)."""
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

        # Evaluate initial energy at the counterfactual no-evidence forecast state at t + dt
        prior_hidden = self.prior_mean[self.hidden_slice].copy()
        initial_energy = self._terms_for(
            prior_hidden, self.canvas, self.aperture_mask, self.clamped, self.evidence,
        )[0]
        energy_trajectory = [initial_energy]
        hidden_spikes_per_step: list[int] = []
        error_spikes_per_step: list[int] = []
        s_e = xp.zeros(self._sensory_nodes, dtype=xp.float64)

        if has_observed and beta > 0.0:
            # Replay the same physical interval [t, t + dt] from the saved start-of-interval
            # state with sensory evidence active, so the RC clock advances by exactly one dt.
            hidden, v_h, v_e_start, prev_spikes = (arr.copy() for arr in self._interval_start)
            v_e = xp.where(sensory_observed, v_e_start, 0.0)
            root_activity = xp.where(self.aperture_mask, xp.tanh(self.canvas), 0.0)
            external_drive = self.canvas_weights @ root_activity

            for _ in range(steps):
                (
                    _, _, _, _, es, _, _, j,
                ) = self._terms_for(
                    hidden, self.canvas, self.aperture_mask, self.clamped, self.evidence,
                )
                hidden = alpha * hidden
                v_e = xp.where(sensory_observed, alpha * v_e + beta * es, 0.0)
                s_e, v_e, err_fired = self._emit_ternary_spikes(v_e)
                hidden_drive = (
                    external_drive
                    + self.recurrent_weights @ (self.spike_threshold * prev_spikes)
                    + j.T @ s_e
                )
                hidden_current = xp.clip(hidden_drive, -clip, clip)
                v_h = alpha * v_h + beta * hidden_current
                s_h, v_h, hid_fired = self._emit_ternary_spikes(v_h)
                hidden = hidden + self.spike_threshold * s_h
                prev_spikes = s_h
                hidden_spikes_per_step.append(hid_fired)
                error_spikes_per_step.append(err_fired)
                step_energy = self._terms_for(
                    hidden, self.canvas, self.aperture_mask, self.clamped, self.evidence,
                )[0]
                energy_trajectory.append(step_energy)

            self.mean[self.hidden_slice] = hidden
            self.hidden_potential = v_h
            self.error_potential = v_e
            self.last_hidden_spikes = prev_spikes
            self.last_error_spikes = s_e
            self.condition_hidden_spikes = int(sum(hidden_spikes_per_step))
            self.condition_error_spikes = int(sum(error_spikes_per_step))
            self.committed_hidden_spikes = self.condition_hidden_spikes
            self.hidden_spike_trajectory = tuple(hidden_spikes_per_step)
            self.error_spike_trajectory = tuple(error_spikes_per_step)
        else:
            # No sensory evidence (or zero input gain): the forecast rollout over [t, t + dt]
            # is already the exact state at t + dt; do not apply a second dt leak.
            self.error_potential = xp.where(sensory_observed, self.error_potential, 0.0)
            self.last_error_spikes = s_e
            self.condition_hidden_spikes = 0
            self.condition_error_spikes = 0
            self.committed_hidden_spikes = self.forecast_hidden_spikes
            self.hidden_spike_trajectory = self._forecast_spike_trajectory
            self.error_spike_trajectory = (0,) * steps

        self.inference_energy = tuple(energy_trajectory)

        self._capture()
        direction = -self.canvas_gradient.copy()
        if self.learning_enabled and self.learning_rate > 0 and has_observed:
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

    def candidate_information_gains(self, canvases, dt, *, aperture_masks=None):
        raise RuntimeError(
            "Gaussian candidate information gain is unavailable for RCSpikingPredictiveCodingGraph",
        )

    def candidate_information_gain(self, canvas, dt, *, aperture_mask=None):
        raise RuntimeError(
            "Gaussian candidate information gain is unavailable for RCSpikingPredictiveCodingGraph",
        )

    def belief_snapshot(self) -> RCSpikingSnapshot:
        """Return independent host arrays, all evaluated before this frame's learning."""
        def host(value):
            return np.asarray(to_numpy(value)).copy()

        snapshot = {
            name: host(getattr(self, name)) for name in (
                "mean", "prior_mean", "hidden_potential", "prior_hidden_potential",
                "error_potential", "last_hidden_spikes", "last_error_spikes",
                "clamped", "evidence", "aperture_mask",
                "visual_prediction", "audio_prediction",
                "canvas", "canvas_gradient", "hidden_error", "sensory_error",
            )
        }
        snapshot.update({name: host(value) for name, value in self._snapshot_weights.items()})
        snapshot.update(
            model="rc-spiking",
            visual_shape=self.visual_shape,
            visual_slice=(self.visual_slice.start, self.visual_slice.stop),
            audio_slice=(self.audio_slice.start, self.audio_slice.stop),
            hidden_slice=(self.hidden_slice.start, self.hidden_slice.stop),
            learning_enabled=self.learning_enabled,
            learning_rate=self.learning_rate,
            updates=self.updates,
            stage=self.stage,
            energy=self.energy,
            inference_energy=self.inference_energy,
            forecast_hidden_spikes=self.forecast_hidden_spikes,
            condition_hidden_spikes=self.condition_hidden_spikes,
            condition_error_spikes=self.condition_error_spikes,
            committed_hidden_spikes=self.committed_hidden_spikes,
            hidden_spike_trajectory=self.hidden_spike_trajectory,
            error_spike_trajectory=self.error_spike_trajectory,
            spike_threshold=self.spike_threshold,
            memory_time_constant=self.memory_time_constant,
            membrane_capacitance=self.membrane_capacitance,
            membrane_resistance=self.membrane_resistance,
            inference_steps=self.config.inference_steps,
            inference_rate=self.config.inference_rate,
        )
        return snapshot
