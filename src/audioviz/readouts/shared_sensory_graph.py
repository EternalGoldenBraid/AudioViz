from __future__ import annotations

from typing import Literal, Sequence, TypedDict

import numpy as np

from audioviz.utils.array_backend import to_numpy
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS


class SharedGraphSnapshot(TypedDict):
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    learned_mean: np.ndarray
    learned_covariance: np.ndarray
    learned_precision: np.ndarray
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


def _validate_positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    out = int(value)
    if allow_zero:
        if out < 0:
            raise ValueError(f"{name} must be non-negative")
    elif out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _validate_learning_rate(rate: float) -> float:
    if isinstance(rate, bool) or not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError("learning_rate must be finite and in [0, 1]")
    return float(rate)


class SharedSensoryGraph:
    """Compact fully connected learned Gaussian sensory/latent graph with exact hard evidence clamps.

    Maintains one persistent joint Gaussian belief N(m, P) over:
      - compact visual RGB nodes: shape (*visual_shape, 3), flattened in row-major order
      - global spectral audio nodes: shape (audio_nodes,)
      - free internal hidden nodes: shape (hidden_nodes,)

    A single dense SPD population distribution N(mu, C) (with precision C^-1) is
    initialized with exchangeable seeded off-diagonal correlations across all
    nodes (no hardwired modality-specific hidden blocks). During `forecast(dt)`,
    the joint belief relaxes toward (mu, C + 0.5 * D * tau * I) under an exact
    held-parameter Ornstein-Uhlenbeck transition. During `condition(...)`, only
    actually observed sensory coordinates O are hard-clamped to y_O (inferring
    all remaining unclamped sensory and hidden coordinates U via exact dense
    Gaussian conditioning and zeroing posterior variance/covariance on O);
    unobserved modalities remain unclamped rather than injected as zeros.

    When learning is enabled and at least one sensory node is clamped, `mu` and
    `C` update via regularized masked online Gaussian moment/EM matching using
    the population conditional N(mu, C) | (x_O = y_O) (not the filter posterior
    as pseudo-data). This learns co-occurrence mean and covariance structure
    with a positive diagonal floor; it is not a proof of spatial specialization
    or an exact multi-step temporal EM solver.
    """

    MAX_NODES = 96

    def __init__(
        self,
        *,
        visual_shape: tuple[int, int] = (4, 4),
        audio_nodes: int = AUDIO_SPECTRAL_BANDS,
        hidden_nodes: int = 8,
        initial_variance: float = 1.0,
        process_variance: float = 0.01,
        memory_time_constant: float = 1 / 3,
        learning_rate: float = 0.05,
        learning_enabled: bool = True,
        variance_floor: float = 1e-4,
        correlation_scale: float = 0.15,
        seed: int = 0,
        backend=np,
    ):
        if len(visual_shape) != 2:
            raise ValueError("visual_shape must be a 2-tuple of (rows, cols)")
        rows = _validate_positive_int(visual_shape[0], "visual_shape[0]")
        cols = _validate_positive_int(visual_shape[1], "visual_shape[1]")
        if rows > 4 or cols > 4:
            raise ValueError("visual_shape dimensions must be at most 4x4")
        audio_count = _validate_positive_int(audio_nodes, "audio_nodes", allow_zero=True)
        hidden_count = _validate_positive_int(hidden_nodes, "hidden_nodes")
        visual_count = rows * cols * 3
        total_nodes = visual_count + audio_count + hidden_count
        if total_nodes > self.MAX_NODES:
            raise ValueError(f"total graph nodes ({total_nodes}) must not exceed {self.MAX_NODES}")

        for name, val in (
            ("initial_variance", initial_variance),
            ("process_variance", process_variance),
            ("variance_floor", variance_floor),
            ("correlation_scale", correlation_scale),
        ):
            if isinstance(val, bool) or not np.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        if isinstance(memory_time_constant, bool) or not np.isfinite(memory_time_constant) or memory_time_constant <= 0:
            raise ValueError("memory_time_constant must be finite and positive")
        if not isinstance(learning_enabled, bool):
            raise ValueError("learning_enabled must be a bool")

        self.visual_shape = (rows, cols)
        self.visual_nodes = visual_count
        self.audio_nodes = audio_count
        self.hidden_nodes = hidden_count
        self.total_nodes = total_nodes
        self.visual_slice = slice(0, visual_count)
        self.audio_slice = slice(visual_count, visual_count + audio_count)
        self.hidden_slice = slice(visual_count + audio_count, total_nodes)

        self.initial_variance = float(initial_variance)
        self.process_variance = float(process_variance)
        self.memory_time_constant = float(memory_time_constant)
        self.variance_floor = float(variance_floor)
        self.correlation_scale = float(correlation_scale)
        self.seed = int(seed)
        self.backend = backend

        self.learning_enabled = bool(learning_enabled)
        self.learning_rate = _validate_learning_rate(learning_rate)
        self.updates = 0

        # Initialize exchangeable seeded SPD population covariance C and mean mu on host, then move to backend.
        rng = np.random.default_rng(self.seed)
        raw = rng.standard_normal((total_nodes, total_nodes)) / np.sqrt(total_nodes)
        sym = (raw @ raw.T) / float(total_nodes)
        diag = np.sqrt(np.maximum(np.diag(sym), 1e-12))
        corr = sym / (diag[:, None] * diag[None, :])
        np.fill_diagonal(corr, 0.0)
        eig_min = float(np.linalg.eigvalsh(corr).min()) if total_nodes > 1 else 0.0
        scale = float(self.correlation_scale)
        if eig_min < 0.0 and scale * abs(eig_min) >= 0.8:
            scale = 0.8 / abs(eig_min)
        base_var = max(self.initial_variance, self.variance_floor)
        c_init = base_var * (np.eye(total_nodes, dtype=np.float64) + scale * corr)
        c_init = 0.5 * (c_init + c_init.T)
        # Ensure strict variance_floor lower bound on eigenvalues.
        vals, vecs = np.linalg.eigh(c_init)
        vals = np.maximum(vals, self.variance_floor)
        c_init = (vecs * vals) @ vecs.T
        c_init = 0.5 * (c_init + c_init.T)

        self.learned_mean = backend.zeros((total_nodes,), dtype=backend.float64)
        self.learned_covariance = backend.asarray(c_init, dtype=backend.float64)

        self.mean = backend.zeros((total_nodes,), dtype=backend.float64)
        self.covariance = backend.zeros((total_nodes, total_nodes), dtype=backend.float64)
        self.prior_mean = backend.zeros((total_nodes,), dtype=backend.float64)
        self.prior_covariance = backend.zeros((total_nodes, total_nodes), dtype=backend.float64)
        self.clamped = backend.zeros((total_nodes,), dtype=bool)
        self.evidence = backend.full((total_nodes,), np.nan, dtype=backend.float64)
        self.visual_prediction = None
        self.audio_prediction = None
        self._pending = False
        self.reset()

    @property
    def stage(self) -> Literal["forecast", "posterior"]:
        return "forecast" if self._pending else "posterior"

    @property
    def update_count(self) -> int:
        return self.updates

    @property
    def learned_precision(self):
        xp = self.backend
        eye = xp.eye(self.total_nodes, dtype=xp.float64)
        return xp.linalg.solve(self.learned_covariance, eye)

    def set_learning(self, enabled: bool | None = None, rate: float | None = None) -> None:
        if enabled is not None:
            if not isinstance(enabled, bool):
                raise ValueError("learning enabled must be a bool")
            self.learning_enabled = enabled
        if rate is not None:
            self.learning_rate = _validate_learning_rate(rate)

    def reset(self) -> None:
        """Reset transient posterior/forecast belief and clamp state while preserving learned mu and C."""
        xp = self.backend
        self.mean = self.learned_mean.copy()
        self.covariance = self.learned_covariance.copy()
        self.prior_mean = self.mean.copy()
        self.prior_covariance = self.covariance.copy()
        self.clamped = xp.zeros((self.total_nodes,), dtype=bool)
        self.evidence = xp.full((self.total_nodes,), np.nan, dtype=xp.float64)
        self.visual_prediction = None
        self.audio_prediction = None
        self._pending = False

    def _transition(self, dt: float) -> tuple[float, float, float, float]:
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("SharedSensoryGraph timestep dt must be finite and positive")
        tau = self.memory_time_constant
        rho = float(np.exp(-float(dt) / tau))
        blend = float(-np.expm1(-float(dt) / tau))
        factor = float(np.exp(-2.0 * float(dt) / tau))
        one_minus_factor = float(-np.expm1(-2.0 * float(dt) / tau))
        return rho, blend, factor, one_minus_factor

    def _ForecastArrays(self, dt: float):
        xp = self.backend
        rho, blend, factor, one_minus_factor = self._transition(dt)
        stationary_cov = self.learned_covariance + (
            0.5 * self.process_variance * self.memory_time_constant
        ) * xp.eye(self.total_nodes, dtype=xp.float64)
        m_minus = rho * self.mean + blend * self.learned_mean
        p_minus = factor * self.covariance + one_minus_factor * stationary_cov
        p_minus = 0.5 * (p_minus + p_minus.T)
        return m_minus, p_minus

    def forecast(self, dt: float):
        """Advance persistent belief via exact held-parameter OU toward learned (mu, C + D*tau/2 I)."""
        if self._pending:
            raise RuntimeError("Condition the previous shared-graph forecast before forecasting again")
        xp = self.backend
        m_minus, p_minus = self._ForecastArrays(dt)
        self.mean = m_minus
        self.covariance = p_minus
        self.prior_mean = m_minus.copy()
        self.prior_covariance = p_minus.copy()
        self.clamped = xp.zeros((self.total_nodes,), dtype=bool)
        self.evidence = xp.full((self.total_nodes,), np.nan, dtype=xp.float64)
        self.visual_prediction = self.mean[self.visual_slice].reshape(*self.visual_shape, 3).astype(xp.float32)
        self.audio_prediction = (
            self.mean[self.audio_slice].astype(xp.float32).copy() if self.audio_nodes > 0 else None
        )
        self._pending = True
        return self.visual_prediction, self.audio_prediction

    def predict(self, dt: float):
        """Alias for `forecast(dt)`."""
        return self.forecast(dt)

    def _resolve_observation(
        self,
        visual=None,
        audio=None,
        *,
        visual_mask=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return host 1D boolean clamp mask and host 1D float64 evidence vector (NaN where unclamped)."""
        clamped_host = np.zeros((self.total_nodes,), dtype=bool)
        evidence_host = np.full((self.total_nodes,), np.nan, dtype=np.float64)

        if visual_mask is not None:
            mask_arr = np.asarray(to_numpy(visual_mask), dtype=bool)
            if mask_arr.shape == self.visual_shape:
                mask_arr = np.repeat(mask_arr[:, :, None], 3, axis=2)
            elif mask_arr.shape != (*self.visual_shape, 3):
                raise ValueError(
                    f"visual_mask shape {mask_arr.shape} must match {self.visual_shape} or {(*self.visual_shape, 3)}"
                )
            flat_vis_mask = mask_arr.reshape(-1)
        else:
            flat_vis_mask = np.ones((self.visual_nodes,), dtype=bool)

        if visual is not None:
            vis_arr = np.asarray(to_numpy(visual), dtype=np.float64)
            if vis_arr.shape == (*self.visual_shape, 3):
                vis_flat = vis_arr.reshape(-1)
            elif vis_arr.shape == (self.visual_nodes,):
                vis_flat = vis_arr
            else:
                raise ValueError(
                    f"compact visual observation shape {vis_arr.shape} must match {(*self.visual_shape, 3)}"
                )
            if not np.all(np.isfinite(vis_flat[flat_vis_mask])):
                raise ValueError("clamped visual observations must be finite")
            clamped_host[self.visual_slice] = flat_vis_mask
            vis_ev = np.full((self.visual_nodes,), np.nan, dtype=np.float64)
            vis_ev[flat_vis_mask] = vis_flat[flat_vis_mask]
            evidence_host[self.visual_slice] = vis_ev

        if audio is not None:
            if self.audio_nodes == 0:
                raise ValueError("audio observation supplied when audio_nodes == 0")
            aud_arr = np.asarray(to_numpy(audio), dtype=np.float64)
            if aud_arr.shape != (self.audio_nodes,):
                raise ValueError(f"audio observation shape {aud_arr.shape} must match ({self.audio_nodes},)")
            if not np.all(np.isfinite(aud_arr)):
                raise ValueError("audio observation must be finite")
            clamped_host[self.audio_slice] = True
            evidence_host[self.audio_slice] = aud_arr

        return clamped_host, evidence_host

    @staticmethod
    def _conditional_gaussian(
        xp,
        mean,
        cov,
        obs_idx: np.ndarray,
        unobs_idx: np.ndarray,
        y_obs,
        jitter: float = 1e-10,
    ):
        """Exact Gaussian conditioning of N(mean, cov) on x[obs_idx] = y_obs.

        Observed posterior means equal y_obs and observed rows/cols of the
        posterior covariance are identically zero.
        """
        n = mean.shape[0]
        if obs_idx.size == 0:
            return mean.copy(), cov.copy()
        obs = xp.asarray(obs_idx, dtype=xp.int64)
        y = xp.asarray(y_obs, dtype=xp.float64)
        post_mean = mean.copy()
        post_mean[obs] = y
        post_cov = xp.zeros((n, n), dtype=xp.float64)
        if unobs_idx.size == 0:
            return post_mean, post_cov

        unobs = xp.asarray(unobs_idx, dtype=xp.int64)
        c_oo = cov[obs[:, None], obs[None, :]]
        c_uo = cov[unobs[:, None], obs[None, :]]
        c_uu = cov[unobs[:, None], unobs[None, :]]

        # Symmetrize c_oo and solve cleanly (fall back to tiny diagonal jitter only if nearly singular).
        c_oo = 0.5 * (c_oo + c_oo.T)
        try:
            # Solve c_oo @ K^T = c_uo^T  => K = (solve(c_oo, c_uo^T))^T
            k_t = xp.linalg.solve(c_oo, c_uo.T)
        except np.linalg.LinAlgError:
            c_oo_reg = c_oo + jitter * xp.eye(obs_idx.size, dtype=xp.float64)
            k_t = xp.linalg.solve(c_oo_reg, c_uo.T)
        # If prior covariance on observed nodes was already zero/rank-deficient so solve produced NaN/Inf:
        if not bool(xp.all(xp.isfinite(k_t))):
            c_oo_reg = c_oo + jitter * xp.eye(obs_idx.size, dtype=xp.float64)
            k_t = xp.linalg.solve(c_oo_reg, c_uo.T)

        gain = k_t.T
        innovation_o = y - mean[obs]
        post_mean[unobs] = mean[unobs] + gain @ innovation_o
        cond_uu = c_uu - gain @ c_uo.T
        cond_uu = 0.5 * (cond_uu + cond_uu.T)
        # Clamp tiny negative numerical eigenvalues on unobserved block.
        vals, vecs = xp.linalg.eigh(cond_uu)
        vals = xp.maximum(vals, 0.0)
        cond_uu = (vecs * vals) @ vecs.T
        cond_uu = 0.5 * (cond_uu + cond_uu.T)
        post_cov[unobs[:, None], unobs[None, :]] = cond_uu
        return post_mean, post_cov

    def condition(self, visual=None, audio=None, *, visual_mask=None):
        """Hard-clamp observed sensory subset O to y_O, infer free nodes U, and optionally update (mu, C).

        Returns the full joint innovation vector `m^+ - m^-` of shape `(total_nodes,)`.
        When no sensory coordinates are observed (`O` is empty), the belief coasts
        (`m^+ = m^-`, `P^+ = P^-`, innovation is zero) and learned parameters
        `(mu, C)` are untouched.
        """
        if not self._pending:
            raise RuntimeError("Forecast the shared sensory graph before conditioning on observations")
        xp = self.backend
        clamped_host, evidence_host = self._resolve_observation(
            visual, audio, visual_mask=visual_mask,
        )
        self.clamped = xp.asarray(clamped_host, dtype=bool)
        self.evidence = xp.asarray(evidence_host, dtype=xp.float64)

        obs_idx = np.flatnonzero(clamped_host)
        unobs_idx = np.flatnonzero(~clamped_host)

        if obs_idx.size == 0:
            self._pending = False
            return xp.zeros((self.total_nodes,), dtype=xp.float64)

        y_obs = xp.asarray(evidence_host[obs_idx], dtype=xp.float64)
        post_mean, post_cov = self._conditional_gaussian(
            xp, self.prior_mean, self.prior_covariance, obs_idx, unobs_idx, y_obs,
        )
        innovation = post_mean - self.prior_mean
        self.mean = post_mean
        self.covariance = post_cov

        if self.learning_enabled and self.learning_rate > 0.0:
            # Masked online Gaussian moment/EM update from population N(mu, C) | (x_O = y_O),
            # NOT using the temporal filter posterior as pseudo-data.
            u_state, v_cov = self._conditional_gaussian(
                xp, self.learned_mean, self.learned_covariance, obs_idx, unobs_idx, y_obs,
            )
            alpha = self.learning_rate
            diff = u_state - self.learned_mean
            self.learned_mean = self.learned_mean + alpha * diff
            floor_eye = (alpha * self.variance_floor) * xp.eye(self.total_nodes, dtype=xp.float64)
            c_next = (
                (1.0 - alpha) * self.learned_covariance
                + alpha * v_cov
                + (alpha * (1.0 - alpha)) * xp.outer(diff, diff)
                + floor_eye
            )
            c_next = 0.5 * (c_next + c_next.T)
            vals, vecs = xp.linalg.eigh(c_next)
            vals = xp.maximum(vals, self.variance_floor)
            c_next = (vecs * vals) @ vecs.T
            self.learned_covariance = 0.5 * (c_next + c_next.T)
            self.updates += 1

        self._pending = False
        return innovation

    def _normalize_sensory_indices(self, indices) -> np.ndarray:
        """Normalize a boolean mask or integer index sequence of prospective clamped sensory nodes."""
        sensory_limit = self.visual_nodes + self.audio_nodes
        if indices is None:
            return np.empty((0,), dtype=np.int64)
        arr = np.asarray(to_numpy(indices))
        if arr.dtype == bool:
            if arr.shape == self.visual_shape:
                arr = np.repeat(arr[:, :, None], 3, axis=2).reshape(-1)
            elif arr.shape == (*self.visual_shape, 3):
                arr = arr.reshape(-1)
            elif arr.ndim != 1 or arr.size not in (self.visual_nodes, sensory_limit, self.total_nodes):
                raise ValueError("boolean indices mask does not match visual, sensory, or total node count")
            selected = np.flatnonzero(arr)
        else:
            if arr.size == 0:
                return np.empty((0,), dtype=np.int64)
            if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
                raise ValueError("indices must be a 1D integer sequence or boolean mask")
            selected = np.unique(arr.astype(np.int64))
        if np.any(selected < 0) or np.any(selected >= sensory_limit):
            raise ValueError(
                f"prospective clamped indices must be sensory node indices in [0, {sensory_limit})"
            )
        return selected

    def information_gain(self, indices: Sequence[int] | np.ndarray, *, dt: float | None = None) -> float:
        """Finite mutual information I(X_H; X_S | D_t) in nats between free hidden nodes H and prospective clamped sensory nodes S.

        Hard-clamping a continuous variable has infinite self-information I(X_S; X_S),
        whereas the information that clamping sensory subset S provides about the
        unclamped internal hidden nodes H is finite whenever P_{H|S}^- is non-singular:
            IG(S) = 0.5 * ln(det(P_{HH}^-) / det(P_{HH}^- - P_{HS}^- (P_{SS}^-)^{-1} P_{SH}^-))
        If `dt` is provided while in the `posterior` stage, evaluates on the one-step
        OU forecast `(m^-, P^-)` for `dt` without mutating stored state; if already
        in the `forecast` stage and `dt is None`, uses the current pending `prior_covariance`.
        """
        s_idx = self._normalize_sensory_indices(indices)
        if s_idx.size == 0 or self.hidden_nodes == 0:
            return 0.0

        if dt is not None:
            _, cov = self._ForecastArrays(dt)
        elif self._pending:
            cov = self.prior_covariance
        else:
            raise ValueError("Supply dt when scoring information_gain from a posterior state")

        cov_host = np.asarray(to_numpy(cov), dtype=np.float64)
        h_idx = np.arange(self.hidden_slice.start, self.hidden_slice.stop, dtype=np.int64)

        p_hh = 0.5 * (cov_host[np.ix_(h_idx, h_idx)] + cov_host[np.ix_(h_idx, h_idx)].T)
        p_ss = 0.5 * (cov_host[np.ix_(s_idx, s_idx)] + cov_host[np.ix_(s_idx, s_idx)].T)
        p_hs = cov_host[np.ix_(h_idx, s_idx)]

        # Filter out any sensory coordinates in s_idx whose prior variance is already zero.
        s_Keep = np.diag(p_ss) > 1e-12
        if not np.any(s_Keep):
            return 0.0
        p_ss = p_ss[np.ix_(s_Keep, s_Keep)]
        p_hs = p_hs[:, s_Keep]

        try:
            k_t = np.linalg.solve(p_ss, p_hs.T)
        except np.linalg.LinAlgError:
            k_t = np.linalg.lstsq(p_ss, p_hs.T, rcond=1e-12)[0]
        p_hh_cond = p_hh - p_hs @ k_t
        p_hh_cond = 0.5 * (p_hh_cond + p_hh_cond.T)

        eig_prior = np.maximum(np.linalg.eigvalsh(p_hh), 1e-15)
        eig_post = np.maximum(np.linalg.eigvalsh(p_hh_cond), 1e-15)
        ig = 0.5 * float(np.sum(np.log(eig_prior)) - np.sum(np.log(eig_post)))
        return max(0.0, ig)

    def belief_snapshot(self) -> SharedGraphSnapshot:
        """Return a bounded host snapshot of current joint belief, learned model, and clamp state."""
        cov_host = np.asarray(to_numpy(self.covariance), dtype=np.float64)
        vis_pred = (
            np.asarray(to_numpy(self.visual_prediction), dtype=np.float32)
            if self.visual_prediction is not None
            else np.asarray(
                to_numpy(self.mean[self.visual_slice]).reshape(*self.visual_shape, 3),
                dtype=np.float32,
            )
        )
        aud_pred = (
            np.asarray(to_numpy(self.audio_prediction), dtype=np.float32)
            if self.audio_prediction is not None
            else np.asarray(to_numpy(self.mean[self.audio_slice]), dtype=np.float32)
        )
        return {
            "mean": np.asarray(to_numpy(self.mean), dtype=np.float64),
            "variance": np.maximum(np.diag(cov_host), 0.0).copy(),
            "covariance": cov_host,
            "prior_mean": np.asarray(to_numpy(self.prior_mean), dtype=np.float64),
            "prior_covariance": np.asarray(to_numpy(self.prior_covariance), dtype=np.float64),
            "learned_mean": np.asarray(to_numpy(self.learned_mean), dtype=np.float64),
            "learned_covariance": np.asarray(to_numpy(self.learned_covariance), dtype=np.float64),
            "learned_precision": np.asarray(to_numpy(self.learned_precision), dtype=np.float64),
            "clamped": np.asarray(to_numpy(self.clamped), dtype=bool),
            "evidence": np.asarray(to_numpy(self.evidence), dtype=np.float64),
            "visual_prediction": vis_pred,
            "audio_prediction": aud_pred,
            "visual_shape": self.visual_shape,
            "visual_slice": (self.visual_slice.start, self.visual_slice.stop),
            "audio_slice": (self.audio_slice.start, self.audio_slice.stop),
            "hidden_slice": (self.hidden_slice.start, self.hidden_slice.stop),
            "learning_enabled": self.learning_enabled,
            "learning_rate": self.learning_rate,
            "updates": self.updates,
            "stage": self.stage,
        }
