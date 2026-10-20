from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypedDict

import numpy as np

from audioviz.readouts.diagnostics import EnergyStatistics, mean_energy
from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty
from audioviz.utils.array_backend import canvas_direction, project_channels, spatial_mean
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS
from audioviz.utils.opengl_compute import GLArray, GLArrayView


class AudioBeliefSnapshot(TypedDict):
    mean: np.ndarray
    variance: np.ndarray
    sensory_mean: np.ndarray
    sensory_covariance: np.ndarray
    precision: float
    stage: Literal["forecast", "posterior"]


def _validate_audio_precision(value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value < 0 or value > np.finfo(np.float32).max:
        raise ValueError("audio precision must be finite, non-negative and representable as float32")
    return float(value)


class GaussianAudioUncertainty:
    """Common-mode temporal Gaussian mean/covariance filter for pooled audio spectrum.

    Audio hidden state is modeled as h_{r,t} = m_{r,t} + xi_t with a single
    common-mode perturbation xi_t ~ N(0, Diag(v_{a,t})) shared across all pixels
    (Cov(h_r, h_s) = Diag(v_a) for all r, s; rank-K spatial covariance U P_a U^T
    where U = 1_N (x) I_K). Under this explicit common-mode hypothesis, the
    frozen-affine pooled spectral decoder g_a(h) = W_v (1/N sum_r tanh(h_r))
    reduces exactly to a K x K system using the spatially averaged slope
    bar{s} = 1/N sum_r (1 - tanh^2(m_r^-)). Posterior covariance C_a is
    projected onto its diagonal in common-mode space. This does not represent
    independent per-pixel audio variances or physical sound localization.
    """

    def __init__(
        self, readout, *, initial_variance=1.0, process_variance=0.01,
        memory_time_constant=1/3, precision=1.0, backend=np, device=None,
    ):
        for name, value in (("initial_variance", initial_variance), ("process_variance", process_variance)):
            if isinstance(value, bool) or not np.isfinite(value) or value < 0 or value > np.finfo(np.float32).max:
                raise ValueError(f"{name} must be finite, non-negative and representable as float32")
        if np.float32(initial_variance) == 0:
            raise ValueError("initial_variance must be positive in float32")
        if (
            isinstance(memory_time_constant, bool)
            or not np.isfinite(memory_time_constant) or memory_time_constant <= 0
            or memory_time_constant > np.finfo(np.float32).max
            or np.float32(memory_time_constant) == 0
        ):
            raise ValueError("memory_time_constant must be positive and representable as float32")
        self.readout = readout
        self.shape = readout.canvas_shape[:2]
        self.channels = readout.config.hidden_channels
        self.backend = backend
        self.device = device
        self.initial_variance = float(initial_variance)
        self.process_variance = float(process_variance)
        self.memory_time_constant = float(memory_time_constant)
        self._precision = _validate_audio_precision(precision)
        self.prediction = None
        self._prior = None
        self.variance = (
            GLArray(device, (self.channels,))
            if device is not None
            else backend.full((self.channels,), initial_variance, dtype=backend.float32)
        )
        shape = (*self.shape, self.channels)
        self._frozen_hidden = (
            GLArray(device, shape) if device is not None else backend.zeros(shape, dtype=readout.hidden.dtype)
        )
        self._correction = GLArray(device, readout.canvas_shape) if device is not None else None
        self._delta = GLArray(device, (self.channels,)) if device is not None else None
        self._initialized = device is None
        self._pending = False
        self.validate_config(readout.config)

    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, value: float) -> None:
        self._precision = _validate_audio_precision(value)

    def set_precision(self, value: float) -> None:
        self.precision = value

    @staticmethod
    def validate_config(config):
        if config.cross_modal_enabled or config.learning_enabled:
            raise ValueError("Gaussian audio uncertainty currently requires uncoupled, fixed-weight inference")

    def _transition(self, dt):
        return GaussianSensoryUncertainty._transition(self, dt)

    def reset(self):
        if self.device is None:
            self.readout.reset()
            self.variance.fill(self.initial_variance)
        else:
            self.readout.hidden.fill(0)
            self.readout.hidden_error.fill(0)
            self.readout.visual_error.fill(0)
            self.readout.model.audio_prediction.fill(0)
            self.readout.model.spectrum = None
            self.readout.diagnostics = None
            self.readout.stream_state_prediction = None
            self._initialized = False
        self._pending = False
        self._prior = self.prediction = None

    def _ensure_initialized(self):
        if not self._initialized:
            self.device.run(
                """layout(std430,binding=0) buffer Variance {float variance[];};
                uniform int count;uniform float initial;
                void main(){int i=int(gl_GlobalInvocationID.x);if(i<count)variance[i]=initial;}""",
                [self.variance], self.variance.size, count=self.variance.size, initial=self.initial_variance,
            )
            self._initialized = True

    def predict(self, prior, *, dt):
        """Advance per-pixel audio mean and common-mode K-variance before new evidence."""
        self.validate_config(self.readout.config)
        if self._pending:
            raise RuntimeError("Condition the previous Gaussian forecast before predicting again")
        retention, blend, factor, increment = self._transition(dt)
        self._ensure_initialized()
        if self.device is None:
            xp = self.backend
            self._prior = self.readout._coerce(prior, name="canvas prior").copy()
            drive = project_channels(xp.tanh(self._prior), self.readout.canvas_weights.T)
            self.readout.hidden[:] = retention * self.readout.hidden + blend * drive
            self.variance[:] = factor * self.variance + increment
            self._frozen_hidden[:] = self.readout.hidden
            self.prediction = self.readout._capture_prediction(include_stream_state=False)
        else:
            model = self.readout.model
            if not isinstance(prior, GLArray) or prior.device is not self.device or prior.shape != model.shape:
                raise ValueError("Shader inference requires a native prior from its own engine")
            model._copy_prior(prior)
            self._prior = prior
            self.prediction = model.audio_prediction
            pixels = int(np.prod(self.shape))
            self.device.run(
                self._native_source() + """
                layout(std430,binding=4) readonly buffer Prior {float prior[];};
                uniform float retention,blend,factor,increment;
                void main(){int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                    for(int k=0;k<H;++k){int i=p*H+k;
                        float drive=0.0;
                        for(int c=0;c<3;++c)drive+=weights[WC+k*3+c]*tanh(prior[p*3+c]);
                        float next_h=retention*hidden[HIDDEN_OFFSET+i]+blend*drive;
                        hidden[HIDDEN_OFFSET+i]=next_h;frozen[i]=next_h;
                        if(p==0)variance[k]=factor*variance[k]+increment;
                    }
                }""", [*self._native_arrays(), self._prior], pixels,
                retention=retention, blend=blend, factor=factor, increment=increment,
            )
            pooled = self.device.reduce(
                "layout(std430,binding=0) readonly buffer Frozen {float frozen[];};",
                f"tanh(frozen[p*{self.channels}+c])",
                [self._frozen_hidden], pixels, self.channels,
            )
            self.device.run(
                self._native_source() + """
                layout(std430,binding=4) readonly buffer Pooled {float pooled[];};
                layout(std430,binding=5) buffer Predictions {float predictions[];};
                void main(){int o=int(gl_GlobalInvocationID.x);if(o>=B)return;
                    float sum=0.0;
                    for(int k=0;k<H;++k)sum+=weights[WD+o*H+k]*(pooled[k]/float(N));
                    predictions[PRED_OFFSET+o]=sum;
                }""", [*self._native_arrays(), pooled, model.predictions], AUDIO_SPECTRAL_BANDS,
            )
            self.device.release(pooled)
            model._validate(model.hidden, self.prediction)
            self.readout.diagnostics = None
            self.readout.stream_state_prediction = None
        self._pending = True
        return self.prediction

    def _pooled_jacobian(self, hidden):
        xp = self.backend
        mean_slope = spatial_mean(1 - xp.tanh(hidden) ** 2)
        return xp.asarray(self.readout.visual_weights) * mean_slope[None, :]

    @staticmethod
    def _solve_common_mode(j_bar, variance, residual, lam, xp=np):
        dtype = j_bar.dtype
        v = xp.maximum(xp.asarray(variance, dtype=dtype), 0)
        sqrt_v = xp.sqrt(v)
        scaled_j = j_bar * sqrt_v[None, :]
        system = xp.eye(j_bar.shape[1], dtype=dtype) + dtype.type(lam) * (scaled_j.T @ scaled_j)
        inv_system = xp.linalg.inv(system)
        rhs = dtype.type(lam) * (scaled_j.T @ xp.asarray(residual, dtype=dtype))
        z = inv_system @ rhs
        delta = sqrt_v * z
        normalized = xp.where(v > 0, z, 0)
        posterior_variance = xp.maximum(v * xp.diag(inv_system), 0).astype(variance.dtype)
        return delta, normalized, posterior_variance

    def condition(self, observation, precision: float | None = None):
        """Apply one common-mode frozen-affine posterior and return deterministic canvas feedback."""
        self.validate_config(self.readout.config)
        if not self._pending:
            raise RuntimeError("Gaussian conditioning requires a pre-evidence forecast")
        lam = self._precision if precision is None else _validate_audio_precision(precision)
        if precision is not None:
            self._precision = lam
        if self.device is not None:
            correction = self._condition_native(observation, lam)
        else:
            xp = self.backend
            observed = observation is not None and lam > 0
            observation = self.readout._coerce_observation(observation) if observation is not None else None
            delta = xp.zeros((self.channels,), dtype=self.readout.hidden.dtype)
            residual = xp.zeros_like(self.prediction, dtype=self.readout.hidden.dtype)
            normalized = xp.zeros_like(delta)
            energy = []
            if observed:
                residual = observation - self.prediction.astype(self.readout.hidden.dtype)
                j_bar = self._pooled_jacobian(self._frozen_hidden)
                delta, normalized, next_variance = self._solve_common_mode(
                    j_bar, self.variance, residual, lam, xp=xp,
                )
                self.variance[:] = next_variance
                if self.readout.diagnostics_enabled:
                    sqrt_lam = xp.sqrt(xp.asarray(lam, dtype=residual.dtype))
                    energy.append(mean_energy(xp.zeros_like(self.readout.hidden), sqrt_lam * residual))
                residual = residual - j_bar @ delta
                if self.readout.diagnostics_enabled:
                    full_norm = xp.broadcast_to(normalized, self.readout.hidden.shape)
                    energy.append(mean_energy(full_norm, sqrt_lam * residual))
            self.readout.hidden[:] = self._frozen_hidden + delta
            feedback = xp.broadcast_to(delta @ self.readout.canvas_weights, self._prior.shape)
            correction = canvas_direction(
                self.readout.config.canvas_rate, xp.tanh(self._prior), feedback,
            )
            self.readout._require_finite(self.readout.hidden, self.variance, correction)
            full_normalized = xp.broadcast_to(normalized, self.readout.hidden.shape).copy()
            self.readout._finish_inference(
                self._prior + correction, observation if observed else None, energy,
                hidden_error=full_normalized, visual_error=residual, learn=False,
            )
            if self.readout.diagnostics is not None and observed and lam != 1.0:
                visual = self.readout.diagnostics.visual
                self.readout.diagnostics = replace(
                    self.readout.diagnostics,
                    visual=EnergyStatistics(
                        visual.mean * lam,
                        visual.variance * (lam * lam),
                        tuple(x * lam for x in visual.channel_means),
                    ),
                )
        self._pending = False
        return correction

    def _condition_native(self, observation, lam: float):
        """Native OpenGL conditioning with an explicit bounded host summary solve.

        The spatial slope average is reduced on the GPU (`GLCompute.reduce`) and
        downloaded as a tiny `(H,)` vector along with `(H,)` prior variance and
        `(16,)` predicted spectrum (`2*H + 16` floats total) to solve the `H x H`
        common-mode system on the host before uploading the `(H,)` shift,
        posterior variance, and `(16,)` residual back to the GPU.
        """
        model = self.readout.model
        pixels = int(np.prod(self.shape))
        observed = observation is not None and lam > 0
        if observation is not None:
            if isinstance(observation, GLArray):
                model._upload_evidence(observation, (AUDIO_SPECTRAL_BANDS,), 3 * pixels)
                host_obs = GLArrayView(model.evidence, (AUDIO_SPECTRAL_BANDS,), 3 * pixels).download()
            else:
                host_obs = np.asarray(observation, dtype=np.float32)
                if host_obs.shape != (AUDIO_SPECTRAL_BANDS,) or not np.all(np.isfinite(host_obs)):
                    raise ValueError(f"Shader evidence must be finite and have shape {(AUDIO_SPECTRAL_BANDS,)}")
                model._upload_evidence(host_obs, (AUDIO_SPECTRAL_BANDS,), 3 * pixels)
        else:
            host_obs = None
        model.spectrum = host_obs.copy() if observed else None
        collect = self.readout.diagnostics_enabled
        trajectory = []
        delta = np.zeros(self.channels, dtype=np.float32)
        residual_after = np.zeros(AUDIO_SPECTRAL_BANDS, dtype=np.float32)
        if observed:
            slope_sum = self.device.reduce(
                "layout(std430,binding=0) readonly buffer Frozen {float frozen[];};",
                f"1.0 - pow(tanh(frozen[p*{self.channels}+c]), 2.0)",
                [self._frozen_hidden], pixels, self.channels,
            )
            mean_slope = slope_sum.download().reshape(-1).astype(np.float64) / pixels
            self.device.release(slope_sum)
            v_prior = self.variance.download().astype(np.float64)
            y_pred = self.prediction.download().astype(np.float64)
            _, vw, _, _, _ = self.readout.weights()
            j_bar = vw.astype(np.float64) * mean_slope[None, :]
            residual_before = host_obs.astype(np.float64) - y_pred
            delta64, norm64, v_post = self._solve_common_mode(
                j_bar, v_prior, residual_before, lam, xp=np,
            )
            delta = delta64.astype(np.float32)
            residual_after = (residual_before - j_bar @ delta64).astype(np.float32)
            self.variance.upload(v_post.astype(np.float32))
            if collect:
                e_before = float(0.5 * lam * np.sum(residual_before ** 2))
                e_after = float(0.5 * np.sum(norm64 ** 2) + 0.5 * lam * np.sum((residual_before - j_bar @ delta64) ** 2))
                trajectory.extend((e_before, e_after))
        else:
            v_prior = self.variance.download()
        self.readout._trajectory = trajectory
        self._delta.upload(delta)
        self.readout.visual_error.upload(residual_after)
        prior_var = GLArray(self.device, (self.channels,), v_prior.astype(np.float32))
        self.device.run(
            self._native_source() + """
            layout(std430,binding=4) readonly buffer Prior {float prior[];};
            layout(std430,binding=5) readonly buffer Delta {float delta[];};
            layout(std430,binding=6) readonly buffer PriorVar {float prior_var[];};
            layout(std430,binding=7) writeonly buffer Correction {float correction[];};
            layout(std430,binding=8) buffer Errors {float errors[];};
            layout(std430,binding=9) buffer Canvas {float canvas[];};
            uniform float canvas_rate;
            void main(){int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                vec3 feedback=vec3(0.0);
                for(int k=0;k<H;++k){
                    int i=p*H+k;
                    float d=delta[k],v=prior_var[k];
                    hidden[HIDDEN_OFFSET+i]=frozen[i]+d;
                    errors[HIDDEN_OFFSET+i]=v>0.0?d/sqrt(v):0.0;
                    for(int c=0;c<3;++c)feedback[c]+=weights[WC+k*3+c]*d;
                }
                for(int c=0;c<3;++c){
                    float value=prior[p*3+c],a=tanh(value);
                    float change=canvas_rate*(1.0-a*a)*feedback[c];
                    correction[p*3+c]=change;
                    canvas[p*3+c]+=change;
                }
            }""",
            [
                *self._native_arrays(), self._prior, self._delta, prior_var,
                self._correction, model.errors, model.canvas,
            ],
            pixels, canvas_rate=float(self.readout.config.canvas_rate),
        )
        self.device.release(prior_var)
        model._validate(model.hidden, self.variance, self._correction)
        model._trajectory = trajectory
        if collect:
            model._diagnostics(None, None, branches=(model.audio,), audio_precision=lam)
        return self._correction

    def expected_information_gain(
        self, precision: float | None = None, *, prior=None, dt: float | None = None,
    ) -> float:
        """One-step common-mode sensory MI 0.5 * logdet(I_B + lambda_a * J_bar * P_a^- * J_bar^T)."""
        self.validate_config(self.readout.config)
        lam = self._precision if precision is None else _validate_audio_precision(precision)
        if self._pending:
            if prior is not None or dt is not None:
                raise RuntimeError("Condition the current forecast before scoring the next observation")
            if self.device is None:
                j_bar = to_numpy_64(self._pooled_jacobian(self._frozen_hidden))
                v_forecast = to_numpy_64(self.variance)
            else:
                pixels = int(np.prod(self.shape))
                slope_sum = self.device.reduce(
                    "layout(std430,binding=0) readonly buffer Frozen {float frozen[];};",
                    f"1.0 - pow(tanh(frozen[p*{self.channels}+c]), 2.0)",
                    [self._frozen_hidden], pixels, self.channels,
                )
                mean_slope = slope_sum.download().reshape(-1).astype(np.float64) / pixels
                self.device.release(slope_sum)
                _, vw, _, _, _ = self.readout.weights()
                j_bar = vw.astype(np.float64) * mean_slope[None, :]
                v_forecast = self.variance.download().astype(np.float64)
        else:
            if prior is None or dt is None:
                raise ValueError("expected_information_gain requires prior and dt when no forecast is pending")
            retention, blend, factor, increment = self._transition(dt)
            self._ensure_initialized()
            if self.device is None:
                xp = self.backend
                canvas = self.readout._coerce(prior, name="canvas prior")
                drive = project_channels(xp.tanh(canvas), self.readout.canvas_weights.T)
                forecast_hidden = retention * self.readout.hidden + blend * drive
                j_bar = to_numpy_64(self._pooled_jacobian(forecast_hidden))
                v_forecast = to_numpy_64(factor * self.variance + increment)
            else:
                model = self.readout.model
                if not isinstance(prior, GLArray) or prior.device is not self.device or prior.shape != model.shape:
                    raise ValueError("Shader inference requires a native prior from its own engine")
                pixels = int(np.prod(self.shape))
                slope_sum = self.device.reduce(
                    self._native_source() + """
                    layout(std430,binding=4) readonly buffer Prior {float prior[];};
                    uniform float retention,blend;
                    float forecast_slope(int p,int k){
                        float drive=0.0;
                        for(int c=0;c<3;++c)drive+=weights[WC+k*3+c]*tanh(prior[p*3+c]);
                        float h=retention*hidden[HIDDEN_OFFSET+p*H+k]+blend*drive;
                        float a=tanh(h);
                        return 1.0-a*a;
                    }
                    """,
                    "forecast_slope(p,c)",
                    [*self._native_arrays(), prior], pixels, self.channels,
                    retention=retention, blend=blend,
                )
                mean_slope = slope_sum.download().reshape(-1).astype(np.float64) / pixels
                self.device.release(slope_sum)
                _, vw, _, _, _ = self.readout.weights()
                j_bar = vw.astype(np.float64) * mean_slope[None, :]
                v_forecast = factor * self.variance.download().astype(np.float64) + increment
        if lam == 0:
            return 0.0
        v_forecast = np.maximum(v_forecast, 0.0)
        scaled_j = j_bar * np.sqrt(v_forecast)[None, :]
        system = np.eye(self.channels, dtype=np.float64) + lam * (scaled_j.T @ scaled_j)
        _, logdet = np.linalg.slogdet(system)
        return float(max(0.0, 0.5 * logdet))

    def belief_snapshot(self) -> AudioBeliefSnapshot:
        """Return bounded host snapshot of current stored common-mode audio belief."""
        self.validate_config(self.readout.config)
        self._ensure_initialized()
        mean = np.zeros(self.channels, dtype=np.float32)
        if self.device is None:
            hidden_source = self._frozen_hidden if self._pending else self.readout.hidden
            variance = to_numpy_64(self.variance).astype(np.float32, copy=True)
            activity = self.backend.tanh(hidden_source)
            mean_activity = to_numpy_64(spatial_mean(activity))
            mean_slope = to_numpy_64(spatial_mean(1.0 - activity ** 2))
            vw = to_numpy_64(self.readout.visual_weights)
        else:
            pixels = int(np.prod(self.shape))
            hidden_source = self._frozen_hidden if self._pending else self.readout.model.hidden
            offset = 0 if self._pending else pixels * self.channels
            act_sum = self.device.reduce(
                "layout(std430,binding=0) readonly buffer Hidden {float hidden[];};uniform int offset;",
                f"tanh(hidden[offset+p*{self.channels}+c])",
                [hidden_source], pixels, self.channels, offset=offset,
            )
            slope_sum = self.device.reduce(
                "layout(std430,binding=0) readonly buffer Hidden {float hidden[];};uniform int offset;",
                f"1.0 - pow(tanh(hidden[offset+p*{self.channels}+c]), 2.0)",
                [hidden_source], pixels, self.channels, offset=offset,
            )
            mean_activity = act_sum.download().reshape(-1).astype(np.float64) / pixels
            mean_slope = slope_sum.download().reshape(-1).astype(np.float64) / pixels
            self.device.release(act_sum, slope_sum)
            variance = self.variance.download().astype(np.float32, copy=True)
            _, vw_f32, _, _, _ = self.readout.weights()
            vw = vw_f32.astype(np.float64, copy=False)
        var64 = np.maximum(variance.astype(np.float64), 0.0)
        sensory_mean = (vw @ mean_activity).astype(np.float32)
        j_bar = vw * mean_slope[None, :]
        sensory_cov = ((j_bar * var64[None, :]) @ j_bar.T).astype(np.float32)
        for arr in (mean, variance, sensory_mean, sensory_cov):
            arr.setflags(write=False)
        return {
            "mean": mean,
            "variance": variance,
            "sensory_mean": sensory_mean,
            "sensory_covariance": sensory_cov,
            "precision": float(self._precision),
            "stage": "forecast" if self._pending else "posterior",
        }

    def _native_arrays(self):
        model = self.readout.model
        return [self.variance, model.hidden, model.weights, self._frozen_hidden]

    def _native_source(self):
        offset = self.readout.model.weight_offsets[1]
        h = self.channels
        pixels = int(np.prod(self.shape))
        return f"""
        const int H={h},N={pixels},B={AUDIO_SPECTRAL_BANDS};
        const int WC={offset},WD={offset+h*3},HIDDEN_OFFSET={pixels*h},PRED_OFFSET={3*pixels};
        layout(std430,binding=0) buffer Variance {{float variance[];}};
        layout(std430,binding=1) buffer Hidden {{float hidden[];}};
        layout(std430,binding=2) readonly buffer Weights {{float weights[];}};
        layout(std430,binding=3) buffer Frozen {{float frozen[];}};
        """


def to_numpy_64(array) -> np.ndarray:
    from audioviz.utils.array_backend import to_numpy

    return np.asarray(to_numpy(array), dtype=np.float64)
