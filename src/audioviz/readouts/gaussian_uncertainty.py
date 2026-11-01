from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypedDict

import numpy as np

from audioviz.readouts.spatial_precision import GAUSSIAN_PRECISION_GLSL, GaussianSpatialPrecision
from audioviz.readouts.diagnostics import mean_energy
from audioviz.utils.array_backend import canvas_direction, project_channels, to_numpy
from audioviz.utils.opengl_compute import GLArray
from audioviz.utils.spatial_preview import preview_indices


class VisualBeliefSnapshot(TypedDict):
    mean: np.ndarray
    variance: np.ndarray
    sensory_mean: np.ndarray
    sensory_covariance: np.ndarray
    pixel: tuple[int, int]
    gaze: tuple[float, float]
    stage: Literal["forecast", "posterior"]


class GaussianSensoryUncertainty:
    """Temporal Gaussian mean/covariance filter for a fixed-weight RGB predictor.

    Hidden covariance is projected onto its diagonal after conditioning. The
    wave canvas is a known input; its uncertainty and parameter uncertainty
    are outside this conditional approximation.
    """

    def __init__(
        self, readout, *, initial_variance=1.0, process_variance=0.01,
        memory_time_constant=1/3, backend=np, device=None,
    ):
        for name, value in (("initial_variance", initial_variance), ("process_variance", process_variance)):
            if not np.isfinite(value) or value < 0 or value > np.finfo(np.float32).max:
                raise ValueError(f"{name} must be finite, non-negative and representable as float32")
        if np.float32(initial_variance) == 0:
            raise ValueError("initial_variance must be positive in float32")
        if (
            not np.isfinite(memory_time_constant) or memory_time_constant <= 0
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
        self.prediction = None
        self._prior = None
        shape = (*self.shape, self.channels)
        self.variance = (
            GLArray(device, shape) if device is not None
            else backend.full(shape, initial_variance, dtype=backend.float32)
        )
        self._frozen_hidden = (
            GLArray(device, shape) if device is not None else backend.zeros(shape, dtype=readout.hidden.dtype)
        )
        self._initialized = device is None
        self._pending = False
        self._indices = self._moments = None
        self._index_shape = None
        self.validate_config(readout.config)

    @staticmethod
    def validate_config(config):
        if config.cross_modal_enabled or config.learning_enabled:
            raise ValueError("Gaussian visual uncertainty currently requires uncoupled, fixed-weight inference")

    def _transition(self, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("uncertainty timestep must be finite and non-negative")
        elapsed = float(dt)/self.memory_time_constant
        retention = float(np.exp(-elapsed))
        blend = float(-np.expm1(-elapsed))
        factor = float(np.exp(-2*elapsed))
        increment = self.process_variance*(.5*self.memory_time_constant)*float(-np.expm1(-2*elapsed))
        if not np.isfinite(increment) or increment > np.finfo(np.float32).max:
            raise ValueError("process variance increment must be representable as float32")
        return retention, blend, factor, float(increment)

    def reset(self):
        self.readout.reset()
        if self.device is None:
            self.variance.fill(self.initial_variance)
        else:
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
        """Advance and freeze one shared mean/covariance prior before new evidence."""
        self.validate_config(self.readout.config)
        if self._pending:
            raise RuntimeError("Condition the previous Gaussian forecast before predicting again")
        retention, blend, factor, increment = self._transition(dt)
        self._ensure_initialized()
        if self.device is None:
            xp = self.backend
            self._prior = self.readout._coerce(prior, name="canvas prior").copy()
            drive = project_channels(xp.tanh(self._prior), self.readout.canvas_weights.T)
            self.readout.hidden[:] = retention*self.readout.hidden + blend*drive
            self.variance[:] = factor * self.variance + increment
            self._frozen_hidden[:] = self.readout.hidden
            self.prediction = self.readout._capture_prediction(include_stream_state=False)
        else:
            model = self.readout.model
            model._copy_prior(prior)
            self._prior = prior
            self.prediction = model.camera_prediction
            self.device.run(
                self._native_source() + """
                layout(std430,binding=4) readonly buffer Prior {float prior[];};
                layout(std430,binding=5) writeonly buffer Prediction {float prediction[];};
                uniform float retention,blend,factor,increment;
                void main(){int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                    for(int k=0;k<H;++k){int i=p*H+k;
                        float drive=0.0;
                        for(int c=0;c<3;++c)drive+=weights[WC+k*3+c]*tanh(prior[p*3+c]);
                        hidden[i]=retention*hidden[i]+blend*drive;
                        variance[i]=factor*variance[i]+increment;frozen[i]=hidden[i];
                    }
                    vec3 y=decoded(p);
                    for(int c=0;c<3;++c)prediction[p*3+c]=y[c];
                }""", [*self._native_arrays(), self._prior, self.prediction], int(np.prod(self.shape)),
                retention=retention, blend=blend, factor=factor, increment=increment,
            )
            model._validate(model.hidden, self.prediction)
            self.readout.diagnostics = None
            self.readout.stream_state_prediction = None
        self._pending = True
        return self.prediction

    def _jacobian(self, hidden):
        xp = self.backend
        weights = xp.asarray(self.readout.visual_weights)
        return weights * (1 - xp.tanh(hidden)**2)[..., None, :]

    @staticmethod
    def _components(j, variance):
        return tuple((j[..., a, :] * variance * j[..., b, :]).sum(axis=-1)
                     for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))

    def condition(self, observation, precision: GaussianSpatialPrecision | None = None):
        """Apply one frozen-affine posterior and return deterministic canvas feedback."""
        self.validate_config(self.readout.config)
        if not self._pending:
            raise RuntimeError("Gaussian conditioning requires a pre-evidence forecast")
        precision = self.readout.config.spatial_precision if precision is None else precision
        precision = GaussianSpatialPrecision(baseline=1, peak=0) if precision is None else precision
        if precision != self.readout.config.spatial_precision:
            self.readout.update_config(spatial_precision=precision)
        if self.device is not None:
            correction = self._condition_native(observation, precision)
        else:
            xp = self.backend
            observed = observation is not None and precision.sample
            observation = self.readout._coerce_observation(observation) if observation is not None else None
            delta = xp.zeros_like(self.readout.hidden)
            residual = xp.zeros_like(self.prediction)
            normalized = xp.zeros_like(delta)
            energy = []
            if observed:
                pi = precision.field(self.shape, backend=xp)
                residual = observation-self.prediction
                j = self._jacobian(self._frozen_hidden)
                a, b, c, d, e, f = self._components(j, self.variance)
                a, b, c, d, e, f = 1+pi*a, 1+pi*b, 1+pi*c, pi*d, pi*e, pi*f
                aa, bb, cc = b*c-f*f, a*c-e*e, a*b-d*d
                dd, ee, ff = e*f-d*c, d*f-b*e, d*e-a*f
                determinant = a*aa+d*dd+e*ee
                x, y, z = j[..., 0, :], j[..., 1, :], j[..., 2, :]
                u, v, w = (residual[..., k] for k in range(3))
                dx = (aa*u+dd*v+ee*w)/determinant
                dy = (dd*u+bb*v+ff*w)/determinant
                dz = (ee*u+ff*v+cc*w)/determinant
                delta = pi[..., None]*self.variance*(
                    x*dx[..., None]+y*dy[..., None]+z*dz[..., None]
                )
                normalized = delta/xp.sqrt(xp.where(self.variance > 0, self.variance, 1))
                quadratic = (
                    aa[..., None]*x*x+bb[..., None]*y*y+cc[..., None]*z*z
                    +2*(dd[..., None]*x*y+ee[..., None]*x*z+ff[..., None]*y*z)
                )/determinant[..., None]
                self.variance[:] = xp.maximum(
                    self.variance-pi[..., None]*self.variance**2*quadratic, 0,
                )
                if self.readout.diagnostics_enabled:
                    energy.append(mean_energy(xp.zeros_like(delta), xp.sqrt(pi[..., None])*residual))
                residual = residual-xp.einsum("...ck,...k->...c", j, delta)
                if self.readout.diagnostics_enabled:
                    energy.append(mean_energy(normalized, xp.sqrt(pi[..., None])*residual))
            self.readout.hidden[:] = self._frozen_hidden+delta
            correction = canvas_direction(
                self.readout.config.canvas_rate, xp.tanh(self._prior),
                project_channels(delta, self.readout.canvas_weights),
            )
            self.readout._require_finite(self.readout.hidden, self.variance, correction)
            self.readout._finish_inference(
                self._prior+correction, observation if observed else None, energy,
                hidden_error=normalized, visual_error=residual, learn=False,
            )
        self._pending = False
        return correction

    def _condition_native(self, observation, precision):
        model = self.readout.model
        observed = observation is not None and precision.sample
        evidence = model._upload_evidence(observation, model.shape, 0) if observation is not None else model.evidence
        model.observation = evidence if observed else None
        x, y, width = precision.pixel_parameters(self.shape)
        source = self._native_source() + GAUSSIAN_PRECISION_GLSL + f"""
            layout(std430,binding=4) readonly buffer Prior {{float prior[];}};
            layout(std430,binding=5) readonly buffer Observation {{float observation[];}};
            layout(std430,binding=6) writeonly buffer Correction {{float correction[];}};
            layout(std430,binding=7) buffer Errors {{float errors[];}};
            layout(std430,binding=8) buffer Canvas {{float canvas[];}};
            uniform int observed;
            uniform float gaze_x,gaze_y,width,baseline,peak,canvas_rate;
            const int VISUAL=2*N*H;
        """
        arrays = [*self._native_arrays(), self._prior, evidence, model.correction, model.errors, model.canvas]
        uniforms = dict(
            observed=int(observed), gaze_x=float(x), gaze_y=float(y), width=float(width),
            baseline=float(precision.baseline), peak=float(precision.peak),
            canvas_rate=float(self.readout.config.canvas_rate),
        )
        collect = self.readout.diagnostics_enabled
        self.readout._trajectory = trajectory = []
        if collect and observed:
            self.device.run(source + """
                void main(){int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                    vec3 expected=decoded(p);
                    for(int c=0;c<3;++c)errors[VISUAL+p*3+c]=observation[p*3+c]-expected[c];
                }""", arrays, int(np.prod(self.shape)), **uniforms,
            )
            trajectory.append(float(model._means(model.camera.visual_error, energy=True, precision=True).sum()))
        self.device.run(source + f"""
            void main(){{int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                vec3 residual=vec3(0.0);
                if(observed==1)for(int c=0;c<3;++c)residual[c]=observation[p*3+c];
                if(observed==1)residual-=decoded(p);
                float pi=observed==1?gaussian_precision(
                    vec2(float(p%{self.shape[1]}),float(p/{self.shape[1]}))-vec2(gaze_x,gaze_y),
                    width,baseline,peak):0.0;
                mat3 cov=mat3(0.0);
                for(int k=0;k<H;++k){{
                    vec3 j=decoder_column(k)*(1.0-pow(tanh(frozen[p*H+k]),2.0));
                    cov+=variance[p*H+k]*outerProduct(j,j);
                }}
                mat3 inv=inverse(mat3(1.0)+pi*cov);
                vec3 u=inv*residual,feedback=vec3(0.0),decoded_delta=vec3(0.0);
                for(int k=0;k<H;++k){{
                    int i=p*H+k;
                    vec3 j=decoder_column(k)*(1.0-pow(tanh(frozen[i]),2.0));
                    float v=variance[i],delta=pi*v*dot(j,u);
                    hidden[i]=frozen[i]+delta;
                    errors[i]=v>0.0?delta/sqrt(v):0.0;
                    variance[i]=max(v-pi*v*v*dot(j,inv*j),0.0);
                    decoded_delta+=j*delta;
                    for(int c=0;c<3;++c)feedback[c]+=weights[WC+k*3+c]*delta;
                }}
                for(int c=0;c<3;++c){{
                    float value=prior[p*3+c],a=tanh(value);
                    float change=canvas_rate*(1.0-a*a)*feedback[c];
                    correction[p*3+c]=change;canvas[p*3+c]+=change;
                    errors[VISUAL+p*3+c]=residual[c]-decoded_delta[c];
                }}
            }}""", arrays, int(np.prod(self.shape)), **uniforms,
        )
        model._validate(model.hidden, self.variance, model.correction)
        if collect:
            if observed:
                trajectory.append(float(
                    model._means(model.camera.hidden_error, energy=True).sum()
                    +model._means(model.camera.visual_error, energy=True, precision=True).sum()
                ))
            model._trajectory = trajectory
            model._diagnostics(None, None, branches=(model.camera,))
        return model.correction

    def _native_arrays(self):
        model = self.readout.model
        return [self.variance, model.hidden, model.weights, self._frozen_hidden]

    def _native_source(self):
        offset = self.readout.model.weight_offsets[0]
        h = self.channels
        return f"""
        const int H={h},N={int(np.prod(self.shape))},WC={offset},WD={offset+h*3};
        layout(std430,binding=0) buffer Variance {{float variance[];}};
        layout(std430,binding=1) buffer Hidden {{float hidden[];}};
        layout(std430,binding=2) readonly buffer Weights {{float weights[];}};
        layout(std430,binding=3) buffer Frozen {{float frozen[];}};
        vec3 decoder_column(int k){{
            return vec3(weights[WD+k],weights[WD+H+k],weights[WD+2*H+k]);
        }}
        vec3 decoded(int p){{
            vec3 result=vec3(0.0);
            for(int k=0;k<H;++k)result+=decoder_column(k)*tanh(frozen[p*H+k]);
            return result;
        }}
        """

    def expected_information_gain(
        self, precision: GaussianSpatialPrecision, *, prior, dt, grid_shape=(9, 13), max_side=64,
    ):
        """One-step sensory-state MI, conditional on a held-fixed point canvas.

        Full local RGB covariance is retained for scoring. Only spatial
        quadrature and the diagonal hidden-state approximation discard terms.
        Spatial samples are weighted by their represented pixel area.
        """
        self.validate_config(self.readout.config)
        if self._pending:
            raise RuntimeError("Condition the current forecast before scoring the next observation")
        if len(grid_shape) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in grid_shape
        ):
            raise ValueError("candidate grid must contain two integers of at least two")
        if prior.shape != (*self.shape, 3):
            raise ValueError("uncertainty planning prior must match the RGB canvas")
        retention, rate, factor, increment = self._transition(dt)
        self._ensure_initialized()
        rows, cols = preview_indices(self.shape, max_side=max_side)
        row_weights = _pixel_area_weights(rows, self.shape[0])
        col_weights = _pixel_area_weights(cols, self.shape[1])
        scale = 0.5
        if self.device is None:
            xp = self.backend
            r, c = xp.asarray(rows), xp.asarray(cols)
            area = xp.asarray(row_weights)[:, None]*xp.asarray(col_weights)[None, :]
            hidden = self.readout.hidden[r[:, None], c[None, :]]
            canvas = prior[r[:, None], c[None, :]]
            forecast = retention*hidden + rate*xp.einsum(
                "...c,kc->...k", xp.tanh(canvas), self.readout.canvas_weights,
            )
            v = factor*self.variance[r[:, None], c[None, :]] + increment
            a, b, c0, d, e, f = self._components(self._jacobian(forecast), v)
            trace = a+b+c0
            minor = xp.maximum(a*b+a*c0+b*c0-d*d-e*e-f*f, 0)
            determinant = xp.maximum(a*b*c0+2*d*e*f-a*f*f-b*e*e-c0*d*d, 0)
            scores = xp.empty(grid_shape, dtype=xp.float32)
            for iy, y in enumerate(np.linspace(0, 1, grid_shape[0])):
                for ix, x in enumerate(np.linspace(0, 1, grid_shape[1])):
                    pi = replace(precision, gaze=(float(x), float(y))).field(
                        self.shape, backend=xp, indices=(r, c),
                    )
                    scores[iy, ix] = scale*(area*xp.log1p(
                        pi*(trace+pi*(minor+pi*determinant)),
                    )).sum()
            scores = to_numpy(scores)
        else:
            if not isinstance(prior, GLArray) or prior.device is not self.device:
                raise ValueError("uncertainty planning requires a prior on its own OpenGL device")
            index_shape = (len(rows), len(cols))
            if self._index_shape != index_shape:
                if self._indices is not None:
                    self.device.release(self._indices, self._moments)
                self._indices = GLArray(
                    self.device, (2*(len(rows)+len(cols)),),
                    np.concatenate((rows, cols, row_weights, col_weights)).astype(np.float32),
                )
                self._moments = GLArray(self.device, (len(rows)*len(cols), 3))
                self._index_shape = index_shape
            self.device.run(
                self._native_source() + f"""
                layout(std430,binding=4) readonly buffer Prior {{float prior[];}};
                layout(std430,binding=5) readonly buffer Indices {{float indices[];}};
                layout(std430,binding=6) writeonly buffer Moments {{float moments[];}};
                uniform int nr,nc;uniform float retention,rate,factor,increment;
                void main(){{int i=int(gl_GlobalInvocationID.x);if(i>=nr*nc)return;
                    int y=int(indices[i/nc]),x=int(indices[nr+i%nc]),p=y*{self.shape[1]}+x;
                    mat3 cov=mat3(0.0);
                    for(int k=0;k<H;++k){{
                        float drive=0.0;
                        for(int c=0;c<3;++c)drive+=weights[WC+k*3+c]*tanh(prior[p*3+c]);
                        float h=retention*hidden[p*H+k]+rate*drive;
                        vec3 j=decoder_column(k)*(1.0-pow(tanh(h),2.0));
                        cov+=(factor*variance[p*H+k]+increment)*outerProduct(j,j);
                    }}
                    float a=cov[0][0],b=cov[1][1],c=cov[2][2];
                    float d=cov[0][1],e=cov[0][2],f=cov[1][2];
                    moments[i*3]=a+b+c;
                    moments[i*3+1]=max(a*b+a*c+b*c-d*d-e*e-f*f,0.0);
                    moments[i*3+2]=max(determinant(cov),0.0);
                }}""", [*self._native_arrays(), prior, self._indices, self._moments],
                len(rows)*len(cols), nr=len(rows), nc=len(cols), retention=retention,
                rate=rate, factor=factor, increment=increment,
            )
            _, _, width = precision.pixel_parameters(self.shape)
            result = self.device.reduce(
                GAUSSIAN_PRECISION_GLSL + f"""
                layout(std430,binding=0) readonly buffer Moments {{float moments[];}};
                layout(std430,binding=1) readonly buffer Indices {{float indices[];}};
                uniform int nr,nc,sampling;uniform float width,baseline,peak;
                float gain(int p,int candidate){{
                    if(sampling==0)return 0.0;
                    vec2 gaze=vec2(float(candidate%{grid_shape[1]})/{grid_shape[1]-1}.0*{self.shape[1]-1}.0,
                        float(candidate/{grid_shape[1]})/{grid_shape[0]-1}.0*{self.shape[0]-1}.0);
                    vec2 r=vec2(indices[nr+p%nc],indices[p/nc]);
                    float pi=gaussian_precision(r-gaze,width,baseline,peak);
                    float z=pi*(moments[p*3]+pi*(moments[p*3+1]+pi*moments[p*3+2]));
                    float u=1.0+z;
                    float area=indices[nr+nc+p/nc]*indices[nr+nc+nr+p%nc];
                    return area*(u==1.0?z:log(u)*(z/(u-1.0)));
                }}""", "gain(p,c)", [self._moments, self._indices], len(rows)*len(cols),
                grid_shape[0]*grid_shape[1], nr=len(rows), nc=len(cols), sampling=int(precision.sample),
                width=float(width), baseline=float(precision.baseline), peak=float(precision.peak),
            )
            scores = result.download().reshape(grid_shape)*scale
            self.device.release(result)
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise FloatingPointError("Gaussian uncertainty diverged; reduce variance settings or timestep")
        return scores

    def belief_snapshot(self, gaze: tuple[float, float] | None = None) -> VisualBeliefSnapshot:
        """Return bounded host snapshot of current stored projected visual belief at nearest gaze pixel."""
        self.validate_config(self.readout.config)
        self._ensure_initialized()
        if gaze is None:
            precision = self.readout.config.spatial_precision
            gaze = precision.gaze if precision is not None else (0.5, 0.5)
        if len(gaze) != 2 or any(not np.isfinite(v) or not 0 <= v <= 1 for v in gaze):
            raise ValueError("gaze must contain two finite coordinates in [0, 1]")
        rows, cols = self.shape
        col = int(np.clip(round(float(gaze[0]) * (cols - 1)), 0, cols - 1))
        row = int(np.clip(round(float(gaze[1]) * (rows - 1)), 0, rows - 1))
        pixel = row * cols + col
        if self.device is None:
            hidden_source = self._frozen_hidden if self._pending else self.readout.hidden
            mean = to_numpy(hidden_source[row, col]).astype(np.float32, copy=True)
            variance = to_numpy(self.variance[row, col]).astype(np.float32, copy=True)
            vw = to_numpy(self.readout.visual_weights).astype(np.float64, copy=False)
        else:
            hidden_source = self._frozen_hidden if self._pending else self.readout.model.hidden
            sample = GLArray(self.device, (2 * self.channels,))
            self.device.run(
                """layout(std430,binding=0) readonly buffer Hidden {float hidden[];};
                layout(std430,binding=1) readonly buffer Variance {float variance[];};
                layout(std430,binding=2) writeonly buffer Out {float out_values[];};
                uniform int base,channels;
                void main(){int k=int(gl_GlobalInvocationID.x);if(k>=channels)return;
                    out_values[k]=hidden[base+k];
                    out_values[channels+k]=variance[base+k];
                }""",
                [hidden_source, self.variance, sample], self.channels,
                base=pixel * self.channels, channels=self.channels,
            )
            values = sample.download()
            self.device.release(sample)
            mean = values[:self.channels].astype(np.float32, copy=True)
            variance = values[self.channels:].astype(np.float32, copy=True)
            _, vw_f32, _, _, _ = self.readout.weights()
            vw = vw_f32.astype(np.float64, copy=False)
        mean64 = mean.astype(np.float64)
        var64 = np.maximum(variance.astype(np.float64), 0.0)
        sensory_mean = (vw @ np.tanh(mean64)).astype(np.float32)
        j = vw * (1.0 - np.tanh(mean64) ** 2)[None, :]
        sensory_cov = ((j * var64[None, :]) @ j.T).astype(np.float32)
        for arr in (mean, variance, sensory_mean, sensory_cov):
            arr.setflags(write=False)
        return {
            "mean": mean,
            "variance": variance,
            "sensory_mean": sensory_mean,
            "sensory_covariance": sensory_cov,
            "pixel": (row, col),
            "gaze": (float(gaze[0]), float(gaze[1])),
            "stage": "forecast" if self._pending else "posterior",
        }


def _pixel_area_weights(indices, size):
    """Pixel-cell widths, with outer image boundaries half a pixel past the endpoints."""
    edges = np.empty(len(indices)+1, dtype=np.float64)
    edges[0], edges[-1] = -.5, size-.5
    edges[1:-1] = .5*(indices[:-1]+indices[1:])
    return np.diff(edges).astype(np.float32)
