from __future__ import annotations

from dataclasses import replace

import numpy as np

from audioviz.readouts.spatial_precision import GAUSSIAN_PRECISION_GLSL, GaussianSpatialPrecision
from audioviz.utils.array_backend import to_numpy
from audioviz.utils.opengl_compute import GLArray
from audioviz.utils.spatial_preview import preview_indices


class GaussianSensoryUncertainty:
    """Local Gaussian covariance filter for an uncoupled, fixed-weight RGB predictor.

    Hidden covariance is projected onto its diagonal after conditioning. The
    wave canvas is a known input; its uncertainty and parameter uncertainty
    are outside this conditional approximation.
    """

    def __init__(self, readout, *, initial_variance=1.0, process_variance=0.01, backend=np, device=None):
        for name, value in (("initial_variance", initial_variance), ("process_variance", process_variance)):
            if not np.isfinite(value) or value < 0 or value > np.finfo(np.float32).max:
                raise ValueError(f"{name} must be finite, non-negative and representable as float32")
        if np.float32(initial_variance) == 0:
            raise ValueError("initial_variance must be positive in float32")
        self.readout = readout
        self.shape = readout.canvas_shape[:2]
        self.channels = readout.config.hidden_channels
        self.backend = backend
        self.device = device
        self.initial_variance = float(initial_variance)
        self.process_variance = float(process_variance)
        shape = (*self.shape, self.channels)
        self.variance = (
            GLArray(device, shape) if device is not None
            else backend.full(shape, initial_variance, dtype=backend.float32)
        )
        self._frozen_hidden = (
            GLArray(device, shape) if device is not None else backend.zeros(shape, dtype=backend.float32)
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

    def _process_increment(self, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("uncertainty timestep must be finite and non-negative")
        increment = self.process_variance * float(dt)
        if not np.isfinite(increment) or increment > np.finfo(np.float32).max:
            raise ValueError("process variance increment must be representable as float32")
        return float(increment)

    def reset(self):
        if self.device is None:
            self.variance.fill(self.initial_variance)
        else:
            self._initialized = False
        self._pending = False

    def _ensure_initialized(self):
        if not self._initialized:
            self.device.run(
                """layout(std430,binding=0) buffer Variance {float variance[];};
                uniform int count;uniform float initial;
                void main(){int i=int(gl_GlobalInvocationID.x);if(i<count)variance[i]=initial;}""",
                [self.variance], self.variance.size, count=self.variance.size, initial=self.initial_variance,
            )
            self._initialized = True

    def predict(self, *, dt):
        """Called immediately after the mean predictor, before any new evidence."""
        self.validate_config(self.readout.config)
        if self._pending:
            raise RuntimeError("Condition the previous Gaussian forecast before predicting again")
        increment = self._process_increment(dt)
        self._ensure_initialized()
        factor = float((1 - self.readout.config.inference_rate) ** 2)
        if self.device is None:
            self.variance[:] = factor * self.variance + increment
            self._frozen_hidden[:] = self.readout.hidden
        else:
            self.device.run(
                self._native_source() + """
                uniform float factor,increment;
                void main(){int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                    for(int k=0;k<H;++k){int i=p*H+k;
                        variance[i]=factor*variance[i]+increment;frozen[i]=hidden[i];
                    }
                }""", self._native_arrays(), int(np.prod(self.shape)), factor=factor, increment=increment,
            )
        self._pending = True

    def _jacobian(self, hidden):
        xp = self.backend
        weights = xp.asarray(self.readout.visual_weights, dtype=xp.float32)
        return weights * (1 - xp.tanh(hidden)**2)[..., None, :]

    @staticmethod
    def _components(j, variance):
        return tuple((j[..., a, :] * variance * j[..., b, :]).sum(axis=-1)
                     for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))

    def condition(self, precision: GaussianSpatialPrecision | None, *, observed):
        """Condition once, using the frozen pre-evidence decoder linearization."""
        if not self._pending:
            raise RuntimeError("Gaussian conditioning requires a pre-evidence forecast")
        if not isinstance(observed, bool):
            raise ValueError("observed must be a boolean")
        precision = GaussianSpatialPrecision(baseline=1, peak=0) if precision is None else precision
        if observed and precision.sample:
            if self.device is None:
                xp = self.backend
                pi = precision.field(self.shape, backend=xp)
                j = self._jacobian(self._frozen_hidden)
                a, b, c, d, e, f = self._components(j, self.variance)
                a, b, c, d, e, f = 1 + pi*a, 1 + pi*b, 1 + pi*c, pi*d, pi*e, pi*f
                aa, bb, cc = b*c-f*f, a*c-e*e, a*b-d*d
                dd, ee, ff = e*f-d*c, d*f-b*e, d*e-a*f
                determinant = a*aa + d*dd + e*ee
                x, y, z = j[..., 0, :], j[..., 1, :], j[..., 2, :]
                quadratic = (
                    aa[..., None]*x*x + bb[..., None]*y*y + cc[..., None]*z*z
                    + 2*(dd[..., None]*x*y + ee[..., None]*x*z + ff[..., None]*y*z)
                ) / determinant[..., None]
                self.variance[:] = xp.maximum(
                    self.variance - pi[..., None]*self.variance**2*quadratic, 0,
                )
            else:
                x, y, width = precision.pixel_parameters(self.shape)
                self.device.run(
                    self._native_source() + GAUSSIAN_PRECISION_GLSL + f"""
                    uniform float gaze_x,gaze_y,width,baseline,peak;
                    void main(){{int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                        float pi=gaussian_precision(vec2(float(p%{self.shape[1]}),float(p/{self.shape[1]}))
                            -vec2(gaze_x,gaze_y),width,baseline,peak);
                        mat3 cov=mat3(0.0);
                        for(int k=0;k<H;++k){{
                            vec3 j=decoder_column(k)*(1.0-pow(tanh(frozen[p*H+k]),2.0));
                            cov+=variance[p*H+k]*outerProduct(j,j);
                        }}
                        mat3 inv=inverse(mat3(1.0)+pi*cov);
                        for(int k=0;k<H;++k){{
                            vec3 j=decoder_column(k)*(1.0-pow(tanh(frozen[p*H+k]),2.0));
                            float v=variance[p*H+k];
                            variance[p*H+k]=max(v-pi*v*v*dot(j,inv*j),0.0);
                        }}
                    }}""", self._native_arrays(), int(np.prod(self.shape)),
                    gaze_x=float(x), gaze_y=float(y), width=float(width),
                    baseline=float(precision.baseline), peak=float(precision.peak),
                )
        self._pending = False

    def _native_arrays(self):
        model = self.readout.model
        return [self.variance, model.hidden, model.weights, self._frozen_hidden]

    def _native_source(self):
        offset = self.readout.model.weight_offsets[0]
        h = self.channels
        return f"""
        const int H={h},N={int(np.prod(self.shape))},WC={offset},WD={offset+h*3};
        layout(std430,binding=0) buffer Variance {{float variance[];}};
        layout(std430,binding=1) readonly buffer Hidden {{float hidden[];}};
        layout(std430,binding=2) readonly buffer Weights {{float weights[];}};
        layout(std430,binding=3) buffer Frozen {{float frozen[];}};
        vec3 decoder_column(int k){{
            return vec3(weights[WD+k],weights[WD+H+k],weights[WD+2*H+k]);
        }}
        """

    def expected_information_gain(
        self, precision: GaussianSpatialPrecision, *, prior, dt, grid_shape=(9, 13), max_side=64,
    ):
        """One-step sensory-state MI, conditional on a held-fixed point canvas.

        Full local RGB covariance is retained for scoring. Only spatial
        quadrature and the diagonal hidden-state approximation discard terms.
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
        increment = self._process_increment(dt)
        self._ensure_initialized()
        rows, cols = preview_indices(self.shape, max_side=max_side)
        rate = float(self.readout.config.inference_rate)
        factor = float((1 - rate)**2)
        scale = 0.5 * np.prod(self.shape) / (len(rows)*len(cols))
        if self.device is None:
            xp = self.backend
            r, c = xp.asarray(rows), xp.asarray(cols)
            hidden = self.readout.hidden[r[:, None], c[None, :]]
            canvas = prior[r[:, None], c[None, :]]
            forecast = (1-rate)*hidden + rate*xp.einsum(
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
                    scores[iy, ix] = scale*xp.log1p(pi*(trace+pi*(minor+pi*determinant))).sum()
            scores = to_numpy(scores)
        else:
            if not isinstance(prior, GLArray) or prior.device is not self.device:
                raise ValueError("uncertainty planning requires a prior on its own OpenGL device")
            index_shape = (len(rows), len(cols))
            if self._index_shape != index_shape:
                if self._indices is not None:
                    self.device.release(self._indices, self._moments)
                self._indices = GLArray(
                    self.device, (len(rows)+len(cols),), np.concatenate((rows, cols)).astype(np.float32),
                )
                self._moments = GLArray(self.device, (len(rows)*len(cols), 3))
                self._index_shape = index_shape
            self.device.run(
                self._native_source() + f"""
                layout(std430,binding=4) readonly buffer Prior {{float prior[];}};
                layout(std430,binding=5) readonly buffer Indices {{float indices[];}};
                layout(std430,binding=6) writeonly buffer Moments {{float moments[];}};
                uniform int nr,nc;uniform float rate,factor,increment;
                void main(){{int i=int(gl_GlobalInvocationID.x);if(i>=nr*nc)return;
                    int y=int(indices[i/nc]),x=int(indices[nr+i%nc]),p=y*{self.shape[1]}+x;
                    mat3 cov=mat3(0.0);
                    for(int k=0;k<H;++k){{
                        float drive=0.0;
                        for(int c=0;c<3;++c)drive+=weights[WC+k*3+c]*tanh(prior[p*3+c]);
                        float h=(1.0-rate)*hidden[p*H+k]+rate*drive;
                        vec3 j=decoder_column(k)*(1.0-pow(tanh(h),2.0));
                        cov+=(factor*variance[p*H+k]+increment)*outerProduct(j,j);
                    }}
                    float a=cov[0][0],b=cov[1][1],c=cov[2][2];
                    float d=cov[0][1],e=cov[0][2],f=cov[1][2];
                    moments[i*3]=a+b+c;
                    moments[i*3+1]=max(a*b+a*c+b*c-d*d-e*e-f*f,0.0);
                    moments[i*3+2]=max(determinant(cov),0.0);
                }}""", [*self._native_arrays(), prior, self._indices, self._moments],
                len(rows)*len(cols), nr=len(rows), nc=len(cols), rate=rate, factor=factor, increment=increment,
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
                    return u==1.0?z:log(u)*(z/(u-1.0));
                }}""", "gain(p,c)", [self._moments, self._indices], len(rows)*len(cols),
                grid_shape[0]*grid_shape[1], nr=len(rows), nc=len(cols), sampling=int(precision.sample),
                width=float(width), baseline=float(precision.baseline), peak=float(precision.peak),
            )
            scores = result.download().reshape(grid_shape)*scale
            self.device.release(result)
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise FloatingPointError("Gaussian uncertainty diverged; reduce variance settings or timestep")
        return scores
