from __future__ import annotations

from dataclasses import replace

import numpy as np

from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.diagnostics import (
    EnergyStatistics, PredictiveCodingDiagnostics, SensoryPreview, StreamStateDiagnostics,
    copy_weights,
)
from audioviz.readouts.predictive_coding import PredictiveCodingRGBReadout
from audioviz.readouts.spatial_precision import GAUSSIAN_PRECISION_GLSL
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS
from audioviz.utils.opengl_compute import GLArray, GLArrayView


class OpenGLReadout:
    """Presentation/configuration facade over one branch of native shader state."""

    def __init__(self, model, index):
        self.model = model
        self.index = index
        self.config = model.config
        self.canvas_shape = model.shape
        self.diagnostics_enabled = False
        self.spatial_preview_enabled = False
        self.diagnostics = None
        self.stream_state_prediction = None

    @property
    def hidden(self):
        return GLArrayView(
            self.model.hidden, (*self.model.shape[:2], self.config.hidden_channels),
            self.index * self.model.pixels * self.config.hidden_channels,
        )

    @property
    def hidden_error(self):
        return GLArrayView(
            self.model.errors, self.hidden.shape,
            self.index * self.model.pixels * self.config.hidden_channels,
        )

    @property
    def visual_error(self):
        offset = 2 * self.model.pixels * self.config.hidden_channels
        return GLArrayView(
            self.model.errors, self.model.shape if self.index == 0 else (AUDIO_SPECTRAL_BANDS,),
            offset if self.index == 0 else offset + self.model.pixels * 3,
        )

    def weights(self):
        block = self.model.weight_offsets[self.index]
        values = self.model.weights.download()
        h = self.config.hidden_channels
        outputs = 3 if self.index == 0 else AUDIO_SPECTRAL_BANDS
        return (
            values[block: block + h * 3].reshape(h, 3).copy(),
            values[block + h * 3: block + h * (3 + outputs)].reshape(outputs, h).copy(),
            values[block + h * (3 + outputs): block + h * (4 + outputs)].reshape(1, h).copy(),
            values[block + h * (4 + outputs): block + h * (4 + outputs) + 1].copy(),
            values[block + h * (4 + outputs) + 1: block + h * (4 + outputs) + 1 + h * h].reshape(h, h).copy(),
        )

    @property
    def canvas_weights(self):
        return self.weights()[0]

    @property
    def visual_weights(self):
        return self.weights()[1]

    def set_diagnostics_enabled(self, enabled):
        self.diagnostics_enabled = bool(enabled)
        self.diagnostics = None
        if not enabled:
            self.spatial_preview_enabled = False

    def set_spatial_preview_enabled(self, enabled):
        self.spatial_preview_enabled = bool(enabled) and self.diagnostics_enabled

    def update_config(self, **changes):
        self.model.update_config(**changes)

    def reset(self):
        self.model.reset()


class OpenGLPredictiveModel:
    """Float32 GLSL implementation of the existing synchronous predictive model."""

    def __init__(self, engine, config, *, audio_enabled):
        if config.dtype != "float32":
            raise ValueError("OpenGL inference requires explicit dtype='float32'")
        if config.cross_modal_enabled and not audio_enabled:
            raise ValueError("Cross-modal inference requires both readouts")
        self.engine = engine
        self.device = engine.propagator.compute
        self.config = config
        self.shape = engine.canvas_shape
        self.pixels = int(np.prod(self.shape[:2]))
        self.branch_count = 2 if audio_enabled else 1
        self.hidden = GLArray(self.device, (2, self.pixels, config.hidden_channels))
        self.next_hidden = GLArray(self.device, self.hidden.shape)
        self.canvas = GLArray(self.device, self.shape)
        self.next_canvas = GLArray(self.device, self.shape)
        self.errors = GLArray(
            self.device, (2 * self.pixels * config.hidden_channels + 3 * self.pixels + AUDIO_SPECTRAL_BANDS,)
        )
        self.evidence = GLArray(self.device, (3 * self.pixels + AUDIO_SPECTRAL_BANDS,))
        self.predictions = GLArray(self.device, (3 * self.pixels + AUDIO_SPECTRAL_BANDS + 4,))
        self.correction = GLArray(self.device, self.shape)
        self.camera_image = GLArray(self.device, self.shape)
        self._raw_camera = None
        self._camera_indices = None
        self.camera_prediction = GLArrayView(self.predictions, self.shape, 0)
        self.audio_prediction = GLArrayView(self.predictions, (AUDIO_SPECTRAL_BANDS,), 3 * self.pixels)
        self._pooled = None
        self.observation = None
        self.spectrum = None
        self.states = (False, False)
        self._trajectory = []
        self.weight_offsets = []
        parameters = []
        # Initialization is small and seeded identically to the array reference.
        for cls in (PredictiveCodingRGBReadout, PredictiveCodingAudioReadout):
            branch = cls(canvas_shape=(1, 1, 3), config=config)
            self.weight_offsets.append(sum(x.size for x in parameters))
            parameters.extend((
                branch.canvas_weights.ravel(), branch.visual_weights.ravel(),
                branch.stream_state_weights.ravel(), branch.stream_state_bias,
                np.zeros(config.hidden_channels**2, dtype=np.float32),
            ))
        initial = np.concatenate(parameters).astype(np.float32)
        self.weights = GLArray(self.device, initial.shape, initial)
        self.camera = OpenGLReadout(self, 0)
        self.audio = OpenGLReadout(self, 1) if audio_enabled else None
        self._source = self._make_source()

    @property
    def branches(self):
        return (self.camera,) if self.audio is None else (self.camera, self.audio)

    def update_config(self, **changes):
        config = replace(self.config, **changes)
        if config.hidden_channels != self.config.hidden_channels or config.dtype != self.config.dtype:
            raise ValueError("hidden_channels and dtype require rebuilding OpenGL readouts")
        if config.cross_modal_enabled and self.audio is None:
            raise ValueError("Cross-modal inference requires both readouts")
        self.config = config
        for branch in self.branches:
            branch.config = config

    def _make_source(self):
        h, n, bands = self.config.hidden_channels, self.pixels, AUDIO_SPECTRAL_BANDS
        return self._precision_source() + f"""
        const int H={h},N={n},B={bands};
        layout(std430,binding=0) readonly buffer Canvas {{float canvas[];}};
        layout(std430,binding=1) readonly buffer Hidden {{float hidden[];}};
        layout(std430,binding=2) buffer Weights {{float weights[];}};
        layout(std430,binding=3) readonly buffer Pooled {{float pooled[];}};
        layout(std430,binding=4) readonly buffer Evidence {{float evidence[];}};
        layout(std430,binding=5) buffer Errors {{float errors[];}};
        layout(std430,binding=6) writeonly buffer NextCanvas {{float next_canvas[];}};
        layout(std430,binding=7) writeonly buffer NextHidden {{float next_hidden[];}};
        uniform int coupled,branches,camera_present,audio_present,state0,state1,predicting;
        uniform float hidden_rate,canvas_rate;
        int base(int b){{return b==0?{self.weight_offsets[0]}:{self.weight_offsets[1]};}}
        int outputs(int b){{return b==0?3:B;}}
        int decoder(int b){{return base(b)+H*3;}}
        int state_weights(int b){{return decoder(b)+outputs(b)*H;}}
        int cross_weights(int b){{return state_weights(b)+H+1;}}
        float activity(int b,int p,int k){{return tanh(hidden[(b*N+p)*H+k]);}}
        float mean_activity(int b,int k){{return pooled[b*H+k]/float(N);}}
        float state_logit(int b){{
            float logit=weights[state_weights(b)+H];
            for(int k=0;k<H;++k)logit+=weights[state_weights(b)+k]*mean_activity(b,k);
            return logit;
        }}
        float state_prediction(int b){{
            return 0.5*(1.0+tanh(0.5*state_logit(b)));
        }}
        float state_direction(int b){{
            float pred=state_prediction(b);
            return (float(b==0?state0:state1)-pred)*pred*(1.0-pred);
        }}
        float sensory_prediction(int b,int p,int o){{
            float v=0.0;
            for(int k=0;k<H;++k)v+=weights[decoder(b)+o*H+k]*
                (b==0?activity(b,p,k):mean_activity(b,k));
            return v;
        }}
        float sensory_error(int b,int p,int o){{
            if(predicting==1 || (b==0?camera_present:audio_present)==0)return 0.0;
            return evidence[b==0?p*3+o:N*3+o]-sensory_prediction(b,p,o);
        }}
        float hidden_error(int b,int p,int k){{
            return errors[(b*N+p)*H+k];
        }}
        """

    def _precision_source(self):
        return GAUSSIAN_PRECISION_GLSL + f"""
        uniform int spatial_precision_enabled,spatial_sample;
        uniform float gaze_x,gaze_y,foveal_width,peripheral_precision,peak_precision;
        float camera_precision(int p){{
            if(spatial_precision_enabled==0)return 1.0;
            if(spatial_sample==0)return 0.0;
            vec2 r=vec2(float(p%{self.shape[1]}),float(p/{self.shape[1]}));
            return gaussian_precision(r-vec2(gaze_x,gaze_y),foveal_width,
                peripheral_precision,peak_precision);
        }}
        """

    def _precision_uniforms(self):
        precision = self.config.spatial_precision
        if precision is None:
            return dict(spatial_precision_enabled=0)
        x, y, width = precision.pixel_parameters(self.shape[:2])
        return dict(
            spatial_precision_enabled=1,
            spatial_sample=int(precision.sample),
            gaze_x=float(x), gaze_y=float(y), foveal_width=float(width),
            peripheral_precision=float(precision.baseline), peak_precision=float(precision.peak),
        )

    def _arrays(self):
        return [
            self.canvas, self.hidden, self.weights, self._pooled,
            self.evidence, self.errors, self.next_canvas, self.next_hidden,
        ]

    def _uniforms(self, *, predicting=False):
        return dict(
            coupled=int(self.config.cross_modal_enabled), branches=self.branch_count,
            camera_present=int(self.observation is not None),
            audio_present=int(self.spectrum is not None), state0=int(self.states[0]),
            state1=int(self.states[1]), predicting=int(predicting),
            hidden_rate=self.config.inference_rate, canvas_rate=self.config.canvas_rate,
            **self._precision_uniforms(),
        )

    def _pool(self):
        if self._pooled is not None:
            self.device.release(self._pooled)
        h = self.config.hidden_channels
        self._pooled = self.device.reduce(
            "layout(std430,binding=0) readonly buffer Hidden {float hidden[];};",
            f"tanh(hidden[((c/{h})*{self.pixels}+p)*{h}+c%{h}])",
            [self.hidden], self.pixels, 2 * h,
        )

    def _errors(self, *, predicting=False):
        self.device.run(self._source + """
        void main(){
            int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
            for(int b=0;b<branches;++b){
                for(int k=0;k<H;++k){
                    float e=hidden[(b*N+p)*H+k];
                    for(int c=0;c<3;++c)e-=weights[base(b)+k*3+c]*tanh(canvas[p*3+c]);
                    if(coupled==1)for(int j=0;j<H;++j)
                        e-=weights[cross_weights(b)+k*H+j]*activity(1-b,p,j);
                    errors[(b*N+p)*H+k]=e;
                }
            }
            for(int c=0;c<3;++c)errors[2*N*H+p*3+c]=sensory_error(0,p,c);
            if(p==0)for(int o=0;o<B;++o)errors[2*N*H+3*N+o]=sensory_error(1,0,o);
        }""", self._arrays(), self.pixels, **self._uniforms(predicting=predicting))

    def _advance(self, *, predicting=False):
        self.device.run(self._source + """
        void main(){
            int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
            for(int c=0;c<3;++c){
                float change=0.0;
                for(int b=0;b<branches;++b)for(int k=0;k<H;++k)
                    change+=weights[base(b)+k*3+c]*hidden_error(b,p,k);
                float a=tanh(canvas[p*3+c]);
                next_canvas[p*3+c]=canvas[p*3+c]+
                    (predicting==1?0.0:canvas_rate*(1.0-a*a)*change);
            }
            for(int b=0;b<branches;++b)for(int k=0;k<H;++k){
                float feedback=0.0,a=activity(b,p,k);
                if(predicting==0){
                    for(int o=0;o<outputs(b);++o)
                        feedback+=weights[decoder(b)+o*H+k]*
                            errors[2*N*H+(b==0?p*3+o:3*N+o)]*
                            (b==0?camera_precision(p):1.0);
                    feedback+=state_direction(b)*weights[state_weights(b)+k];
                }
                if(coupled==1)for(int j=0;j<H;++j)
                    feedback+=weights[cross_weights(1-b)+j*H+k]*hidden_error(1-b,p,j);
                next_hidden[(b*N+p)*H+k]=hidden[(b*N+p)*H+k]+hidden_rate*
                    (-hidden_error(b,p,k)+(1.0-a*a)*feedback);
            }
        }""", self._arrays(), self.pixels, **self._uniforms(predicting=predicting))
        self.hidden, self.next_hidden = self.next_hidden, self.hidden
        self.canvas, self.next_canvas = self.next_canvas, self.canvas

    def _copy_prior(self, prior):
        if not isinstance(prior, GLArray) or prior.device is not self.device or prior.shape != self.shape:
            raise ValueError("Shader inference requires a native prior from its own engine")
        self.device.run(
            """layout(std430,binding=0) readonly buffer Input {float values[];};
            layout(std430,binding=1) writeonly buffer Output {float result[];};
            uniform int count;
            void main(){int i=int(gl_GlobalInvocationID.x);if(i<count)result[i]=values[i];}""",
            [prior, self.canvas], self.canvas.size, count=self.canvas.size,
        )

    def predict(self, prior):
        self._copy_prior(prior)
        self._pool()
        self._errors(predicting=True)
        self._advance(predicting=True)
        self._pool()
        # The frozen expectation is separate from subsequent inference scratch.
        self.device.run(
            self._source.replace(
                "layout(std430,binding=6) writeonly buffer NextCanvas {float next_canvas[];};",
                "layout(std430,binding=6) writeonly buffer Prediction {float prediction[];};",
            ) + """
            void main(){
                int p=int(gl_GlobalInvocationID.x);if(p>=N)return;
                for(int c=0;c<3;++c)prediction[p*3+c]=sensory_prediction(0,p,c);
                if(p==0){
                    for(int o=0;o<B;++o)prediction[N*3+o]=sensory_prediction(1,0,o);
                    for(int b=0;b<2;++b){
                        prediction[N*3+B+b]=state_prediction(b);
                        prediction[N*3+B+2+b]=state_logit(b);
                    }
                }
            }""",
            [*self._arrays()[:6], self.predictions, self.next_hidden],
            self.pixels, **self._uniforms(predicting=True),
        )
        self._validate(self.hidden, self.predictions)
        states = GLArrayView(self.predictions, (2,), 3 * self.pixels + AUDIO_SPECTRAL_BANDS).download()
        for branch in self.branches:
            branch.diagnostics = None
            branch.stream_state_prediction = float(states[branch.index])
        return self.camera_prediction, self.audio_prediction if self.audio is not None else None

    def infer(self, prior, image, spectrum, *, states):
        self._copy_prior(prior)
        self.observation = image
        self.spectrum = spectrum
        if len(states) != 2 or any(not isinstance(state, bool) for state in states):
            raise ValueError("Shader stream-state evidence requires two boolean flags")
        self.states = states
        for value, shape, offset in ((image, self.shape, 0), (spectrum, (AUDIO_SPECTRAL_BANDS,), 3 * self.pixels)):
            if value is not None:
                self._upload_evidence(value, shape, offset)
        collect = any(b.diagnostics_enabled for b in self.branches)
        self._trajectory = []
        for _ in range(self.config.inference_steps):
            self._pool()
            self._errors()
            if collect:
                self._trajectory.append(self._energy())
            self._advance()
        self._pool()
        self._errors()
        self._validate(self.canvas, self.hidden, self.errors, self._pooled)
        if collect:
            self._trajectory.append(self._energy())
        before = self.weights.download() if collect else None
        stream_predictions = self._state_predictions()
        if self.config.learning_enabled and self.config.learning_rate:
            self._learn()
        if collect:
            self._diagnostics(before, stream_predictions)
        self.device.run(
            """layout(std430,binding=0) readonly buffer Canvas {float canvas[];};
            layout(std430,binding=1) readonly buffer Prior {float prior[];};
            layout(std430,binding=2) writeonly buffer Delta {float delta[];};
            uniform int count;
            void main(){int i=int(gl_GlobalInvocationID.x);if(i<count)delta[i]=canvas[i]-prior[i];}""",
            [self.canvas, prior, self.correction], self.canvas.size, count=self.canvas.size,
        )
        self._validate(self.correction)
        return self.correction

    def _upload_evidence(self, value, shape, offset):
        target = GLArrayView(self.evidence, shape, offset)
        if isinstance(value, GLArray):
            if value.shape != shape or value.device is not self.device:
                raise ValueError(f"Shader evidence must have shape {shape} on its own context")
            self._validate(value)
            self.device.run(
                """layout(std430,binding=0) readonly buffer Input {float values[];};
                layout(std430,binding=1) writeonly buffer Evidence {float evidence[];};
                uniform int count,offset;
                void main(){int i=int(gl_GlobalInvocationID.x);if(i<count)evidence[offset+i]=values[i];}""",
                [value, self.evidence], value.size, count=value.size, offset=offset,
            )
        else:
            value = np.asarray(value, dtype=np.float32)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Shader evidence must be finite and have shape {shape}")
            target.upload(value)
        return target

    def preprocess_camera(self, frame, *, gain=1.):
        values = np.asarray(frame)
        if values.ndim not in (2, 3) or not values.size:
            raise ValueError("Camera frame must be a nonempty image")
        if values.ndim == 3 and values.shape[2] < 3:
            raise ValueError("Camera frame must provide at least three channels")
        rows, cols = self.shape[:2]
        if self._raw_camera is None or self._raw_camera.shape != values.shape:
            if self._raw_camera is not None:
                self.device.release(self._raw_camera, self._camera_indices)
            self._raw_camera = GLArray(self.device, values.shape)
            y = np.rint(np.linspace(0, values.shape[0]-1, rows))
            x = np.rint(np.linspace(0, values.shape[1]-1, cols))
            self._camera_indices = GLArray(self.device, (rows+cols,), np.concatenate((y, x)))
        scale = 255. if values.dtype.kind in ("i", "u") else (
            255. if np.nanmax(values[..., :3] if values.ndim == 3 else values) > 1 else 1.
        )
        self._raw_camera.upload(values.astype(np.float32))
        self.device.run(
            """layout(std430,binding=0) readonly buffer Raw {float raw[];};
            layout(std430,binding=1) readonly buffer Indices {float indices[];};
            layout(std430,binding=2) writeonly buffer Image {float image[];};
            uniform int count,rows,cols,src_cols,src_channels;
            uniform float scale,gain;
            void main(){
                int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
                int p=i/3,c=i%3,y=int(indices[p/cols]),x=int(indices[rows+p%cols]);
                float v=raw[(y*src_cols+x)*src_channels+(src_channels==1?0:2-c)]/scale;
                if(isnan(v))v=0.0;
                image[i]=clamp(v,0.0,1.0)*gain;
            }""", [self._raw_camera, self._camera_indices, self.camera_image], self.camera_image.size,
            count=self.camera_image.size, rows=rows, cols=cols, src_cols=values.shape[1],
            src_channels=1 if values.ndim==2 else values.shape[2], scale=scale, gain=gain,
        )
        return self.camera_image

    def _validate(self, *arrays):
        for array in arrays:
            invalid = self.device.reduce(
                "layout(std430,binding=0) readonly buffer Input {float values[];};",
                "(isnan(values[p]) || isinf(values[p])) ? 1.0 : 0.0",
                [array], array.size, 1,
            )
            failed = invalid.download()[0, 0] != 0
            self.device.release(invalid)
            if failed:
                raise FloatingPointError("Predictive coding diverged; reduce inference or learning rates.")

    def _state_predictions(self):
        result = GLArray(self.device, (4,))
        self.device.run(
            self._source.replace(
                "layout(std430,binding=6) writeonly buffer NextCanvas {float next_canvas[];};",
                "layout(std430,binding=6) writeonly buffer Output {float result[];};",
            ) + """void main(){int b=int(gl_GlobalInvocationID.x);
            if(b<2){result[b]=state_prediction(b);result[2+b]=state_logit(b);}}""",
            [*self._arrays()[:6], result, self.next_hidden], 2, **self._uniforms(),
        )
        predictions = result.download()
        self.device.release(result)
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError("Predictive coding stream-state logits diverged")
        return predictions[:2]

    def _learn(self):
        # Each parameter has its own spatial reduction, then a single update.
        declarations = self._source.replace(
            "layout(std430,binding=6) writeonly buffer NextCanvas {float next_canvas[];};", ""
        ).replace(
            "layout(std430,binding=7) writeonly buffer NextHidden {float next_hidden[];};", ""
        ) + """
        float gradient(int p,int index){
            int b=index<base(1)?0:1;
            if(b>=branches)return 0.0;
            int i=index-base(b);
            if(i<H*3)return hidden_error(b,p,i/3)*tanh(canvas[p*3+i%3]);
            i-=H*3;
            if(i<outputs(b)*H){
                if((b==0?camera_present:audio_present)==0)return 0.0;
                int o=i/H,k=i%H;
                return errors[2*N*H+(b==0?p*3+o:3*N+o)]*
                    (b==0?camera_precision(p)*activity(b,p,k):mean_activity(b,k));
            }
            i-=outputs(b)*H;
            if(i<H)return state_direction(b)*mean_activity(b,i);
            if(i==H)return state_direction(b);
            i-=H+1;
            return coupled==1?hidden_error(b,p,i/H)*activity(1-b,p,i%H):0.0;
        }"""
        gradient = self.device.reduce(
            declarations, "gradient(p,c)", self._arrays()[:6],
            self.pixels, self.weights.size, **self._uniforms(),
        )
        self.device.run(
            self._source.replace(
                "layout(std430,binding=6) writeonly buffer NextCanvas {float next_canvas[];};",
                "layout(std430,binding=6) readonly buffer Gradient {float gradient[];};",
            ) + """
            uniform int count;
            uniform float learning_rate,decay,gradient_clip,weight_clip;
            void main(){
                int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
                int b=i<base(1)?0:1;if(b>=branches)return;
                int j=i-base(b);
                if(j>=H*3 && j<H*(3+outputs(b)) && (b==0?camera_present:audio_present)==0)return;
                if(j>=H*(4+outputs(b))+1 && coupled==0)return;
                float direction=gradient[i]/float(N)-decay*weights[i];
                weights[i]=clamp(weights[i]+learning_rate*clamp(direction,-gradient_clip,gradient_clip),
                    -weight_clip,weight_clip);
            }""", [*self._arrays()[:6], gradient, self.next_hidden], self.weights.size,
            count=self.weights.size, learning_rate=self.config.learning_rate,
            decay=self.config.weight_decay, gradient_clip=self.config.gradient_clip,
            weight_clip=self.config.weight_clip, **self._uniforms(),
        )
        self.device.release(gradient)
        self._validate(self.weights)

    def _means(self, array, *, activity=False, energy=False, fourth=False, precision=False):
        channels = array.shape[-1]
        count = array.size // channels
        value = "values[offset+p*channels+c]"
        if activity:
            value = f"tanh({value})"
        if energy or fourth:
            value = f"0.5*({value})*({value})"
            if precision:
                value = f"camera_precision(p)*({value})"
        if fourth:
            value = f"({value})*({value})"
        result = self.device.reduce(
            (self._precision_source() if precision else "") +
            "layout(std430,binding=0) readonly buffer Input {float values[];};uniform int offset;",
            value, [array], count, channels, offset=array.offset,
            **(self._precision_uniforms() if precision else {}),
        )
        host = result.download().reshape(-1) / count
        self.device.release(result)
        return host

    def _statistics(self, array, *, precision=False):
        means = self._means(array, energy=True, precision=precision)
        second = self._means(array, fourth=True, precision=precision)
        return EnergyStatistics(
            float(means.mean()), max(0.0, float(second.mean() - means.mean()**2)),
            tuple(float(x) for x in means),
        )

    def _energy(self):
        total = 0.
        predictions = self._state_predictions()
        for branch in self.branches:
            total += self._means(branch.hidden_error, energy=True).sum()
            total += self._means(branch.visual_error, energy=True, precision=branch.index == 0).sum()
            total += .5 * (float(self.states[branch.index]) - predictions[branch.index])**2
        return float(total)

    def _diagnostics(self, before, stream_predictions, *, branches=None, audio_precision=1.0):
        after = self.weights.download() if before is not None else None
        for branch in (self.branches if branches is None else branches):
            if not branch.diagnostics_enabled:
                continue
            b = branch.index
            cw, vw, sw, bias, cross = branch.weights()
            start = self.weight_offsets[b]
            end = self.weight_offsets[b + 1] if b == 0 else self.weights.size
            observed = self.observation if b == 0 else self.spectrum
            prediction = self.camera_prediction if b == 0 else self.audio_prediction
            present = observed is not None
            observation_means = (
                (self._means(observed) if isinstance(observed, GLArray) else observed.mean(axis=(0, 1)))
                if b == 0 else observed
            ) if present else self._final_sensory_means(b)
            visual_stats = (
                self._statistics(branch.visual_error, precision=b == 0)
                if present else EnergyStatistics(np.nan, np.nan, (np.nan,) * vw.shape[0])
            )
            if present and b == 1 and audio_precision != 1.0:
                scale = float(audio_precision)
                visual_stats = EnergyStatistics(
                    visual_stats.mean * scale,
                    visual_stats.variance * (scale * scale),
                    tuple(x * scale for x in visual_stats.channel_means),
                )
            branch.diagnostics = PredictiveCodingDiagnostics(
                hidden=self._statistics(branch.hidden_error),
                visual=visual_stats,
                inference_energy=tuple(self._trajectory),
                canvas_means=tuple(float(x) for x in self._means(self.canvas)),
                hidden_means=tuple(float(x) for x in self._means(branch.hidden)),
                observation_means=tuple(float(x) for x in observation_means),
                canvas_weights=copy_weights(cw), visual_weights=copy_weights(vw),
                learning_enabled=self.config.learning_enabled, learning_rate=self.config.learning_rate,
                weight_update_norm=float(np.linalg.norm(after[start:end]-before[start:end]))
                if before is not None else 0.0,
                spatial=SensoryPreview(
                    branch.hidden.preview(),
                    prediction.preview() if b == 0 else copy_weights(prediction.download()),
                    self._observation_preview(observed) if b == 0 else copy_weights(observed),
                ) if present and branch.spatial_preview_enabled else None,
                observation_present=present,
                cross_modal_weights=copy_weights(cross) if self.config.cross_modal_enabled else None,
                cross_parent_means=tuple(float(x) for x in self._means(self.branches[1-b].hidden))
                if self.config.cross_modal_enabled else (),
                stream_state=StreamStateDiagnostics(
                    self.states[b], float(stream_predictions[b]),
                    float(.5*(float(self.states[b])-stream_predictions[b])**2),
                    copy_weights(sw), float(bias[0]),
                ) if stream_predictions is not None else None,
            )

    def _final_sensory_means(self, branch):
        # Missing samples still expose final model expectations in diagnostics.
        if branch == 0:
            means = self._means(self.camera.hidden, activity=True)
        else:
            means = self._means(self.audio.hidden, activity=True)
        return means @ self.branches[branch].visual_weights.T

    @staticmethod
    def _observation_preview(observed):
        from audioviz.utils.spatial_preview import spatial_preview

        return spatial_preview(observed)

    def reset(self):
        for array in (self.hidden, self.next_hidden, self.canvas, self.next_canvas,
                      self.errors, self.predictions, self.correction):
            if array._handle is not None:
                array.fill(0)
        self.observation = self.spectrum = None
        for branch in self.branches:
            branch.diagnostics = None
            branch.stream_state_prediction = None
