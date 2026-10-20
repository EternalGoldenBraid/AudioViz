from dataclasses import replace

import numpy as np

from audioviz.utils.opengl_compute import GLArray


class SharedSensoryBranch:
    """A sensory view of one joint belief, with no separate per-pixel hidden state."""

    def __init__(self, graph, config, canvas_shape, backend):
        self.graph = graph
        self.config = config
        self.canvas_shape = canvas_shape
        self.backend = backend
        self.diagnostics = None
        self.diagnostics_enabled = False
        self.stream_state_prediction = None

    @property
    def hidden(self):
        return self.graph.mean[self.graph.hidden_slice].reshape(1, 1, -1)

    def update_config(self, **changes):
        self.config = replace(self.config, **changes)

    def set_diagnostics_enabled(self, enabled):
        if enabled:
            raise RuntimeError("Use joint graph belief diagnostics, not legacy readout energy diagnostics")

    def set_spatial_preview_enabled(self, enabled):
        if enabled:
            raise RuntimeError("Shared graph state is available through joint belief diagnostics")

    def predict(self, prior):
        raise RuntimeError("Forecast shared sensory branches together through the orchestrator")

    def infer(self, prior, observation, *, stream_state=None):
        raise RuntimeError("Condition shared sensory branches together through the orchestrator")

    def reset(self):
        self.graph.reset()


class SharedSensoryIO:
    """Route fixed circular slots through world coordinates without resizing weights."""

    def __init__(self, shape, *, radius=8, backend=np, device=None):
        self.shape = shape
        self.backend = backend
        self.device = device
        if isinstance(radius, bool) or not isinstance(radius, int) or not 1 <= radius <= 16:
            raise ValueError("foveal radius must be an integer between 1 and 16 pixels")
        self.radius = radius
        self.offsets = np.array([
            (y, x) for y in range(-radius, radius+1) for x in range(-radius, radius+1)
            if y*y+x*x <= radius*radius
        ], dtype=np.intp)
        self.visual_shape = (len(self.offsets), 1)
        self.gaze = (.5, .5)
        self.positions, self.valid = self.route(self.gaze)
        self._previous_positions = self.positions.copy()
        self._previous_valid = np.zeros(len(self.offsets), dtype=bool)
        self._candidate_indices = None
        self._candidate_samples = None
        if device is None:
            self._prediction = backend.full(shape, .5, dtype=backend.float32)
            self._correction = backend.zeros(shape, dtype=backend.float32)
        else:
            self._indices = GLArray(
                device, (len(self.offsets), 3), np.column_stack((self.positions, self.valid)),
            )
            self._samples = GLArray(device, (*self.visual_shape, 3))
            self._render_values = GLArray(device, (*self.visual_shape, 3))
            self._prediction = GLArray(device, shape, np.full(shape, .5, dtype=np.float32))
            self._correction = GLArray(device, shape)

    def route(self, gaze):
        positions, valid = self._routes([gaze])
        return positions[0], valid[0]

    def _routes(self, gazes):
        values = np.asarray(gazes, dtype=np.float64)
        if (
            values.ndim != 2 or values.shape[1] != 2 or not len(values)
            or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
        ):
            raise ValueError("gazes must contain pairs of normalized coordinates")
        centers = np.rint(values[:, ::-1]*(np.asarray(self.shape[:2])-1)).astype(np.intp)
        positions = self.offsets[None, :, :]+centers[:, None, :]
        return positions, np.all((positions >= 0) & (positions < self.shape[:2]), axis=-1)

    def set_gaze(self, gaze):
        self.gaze = tuple(gaze)
        self.positions, self.valid = self.route(gaze)
        if self.device is not None:
            self._indices.upload(np.column_stack((self.positions, self.valid)))

    def _gather(self, image, positions, valid, indices=None, output=None):
        if tuple(image.shape) != tuple(self.shape):
            raise ValueError("Shared graph camera observation must match the RGB canvas")
        if self.device is None:
            xp = self.backend
            result = xp.zeros((*valid.shape, 3), dtype=xp.float32)
            rows, cols = positions[valid].T
            result[xp.asarray(valid)] = image[xp.asarray(rows), xp.asarray(cols)]
            return result
        self.device.run(
            """
            layout(std430,binding=0) readonly buffer Image {float image[];};
            layout(std430,binding=1) readonly buffer Indices {float indices[];};
            layout(std430,binding=2) writeonly buffer Samples {float samples[];};
            uniform int count,width,offset;
            void main(){
                int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
                int p=i/3,c=i%3,y=int(indices[p*3]),x=int(indices[p*3+1]);
                samples[i]=indices[p*3+2]>0.0?image[offset+(y*width+x)*3+c]:0.0;
            }""",
            [image, indices, output], output.size, count=output.size,
            width=self.shape[1], offset=image.offset,
        )
        return output.download()

    def sample(self, image):
        return self._gather(
            image, self.positions, self.valid,
            getattr(self, "_indices", None), getattr(self, "_samples", None),
        ).reshape(*self.visual_shape, 3)

    def sample_candidates(self, image, gazes):
        positions, valid = self._routes(gazes)
        if self.device is None:
            return self._gather(image, positions, valid), valid
        values = np.concatenate((positions, valid[..., None]), axis=-1)
        if self._candidate_indices is None or self._candidate_indices.shape != values.shape:
            if self._candidate_indices is not None:
                self.device.release(self._candidate_indices, self._candidate_samples)
            self._candidate_indices = GLArray(self.device, values.shape, values)
            self._candidate_samples = GLArray(self.device, values.shape)
        else:
            self._candidate_indices.upload(values)
        return self._gather(
            image, positions, valid, self._candidate_indices, self._candidate_samples,
        ), valid

    def nearest_pixel(self, gaze):
        index = int(np.flatnonzero(np.all(self.offsets == 0, axis=1))[0])
        return (index, 0), tuple(int(v) for v in self.route(gaze)[0][index])

    def _write(self, output, values, positions, valid):
        values = values.reshape(*self.visual_shape, 3)
        if self.device is None:
            rows, cols = positions[valid].T
            output[self.backend.asarray(rows), self.backend.asarray(cols)] = self.backend.asarray(
                values, dtype=self.backend.float32,
            ).reshape(-1, 3)[self.backend.asarray(valid)]
            return output
        self._indices.upload(np.column_stack((positions, valid)))
        self._render_values.upload(values)
        self.device.run(
            """
            layout(std430,binding=0) readonly buffer Values {float values[];};
            layout(std430,binding=1) readonly buffer Indices {float indices[];};
            layout(std430,binding=2) buffer Output {float result[];};
            uniform int count,width;
            void main(){
                int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
                int p=i/3,c=i%3;if(indices[p*3+2]==0.0)return;
                int y=int(indices[p*3]),x=int(indices[p*3+1]);
                result[(y*width+x)*3+c]=values[i];
            }""",
            [self._render_values, self._indices, output],
            self._render_values.size, count=self._render_values.size, width=self.shape[1],
        )
        self._indices.upload(np.column_stack((self.positions, self.valid)))
        return output

    def scatter(self, values):
        """Sampling's exact adjoint; clear only the previous correction's support."""
        zeros = self.backend.zeros((*self.visual_shape, 3), dtype=self.backend.float32)
        self._write(self._correction, zeros, self._previous_positions, self._previous_valid)
        self._previous_positions = self.positions.copy()
        self._previous_valid = self.valid.copy()
        return self._write(self._correction, values, self.positions, self.valid)

    def render(self, values):
        return self._write(self._prediction, values, self.positions, self.valid)

    def reset(self):
        self._prediction = (
            self.backend.full(self.shape, .5, dtype=self.backend.float32)
            if self.device is None else self._prediction
        )
        if self.device is not None:
            self.device.run(
                "layout(std430,binding=0) buffer Output {float values[];};"
                "uniform int count;void main(){int i=int(gl_GlobalInvocationID.x);"
                "if(i<count)values[i]=0.5;}",
                [self._prediction], self._prediction.size, count=self._prediction.size,
            )
        self._correction.fill(0)
        self._previous_valid.fill(False)
