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
    """Bounded RGB gather and device-native interpolation around the compact graph."""

    def __init__(self, shape, visual_shape, *, backend=np, device=None):
        self.shape = shape
        self.visual_shape = visual_shape
        self.backend = backend
        self.device = device
        rows, cols = shape[:2]
        self.sample_rows = np.rint(np.linspace(0, rows-1, visual_shape[0])).astype(np.intp)
        self.sample_cols = np.rint(np.linspace(0, cols-1, visual_shape[1])).astype(np.intp)
        self._rows = backend.asarray(self.sample_rows)
        self._cols = backend.asarray(self.sample_cols)
        x = np.interp(np.arange(cols), self.sample_cols, np.arange(visual_shape[1]))
        y = np.interp(np.arange(rows), self.sample_rows, np.arange(visual_shape[0]))
        if device is None:
            self._x = backend.asarray(x, dtype=backend.float32)
            self._y = backend.asarray(y, dtype=backend.float32)
        else:
            self._indices = GLArray(device, (sum(visual_shape),), np.concatenate((self.sample_rows, self.sample_cols)))
            self._samples = GLArray(device, (*visual_shape, 3))
            self._x = GLArray(device, (cols,), x)
            self._y = GLArray(device, (rows,), y)
            self._render_values = GLArray(device, (*visual_shape, 3))
            self._prediction = GLArray(device, shape)
            self._correction = GLArray(device, shape)
            row_ports = np.full(rows, -1, dtype=np.float32)
            col_ports = np.full(cols, -1, dtype=np.float32)
            row_ports[self.sample_rows] = np.arange(visual_shape[0])
            col_ports[self.sample_cols] = np.arange(visual_shape[1])
            self._row_ports = GLArray(device, (rows,), row_ports)
            self._col_ports = GLArray(device, (cols,), col_ports)

    def sample(self, image):
        if tuple(image.shape) != tuple(self.shape):
            raise ValueError("Shared graph camera observation must match the RGB canvas")
        if self.device is None:
            return image[self._rows[:, None], self._cols[None, :]]
        self.device.run(
            """
            layout(std430,binding=0) readonly buffer Image {float image[];};
            layout(std430,binding=1) readonly buffer Indices {float indices[];};
            layout(std430,binding=2) writeonly buffer Samples {float samples[];};
            uniform int count,nr,nc,width,offset;
            void main(){
                int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
                int p=i/3,c=i%3,y=int(indices[p/nc]),x=int(indices[nr+p%nc]);
                samples[i]=image[offset+(y*width+x)*3+c];
            }""",
            [image, self._indices, self._samples], self._samples.size,
            count=self._samples.size, nr=self.visual_shape[0], nc=self.visual_shape[1],
            width=self.shape[1], offset=image.offset,
        )
        return self._samples.download()

    def visual_mask(self, precision):
        """Observe the gaze sample and sample centers inside its one-sigma fovea."""
        if not precision.sample:
            return np.zeros(self.visual_shape, dtype=bool)
        if precision.peak == 0:
            return np.ones(self.visual_shape, dtype=bool)
        x, y, radius = precision.pixel_parameters(self.shape[:2])
        distances = (
            (self.sample_rows[:, None]-y)**2 + (self.sample_cols[None, :]-x)**2
        )
        mask = distances <= radius**2
        mask.flat[distances.argmin()] = True
        return mask

    def nearest_pixel(self, gaze):
        row = int(np.abs(self.sample_rows-gaze[1]*(self.shape[0]-1)).argmin())
        col = int(np.abs(self.sample_cols-gaze[0]*(self.shape[1]-1)).argmin())
        return (row, col), (int(self.sample_rows[row]), int(self.sample_cols[col]))

    def scatter(self, values):
        """Apply the adjoint of sampling to a gradient on the canvas root ports."""
        values = values.reshape(*self.visual_shape, 3)
        if self.device is None:
            result = self.backend.zeros(self.shape, dtype=self.backend.float32)
            result[self._rows[:, None], self._cols[None, :]] = values
            return result
        self._render_values.upload(values)
        self.device.run(
            """
            layout(std430,binding=0) readonly buffer Values {float values[];};
            layout(std430,binding=1) readonly buffer Rows {float rows[];};
            layout(std430,binding=2) readonly buffer Cols {float cols[];};
            layout(std430,binding=3) writeonly buffer Output {float result[];};
            uniform int count,width,nc;
            void main(){
                int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
                int p=i/3,c=i%3,y=int(rows[p/width]),x=int(cols[p%width]);
                result[i]=(y<0||x<0)?0.0:values[(y*nc+x)*3+c];
            }""",
            [self._render_values, self._row_ports, self._col_ports, self._correction],
            self._correction.size, count=self._correction.size,
            width=self.shape[1], nc=self.visual_shape[1],
        )
        return self._correction

    def render(self, values, *, correction=False):
        values = values.reshape(*self.visual_shape, 3)
        if self.device is not None:
            self._render_values.upload(values)
            output = self._correction if correction else self._prediction
            self.device.run(
                """
                layout(std430,binding=0) readonly buffer Values {float values[];};
                layout(std430,binding=1) readonly buffer X {float xcoord[];};
                layout(std430,binding=2) readonly buffer Y {float ycoord[];};
                layout(std430,binding=3) writeonly buffer Output {float result[];};
                uniform int count,width,nr,nc;
                void main(){
                    int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
                    int p=i/3,c=i%3;
                    float x=xcoord[p%width],y=ycoord[p/width];
                    int x0=int(x),y0=int(y),x1=min(x0+1,nc-1),y1=min(y0+1,nr-1);
                    float top=mix(values[(y0*nc+x0)*3+c],values[(y0*nc+x1)*3+c],fract(x));
                    float bottom=mix(values[(y1*nc+x0)*3+c],values[(y1*nc+x1)*3+c],fract(x));
                    result[i]=mix(top,bottom,fract(y));
                }""",
                [self._render_values, self._x, self._y, output], output.size,
                count=output.size, width=self.shape[1], nr=self.visual_shape[0], nc=self.visual_shape[1],
            )
            return output
        xp = self.backend
        values = xp.asarray(values, dtype=xp.float32)
        x0, y0 = self._x.astype(xp.intp), self._y.astype(xp.intp)
        x1 = xp.minimum(x0+1, self.visual_shape[1]-1)
        y1 = xp.minimum(y0+1, self.visual_shape[0]-1)
        fx, fy = (self._x-x0)[None, :, None], (self._y-y0)[:, None, None]
        top = values[y0[:, None], x0[None, :]]*(1-fx) + values[y0[:, None], x1[None, :]]*fx
        bottom = values[y1[:, None], x0[None, :]]*(1-fx) + values[y1[:, None], x1[None, :]]*fx
        return (top*(1-fy)+bottom*fy).astype(xp.float32, copy=False)
