from __future__ import annotations

import numpy as np

from audioviz.utils.spatial_preview import preview_indices


class GLArray:
    """Explicit native storage; host access is limited to download or preview."""

    def __init__(self, device, shape, values=None):
        self.device = device
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.dtype = np.dtype("float32")
        self.offset = 0
        self._handle = None
        self._initial = values

    def __len__(self):
        return self.shape[0]

    @property
    def handle(self):
        self.device.ensure()
        if self._handle is None:
            gl = self.device.gl
            if self.size * 4 > int(gl.glGetIntegerv(gl.GL_MAX_SHADER_STORAGE_BLOCK_SIZE)):
                raise ValueError("Native array exceeds this OpenGL driver's shader-storage block limit")
            self._handle = int(gl.glGenBuffers(1))
            self.device.buffers.add(self._handle)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._handle)
            values = (
                np.zeros(self.size, dtype=np.float32) if self._initial is None
                else np.ascontiguousarray(self._initial, dtype=np.float32)
            )
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, self.size * 4, values, gl.GL_DYNAMIC_COPY)
            self._initial = None
        return self._handle

    def upload(self, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.shape != self.shape:
            raise ValueError(f"OpenGL upload must have shape {self.shape}")
        handle = self.handle
        gl = self.device.gl
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, handle)
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, self.offset * 4, values.nbytes, values)

    def download(self):
        handle = self.handle
        gl = self.device.gl
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, handle)
        data = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, self.offset * 4, self.size * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(self.shape).copy()

    def fill(self, value):
        if value != 0:
            raise ValueError("Native state reset only supports zero")
        self.device.run(
            "layout(std430,binding=0) buffer Output { float result[]; };"
            "uniform int count,offset; void main(){uint i=gl_GlobalInvocationID.x;"
            "if(i<uint(count)) result[offset+i]=0.0;}",
            [self], self.size, count=self.size, offset=self.offset,
        )

    def preview(self, *, max_side=64):
        rows, cols = preview_indices(self.shape, max_side=max_side)
        channels = self.shape[2] if self.ndim == 3 else 1
        output = GLArray(self.device, (len(rows), len(cols), channels))
        indices = GLArray(self.device, (len(rows) + len(cols),))
        indices.upload(np.concatenate((rows, cols)).astype(np.float32))
        self.device.run(
            """
            layout(std430,binding=0) readonly buffer Input { float values[]; };
            layout(std430,binding=1) readonly buffer Indices { float indices[]; };
            layout(std430,binding=2) writeonly buffer Output { float result[]; };
            uniform int nr,nc,width,channels,count,offset;
            void main(){int i=int(gl_GlobalInvocationID.x); if(i>=count)return;
            int c=i%channels,p=i/channels,y=int(indices[p/nc]),x=int(indices[nr+p%nc]);
            result[i]=values[offset+(y*width+x)*channels+c];}
            """, [self, indices, output], output.size,
            nr=len(rows), nc=len(cols), width=self.shape[1], channels=channels,
            count=output.size, offset=self.offset,
        )
        result = output.download()
        self.device.release(output, indices)
        result.setflags(write=False)
        return result


class GLArrayView(GLArray):
    def __init__(self, base, shape, offset):
        super().__init__(base.device, shape)
        self.base = base
        self.offset = offset

    @property
    def handle(self):
        return self.base.handle


class GLCompute:
    def __init__(self, propagator):
        self.propagator = propagator
        self.gl = None
        self.programs = {}
        self.buffers = set()
        self.closed = False

    def ensure(self):
        if self.closed:
            raise RuntimeError("OpenGL compute resources have been closed")
        self.propagator._ensure_initialized()
        self.gl = self.propagator._gl

    def run(self, body, arrays, work_items, *, groups_y=1, **uniforms):
        """Upload scalar uniforms with Python types matching their GLSL declarations."""
        self.ensure()
        gl = self.gl
        if body not in self.programs:
            indexed = body.replace("gl_GlobalInvocationID.x", "linear_index()").replace(
                "gl_WorkGroupID.x", "linear_group()"
            )
            self.programs[body] = self.propagator._compile_compute_program(
                """#version 430
                layout(local_size_x=256) in;
                uint linear_group(){return gl_WorkGroupID.x+gl_WorkGroupID.z*gl_NumWorkGroups.x;}
                uint linear_index(){return linear_group()*256u+gl_LocalInvocationID.x;}
                """ + indexed
            )
        program = self.programs[body]
        gl.glUseProgram(program)
        for index, array in enumerate(arrays):
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, index, array.handle)
        for name, value in uniforms.items():
            location = gl.glGetUniformLocation(program, name)
            if isinstance(value, (int, np.integer)):
                gl.glUniform1i(location, int(value))
            else:
                gl.glUniform1f(location, float(value))
        groups = (work_items + 255) // 256
        groups_x = min(groups, 65535)
        if groups_y > 65535:
            raise ValueError("Shader reduction exceeds the portable dispatch channel limit")
        self.propagator._gl_compute.dispatch_compute(groups_x, groups_y, (groups + groups_x - 1) // groups_x)
        self.propagator._gl_compute.memory_barrier(
            gl.GL_SHADER_STORAGE_BARRIER_BIT | gl.GL_BUFFER_UPDATE_BARRIER_BIT
            | gl.GL_TEXTURE_FETCH_BARRIER_BIT | gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
        )

    def reduce(self, declarations, expression, arrays, count, channels, **uniforms):
        """Hierarchical sum, with one independent reduction per output channel."""
        groups = (count + 255) // 256
        partial = GLArray(self, (groups, channels))
        body = declarations + f"""
        layout(std430,binding={len(arrays)}) writeonly buffer Output {{float result[];}};
        uniform int count,channels;
        shared float scratch[256];
        void main(){{
            int p=int(gl_GlobalInvocationID.x), c=int(gl_WorkGroupID.y);
            uint t=gl_LocalInvocationID.x;
            scratch[t]=p<count ? ({expression}) : 0.0;
            barrier();
            for(uint s=128u;s>0u;s>>=1u){{
                if(t<s)scratch[t]+=scratch[t+s]; barrier();
            }}
            if(t==0u && int(gl_WorkGroupID.x)*256<count)
                result[int(gl_WorkGroupID.x)*channels+c]=scratch[0];
        }}"""
        self.run(body, [*arrays, partial], count, groups_y=channels, count=count, channels=channels, **uniforms)
        while groups > 1:
            previous = partial
            partial = self.reduce(
                "layout(std430,binding=0) readonly buffer Input {float values[];};",
                "values[p*channels+c]", [previous], groups, channels,
            )
            self.release(previous)
            groups = 1
        return partial

    def release(self, *arrays):
        for array in arrays:
            if array._handle is not None:
                self.gl.glDeleteBuffers(1, [array._handle])
                self.buffers.remove(array._handle)
                array._handle = None

    def close(self):
        if self.closed:
            return
        if self.programs or self.buffers:
            self.ensure()
            for program in self.programs.values():
                self.gl.glDeleteProgram(program)
            if self.buffers:
                self.gl.glDeleteBuffers(len(self.buffers), list(self.buffers))
        self.programs.clear()
        self.buffers.clear()
        self.closed = True
