from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np

from audioviz.physics.wave_propagator import BoundaryCondition, coerce_boundary_condition


COMPUTE_SHADER_SOURCE = """
#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(FIELD_FORMAT, binding = 0) uniform readonly image2D z_current;
layout(FIELD_FORMAT, binding = 1) uniform readonly image2D z_old;
layout(FIELD_FORMAT, binding = 2) uniform readonly image2D excitation;
layout(FIELD_FORMAT, binding = 3) uniform writeonly image2D z_next;
layout(FIELD_FORMAT, binding = 4) uniform writeonly image2D z_current_after_excitation;

uniform ivec2 grid_shape;
uniform float c2_dt2;
uniform float damping;
uniform int boundary_condition;

const int BOUNDARY_CONDITION_CYCLIC = 0;
const int BOUNDARY_CONDITION_NEUMANN = 1;

vec4 current_with_excitation(ivec2 p) {
    return imageLoad(z_current, p) + imageLoad(excitation, p);
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int width = grid_shape.x;
    int height = grid_shape.y;

    if (p.x >= width || p.y >= height) {
        return;
    }

    vec4 center = current_with_excitation(p);
    vec4 neighbor_sum = vec4(0.0);
    float degree = 0.0;

    if (boundary_condition == BOUNDARY_CONDITION_CYCLIC) {
        ivec2 left = ivec2((p.x + width - 1) % width, p.y);
        ivec2 right = ivec2((p.x + 1) % width, p.y);
        ivec2 up = ivec2(p.x, (p.y + height - 1) % height);
        ivec2 down = ivec2(p.x, (p.y + 1) % height);
        neighbor_sum =
            current_with_excitation(left)
            + current_with_excitation(right)
            + current_with_excitation(up)
            + current_with_excitation(down);
        degree = 4.0;
    } else {
        if (p.x > 0) {
            neighbor_sum += current_with_excitation(ivec2(p.x - 1, p.y));
            degree += 1.0;
        }
        if (p.x + 1 < width) {
            neighbor_sum += current_with_excitation(ivec2(p.x + 1, p.y));
            degree += 1.0;
        }
        if (p.y > 0) {
            neighbor_sum += current_with_excitation(ivec2(p.x, p.y - 1));
            degree += 1.0;
        }
        if (p.y + 1 < height) {
            neighbor_sum += current_with_excitation(ivec2(p.x, p.y + 1));
            degree += 1.0;
        }
    }

    vec4 laplacian = neighbor_sum - degree * center;

    vec4 next = 2.0 * center - imageLoad(z_old, p) + c2_dt2 * laplacian;
    next *= damping;

    // This pass is synchronous in time: all nodes read t and write t+1.
    // Future Loihi-like neuromorphic backends could use event-driven,
    // asynchronous local updates when activity crosses a threshold.
    imageStore(z_current_after_excitation, p, center);
    imageStore(z_next, p, next);
}
"""


class WavePropagatorOpenGL:
    """OpenGL compute-shader implementation of the 2D wave update.

    The public interface intentionally matches the CPU/CuPy propagators. It
    uses ping-pong textures internally, exposes native inference storage and
    downloads state only through the explicit get_state boundary.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        dx: float,
        dt: float,
        speed: float,
        damping: float,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.CYCLIC,
        use_current_context: bool = False,
    ):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping
        self.c2_dt2 = (self.c * self.dt / self.dx) ** 2
        self.boundary_condition = coerce_boundary_condition(boundary_condition)
        self.use_current_context = use_current_context

        self._context = None
        self._surface = None
        self._gl = None
        self._gl_compute = None

        self._rows, self._cols = self.shape[:2]
        self.channels = self.shape[2] if len(self.shape) == 3 else 1
        if self.channels not in (1, 3):
            raise ValueError("OpenGL wave state requires one or three channels")
        self._format_name = "rgba32f" if self.channels == 3 else "r32f"
        self._owner_context = None
        self._closed = False
        from audioviz.utils.opengl_compute import GLCompute

        self.compute = GLCompute(self)
        self._native_current = None
        self._native_prior = None
        self._pending_excitation = np.zeros(self.shape, dtype=np.float32)
        self._pending_has_excitation = False
        self._excitation_is_zero = True
        self._readback = np.zeros(self.shape, dtype=np.float32)

        self._program = None
        self._textures = []
        self._tex_old = None
        self._tex_current = None
        self._tex_next = None
        self._tex_current_after = None
        self._tex_excitation = None

        if not self.use_current_context:
            self._ensure_initialized()

    def add_excitation(self, excitation: np.ndarray) -> None:
        assert excitation.shape == self.shape
        self._pending_excitation += np.asarray(excitation, dtype=np.float32)
        self._pending_has_excitation = True

    def step(self) -> None:
        self._ensure_initialized()
        gl = self._gl
        self._prepare_excitation()

        gl.glUseProgram(self._program)
        gl.glUniform2i(
            gl.glGetUniformLocation(self._program, "grid_shape"),
            self._cols,
            self._rows,
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(self._program, "c2_dt2"),
            float(self.c2_dt2),
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(self._program, "damping"),
            float(self.damping),
        )
        gl.glUniform1i(
            gl.glGetUniformLocation(self._program, "boundary_condition"),
            0 if self.boundary_condition is BoundaryCondition.CYCLIC else 1,
        )

        self._bind_image(0, self._tex_current, gl.GL_READ_ONLY)
        self._bind_image(1, self._tex_old, gl.GL_READ_ONLY)
        self._bind_image(2, self._tex_excitation, gl.GL_READ_ONLY)
        self._bind_image(3, self._tex_next, gl.GL_WRITE_ONLY)
        self._bind_image(4, self._tex_current_after, gl.GL_WRITE_ONLY)

        groups_x = (self._cols + 15) // 16
        groups_y = (self._rows + 15) // 16
        groups_z = 1
        self._gl_compute.dispatch_compute(groups_x, groups_y, groups_z)
        self._gl_compute.memory_barrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        (
            self._tex_old,
            self._tex_current,
            self._tex_next,
            self._tex_current_after,
        ) = (
            self._tex_current_after,
            self._tex_next,
            self._tex_old,
            self._tex_current,
        )
        self._copy_to_native()

    def _prepare_excitation(self):
        gl = self._gl
        if self._pending_has_excitation:
            self._upload_texture(self._tex_excitation, self._pending_excitation)
            self._pending_excitation.fill(0.0)
            self._pending_has_excitation = False
            self._excitation_is_zero = False
        elif not self._excitation_is_zero:
            self._bind_image(0, self._tex_excitation, gl.GL_WRITE_ONLY)
            self.compute.run(
                f"""layout({self._format_name},binding=0) uniform writeonly image2D excitation;
                uniform int width,count;
                void main(){{int p=int(gl_GlobalInvocationID.x);if(p<count)
                imageStore(excitation,ivec2(p%width,p/width),vec4(0.0));}}""",
                [], self._rows * self._cols, width=self._cols, count=self._rows * self._cols,
            )
            self._excitation_is_zero = True

    def hold(self, excitation=None):
        """Keep the corrected field stationary, with zero wave velocity."""
        self._ensure_initialized()
        if excitation is not None:
            self.add_excitation(excitation)
        self._prepare_excitation()
        gl = self._gl
        self._bind_image(0, self._tex_current, gl.GL_READ_WRITE)
        self._bind_image(1, self._tex_old, gl.GL_WRITE_ONLY)
        self._bind_image(2, self._tex_excitation, gl.GL_READ_ONLY)
        self.compute.run(
            f"""
            layout({self._format_name},binding=0) uniform image2D current;
            layout({self._format_name},binding=1) uniform writeonly image2D previous;
            layout({self._format_name},binding=2) uniform readonly image2D excitation;
            uniform int width,count;
            void main(){{int p=int(gl_GlobalInvocationID.x);if(p>=count)return;
            ivec2 xy=ivec2(p%width,p/width);
            vec4 value=imageLoad(current,xy)+imageLoad(excitation,xy);
            imageStore(current,xy,value);imageStore(previous,xy,value);}}
            """, [], self._rows*self._cols, width=self._cols, count=self._rows*self._cols,
        )
        self._copy_to_native()

    def _copy_to_native(self):
        from audioviz.utils.opengl_compute import GLArray

        if self._native_current is None:
            self._native_current = GLArray(self.compute, (*self.shape[:2], self.channels))
            self._native_prior = GLArray(self.compute, self._native_current.shape)
        self._bind_image(0, self._tex_current, self._gl.GL_READ_ONLY)
        self.compute.run(
            f"""
            layout({self._format_name},binding=0) uniform readonly image2D field;
            layout(std430,binding=0) writeonly buffer Current {{float current[];}};
            layout(std430,binding=1) writeonly buffer Prior {{float prior[];}};
            uniform int width,channels,count;
            void main(){{int i=int(gl_GlobalInvocationID.x);if(i>=count)return;
            int p=i/channels;float v=imageLoad(field,ivec2(p%width,p/width))[i%channels];
            current[i]=v;prior[i]=v;}}
            """, [self._native_current, self._native_prior],
            self._native_current.size, width=self._cols, channels=self.channels,
            count=self._native_current.size,
        )

    def get_native_state(self, *, prior=False):
        self._ensure_initialized()
        if self._native_current is None:
            self._copy_to_native()
        return self._native_prior if prior else self._native_current

    def apply_correction(self, correction):
        from audioviz.utils.opengl_compute import GLArray

        if not isinstance(correction, GLArray) or correction.device is not self.compute:
            raise ValueError("OpenGL correction requires native storage from its own engine")
        gl = self._gl
        if correction.shape != (*self.shape[:2], self.channels):
            raise ValueError("OpenGL correction must match canvas shape")
        self._bind_image(0, self._tex_current, gl.GL_READ_WRITE)
        self._bind_image(1, self._tex_old, gl.GL_READ_WRITE)
        self.compute.run(
            f"""
            layout({self._format_name},binding=0) uniform image2D current;
            layout({self._format_name},binding=1) uniform image2D previous;
            layout(std430,binding=0) readonly buffer Delta {{float delta[];}};
            layout(std430,binding=1) writeonly buffer State {{float state[];}};
            uniform int width,channels,count;
            void main(){{int p=int(gl_GlobalInvocationID.x);if(p>=count)return;
            ivec2 xy=ivec2(p%width,p/width);
            vec4 change=vec4(0.0);
            for(int c=0;c<channels;++c)change[c]=delta[p*channels+c];
            vec4 value=imageLoad(current,xy)+change;
            imageStore(current,xy,value);
            imageStore(previous,xy,imageLoad(previous,xy)+change);
            for(int c=0;c<channels;++c)state[p*channels+c]=value[c];}}
            """, [correction, self.get_native_state()], self._rows*self._cols,
            width=self._cols, channels=self.channels, count=self._rows*self._cols,
        )

    def get_state(self) -> np.ndarray:
        self._ensure_initialized()
        self._readback[:] = self._download_texture(self._tex_current)
        return self._readback

    def reset(self) -> None:
        self._ensure_initialized()
        zeros = np.zeros(self.shape, dtype=np.float32)
        for texture in self._textures:
            self._upload_texture(texture, zeros)
        self._pending_excitation.fill(0.0)
        self._pending_has_excitation = False
        self._excitation_is_zero = True
        self._readback.fill(0.0)
        if self._native_current is not None:
            self._native_current.fill(0)
            self._native_prior.fill(0)

    def get_current_texture_id(self) -> int:
        self._ensure_initialized()
        return int(self._tex_current)

    def get_texture_shape(self) -> tuple[int, int]:
        return self.shape

    def _ensure_initialized(self) -> None:
        if self._closed:
            raise RuntimeError("OpenGL propagator has been closed")
        if self._program is not None:
            self._make_context_current()
            return

        self._gl = self._load_gl_with_context()
        from PyQt5 import QtGui

        self._owner_context = QtGui.QOpenGLContext.currentContext()
        version = self._owner_context.format()
        if (version.majorVersion(), version.minorVersion()) < (4, 3):
            raise RuntimeError("Shader inference requires OpenGL 4.3 or newer")
        self._gl_compute = self._load_compute_functions()
        self._program = self._compile_compute_program(
            COMPUTE_SHADER_SOURCE.replace("FIELD_FORMAT", self._format_name)
        )
        self._textures = [self._create_float_texture() for _ in range(5)]
        self._tex_old = self._textures[0]
        self._tex_current = self._textures[1]
        self._tex_next = self._textures[2]
        self._tex_current_after = self._textures[3]
        self._tex_excitation = self._textures[4]

    def _load_gl_with_context(self):
        self._make_context_current()
        try:
            from OpenGL import GL as gl
        except ImportError as exc:
            raise RuntimeError(
                "OpenGL shader ripple rendering requires PyOpenGL."
            ) from exc
        return gl

    def _load_compute_functions(self):
        from OpenGL import platform

        def load_function(name: bytes, signature):
            pointer = platform.PLATFORM.getExtensionProcedure(name)
            if not pointer:
                function_name = name.decode("ascii")
                raise RuntimeError(f"OpenGL driver does not expose {function_name}.")
            return signature(pointer)

        dispatch_compute_signature = ctypes.CFUNCTYPE(
            None,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
        )
        memory_barrier_signature = ctypes.CFUNCTYPE(None, ctypes.c_uint)
        bind_image_texture_signature = ctypes.CFUNCTYPE(
            None,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
        )

        return SimpleNamespace(
            dispatch_compute=load_function(
                b"glDispatchCompute",
                dispatch_compute_signature,
            ),
            memory_barrier=load_function(
                b"glMemoryBarrier",
                memory_barrier_signature,
            ),
            bind_image_texture=load_function(
                b"glBindImageTexture",
                bind_image_texture_signature,
            ),
        )

    def _make_context_current(self) -> None:
        try:
            from PyQt5 import QtGui, QtWidgets
        except ImportError as exc:
            raise RuntimeError(
                "OpenGL shader ripple rendering requires PyQt5."
            ) from exc

        if self.use_current_context:
            current = QtGui.QOpenGLContext.currentContext()
            if current is None or (self._owner_context is not None and current is not self._owner_context):
                raise RuntimeError(
                    "OpenGL resources require their renderer-owned context to be current"
                )
            return

        if self._owner_context is not None and QtGui.QOpenGLContext.currentContext() is self._owner_context:
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            raise RuntimeError(
                "OpenGL shader ripple rendering requires a QApplication before "
                "constructing RippleEngine(use_shader=True)."
            )

        if self._context is None:
            surface_format = QtGui.QSurfaceFormat()
            surface_format.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
            surface_format.setVersion(4, 3)
            surface_format.setProfile(QtGui.QSurfaceFormat.CoreProfile)

            self._surface = QtGui.QOffscreenSurface()
            self._surface.setFormat(surface_format)
            self._surface.create()

            self._context = QtGui.QOpenGLContext()
            self._context.setFormat(surface_format)
            if not self._context.create():
                raise RuntimeError("Failed to create an OpenGL 4.3 context.")

        if not self._context.makeCurrent(self._surface):
            raise RuntimeError("Failed to make the OpenGL context current.")

    def _compile_compute_program(self, source: str) -> int:
        gl = self._gl
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)

        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            log = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            gl.glDeleteShader(shader)
            raise RuntimeError(f"Failed to compile OpenGL compute shader:\n{log}")

        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        gl.glDeleteShader(shader)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            gl.glDeleteProgram(program)
            raise RuntimeError(f"Failed to link OpenGL compute shader:\n{log}")

        return program

    def _create_float_texture(self) -> int:
        gl = self._gl
        target = gl.GL_TEXTURE_2D
        internal_format = gl.GL_RGBA32F if self.channels == 3 else gl.GL_R32F
        source_format = gl.GL_RGBA if self.channels == 3 else gl.GL_RED
        source_type = gl.GL_FLOAT
        mip_level = 0
        border = 0
        width = self._cols
        height = self._rows

        texture = gl.glGenTextures(1)
        gl.glBindTexture(target, texture)
        gl.glTexParameteri(target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            target,
            mip_level,
            internal_format,
            width,
            height,
            border,
            source_format,
            source_type,
            np.zeros(
                (self._rows, self._cols, 4) if self.channels == 3 else self.shape,
                dtype=np.float32,
            ),
        )
        gl.glBindTexture(target, 0)
        return int(texture)

    def _upload_texture(self, texture: int, data: np.ndarray) -> None:
        gl = self._gl
        if self.channels == 3:
            packed = np.zeros((*self.shape[:2], 4), dtype=np.float32)
            packed[..., :3] = data
            data = packed
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._cols,
            self._rows,
            gl.GL_RGBA if self.channels == 3 else gl.GL_RED,
            gl.GL_FLOAT,
            np.ascontiguousarray(data, dtype=np.float32),
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _download_texture(self, texture: int) -> np.ndarray:
        gl = self._gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        data = gl.glGetTexImage(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA if self.channels == 3 else gl.GL_RED, gl.GL_FLOAT
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        if self.channels == 3:
            return np.frombuffer(data, dtype=np.float32).reshape(*self.shape[:2], 4)[..., :3]
        return np.frombuffer(data, dtype=np.float32).reshape(self.shape)

    def _bind_image(self, unit: int, texture: int, access: int) -> None:
        mip_level = 0
        layered = self._gl.GL_FALSE
        layer = 0
        image_format = self._gl.GL_RGBA32F if self.channels == 3 else self._gl.GL_R32F
        self._gl_compute.bind_image_texture(
            unit,
            texture,
            mip_level,
            layered,
            layer,
            access,
            image_format,
        )

    def close(self):
        if self._closed:
            return
        if self._program is not None:
            self._make_context_current()
            self.compute.close()
            self._gl.glDeleteProgram(self._program)
            self._gl.glDeleteTextures(self._textures)
        self._closed = True
