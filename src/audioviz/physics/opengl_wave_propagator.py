from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np


COMPUTE_SHADER_SOURCE = """
#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D z_current;
layout(r32f, binding = 1) uniform readonly image2D z_old;
layout(r32f, binding = 2) uniform readonly image2D excitation;
layout(r32f, binding = 3) uniform writeonly image2D z_next;
layout(r32f, binding = 4) uniform writeonly image2D z_current_after_excitation;

uniform ivec2 grid_shape;
uniform float c2_dt2;
uniform float damping;

float current_with_excitation(ivec2 p) {
    return imageLoad(z_current, p).r + imageLoad(excitation, p).r;
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int width = grid_shape.x;
    int height = grid_shape.y;

    if (p.x >= width || p.y >= height) {
        return;
    }

    ivec2 left = ivec2((p.x + width - 1) % width, p.y);
    ivec2 right = ivec2((p.x + 1) % width, p.y);
    ivec2 up = ivec2(p.x, (p.y + height - 1) % height);
    ivec2 down = ivec2(p.x, (p.y + 1) % height);

    float center = current_with_excitation(p);
    float laplacian =
        -4.0 * center
        + current_with_excitation(left)
        + current_with_excitation(right)
        + current_with_excitation(up)
        + current_with_excitation(down);

    float next = 2.0 * center - imageLoad(z_old, p).r + c2_dt2 * laplacian;
    next *= damping;

    // This pass is synchronous in time: all nodes read t and write t+1.
    // Future Loihi-like neuromorphic backends could use event-driven,
    // asynchronous local updates when activity crosses a threshold.
    imageStore(z_current_after_excitation, p, vec4(center, 0.0, 0.0, 1.0));
    imageStore(z_next, p, vec4(next, 0.0, 0.0, 1.0));
}
"""


class WavePropagatorOpenGL:
    """OpenGL compute-shader implementation of the 2D wave update.

    The public interface intentionally matches the CPU/CuPy propagators. It
    uses ping-pong textures internally and reads the current field back into a
    NumPy array for the existing PyQtGraph visualizer.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        dx: float,
        dt: float,
        speed: float,
        damping: float,
        use_current_context: bool = False,
    ):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping
        self.c2_dt2 = (self.c * self.dt / self.dx) ** 2
        self.use_current_context = use_current_context

        self._context = None
        self._surface = None
        self._gl = None
        self._gl_compute = None

        self._rows, self._cols = self.shape
        self._pending_excitation = np.zeros(self.shape, dtype=np.float32)
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

    def step(self) -> None:
        self._ensure_initialized()
        gl = self._gl

        self._upload_texture(self._tex_excitation, self._pending_excitation)
        self._pending_excitation.fill(0.0)

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
        self._readback.fill(0.0)

    def get_current_texture_id(self) -> int:
        self._ensure_initialized()
        return int(self._tex_current)

    def get_texture_shape(self) -> tuple[int, int]:
        return self.shape

    def _ensure_initialized(self) -> None:
        if self._program is not None:
            self._make_context_current()
            return

        self._gl = self._load_gl_with_context()
        self._gl_compute = self._load_compute_functions()
        self._program = self._compile_compute_program(COMPUTE_SHADER_SOURCE)
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

        if QtGui.QOpenGLContext.currentContext() is not None:
            return

        if self.use_current_context:
            raise RuntimeError(
                "OpenGL shader ripple rendering requires the renderer-owned "
                "OpenGL context to be current before stepping the engine."
            )

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
            raise RuntimeError(f"Failed to compile wave compute shader:\n{log}")

        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        gl.glDeleteShader(shader)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to link wave compute shader:\n{log}")

        return program

    def _create_float_texture(self) -> int:
        gl = self._gl
        target = gl.GL_TEXTURE_2D
        internal_format = gl.GL_R32F
        source_format = gl.GL_RED
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
            None,
        )
        gl.glBindTexture(target, 0)
        return int(texture)

    def _upload_texture(self, texture: int, data: np.ndarray) -> None:
        gl = self._gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._cols,
            self._rows,
            gl.GL_RED,
            gl.GL_FLOAT,
            np.ascontiguousarray(data, dtype=np.float32),
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _download_texture(self, texture: int) -> np.ndarray:
        gl = self._gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RED, gl.GL_FLOAT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return np.frombuffer(data, dtype=np.float32).reshape(self.shape)

    def _bind_image(self, unit: int, texture: int, access: int) -> None:
        mip_level = 0
        layered = self._gl.GL_FALSE
        layer = 0
        image_format = self._gl.GL_R32F
        self._gl_compute.bind_image_texture(
            unit,
            texture,
            mip_level,
            layered,
            layer,
            access,
            image_format,
        )
