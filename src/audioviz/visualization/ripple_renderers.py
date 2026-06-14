from __future__ import annotations
from typing import Optional, Protocol

import matplotlib.cm as cm
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets


FIELD_VERTEX_SHADER_SOURCE = """
#version 330 core

out vec2 field_uv;

void main() {
    vec2 positions[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0)
    );
    vec2 tex_coords[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0)
    );

    vec2 position = positions[gl_VertexID];
    field_uv = tex_coords[gl_VertexID];
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


FIELD_FRAGMENT_SHADER_SOURCE = """
#version 330 core

in vec2 field_uv;
out vec4 fragment_color;

uniform sampler2D field_texture;
uniform float value_scale;

void main() {
    float wave_value = texture(field_texture, field_uv).r;
    float signed_level = atan(wave_value * value_scale) / 1.57079632679;

    vec3 negative_color = vec3(0.05, 0.18, 0.75);
    vec3 neutral_color = vec3(0.015, 0.015, 0.02);
    vec3 positive_color = vec3(1.0, 0.48, 0.06);

    vec3 color = signed_level >= 0.0
        ? mix(neutral_color, positive_color, signed_level)
        : mix(neutral_color, negative_color, -signed_level);

    fragment_color = vec4(color, 1.0);
}
"""


class RippleFieldSource(Protocol):
    def get_field_numpy(self) -> np.ndarray:
        """Return the current ripple field as a NumPy array."""


class OpenGLFieldSource(Protocol):
    def get_opengl_field_texture_id(self) -> int:
        """Return the OpenGL texture containing the current ripple field."""

    def get_opengl_field_shape(self) -> tuple[int, int]:
        """Return the field shape as (rows, columns)."""


class NumpyImageRenderer:
    """Render ripple fields through PyQtGraph from a NumPy array."""

    def __init__(
        self,
        *,
        title: str = "Ripple Simulation",
        colormap_name: str = "inferno",
    ):
        self._auto_levels_pending = True
        self.image_item = pg.ImageItem(axisOrder="row-major")
        colormap = cm.get_cmap(colormap_name)
        lookup_table = (colormap(np.linspace(0, 1, 256))[:, :3] * 255).astype(
            np.uint8
        )
        self.image_item.setLookupTable(lookup_table)

        self.plot = pg.PlotItem()
        self.plot.setTitle(title)
        self.plot.invertY(True)
        self.plot.addItem(self.image_item)

        self.widget = pg.GraphicsLayoutWidget()
        self.widget.addItem(self.plot, row=0, col=0)

        self.histogram = pg.HistogramLUTItem(image=self.image_item)
        self.histogram.gradient.loadPreset(colormap_name)
        self.widget.addItem(self.histogram, row=0, col=1)

    def prepare_frame(self) -> bool:
        return True

    def render(self, field_source: RippleFieldSource) -> None:
        field = field_source.get_field_numpy()
        if self._auto_levels_pending:
            max_abs = np.max(np.abs(field))
            if max_abs > 0:
                self.histogram.setLevels(-max_abs, max_abs)
            self._auto_levels_pending = False
        self.image_item.setImage(field, autoLevels=False)


class OpenGLFieldRenderer:
    """Render shader-owned ripple textures without NumPy readback."""

    def __init__(self, *, value_scale: float = 0.08):
        self.widget = OpenGLFieldWidget(value_scale=value_scale)

    def prepare_frame(self) -> bool:
        return self.widget.prepare_frame()

    def render(self, field_source: OpenGLFieldSource) -> None:
        self.widget.set_field_source(field_source)
        self.widget.update()


class OpenGLFieldWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, *, value_scale: float):
        super().__init__()
        surface_format = QtGui.QSurfaceFormat()
        surface_format.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
        surface_format.setVersion(4, 3)
        surface_format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        self.setFormat(surface_format)

        self.value_scale = value_scale
        self._field_source: Optional[OpenGLFieldSource] = None
        self._gl = None
        self._program = None
        self._vertex_array = None

    def prepare_frame(self) -> bool:
        if self.context() is None:
            return False
        self.makeCurrent()
        return True

    def set_field_source(self, field_source: OpenGLFieldSource) -> None:
        self._field_source = field_source

    def initializeGL(self) -> None:
        from OpenGL import GL as gl

        self._gl = gl
        self._program = self._compile_program(
            vertex_source=FIELD_VERTEX_SHADER_SOURCE,
            fragment_source=FIELD_FRAGMENT_SHADER_SOURCE,
        )
        self._vertex_array = self._create_vertex_array()

    def paintGL(self) -> None:
        gl = self._gl
        if gl is None or self._program is None:
            return

        gl.glViewport(0, 0, self.width(), self.height())
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if self._field_source is None:
            return

        texture_id = self._field_source.get_opengl_field_texture_id()
        texture_unit = 0

        gl.glUseProgram(self._program)
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glUniform1i(
            gl.glGetUniformLocation(self._program, "field_texture"),
            texture_unit,
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(self._program, "value_scale"),
            float(self.value_scale),
        )

        gl.glBindVertexArray(self._vertex_array)
        vertex_count = 4
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, vertex_count)
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _compile_program(self, *, vertex_source: str, fragment_source: str) -> int:
        gl = self._gl
        vertex_shader = self._compile_shader(gl.GL_VERTEX_SHADER, vertex_source)
        fragment_shader = self._compile_shader(gl.GL_FRAGMENT_SHADER, fragment_source)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to link field render shader:\n{log}")

        return int(program)

    def _compile_shader(self, shader_type: int, source: str) -> int:
        gl = self._gl
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)

        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            log = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to compile field render shader:\n{log}")

        return int(shader)

    def _create_vertex_array(self) -> int:
        gl = self._gl
        vertex_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vertex_array)
        gl.glBindVertexArray(0)
        return int(vertex_array)
