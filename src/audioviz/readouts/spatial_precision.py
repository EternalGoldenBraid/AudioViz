from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GAUSSIAN_PRECISION_GLSL = """
float gaussian_precision(vec2 delta,float width,float baseline,float peak){
    vec2 scaled=delta/width;
    return baseline+peak*exp(-0.5*dot(scaled,scaled));
}
"""


@dataclass(frozen=True)
class GaussianSpatialPrecision:
    """Gaze in normalized image coordinates; sigma as a fraction of the shorter side."""

    gaze: tuple[float, float] = (0.5, 0.5)
    sigma: float = 0.15
    baseline: float = 0.01
    peak: float = 1.0
    sample: bool = True

    def __post_init__(self):
        if len(self.gaze) != 2 or any(
            not np.isfinite(value) or not 0 <= value <= 1 for value in self.gaze
        ):
            raise ValueError("gaze must contain two finite coordinates in [0, 1]")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if any(not np.isfinite(value) or value < 0 for value in (self.baseline, self.peak)):
            raise ValueError("baseline and peak precision must be finite and non-negative")
        if not np.isfinite(self.baseline + self.peak):
            raise ValueError("total precision must be finite")
        if not isinstance(self.sample, bool):
            raise ValueError("sample must be a boolean")

    def pixel_parameters(self, shape):
        rows, cols = shape
        if rows <= 0 or cols <= 0:
            raise ValueError("precision field requires a nonempty image")
        return self.gaze[0] * (cols - 1), self.gaze[1] * (rows - 1), self.sigma * min(rows, cols)

    def field(self, shape, *, backend=np, dtype=np.float32, indices=None):
        """Evaluate the full grid or the Cartesian product of selected row/column indices."""
        xp = backend
        x, y, width = self.pixel_parameters(shape)
        rows, cols = shape
        v = xp.arange(rows, dtype=dtype) if indices is None else xp.asarray(indices[0], dtype=dtype)
        u = xp.arange(cols, dtype=dtype) if indices is None else xp.asarray(indices[1], dtype=dtype)
        if not self.sample:
            return xp.zeros((len(v), len(u)), dtype=dtype)
        u, v = (u - x) / width, (v - y) / width
        return (
            self.baseline + self.peak * xp.exp(-0.5 * (v[:, None]**2 + u[None, :]**2))
        ).astype(dtype, copy=False)

    def negative_log_likelihood(self, observation, prediction, *, backend=np):
        """Independent Gaussian channels; zero-precision dimensions are unobserved."""
        xp = backend
        observed = xp.asarray(observation, dtype=xp.float64)
        predicted = xp.asarray(prediction, dtype=xp.float64)
        if observed.shape != predicted.shape or observed.ndim != 3 or observed.shape[-1] != 3:
            raise ValueError("observation and prediction must have matching RGB image shapes")
        if not bool(xp.all(xp.isfinite(observed))) or not bool(xp.all(xp.isfinite(predicted))):
            raise ValueError("observation and prediction must contain only finite values")
        precision = self.field(observed.shape[:2], backend=xp, dtype=xp.float64)[..., None]
        present = precision > 0
        safe = xp.where(present, precision, 1.0)
        terms = xp.where(
            present, precision * (observed - predicted)**2 + np.log(2 * np.pi) - xp.log(safe), 0.0,
        )
        return float(0.5 * terms.sum())
