# Active vision: fixed-gaze precision experiment

This implements the stimulus and Stage A foundation of [the research
proposal](https://github.com/EternalGoldenBraid/AudioViz/pull/1). It does not
implement a learned gaze policy, EIG estimator, calibrated predictive
covariance, or a learned circular-motion predictor.

## Run

```bash
uv run python -m audioviz.visualization.active_vision_experiment --backend opengl --gaze 0.8 0.5
```

OpenGL 4.3 is required for the shader backend. `--backend numpy` and
`--backend cupy` run the same inference semantics on the existing array
backends. No camera, microphone, or video codec is required.

The top panels show evaluator ground truth, the Gaussian precision field over
that frame with a gaze cross, and the model's prediction **before the current
frame was incorporated**. The lower canvas is the existing corrected wave
surface. The model weights remain fixed by default. Controls change gaze
location and whether to observe. **Follow mouse** is enabled by default:
hover over any of the three preview images to move gaze directly. Image
coordinates account for scaling and the inverted image Y axis; moving into
the surrounding margins leaves gaze unchanged. Turn the toggle off to
freeze gaze, or position it with the numeric controls.
Gaussian width and baseline/peak precision
remain fixed for the run. The stimulus stops after its configured duration.
Previews update at most 10 Hz and have a maximum side of 256 pixels.

```bash
uv run python -m audioviz.visualization.active_vision_experiment \
  --backend opengl --resolution 240 320 --fps 30 --duration 20 \
  --brownian-scale 3 --seed 7 --sigma 0.15 --baseline 0.01 --peak 1
```

The frame clock advances by `1 / fps` for every generated frame, independent
of wall-clock rendering speed. Display FPS therefore need not equal stimulus
FPS. Parameters are experimental settings, not calibrated camera-noise
estimates.

## Stimulus contract

`CircularDotSource` generates raw, read-only BGR `uint8` frames and separate
evaluator samples containing time, true pixel position, and Brownian offset.
Coordinates are `(x, y)`; resolution is `(rows, columns)`. The default center
is the center of the pixel grid.

The nominal trajectory is
`center + radius * (cos(phase + angular_velocity*t), sin(...))`.
Angular velocity is in radians/second. The first Brownian offset is zero;
subsequent independent increments have distribution
`Normal(0, brownian_scale^2 * dt * I)`, with scale in pixels/sqrt(second).
The seed controls only the evaluator's generator. Zero scale gives the exact
unperturbed trajectory. Positions can leave the image; only the rasterized
disk is clipped, and a fully off-frame dot produces a black frame.
There are `ceil(duration * fps)` samples, starting at time zero.

`CircularDotCapture` adapts this source to the existing camera `read/release`
interface, returning frames only. The generator's seed and true coordinates
are not passed to predictive inference. Ground truth and debugging previews
belong to the evaluator; they must not become inputs to a future gaze policy.
This experiment adds no changes to the production camera capture path.

## Precision contract

```python
from audioviz.readouts import GaussianSpatialPrecision, PredictiveCodingConfig

precision = GaussianSpatialPrecision(
    gaze=(0.8, 0.5), sigma=0.15, baseline=0.01, peak=1.0,
)
config = PredictiveCodingConfig(dtype="float32", spatial_precision=precision)
```

Gaze coordinates range from zero to one along each image axis, with endpoints
at the first/last pixel centers. Sigma is measured as a fraction of the
shorter image dimension, preserving a circular Gaussian in pixel space.
For pixel position `r` and gaze center `g`:

```text
Pi(r) = baseline + peak * exp(-0.5 * ||(r - g) / sigma_pixels||^2)
```

The field is shared across RGB channels. `spatial_precision=None` preserves
the existing unit-precision behavior exactly. Audio observations retain unit
precision and are not spatially gated.
Integral parameter values such as `peak=1` or `gaze=(0, 1)` are valid; the
shader adapter uploads all continuous precision parameters as GLSL floats.

- Sensory feedback and sensory decoder learning use `Pi * (observation - prediction)`.
- Sensory energy is `0.5 * Pi * error^2`, not `0.5 * (Pi * error)^2`.
- Diagnostic sensory energy means/variances follow that same weighted energy.
- Raw error arrays remain raw residuals for compatibility and evaluator inspection.
- `sample=False` sets all spatial precision to zero without stopping the stimulus,
  clearing beliefs, or turning off wave/internal dynamics. The camera's ON/OFF
  context node remains separate from the choice to sample.

NumPy/CuPy cache the field on their own array backend. OpenGL evaluates it
directly in GLSL from gaze and scalar precision parameters; it does not upload
or download a full precision image for inference. The optional CPU field in
the demo is evaluator visualization only.

`precision.negative_log_likelihood(observation, prediction)` supplies the
normalized independent-Gaussian observation likelihood:

```text
NLL = 0.5 * sum(Pi * error^2 + log(2*pi) - log(Pi))
```

This sums over observed RGB dimensions. Exactly zero precision denotes a
missing dimension, excluded from both quadratic and normalization terms;
skipping every dimension gives NLL zero. Positive precision means inverse
observation variance, not predictive belief variance. Observation likelihood
normalization matters when comparing gaze actions, even though its constant
does not affect fixed-gaze state gradients.

The Gaussian spatial envelope is not itself a Gaussian belief over future
dot positions. Brownian motion perturbs the trajectory, not the observation
noise. Before implementing EIG, define and validate the predictive belief,
measurement-noise assumptions, and observation budget consistently with this
precision-dependent likelihood. Do not give that policy the evaluator's raw
frames, true positions, or raw diagnostic residuals.

## Focused verification

```bash
uv run python -m pytest -q tests/sources/test_circular_dot.py tests/readouts/test_spatial_precision.py
QT_QPA_PLATFORM=xcb uv run python -m pytest -q tests/readouts/test_opengl_predictive.py tests/visualization/test_active_vision_experiment.py
```

These cover analytic positions, seeded increments, clipping, Gaussian
geometry, normalized likelihood, inference/learning/energy weighting,
skip behavior, unchanged unit precision, and repeated shader/array parity
with audio, hidden coupling, learning, and missing evidence.
