# Active vision: manual gaze and advisory information gain

This implements the stimulus and Stage A foundation of [the research
proposal](https://github.com/EternalGoldenBraid/AudioViz/pull/1), plus a small
advisory expected-information-gain estimator. It does not implement an
autonomous gaze policy, calibrated predictive covariance, or a learned
circular-motion predictor.

## Run

```bash
uv run python -m audioviz.visualization.active_vision_experiment --backend opengl --gaze 0.8 0.5
```

OpenGL 4.3 is required for the shader backend. `--backend numpy` and
`--backend cupy` run the same inference semantics on the existing array
backends. No camera, microphone, or video codec is required.

The top panels show evaluator ground truth with an optional information-gain
overlay, the Gaussian precision field over
that frame with a gaze cross, and the model's prediction **before the current
frame was incorporated**. The lower canvas is the existing corrected wave
surface. The model weights remain fixed by default. Controls change gaze
location and whether to observe. **Follow mouse** is enabled by default:
hover over any of the three preview images to move gaze directly. Image
coordinates account for scaling and the inverted image Y axis; moving into
the surrounding margins leaves gaze unchanged. Turn the toggle off to
freeze gaze, or position it with the numeric controls.
**Information gain** toggles the advisory overlay. Brighter cells indicate
greater expected uncertainty reduction from an observation centered there;
the white circle marks the highest-scoring candidate, and the arrow points
from current gaze toward it. The status row reports total gain in nats and
normalized target coordinates. The heatmap is scaled to its current score
range; brightness is not an absolute confidence scale. No direction is shown
when there is no expected gain. Gaze never moves automatically. Advice is
restricted to this image, not a physical-camera navigation plan.
If every candidate has equal gain, the status reports no gaze preference and
the direction cue is hidden.
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
noise. Do not give a gaze policy the evaluator's raw frames, true positions,
or raw diagnostic residuals.

## Advisory Gaussian belief

The inference objective already has a Gaussian observation-likelihood
interpretation: the predicted image is its conditional mean, and observation
covariance is `R = diag(1 / Pi)` over observed dimensions. The existing energy
statistics are not predictive covariance.

The advisory estimator adds a **model-linked, conditional Gaussian covariance
filter**. Each pixel has a diagonal hidden-state covariance `P`, stored in
float32 on the inference backend. Its transition is the derivative of the
actual uncoupled prediction step:

```text
h_next = (1 - inference_rate) * h + inference_rate * W_canvas * tanh(canvas)
F = (1 - inference_rate) * I
P_prior = F * P_posterior * F^T + Q * dt * I
J = W_visual * diag(1 - tanh(h_next)^2)
V_sensory = J * P_prior * J^T
```

The full local `3 x 3` RGB covariance is retained for information scoring;
RGB correlations induced by the decoder are not discarded. Given an
observation with precision `Pi`, Gaussian conditioning gives:

```text
P_posterior = diag(P_prior
    - Pi * P_prior * J^T * inverse(I + Pi * V_sensory) * J * P_prior)
IG(action) = 0.5 * sum_pixels(log(det(I + Pi(action) * V_sensory)))
```

Conditioning uses the decoder Jacobian frozen **before** the current evidence,
once per observation, not once per settling iteration. Hidden cross-channel
posterior correlations are discarded when retaining the diagonal. The next
forecast and its sensory covariance use the current model states and actual
decoder weights, so decoder sensitivity and saturation affect advice.
This objective targets sensory-state information, not hidden-state entropy.

`--initial-variance` defaults to `1` (identity hidden covariance);
`--process-variance` defaults to `0.01` in hidden-state variance per stimulus
second. Both are configured assumptions, not learned noise estimates. The
source orchestrator advances covariance with `dt = 1 / fps` during actual
prediction/conditioning, independently of preview rendering. Missing or
skipped observations apply the transition and process uncertainty without
conditioning. Variance can shrink under a contracting transition even without
an observation. With **Observe** off, advice still evaluates the hypothetical
next observation; the status says sampling is paused.

**Scope of the approximation:** the wave canvas and fixed weights are treated
as known point estimates. Planning holds the current canvas fixed for one
prediction step; it does not simulate a future wave trajectory. The covariance
filter is locally linearized around the existing model's mean estimates; the
predictive-coding mean solver is unchanged. This is not the exact joint
posterior of that nonlinear solver, a Hessian-based Laplace calculation, or a
calibrated confidence interval. Wave-state uncertainty, parameter uncertainty,
spatial correlations, stream-state likelihood information, and cross-modal
coupling uncertainty are not propagated in this slice. Enabling learning or
cross-modal coupling while this filter is active raises an explicit error.
Resetting readouts also resets covariance.

The experiment opts in through
`source_orchestrator.enable_visual_uncertainty(dt=1 / fps, ...)`.
Other application entrypoints keep their existing mean-only inference
behavior and incur no covariance-update cost.

This is not an independent coverage tracker: changing the decoder or hidden
mean changes the projected uncertainty. However, a wrong but saturated
predictor can still be overconfident. Calibration and richer covariance
propagation remain necessary before interpreting these scores as trustworthy
real-camera confidence.

Scores use a fixed `9 x 13` candidate grid and nearest-neighbor spatial
quadrature with maximum side 64, scaled to the full pixel count. This is a
coarse approximation: footprints narrower than the sample spacing may be
missed. NumPy/CuPy keep covariance on their own backend. OpenGL updates
covariance in GLSL and projects bounded samples through the actual decoder
before reducing candidate scores, downloading only the small score grid.
Scoring runs at preview cadence (at most 10 Hz), not per settling step.
Neither stimulus truth, seeds, raw images nor error-energy statistics enter
the information objective directly; observed information reaches it through
the model's inferred beliefs. The ground-truth overlay is evaluator-only.

This slice provides no observation-cost objective, automatic skip decision,
or autonomous temporal policy.

## Focused verification

```bash
uv run python -m pytest -q tests/sources/test_circular_dot.py tests/readouts/test_spatial_precision.py tests/readouts/test_gaussian_uncertainty.py
QT_QPA_PLATFORM=xcb uv run python -m pytest -q tests/readouts/test_opengl_predictive.py tests/visualization/test_active_vision_experiment.py
```

These cover analytic positions, seeded increments, clipping, Gaussian
geometry, normalized likelihood, inference/learning/energy weighting,
skip behavior, unchanged unit precision, and repeated shader/array parity
with audio, hidden coupling, learning, and missing evidence. Advisory checks
cover analytic Gaussian entropy reduction, frozen pre-evidence conditioning,
dense-matrix covariance references, actual decoder sensitivity, skipped
evidence, transition/process propagation, bounded quadrature scaling, native
score parity, and manual-only gaze control.
