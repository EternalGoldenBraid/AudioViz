# Active vision: gaze modes, advisory information gain, and headless evaluation

This implements the stimulus and Stage A foundation of [the research
proposal](https://github.com/EternalGoldenBraid/AudioViz/pull/1), an
expected-information-gain estimator, three mutually exclusive experiment gaze
modes (`mouse`, `autonomous`, and `full`), and a headless policy comparison
runner. It does not implement calibrated predictive covariance, a learned
circular-motion predictor, or a physical-camera navigation policy.

## Mathematics diagram and source map

Open [the editable mathematics diagram](predictive-uncertainty-math.drawio)
in the VS Code draw.io extension or draw.io desktop. Its three pages cover
terms/distributions, forecast/conditioning, and the sensory information
objective. The diagram describes the implementation at code snapshot
`4ada271`, including its approximation boundaries.

Each blue code label is a clickable `vscode://file/...:line:column` link into
this checkout at `/home/nicklas/Projects/AudioViz`. These local links work
without publishing the implementation to GitHub; if the checkout moves,
replace that root in the diagram. The repository-relative source links below
are the portable index. Function names remain useful if line numbers move.

| Mathematical responsibility | Implementation |
| --- | --- |
| Existing hidden-mean prediction | [`PredictiveCodingRGBReadout.predict`](../src/audioviz/readouts/predictive_coding.py#L139-L145) |
| Predicted sensory mean | [`_capture_prediction`](../src/audioviz/readouts/predictive_coding.py#L147-L156), [`_predict_observation`](../src/audioviz/readouts/predictive_coding.py#L332-L333) |
| Equivalent native mean transition | [`OpenGLPredictiveModel._advance`](../src/audioviz/readouts/opengl_predictive.py#L271-L299) |
| Hidden covariance initialization and assumptions | [`GaussianSensoryUncertainty.__init__`](../src/audioviz/readouts/gaussian_uncertainty.py#L21-L51) |
| Forecast covariance and freeze pre-evidence mean | [`GaussianSensoryUncertainty.predict`](../src/audioviz/readouts/gaussian_uncertainty.py#L78-L99) |
| Decoder Jacobian and RGB covariance components | [`_jacobian` / `_components`](../src/audioviz/readouts/gaussian_uncertainty.py#L101-L109) |
| Gaussian conditioning and hidden-diagonal projection | [`GaussianSensoryUncertainty.condition`](../src/audioviz/readouts/gaussian_uncertainty.py#L111-L159) |
| Gaze-conditioned scalar precision | [`GaussianSpatialPrecision.field`](../src/audioviz/readouts/spatial_precision.py#L40-L58) |
| Normalized Gaussian measurement likelihood | [`negative_log_likelihood`](../src/audioviz/readouts/spatial_precision.py#L60-L75) |
| Planning forecast, RGB log-determinant, and quadrature | [`expected_information_gain`](../src/audioviz/readouts/gaussian_uncertainty.py#L179-L282) |
| Native moments and candidate-score reduction | [GLSL inside `expected_information_gain`](../src/audioviz/readouts/gaussian_uncertainty.py#L235-L279) |
| Greedy tie/flat-score handling and policy selection | [`has_gaze_preference`](../src/audioviz/sources/gaze.py#L6-L7), [`information_gaze`](../src/audioviz/sources/gaze.py#L10-L23), [`GazePolicy.select`](../src/audioviz/sources/gaze.py#L37-L50) |
| Prediction, pre-capture gaze selection, and conditioning lifecycle | [`predict_visual`](../src/audioviz/sources/orchestrator.py#L175-L181), [`_select_visual_gaze`](../src/audioviz/sources/orchestrator.py#L183-L197), [`correct_from_observations`](../src/audioviz/sources/orchestrator.py#L242-L248) |
| Gaze-mode controls, responsive preview reflow, advisory cue, hypothetical sampling, and finite-run restart | [`resizeEvent`](../src/audioviz/visualization/active_vision_experiment.py#L196-L208), [`_change_gaze_mode` / `_update_precision`](../src/audioviz/visualization/active_vision_experiment.py#L210-L234), [`_update_advisory_marker`](../src/audioviz/visualization/active_vision_experiment.py#L252-L299), [`_after_frame` / `refresh_previews` / `_restart`](../src/audioviz/visualization/active_vision_experiment.py#L318-L386) |
| Headless fixed/random/autonomous evaluation and metrics | [`GazeFrameMetrics` / `GazeRunResult`](../src/audioviz/offline/active_vision.py#L22-L38), [`run_gaze_comparison`](../src/audioviz/offline/active_vision.py#L75-L162) |
| Dense Gaussian mathematical reference | [`test_model_transition_and_frozen_decoder_match_dense_gaussian_reference`](../tests/readouts/test_gaussian_uncertainty.py#L47-L79) |

In the diagram, `lambda` is the scalar precision field called `Pi(r)` below;
the RGB precision matrix is `lambda * I_3`. `m` is the hidden point estimate
stored in `readout.hidden`, while `h` denotes the uncertain hidden variable.
`v = diag(P)` extracts the stored variance vector; `P = Diag(v)` constructs
the conceptual diagonal covariance matrix. These are different operations.

## Run

```bash
uv run python -m audioviz.visualization.active_vision_experiment --backend opengl --gaze-mode mouse --gaze 0.8 0.5
```

OpenGL 4.3 is required for the shader backend. `--backend numpy` and
`--backend cupy` run the same inference semantics on the existing array
backends. No camera, microphone, or video codec is required.

![Active vision experimentation bench](active-vision-bench.png)

The top panels show evaluator ground truth with an optional information-gain
overlay, the Gaussian precision field over
that frame with a gaze cross, and the model's prediction **before the current
frame was incorporated** (aligned in three columns on wide windows `>= 1000 px`
and stacked vertically on narrower windows). The lower canvas is the existing corrected wave
surface. The model weights remain fixed by default. Controls select one of
three mutually exclusive gaze modes and whether to observe (`Observe` remains
an independent gate across all modes):

- **Mouse gaze (`--gaze-mode mouse`, default):** hover over any of the three
  preview images while **Mouse gaze** is selected, or position gaze with the
  numeric controls. Image coordinates account for
  scaling and the inverted image Y axis; moving into the surrounding margins
  leaves gaze unchanged.
- **Autonomous gaze (`--gaze-mode autonomous`):** selects the greedy maximum of
  the expected-information-gain candidate grid at model-step cadence in
  `RippleSourceOrchestrator.predict_visual` before each camera frame is
  captured (not throttled to the 10 Hz preview timer). When `Observe` is off,
  or when all candidates tie / have zero expected gain (within relative
  `1e-6` float32 tolerance), autonomous gaze remains frozen at its current
  position; non-flat ties keep the current gaze if it is among the maximizers
  and otherwise use the first near-maximizer in row-major order. The preview
  arrow uses this same selector. This mode is restricted to the synthetic
  experiment and never reads evaluator ground truth or the unsampled current
  frame.
- **Full canvas (`--gaze-mode full`):** disables spatial gating and applies
  uniform unit precision (`Pi(r) = 1`, `R(r) = I_3`) across the entire frame
  whenever `Observe` is on; switching back to `mouse` or `autonomous` restores
  the Gaussian profile. This is distinct from turning `Observe` off
  (`Pi(r) = 0`, no observation update).

**Information gain** toggles the overlay. Brighter cells indicate
greater expected uncertainty reduction from an observation centered there;
the white circle marks the highest-scoring candidate, and the arrow points
from current gaze toward it. The status row reports total gain in nats and
normalized target coordinates. The heatmap is scaled to its current score
range; brightness is not an absolute confidence scale. No direction is shown
when there is no expected gain or when all candidates tie. Advice and
autonomous selection are restricted to this synthetic image grid, not a
physical-camera navigation plan.
Gaussian width and baseline/peak precision
remain fixed for the run. The stimulus stops after its configured duration
(`--duration`, default `10` s), stopping both the model and preview timers and
showing `Finished: <N> frames - Restart to continue` (during the run, the
status shows `Stimulus time: <elapsed> / <duration> s`). Switching gaze modes
after a run finishes does not automatically restart timers or reset history;
use **Restart** (`ActiveVisionExperiment._restart`) to replay the seeded
stimulus and reset wave/readout/covariance beliefs while preserving the
selected gaze mode, gaze coordinates, and `Observe` state. Previews update at
most 10 Hz and have a maximum side of 256 pixels (the committed
`active-vision-bench.png` screenshot reflects the `b427020` UI snapshot).

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

**Deterministic stimulus pixels do not imply `R = I_3`:** a deterministic
synthetic frame is a noiseless scene realization (zero empirical sampling
jitter across repeats of the same seed); taken literally as a noise distribution
that would approach a degenerate `R -> 0` where Gaussian log-likelihood is
singular. Instead, `R_a(r) = (1 / Pi_a(r)) * I_3` is the **assigned
observation-likelihood covariance** that weights prediction errors in the
variational objective and sets the signal-to-noise ratio in information
scoring:

- In spatial gaze modes (`mouse`, `autonomous`), `Pi_a(r)` varies by pixel so
  `R_a(r) = (1 / Pi_a(r)) * I_3` on observed dimensions (`Pi_a(r) > 0`).
- In `full` canvas mode (`spatial_precision=None` or uniform unit precision),
  `Pi(r) = 1` assigns `R(r) = I_3` everywhere by unit-precision convention,
  not because deterministic pixels inherently have unit variance.
- When `Observe` is off (`sample=False`), `Pi(r) = 0` everywhere (unobserved;
  not inverted, and posterior covariance remains `P_posterior = P_prior`).

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
m_next = (1 - inference_rate) * m + inference_rate * W_canvas * tanh(canvas)
F = (1 - inference_rate) * I
P_prior = F * P_posterior * F^T + q * dt * I
J = W_visual * Diag(1 - tanh(m_next)^2)
V_sensory = J * P_prior * J^T
```

The full local `3 x 3` RGB covariance is retained for information scoring;
RGB correlations induced by the decoder are not discarded. Given an
observation with precision `Pi`, Gaussian conditioning gives:

```text
C = P_prior - Pi * P_prior * J^T * inverse(I + Pi * V_sensory) * J * P_prior
v_posterior = diag(C)
P_posterior = Diag(v_posterior)
IG(action) = 0.5 * sum_pixels(log(det(I + Pi(action) * V_sensory)))
```

Here `q` is the scalar `process_variance`. The information-gain expression
uses ideal conditioning of the local Gaussian **before** discarding hidden
correlations; it is not the entropy difference of the diagonally projected
stored beliefs. The log-determinant form also supports rank-deficient
positive-semidefinite sensory covariance.

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

**Mid-run stationary autonomous gaze—quadrature ranking error vs. conditional-model contraction:**
during a live `1200`-frame (`40` s) native OpenGL run with stimulus time still
advancing, autonomous gaze moves across early frames and then stops changing
after frame `38` (`~1.27` s), remaining at `(0.5, 0.5)` without numerical
divergence or stopped timers. Investigation separates two distinct mechanisms
that must not be conflated:

1. **Immediate cause of the frame-38 freeze (quadrature ranking error):**
   because the wave canvas is treated as a known point estimate (`Var(c) = 0`),
   the uncoupled hidden transition contracts variance by `(1 - inference_rate)^2`
   each frame even on unobserved pixels, approaching the no-evidence floor
   `P_inf = (q * dt) / (1 - (1 - inference_rate)^2) ≈ 0.001754` (at default
   `inference_rate = 0.1`, `q = 0.01`, `dt = 1 / 30`), while border clipping on
   `120 x 160` gives higher scalar precision mass at the center (`~2225.99`)
   than at a corner (`~723.75`). However, `P_inf` and border clipping alone do
   **not** freeze gaze among interior candidates whose unclipped masses are
   nearly tied: reusing unweighted endpoint-inclusive `preview_indices`
   (`48 x 64`, `max_side = 64`) introduces integer-step and boundary-cell
   quadrature bias larger than the `O(1e-3)` interior variance-depletion margin
   (and produces a `6–8%` corner preference that locks corners at frames `30–31`
   in equal-mass and periodic-boundary controls). Holding all current model and
   covariance parameters unchanged (`inference_rate = 0.1`, `q = 0.01`, clipped
   Gaussian, same weights/solver), replacing unweighted `48 x 64` scoring with a
   pixel-area-weighted `48 x 64` prototype or exact full-grid (`120 x 160`)
   scoring eliminates the frame-38 freeze—both continue alternating between
   interior candidates through frame `1200`:

   | Native OpenGL scoring quadrature (`120 x 160`, `1200` frames / `40` s) | Last gaze change | Distinct gazes in final `1` s | Final mean hidden variance | Mean forecast MSE |
   | --- | --- | --- | --- | --- |
   | Current production unweighted `48 x 64` (`preview_indices`) | Frame `38` (`1.27` s) | `1` (locked at `(0.5, 0.5)`) | `0.00175343` | `0.00093626` |
   | Diagnostic pixel-area-weighted `48 x 64` prototype (not yet in runtime) | Frame `1200` (`40.0` s) | `2` (interior alternation) | `0.00175343` | `0.00093164` |
   | Exact full-grid `120 x 160` reference (not yet in runtime) | Frame `1200` (`40.0` s) | `2` (interior alternation) | `0.00175343` | `0.00093177` |

   Sub-sampled area-weighted quadrature still aliases narrow footprints or
   rapidly varying nonlinear integrands, and the resulting interior alternation
   is **not** scene-wide exploration or moving-dot pursuit; greedy one-step
   information-seeking gaze is not globally guaranteed to keep moving across all
   parameter regimes.
2. **Deeper sidecar / solver-parameter inconsistency:** `inference_rate` (`alpha`)
   is reused both as the inner gradient-descent step size in `infer_joint` and
   as the per-frame covariance contraction factor `(1 - alpha)^2`, while the
   predictive-coding mean objective keeps fixed unit prior precision `I` on
   `hidden - W_canvas * tanh(canvas)` rather than weighting by `(P_prior)^-1`
   (and one frame executes a single CFL acoustic wave step `~1.3e-5` s vs.
   `1 / 30` s stimulus `dt`). In a fixed-canvas converged-mean oracle across
   `alpha in {0.05, 0.10, 0.20}`, the converged hidden mean is identical to
   `1e-14` (`0.0996676844244`), whereas the sidecar posterior variance shifts
   nearly fourfold (`0.00330819`, `0.00173870`, and `0.000923584`).

**Architectural rule and next-step boundary:** (1) spatial quadrature
action-ranking error must be checked against candidate margins and a full-grid
reference before changing the model, and (2) numerical optimizer step sizes must
not silently define probabilistic transition dynamics—any subsequent model slice
must explicitly choose between a memoryless static conditional prior and a
temporal conditional model whose retention rate is decoupled from solver
`inference_rate` and whose prior covariance is shared by both mean and
covariance updates.

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

This slice provides no observation-cost objective or automatic skip decision.

## Headless policy comparison

```bash
uv run python -m audioviz.offline.active_vision --backend numpy --resolution 120 160 --duration 2 --seed 7 --policy-seed 29 --output outputs/gaze-comparison.json
```

`audioviz.offline.active_vision` runs matched headless rollouts (`--backend
numpy`, `cupy`, or `opengl`) comparing `fixed` (fixed at the `(0.5, 0.5)` grid
center), `random`, and information-seeking (`autonomous`) gaze policies under an
identical frame count, observation schedule (observing every frame), initial
model state/weights, and Gaussian precision profile (`--policy-seed` controls
random-policy draws independently of the stimulus `--seed`). Because Gaussian
footprints near image borders are clipped by the finite pixel grid, total
precision mass per frame varies with gaze position; the evaluator reports
`precision_mass` as the sum of scalar per-pixel precision `Pi(r)` over the image
grid (not multiplied by the 3 RGB channels) rather than assuming border-clipped
gazes have equal summed precision. Native OpenGL metric readbacks reduce MSE and
mean hidden variance to scalars on-device; candidate scoring downloads only the
`9 x 13` (`117`-value) score grid.

The comparison separates three distinct quantities that must not be conflated:

1. **Expected ideal sensory information gain (`expected_gain_nats`):** one-step
   mutual information `0.5 * sum_r log(det(I_3 + Pi_a(r) * V_sensory(r)))`
   evaluated from the full local `3 x 3` RGB covariance `V_sensory` **before**
   projecting hidden covariance onto its diagonal.
2. **Retained hidden-diagonal uncertainty (`posterior_hidden_variance_mean`):**
   mean of the stored diagonal variance vector `v_posterior = diag(C)` after
   discarding off-diagonal hidden correlations induced by conditioning.
3. **Next-frame pre-evidence prediction error (`forecast_mse`):** evaluator
   image residual comparing the model's pre-observation prediction against the
   next stimulus frame.

Greedy one-step maximization of expected sensory information gain does not
universally guarantee higher cumulative information gain over a rollout, lower
final projected diagonal variance, or lower image prediction error than fixed
or random policies on arbitrary trajectories. Analytical dominance holds only in
specifically constructed regimes.

## Focused verification

```bash
uv run python -m pytest -q tests/sources/test_circular_dot.py tests/sources/test_gaze.py tests/readouts/test_spatial_precision.py tests/readouts/test_gaussian_uncertainty.py tests/offline/test_active_vision.py
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
