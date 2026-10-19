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
`eb35320`, including its approximation boundaries.

Each blue code label is a clickable `vscode://file/...:line:column` link into
this checkout at `/home/nicklas/Projects/AudioViz`. These local links work
without publishing the implementation to GitHub; if the checkout moves,
replace that root in the diagram. The repository-relative source links below
are the portable index. Function names remain useful if line numbers move.

| Mathematical responsibility | Implementation |
| --- | --- |
| Known canvas projection & ordinary PC mean prediction (when Gaussian filter is off) | [`PredictiveCodingRGBReadout.predict`](../src/audioviz/readouts/predictive_coding.py#L139-L145), [`OpenGLPredictiveModel._advance`](../src/audioviz/readouts/opengl_predictive.py#L271-L299) |
| Predicted sensory mean | [`_capture_prediction`](../src/audioviz/readouts/predictive_coding.py#L147-L157), [`_predict_observation`](../src/audioviz/readouts/predictive_coding.py#L334-L335) |
| Temporal Gaussian initialization, assumptions, and exact OU transition | [`GaussianSensoryUncertainty.__init__`](../src/audioviz/readouts/gaussian_uncertainty.py#L22-L59), [`validate_config`](../src/audioviz/readouts/gaussian_uncertainty.py#L62-L64), [`_transition`](../src/audioviz/readouts/gaussian_uncertainty.py#L66-L76) |
| Shared OU hidden-mean / covariance forecast and frozen pre-evidence mean | [`GaussianSensoryUncertainty.predict`](../src/audioviz/readouts/gaussian_uncertainty.py#L97-L138) |
| Decoder Jacobian and RGB covariance components | [`_jacobian` / `_components`](../src/audioviz/readouts/gaussian_uncertainty.py#L140-L148) |
| Joint frozen-affine Gaussian conditioning, hidden-diagonal projection, and deterministic canvas feedback | [`GaussianSensoryUncertainty.condition`](../src/audioviz/readouts/gaussian_uncertainty.py#L150-L210), [`_condition_native`](../src/audioviz/readouts/gaussian_uncertainty.py#L212-L284), [`_finish_inference`](../src/audioviz/readouts/predictive_coding.py#L184-L269), [`_diagnostics`](../src/audioviz/readouts/opengl_predictive.py#L561-L604) |
| Gaze-conditioned scalar precision | [`GaussianSpatialPrecision.field`](../src/audioviz/readouts/spatial_precision.py#L40-L58) |
| Normalized Gaussian measurement likelihood | [`negative_log_likelihood`](../src/audioviz/readouts/spatial_precision.py#L60-L75) |
| Planning forecast, RGB log-determinant, and pixel-area quadrature | [`expected_information_gain`](../src/audioviz/readouts/gaussian_uncertainty.py#L309-L419), [`_pixel_area_weights`](../src/audioviz/readouts/gaussian_uncertainty.py#L422-L427) |
| Native moments and candidate-score reduction | [GLSL inside `expected_information_gain`](../src/audioviz/readouts/gaussian_uncertainty.py#L364-L414) |
| Greedy tie/flat-score handling and policy selection | [`has_gaze_preference`](../src/audioviz/sources/gaze.py#L6-L7), [`information_gaze`](../src/audioviz/sources/gaze.py#L10-L23), [`GazePolicy.select`](../src/audioviz/sources/gaze.py#L37-L50) |
| Opt-in temporal Gaussian wiring, pre-capture gaze selection, and conditioning lifecycle | [`predict_visual`](../src/audioviz/sources/orchestrator.py#L175-L180), [`_select_visual_gaze`](../src/audioviz/sources/orchestrator.py#L182-L196), [`enable_visual_uncertainty`](../src/audioviz/sources/orchestrator.py#L198-L217), [`correct_from_observations`](../src/audioviz/sources/orchestrator.py#L246-L257) |
| Gaze-mode controls, responsive preview reflow, advisory cue, hypothetical sampling, and finite-run restart | [`resizeEvent`](../src/audioviz/visualization/active_vision_experiment.py#L198-L210), [`_change_gaze_mode` / `_update_precision`](../src/audioviz/visualization/active_vision_experiment.py#L212-L236), [`_update_advisory_marker`](../src/audioviz/visualization/active_vision_experiment.py#L254-L301), [`_after_frame` / `refresh_previews` / `_restart`](../src/audioviz/visualization/active_vision_experiment.py#L320-L388), [`main`](../src/audioviz/visualization/active_vision_experiment.py#L397-L433) |
| Headless fixed/random/autonomous evaluation and metrics | [`GazeFrameMetrics` / `GazeRunResult`](../src/audioviz/offline/active_vision.py#L22-L38), [`run_gaze_comparison`](../src/audioviz/offline/active_vision.py#L75-L164), [`main`](../src/audioviz/offline/active_vision.py#L167-L226) |
| Dense/scalar affine references, OU partition invariance, native parity, and quadrature ranking oracle | [`test_model_transition_and_frozen_decoder_match_dense_gaussian_reference`](../tests/readouts/test_gaussian_uncertainty.py#L51-L104), [`test_bounded_gain_preserves_full_grid_action_ranking_after_repeated_center_observations`](../tests/readouts/test_gaussian_uncertainty.py#L181-L212), [`test_temporal_posterior_and_scores_match_scalar_affine_oracle_independently_of_solver`](../tests/readouts/test_gaussian_uncertainty.py#L216-L279), [`test_temporal_mean_and_covariance_match_exact_ou_independently_of_step_partition`](../tests/readouts/test_gaussian_long_run.py#L10-L38), [`test_shader_uncertainty_matches_gaussian_reference_and_downloads_only_gain_grid`](../tests/readouts/test_opengl_predictive.py#L219-L292) |

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
  --brownian-scale 3 --seed 7 --sigma 0.15 --baseline 0.01 --peak 1 \
  --memory-time-constant 0.333333
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
  not inverted, and posterior mean/covariance remain `m_posterior = m_prior`,
  `P_posterior = P_prior`).

## Advisory Gaussian belief

The inference objective already has a Gaussian observation-likelihood
interpretation: the predicted image is its conditional mean, and observation
covariance is `R = diag(1 / Pi)` over observed dimensions. The existing energy
statistics are not predictive covariance.

The experiment opts into a **conditional temporal Gaussian filter**
(`GaussianSensoryUncertainty`) via
`source_orchestrator.enable_visual_uncertainty(dt=1 / fps, initial_variance=1.0, process_variance=0.01, memory_time_constant=1 / 3)`.
Ordinary predictive-coding entrypoints (`visual_uncertainty is None`) keep their
existing multi-step `infer_joint` mean-settling behavior unchanged and incur no
covariance-update cost. When enabled, each pixel has a hidden mean `m` (stored
in `readout.hidden`) and diagonal hidden-state covariance `P = Diag(v)` (stored
in float32 on the inference backend) governed by an exact Ornstein-Uhlenbeck
(OU) transition with physical memory time constant `tau`
(`--memory-time-constant`, default `1 / 3` stimulus seconds, completely
independent of the predictive-coding solver `inference_rate` and
`inference_steps`):

```text
rho = exp(-dt / tau)
Q(dt) = 0.5 * D * tau * (1 - exp(-2 * dt / tau))
m_prior = rho * m_posterior + (1 - rho) * W_canvas * tanh(canvas)
P_prior = rho^2 * P_posterior + Q(dt) * I
y_prior = W_visual * tanh(m_prior)
J = W_visual * Diag(1 - tanh(m_prior)^2)
V_sensory = J * P_prior * J^T
```

The full local `3 x 3` RGB covariance `V_sensory` is retained for information
scoring; RGB correlations induced by the decoder are not discarded. Given an
observation `Y` with scalar precision `Pi`, one frozen-affine Extended Kalman
Filter (EKF) update conditions **both** the hidden mean and covariance jointly,
projects the posterior covariance onto its diagonal, and returns deterministic
observer feedback for the next frame's known wave input:

```text
S = I_3 + Pi * V_sensory
delta_m = Pi * P_prior * J^T * inverse(S) * (Y - y_prior)
m_posterior = m_prior + delta_m
C = P_prior - Pi * P_prior * J^T * inverse(S) * J * P_prior
v_posterior = diag(C)
P_posterior = Diag(v_posterior)
delta_canvas = canvas_rate * (1 - tanh(canvas)^2) * (W_canvas^T * delta_m)
IG(action) = 0.5 * sum_pixels(log(det(I_3 + Pi(action) * V_sensory)))
```

Here `D` is the scalar diffusion intensity `process_variance` (default `0.01` in
hidden-state variance per stimulus second), `tau` is `memory_time_constant`
(default `1 / 3` s), and `initial_variance` `v_0` (default `1.0`) is configured
independently of the stationary unobserved prior variance
`v_inf = 0.5 * D * tau ≈ 0.001667`. Under this exact held-drive OU transition,
multi-step unobserved propagation is invariant to how elapsed time is
partitioned into steps `dt` (`test_gaussian_long_run.py`). For a linear
decoder (`W_visual * h`), `(m_posterior, C)` is the exact one-step conditional
Gaussian posterior (and for the nonlinear `W_visual * tanh(h)` decoder, it is
the exact posterior of the frozen-affine likelihood linearized once at
`m_prior` before current evidence), whereas the stored
`P_posterior = Diag(diag(C))` is a diagonal projection that drops cross-channel
hidden correlations induced by mixed `W_visual` (an identity decoder is exact
for the diagonal entropy oracle because `C` is already diagonal there). The
information-gain expression `0.5 * log(det(I_3 + Pi * V_sensory))` is the exact
sensory/hidden entropy reduction under the full local posterior covariance `C`
**before** discarding off-diagonal hidden correlations; it is generally not the
entropy difference of the diagonally projected stored beliefs `P_posterior`.
The log-determinant form also supports rank-deficient positive-semidefinite
sensory covariance.

Conditioning uses the decoder Jacobian frozen at `m_prior` once per
observation, bypassing `infer_joint` entirely so predictive-coding
`inference_steps` and `inference_rate` are unused (and displayed as read-only
unused fields in the GUI when temporal Gaussian inference is active). Hidden
cross-channel posterior correlations in `C` are discarded when retaining
`v_posterior = diag(C)` (note that for a mixed decoder `W_visual`,
`diag(inverse(P_prior^-1 + Pi * J^T * J))` is given by the Woodbury projection
above, not the elementwise inverse of the precision diagonal). The canvas
correction `delta_canvas` is deterministic observer feedback from the observed
hidden innovation `delta_m` (zero when unobserved or `Observe` is off) for the
subsequent known input `canvas`, not a Gaussian posterior over wave states. When
diagnostics are enabled, the two-point `inference_energy` trajectory records the
frozen-affine objective before and after `delta_m`, `hidden_error` stores the
prior-whitened innovation `delta_m / sqrt(v_prior)` (`0` where `v_prior == 0`),
`visual_error` stores the post-update linearized residual
`(Y - y_prior) - J * delta_m`, and the categorical `stream_state` factor is
omitted (`None`). The next forecast and its sensory covariance use the updated
model states and actual decoder weights, so decoder sensitivity and saturation
affect advice. This objective targets sensory-state information, not
hidden-state entropy.

`--initial-variance` (`1.0`), `--process-variance` (`0.01`), and
`--memory-time-constant` (`1 / 3` s) are configured experimental assumptions,
not learned or calibrated noise estimates. The source orchestrator advances
mean and covariance with `dt = 1 / fps` during actual prediction/conditioning,
independently of preview rendering. Missing or skipped observations (`Observe`
off) apply the OU transition without conditioning (`delta_m = 0`,
`delta_canvas = 0`, `m_posterior = m_prior`, `P_posterior = P_prior`). Variance
relaxes toward `v_inf = 0.5 * D * tau` under `rho < 1` even without an
observation. With **Observe** off, advice still evaluates the hypothetical next
observation; the status says sampling is paused.

**Scope of the approximation:** the wave canvas and fixed weights are treated
as known point estimates. Advisory UI scoring holds the current corrected canvas
fixed for one hypothetical step, whereas autonomous policy scoring in
`predict_visual` evaluates the actual next wave prior after `engine.propagate`;
neither simulates a multi-step future wave trajectory. This conditional filter
is not a joint wave-plus-hidden posterior, an iterated nonlinear MAP + Laplace
approximation, or a calibrated confidence interval (at default uncalibrated
noise/model scales, the pre-evidence forecast and wave surface remain faint and
are not claimed to provide high-contrast stimulus reconstruction). Wave-state
uncertainty, parameter uncertainty, spatial correlations, stream-state
likelihood information, and cross-modal/audio coupling are not propagated in
this slice. Enabling `enable_visual_uncertainty` with an audio readout,
cross-modal coupling, or weight learning raises an explicit `ValueError`.
Resetting readouts also resets hidden mean and covariance.

This is not an independent coverage tracker: changing the decoder or hidden
mean changes the projected uncertainty. However, a wrong but saturated
predictor can still be overconfident. Calibration and richer covariance
propagation remain necessary before interpreting these scores as trustworthy
real-camera confidence.

**Mid-run stationary autonomous gaze—pixel-area quadrature fix vs. conditional-model contraction:**
in a historical `1200`-frame (`40` s) native OpenGL run prior to `46a7f7a` and
`e751ee4` with stimulus time still advancing, the unweighted `preview_indices`
(`48 x 64`) scorer moved autonomous gaze across early frames and then stopped
changing after frame `38` (`~1.27` s), remaining at `(0.5, 0.5)` without
numerical divergence or stopped timers. Investigation separated two distinct
mechanisms that were addressed in two sequential slices:

1. **Immediate cause of the historical frame-38 freeze (quadrature ranking error, fixed in `46a7f7a`):**
   because the wave canvas is treated as a known point estimate (`Var(c) = 0`),
   the uncoupled hidden transition contracts variance each frame even on
   unobserved pixels—historically approaching
   `P_inf = (q * dt) / (1 - (1 - inference_rate)^2) ≈ 0.001754` under the pre-OU
   `(1 - inference_rate)^2` sidecar, and now approaching the physical OU floor
   `v_inf = 0.5 * D * tau ≈ 0.001667`—while border clipping on `120 x 160` gives
   higher scalar precision mass at the center (`~2225.99`) than at a corner
   (`~723.75`). However, the no-evidence floor and border clipping alone do
   **not** freeze gaze among interior candidates whose unclipped masses are
   nearly tied: unweighted endpoint-inclusive `preview_indices` (`48 x 64`,
   `max_side = 64`) introduced integer-step and boundary-cell quadrature bias
   larger than the `O(1e-3)` interior variance-depletion margin, reversing the
   true candidate ranking (at the identical post-240-step historical covariance
   state, unweighted `48 x 64` scored center `5.727252` vs. left `5.723267` with
   a left-minus-center margin of `-0.003985`, whereas exact full-grid `120 x 160`
   scored left `5.836790` vs. center `5.834804` with a left-minus-center margin
   of `+0.001986` and pixel-area-weighted `48 x 64` scored left `5.836767` vs.
   center `5.834881` with a left-minus-center margin of `+0.001886`).
   `_pixel_area_weights` weights each sampled row/column by its Voronoi cell
   width over `[-0.5, size - 0.5]` (`e_0 = -0.5`, `e_N = size - 0.5`, midpoint
   interior edges `e_i = 0.5 * (r_{i-1} + r_i)`, `w_{ij} = w_row[i] * w_col[j]`),
   eliminating the frame-38 freeze across NumPy, CuPy, and OpenGL:

   | Historical pre-OU quadrature ablation (`120 x 160`, `1200` frames / `40` s, native OpenGL) | Last gaze change | Distinct gazes in final `1` s | Final mean hidden variance | Mean forecast MSE |
   | --- | --- | --- | --- | --- |
   | Historical unweighted `48 x 64` baseline (pre-`46a7f7a`) | Frame `38` (`1.27` s) | `1` (locked at `(0.5, 0.5)`) | `0.00175343` | `0.00093626` |
   | Pixel-area-weighted `48 x 64` on pre-OU sidecar (`46a7f7a`) | Frame `1200` (`40.0` s) | `2` (interior alternation) | `0.00175343` | `0.00093164` |
   | Exact full-grid `120 x 160` reference on pre-OU sidecar | Frame `1200` (`40.0` s) | `2` (interior alternation) | `0.00175343` | `0.00093177` |

   Sub-sampled area-weighted quadrature still aliases narrow footprints or
   rapidly varying nonlinear integrands, and the resulting interior alternation
   is **not** scene-wide exploration or moving-dot pursuit; greedy one-step
   information-seeking gaze is not globally guaranteed to keep moving across all
   parameter regimes.
2. **Historical sidecar / solver-parameter inconsistency (resolved within the conditional temporal Gaussian mode in `e751ee4`):**
   prior to `e751ee4`, `inference_rate` (`alpha`) was reused both as the inner
   gradient-descent step size in `infer_joint` and as the per-frame covariance
   contraction factor `(1 - alpha)^2`, while the predictive-coding mean
   objective kept fixed unit prior precision `I` on
   `hidden - W_canvas * tanh(canvas)` rather than weighting by `(P_prior)^-1`
   (and one frame executes a single CFL acoustic wave step `~1.3e-5` s vs.
   `1 / 30` s stimulus `dt`). In a historical fixed-canvas converged-mean oracle
   across `alpha in {0.05, 0.10, 0.20}`, the converged hidden mean was identical
   to `1e-14` (`0.0996676844244`), whereas the sidecar posterior variance shifted
   nearly fourfold (`0.00330819`, `0.00173870`, and `0.000923584`). In `e751ee4`,
   `GaussianSensoryUncertainty` replaces that split with a single physical OU
   time constant `tau = memory_time_constant` and one joint frozen-affine EKF
   update, making posterior means, variances, canvas corrections, and IG scores
   bitwise invariant to `inference_rate` and `inference_steps`
   (`test_temporal_posterior_and_scores_match_scalar_affine_oracle_independently_of_solver`).
   Because wave-canvas uncertainty (`Var(c) = 0`), parameter uncertainty, and
   multi-step action lookahead remain unmodeled and `tau`, `D`, and `v_0` are
   uncalibrated, autonomous gaze under the consistent temporal Gaussian filter
   continues to alternate between the two interior candidates `(0.4167, 0.5)`
   and `(0.5833, 0.5)` through frame `1200` (with `40` s native OpenGL
   `autonomous` / `fixed` / `random` final mean variance `0.00166576` /
   `0.00166576` / `0.00166598`, mean forecast MSE `0.00094218` / `0.00094215` /
   `0.00094234`, and cumulative expected gain `18308.58` / `15917.09` /
   `14433.17` nats) rather than tracking the moving dot or reducing image
   forecast MSE.

**Architectural rules and open boundaries:** (1) spatial quadrature
action-ranking error must be checked against candidate margins and a full-grid
reference (covered by a full-grid ranking regression), and (2) numerical
optimizer step sizes must not silently define probabilistic transition
dynamics—temporal retention is governed by `--memory-time-constant` (`tau`) and
shared across mean and covariance updates. Propagating wave-state uncertainty,
calibrating physical noise/memory parameters, and designing non-myopic or
hysteretic gaze policies remain open extensions outside this slice.

Scores use a fixed `9 x 13` candidate grid and nearest-neighbor spatial
quadrature with maximum side 64 weighted by represented pixel area (`w_{ij} =
w_row[i] * w_col[j]` over `[-0.5, R - 0.5] x [-0.5, C - 0.5]`, summing to the
full pixel count). This remains a
coarse approximation: footprints narrower than the sample spacing may be
missed. NumPy/CuPy keep covariance on their own backend. OpenGL updates
covariance in GLSL and projects bounded samples (packing row/column indices and
pixel-area weights in the cached index buffer) through the actual decoder
before reducing candidate scores, downloading only the small score grid.
Advisory scoring runs at preview cadence (at most 10 Hz); autonomous policy
scoring runs each model step before capture (neither runs per settling step).
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
