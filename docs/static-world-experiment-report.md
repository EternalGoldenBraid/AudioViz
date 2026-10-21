# Static-world vision experiments: results and pause-point

Date: 2026-10-20. Branch: `feature/active-vision-precision`.
Runtime snapshot: `c494c32`.

Research and implementation histories have since been combined on the shared
branch **`research/active-vision-eig`** in draft PR #1. The branch and commit
labels in this report identify the original experimental snapshot. Branch
consolidation did not change the implementation pause recorded below.

## Current baseline: continuous whole-scene beliefs

The active bench now has one continuous model. Spiking is deferred and scene
motion is off by default. Production `main.py` retains its existing model;
the measurements and model-selection commands in the historical sections
below identify earlier experiments, not alternate active bench modes.

```bash
.venv/bin/python -m audioviz.visualization.static_world_bench
```

The represented state is a world-coordinate RGB neural/logit field `z`,
coupled to one shared dense recurrent hidden population `h`. Pixel intensity
is decoded as `sigmoid(z)`, with neutral `0.5` at zero state. No constant
bias drives or spike encodings are involved. The fovea restricts incoming
camera evidence, not the support of neural dynamics. Every scene coordinate
evolves and is decoded each interval, including previously visited locations
outside the current gaze; the rendered forecast is not a last-visit cache.

Forecast and conditioned replay resolve the same fixed physical interval.
Learning modifies connection matrices, not stored RGB screenshots. Turning
Observe off removes sensory evidence but does not remove a nonzero root
drive. Unforced trials therefore keep root feedback at zero.

The bench exposes neural-field/hidden-state norms, whole-scene neutral
distance, observed-coordinate forecast/posterior MSE and an energy trajectory.
No-evidence MSE remains unavailable. Live controls are continuous integration
microsteps/gain, leak time constant and connection learning. The earlier
spike threshold, capacitance and model selector are absent. Existing seeded
Brownian motion is available only by explicit opt-in; its design is deferred.

Gaze has the intended mouse/one-autonomous boundary. Only mouse/locked gaze
is operational in this baseline while the single epistemic objective and
matching uncertainty representation are established. No random fallback,
mixture model or unsupported Gaussian covariance is attached to the new
neural dynamics.

### Whole-scene evolution check

Native Qt/OpenGL trial at `80x120`, radius `32`, six hidden units,
`dt=1/30`, `tau=1/3`, eight integration microsteps and inference gain `0.1`.
The world was static, the root stayed zero and no state was cleared between
phases:

| Intervention | Result |
| --- | --- |
| Observe gaze A `(0.2, 0.5)` for 40 steps, learning off | Neural-field norm in A: `71.377810` |
| Move to non-overlapping gaze B `(0.8, 0.5)` for 100 learning steps | Former A norm falls to `0.488248`; whole-field norm is `84.300165` |
| Turn Observe off for 180 further steps | Whole-field norm falls to `1.200930e-6`; whole-scene forecast MAE from neutral falls from `0.05907949` to `9.962273e-10` |

The camera read count remained `140` throughout the unobserved phase, learning
updates remained `100`, and all learned connection matrices stayed exactly
unchanged. Rendered images showed the formerly observed patch changing after
gaze moved and the complete predicted scene returning to neutral gray.

The energy plot uses additive hidden and sensory components, including leak,
logit coupling and observed residuals. Its conditioned microstep trajectory
starts at the saved interval-start state, not at the counterfactual forecast
endpoint. Forecast MSE still measures that pre-evidence counterfactual.
Inference gain changes the interaction terms relative to passive leak;
monotonic energy descent is not guaranteed for arbitrary gains, numerical
step sizes or clipped currents.

## Historical: canvas ablation and prediction-persistence boundary

This section describes the unchanged PC control. The bias-free RC experiment
and its different resting-state contract are recorded separately below.

Main and the bench now default to a **held root canvas**, with no autonomous
wave propagation or damping. **Controls > Surprise: wave dynamics OFF/ON**
restores the existing wave equation. Explicit drives and sensory-gradient
corrections still update the root when waves are off; disabling waves does
not imply a zero root or disable predictive coding. A held step synchronizes
the current/previous field states, clearing wave velocity. Model, world, and
source clocks continue advancing. New trial retains the wave enable switch,
and traces record the actual enable state per step as well as at export.

The static and Brownian measurements recorded below predate this default
change; they used the enabled wave substrate with a zero root and zero
canvas feedback, where the root stayed zero.

Prediction persistence has separate causes:

- The full-image forecast is a last-visit cache. `SharedSensoryIO.render`
  overwrites only the current in-bounds foveal positions; outside entries
  are never recomputed or aged. This cache is not used as evidence or as
  the model's candidate-scoring state.
- Hidden activity is warm-started with `exp(-dt/tau)` and then relaxed
  against the hidden-only energy. Relaxation additionally removes
  history dependence, with finite settling controlled by steps/rate;
  `tau` alone does not specify the complete temporal dynamics.
- Synaptic parameters, including decoder biases, persist. A learned
  non-neutral prior can remain after transient hidden activity dissipates.

No display fade, weight forgetting, or temporal-model rewrite is included
in the canvas ablation. Choosing one requires specifying which state should
decay and its resting prior; these are not interchangeable mechanisms.

Math peer's reading of
[Difference Predictive Coding for Training Spiking Neural Networks,
ICLR 2026](https://iclr.cc/virtual/2026/poster/10007923):
DiffPC uses non-leaky difference accumulators and signed ternary spikes to
communicate state changes. The reconstructed signal persists when spike
traffic stops; threshold refinement is not prediction decay. Spiking alone
does not expire stale display history or define temporal-world dynamics.

## Follow-up: continuous RC-spiking bench prototype

The bench now offers an explicit `--model rc-spiking` experiment alongside the
unchanged default PC control. It reuses the production camera routing, foveal
gather/scatter and renderer. Production main remains on the PC model.
The numerical graph runs on NumPy or CuPy; OpenGL retains the existing
CPU-float64 graph with GPU pixel routing.

The RC model separates leaky membrane voltages, ternary threshold/reset spikes,
and a leaky decoded hidden state. For `M` microsteps over one physical `dt`,
`alpha = exp(-dt/(M*tau))` and the zero-order-hold current multiplier is
`tau*(1-alpha)*gain/C`. Constant subthreshold current is invariant to
microstep count; discrete spike timing remains resolution-dependent.
Forecast computes the no-evidence counterfactual, and conditioning replays the
same interval from its saved start with evidence. Observe off commits the
forecast without a second interval of leak.

Initial bias-enabled prototype, native Qt/OpenGL-rendered validation at
80x120, radius32, six hidden units,
locked central gaze, static world, `dt=1/30`, `tau=1/3`, `C=.02`,
threshold `.02`, eight microsteps and current gain `.1`:

| Intervention | Result |
| --- | --- |
| Fresh trial, learning/feedback off, eight observed steps | 63 hidden spikes, 234,791 sensory-error spikes; final hidden L2 norm `0.135587` |
| Observe off for 90 further steps, without clearing any model state | Hidden norm `0.000016733`; current foveal forecast MAE from neutral `0.0000006837`; camera reads remain eight |
| Separate fresh trial, learning on for 553 observed steps | Hidden norm `3.943730`; learned hidden-bias norm `2.914397`, output-bias norm `37.494134` |
| Then Observe off for 90 steps | Hidden norm still `3.943730`, 48 committed hidden spikes per step; forecast MAE from neutral `0.349088` |

That learned plateau reproduced the user's report: hidden biases supplied
tonic currents and output biases defined a learned resting image. The user
subsequently chose a **bias-free RC model**: only connection weights learn,
both bias arrays remain zero, and unforced predictions have a neutral `0.5`
resting value. The original PC model retains its biases. Connection weights
are not forgotten, and outside-fovea display pixels remain a last-visit cache
with no cosmetic fade. The bias-enabled measurements above record the
diagnosis, not the corrected model's learning behavior.

The corrected bias-free model was then run with the same 553-step learned
trial. Hidden magnitude at the end of observation was `3.943730`; after
90 Observe-off steps it was `0.000499016`, with zero committed hidden spikes
and current foveal forecast MAE from neutral `0.0000949386`. Both bias arrays
remained zero, and all learned connection matrices were unchanged throughout
the no-evidence interval. The rendered foveal forecast returned to gray
without clearing state or fading the display. NumPy, CuPy and OpenGL bench
regressions also exercise actual camera-routed learning followed by decay.

Controls expose tau, capacitance, derived resistance and spike threshold.
Plots/export distinguish counterfactual and committed spike counts, committed
hidden magnitude and valid-aperture pre-evidence distance from neutral.
Residual energy is not monotonic. Gaussian covariance, EIG and autonomous
gaze are unavailable for this prototype. These results establish its RC
integration and persistence boundaries, not DiffPC equivalence, generalization,
event-driven hardware efficiency or real-time throughput.

### Design checkpoint: gaze objective and peripheral state

The intended live interface has only **mouse** and **autonomous** gaze.
Autonomous gaze must follow one explicitly defined active-inference
epistemic objective: expected reduction of uncertainty about the modeled
state before observing a candidate patch. Random exploration is not a
substitute for that objective. Fixed/random policies used by offline
comparisons do not define additional live behavior.

The PC control currently approximates uncertainty with a single local
Gauss-Newton Gaussian about its hidden state, not a mixture of Gaussians.
Its expected information gain is an epistemic objective, not a complete
expected-free-energy formulation with outcome preferences. The RC graph
does not yet supply the corresponding uncertainty model, so its autonomous
toggle remains disabled pending that formulation.

Observation support and dynamical support are separate: **evidence enters
only through the fovea, while every represented dynamical belief must
continue evolving without evidence**. The present implementation has a
dynamic foveal graph and a last-visit image cache. Its peripheral cached
pixels are not continuously decoded world-coordinate beliefs and have no
transition or decay law. They cannot be made dynamical by changing the
hidden leak constant.

Bias-free learning removes the original constant-current memorization
mechanism, but fitting one static patch still does not establish
generalization. Negative resting voltage, additional scene noise and a
cosmetic peripheral fade do not resolve the missing world-state boundary.
No such changes have been introduced. A whole-scene dynamical belief
requires an explicit state representation and transition law before its
uncertainty can meaningfully guide the single autonomous gaze policy.

## Follow-up: Brownian scene-motion experiment

After branch consolidation, the user authorized making the test world move
before choosing a new gaze mechanism. The bench now starts with seeded Brownian
translation enabled; **gaze-policy and latent/observer sampling design remain
paused**. The original static-patch measurements below are unchanged and do
not describe the moving-world experiment.

The world advances once per fixed model interval, independently of camera
reads and Observe. Its per-axis increment distribution is
`Normal(0, scale^2 * dt)`, with default scale `4` pixels/sqrt(second) and seed
`0`. Rendering rounds the continuous offset to pixels, crops rather than wraps,
and fills exposed pixels with the scene's dark background. The random walk
itself is not clipped or reflected; objects can move out of view.

**Brownian scene motion** and its scale are live controls. Turning motion off
freezes the current position. **New trial** restores the initial position and
RNG while retaining the chosen settings. Use `--static-scene` and Lock gaze to
recover the held-target control. Trace rows record enabled state, scale, and
continuous/rendered offsets; the export also records world seed/time.

Native OpenGL trial: at `80x120`, radius `32`, hidden count `6`, fixed gaze
`(0.5, 0.5)`, `dt=1/30`, default motion scale, and learning/canvas feedback off,
60 steps moved the rendered scene from `(0, 0)` to `(0, 7)` pixels. The first 10 steps advanced the world
with Observe off, zero camera reads, and unavailable MSE. Enabling Observe
then produced 50 camera reads over the remaining 50 steps. A fresh trial
with motion off reproduced the held-world forecast MSE at step 15:
`0.15163090413829192`. These establish the world/measurement separation and
static control, not a new exploration or learning result.

Only the RGB frame crosses the camera interface. Neither gaze selection nor
the predictive model receives evaluator offsets or the world's RNG state.
No pixel noise, random tie-breaker, softmax, or eye-motion policy was added.
With zero root feedback, the model-side symmetry described below can persist:
a moving scene is not evidence that informative autonomous sensing is solved.

## Decision at the original pause

At this earlier pause, implementation of stochastic gaze/world dynamics was
paused. The user does **not**
want a special-case rule that detects equally good candidates and then samples
among them. The direction under consideration is that stochasticity and
sampling belong to the ongoing world/observer dynamics, including ordinary
non-tied situations. Its mathematical formulation remains open.

No random tie-breaker, injected image noise, eye jitter, sampling temperature,
or replacement action distribution has been implemented. The existing greedy
autonomous policy, including its flat/tied-score handling, is still present;
this describes the current implementation, not an endorsed final design.

The coordination request remains to retain the evidence and unresolved
questions below, not to force a gaze-policy choice.

## What now exists

The bench uses the production `RippleWaveVisualizer` /
`RippleSourceOrchestrator` / `SharedPredictiveCodingGraph` pipeline with a
deterministic static scene. It is not another inference implementation.

```bash
.venv/bin/python -m audioviz.visualization.static_world_bench
```

The application and bench currently have these properties:

- Main defaults to vision-only operation: audio source and audio windows are
  disabled, so startup does not initialize audio devices.
- The bench starts paused, with mouse gaze unlocked, learning off, and canvas
  feedback zero. Mouse movement over any large image or the embedded canvas
  selects gaze through the production controls.
- Selecting gaze while paused does not consume evidence. Step/Run forecasts
  and then gathers the selected circular patch before conditioning.
- Lock gaze provides fixed-patch trials. Existing Controls provide inference
  steps/rate, learning, learning rate, feedback, and the independent Observe gate.
- The world outline shows the selected next gaze; prediction/error outlines
  show the latest sampled aperture. The status and trace record sampled gaze.
- New trial restores seeded weights, hidden state, and zero canvas, retaining
  model settings and gaze. Ordinary field/belief reset retains learned weights.
- Trace export retains at most 512 observations; it is a metric log, not a
  complete replay of arbitrary source/wave-control interventions.

Main and the bench use the same autonomous gaze policy. The current hidden
population is densely recurrent, with prescribed root/sensory roles, not
multiple imposed hidden layers or a general symmetric geometry-pruned graph.
The proposed geometric topology initialization remains unimplemented.

## Measurement contract

The Math peer implemented and checked the metric adapter. We keep distinct:

1. **Pre-evidence forecast MSE:** evaluated from the saved prior prediction on
   observed foveal RGB coordinates.
2. **Settled decoder MSE:** evaluated from decoder residuals after conditioning,
   before that observation's parameter update. Clamped sensory state values
   are evidence, not zero-error decoder predictions.
3. **Within-observation total residual energy:** the accepted hidden-relaxation
   trajectory with weights and root held fixed for that conditioning objective.
   Backtracking guarantees a non-increasing total, not necessarily decreasing
   sensory MSE alone.

No visual evidence means MSE is unavailable (`null` in JSON; a gap in plots),
not zero. Missing evidence causes no sensory learning, but finite hidden
settling can still leave a nonzero root gradient.

Across-observation forecast MSE has no general monotonicity guarantee. Moving
gaze changes the evaluated patch. Learning changes parameters; positive canvas
feedback also introduces wave propagation and damping. These are separate
interventions, not interchangeable explanations for an improving curve.

## Controlled static-patch results

The native OpenGL trials were run at bench snapshot `569aaa0` on an NVIDIA
GeForce RTX 3060. OpenGL handles pixels and the wave canvas; the shared
numerical graph in this backend executes in CPU float64. These measurements
are not a claim of all-GPU graph inference or a real-time throughput benchmark.

Common setup:

| Setting | Value |
| --- | --- |
| Scene | `static-geometric-v1`, 80 rows x 120 columns, gain 1 |
| Gaze | Held at `(0.5, 0.5)` |
| Fovea | Radius 32; 3,209 disk slots; 9,627 observed RGB coordinates |
| Hidden population | 6 total recurrent units |
| Initialization | Seed 0; fresh weights and zero wave canvas |
| Model forecast interval | 1/30 s per observation, independent of playback |
| Memory time constant | 1/3 s |
| Maximum hidden updates | 8 per forecast and per conditioning call |
| Initial hidden step size | 0.1, with backtracking |
| Canvas feedback / weight decay | Both zero |

For a fresh fixed-patch trial at the current snapshot, explicitly select
**Lock gaze** before stepping. The old measured default was locked; the current
default is unlocked mouse gaze.

| Regime | Observation | Forecast MSE | Settled MSE | Initial energy | Final energy | Learning updates |
| --- | --- | --- | --- | --- | --- | --- |
| Learning off | 1 | 0.151666411 | 0.151612212 | 730.046269 | 729.794057 | 0 |
| Learning off | 15 | 0.151630904 | 0.151612212 | 729.876929 | 729.794057 | 0 |
| Learning on, rate 0.05 | 1 | 0.151666411 | 0.151612212 | 730.046269 | 729.794057 | 1 |
| Learning on, rate 0.05 | 30 | 0.087504116 | 0.085525018 | 421.688850 | 414.776842 | 30 |

With fixed weights/root, the small forecast transient plateaus while
conditioning still reduces total energy within each observation. Enabling
learning improves the pre-evidence forecast on this held patch.

**Interpretation limit:** individual sensory biases can memorize a single
static patch, including with hidden activity zero for interior-valued targets.
These results do not establish generalization, location-specific world
memory, or an H-rank lower bound on single-patch reconstruction error. Reused
retinotopic weights and a persistent world-coordinate prediction buffer do
not by themselves establish a location-specific generative memory.

Full traces, native screenshots, and executable smoke harnesses are retained
as local session artifacts under
`/home/nicklas/.copilot/session-state/68398376-959e-4b56-92ac-666576d17adb/files/`.
Relevant basenames are `static-world-smoke.json`, `static-world-held.json`,
`static-world-learning.json`, `static-world-moving-gaze.json`, and the
corresponding PNG/log files. They are not committed or portable report assets;
the essential configuration and results are recorded above.

## Why stationary gaze was ambiguous

There were two genuine UI issues: gaze initially started locked, and the
large images did not route mouse movement into the gaze controller. Both are
fixed. Moving-gaze checks now verify actual selected coordinates, sampled
evidence and boundary masks, including selection before capture.

There is also a separate model/policy issue. A NumPy probe of the default
zero-root, fixed-weight scene found **five equally maximal candidates** on
the 9 x 13 gaze grid. Across the first three observations, the existing
autonomous policy held `(0.5, 0.5)`. On the first observation, scores ranged
from approximately 6.019113 to 9.785734; the current center was one maximum.
Boundary candidates have different valid apertures.

For candidates with identical routed root values and aperture masks, the
shared model has no location-specific difference to score. Its non-mutating
candidate forecasts do not inspect the unseen scene. Stationary gaze in
this setting is therefore not, by itself, evidence of a broken sampling wire.

The 41 targeted checks run for the final runtime snapshot covered main
configuration/startup, bench behavior, production live gaze, and metrics,
including NumPy, CuPy, and OpenGL where applicable. A separate native GPU
smoke used real Qt mouse events over all three large images; its rendered
result was inspected. These checks establish wiring and metric contracts,
not the adequacy of an autonomous exploration mechanism.

## Open questions, deliberately unresolved

- **Where does stochasticity live?** World evolution, sensory measurement,
  observer/eye dynamics, latent-state inference, action selection, or a
  coherent joint description? These induce different conditional
  distributions; no placement or noise scale has been chosen.
- **What does "always sampling" mean operationally?** Sampling a world state,
  a belief, a gaze trajectory, or an action distribution must have an explicit
  target distribution and temporal meaning. An equality-test fallback would
  not answer this broader question.
- **Does noise resolve the actual symmetry?** Homogeneous observation noise
  need not distinguish otherwise identical *expected* information-gain
  scores. Random gaze induces an observation distribution through the
  rendering/sampling map, but that map need not be invertible or equivalent
  to independent sensor noise. Small eye motion can also disappear under
  the current integer-pixel routing.
- **Which state remembers location?** The display buffer retains old
  predictions, while weights are retinotopically reused. What generative
  state would support location-specific recall and information seeking
  without looking at evaluator-only image data?
- **How will we distinguish adaptation from memorization?** Fixed-patch
  improvement is established. Transfer to other patches, return visits,
  missing-evidence intervals, and stochastic-world behavior remain questions,
  not completed experiments.
- **How will stochastic experiments remain interpretable?** Sources of
  randomness, seed/reset behavior, timescales, and repeated-trial reporting
  will need to be explicit once the underlying formulation is chosen.

Do not infer from this list that tie-breaking noise, a softmax, an exploration
bonus, or eye tremor has been selected as the next implementation.

## Dzogchen reading and the stance of this pause

At the user's request, we read about Dzogchen, the Tibetan contemplative
tradition often translated as Great Perfection, and a translated instruction
by Jamyang Khyentse Chokyi Lodro. The latter describes examining the arising,
remaining, and departing of thoughts and resting without fabrication, within
its religious/practice context.

The connection here is a **stance of inquiry**: do not manufacture a mechanism
merely to remove the discomfort of an unresolved design question. Retain the
observed symmetry and competing interpretations, and allow the formulation to
become clearer before adding machinery. This is our engineering reflection,
not a claim that Dzogchen prescribes stochastic sampling, proves a learning
algorithm, or implies that adding noise is sufficient.

Sources consulted:

- [Rigpa Wiki: Dzogchen](https://www.rigpawiki.org/index.php?title=Dzogchen).
- [Jamyang Khyentse Chokyi Lodro, *Advice on Recognizing the Nature of Mind*](https://www.lotsawahouse.org/tibetan-masters/jamyang-khyentse-chokyi-lodro/recognizing-nature-of-mind),
  translated by Adam Pearcey, Lotsawa House.

## Reviewable implementation history

| Commit | Boundary |
| --- | --- |
| `4edcb61` | Deterministic scene source and Math peer's metric adapter/tests |
| `569aaa0` | Same-pipeline static-world bench and fixed model forecast clock |
| `fca1485` | Vision-only main defaults |
| `c494c32` | Large-image mouse gaze routed into production sampling |

The established per-pixel application remains separately available in
`../AudioViz-baseline` on local `main` at `e0994e8`. Local review commits have
not been pushed. Unrelated user edits are preserved.
