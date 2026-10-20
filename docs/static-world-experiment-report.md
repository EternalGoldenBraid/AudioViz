# Static-world vision experiments: results and pause-point

Date: 2026-10-20. Branch: `feature/active-vision-precision`.
Runtime snapshot: `c494c32`.

## Decision at this pause

Pause implementation of stochastic gaze/world dynamics. The user does **not**
want a special-case rule that detects equally good candidates and then samples
among them. The direction under consideration is that stochasticity and
sampling belong to the ongoing world/observer dynamics, including ordinary
non-tied situations. Its mathematical formulation remains open.

No random tie-breaker, injected image noise, eye jitter, sampling temperature,
or replacement action distribution has been implemented. The existing greedy
autonomous policy, including its flat/tied-score handling, is still present;
this describes the current implementation, not an endorsed final design.

The immediate coordination request is to retain the evidence and unresolved
questions below, not to start another implementation or force a policy choice.

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
