# Continuous-scene reference

[Project overview](../README.md) | [Experiments](experiments/README.md)

This page describes `ContinuousScenePredictiveCodingGraph`, the numerical model
used by current `main.py` and the static-world bench. Earlier per-pixel, Gaussian
and RC-spiking experiments have different contracts; their results are not
interchangeable with this model.

## Configuration and coordinates

The camera image represents the whole world in this prototype. World resolution
is `(rows, columns)`; a fovea selects a subset of those coordinates. Display
window dimensions do not change the world's neural resolution.

Configure startup settings in [`main.py`](../main.py):

```python
RIPPLE_CONFIG["field"]["resolution"] = (200, 200)
RIPPLE_CONFIG["model"].update(
    foveal_radius=32,
    hidden_units=6,
    memory_time_constant=1/3,
)
RIPPLE_CONFIG["renderer"]["backend"] = "gpu"  # numpy | gpu | opengl
RIPPLE_CONFIG["gaze"]["mode"] = "mouse"  # mouse | autonomous
```

`hidden_units` is the total shared recurrent population, not layers or channels
at every pixel. Hidden recurrence is dense and directed; reciprocal weights need
not be equal. Display depth does not impose a hierarchy. Canvas-root and
sensory-terminal roles are prescribed; geometric pruning is not implemented.

Radius is measured in world pixels. An interior radius-32 disk contains 3,209
RGB locations, or 9,627 scalar visual observations. Borders reduce the in-bounds
support. Allocation guards allow at most 307,200 world pixels, 64 hidden units
and 4,096 foveal locations (integer radius through 36). Oversized configurations
fail without silently reducing resolution. These are software guards, not
measured hardware limits.

Resolution, radius and hidden count are startup settings. Source availability
comes from `RIPPLE_CONFIG["sources"]`; windows come from `APP_CONFIG["windows"]`.
For camera-only operation, disable audio and both audio windows. Enabling audio
initializes the existing audio processor; streaming device selection is stored
in ignored `config/audio_devices.json`. The
[example configuration](../config/audio_devices.example.json) documents null
device values; the command-line picker uses `-1` for a null device.

## State and elapsed-time semantics

Each world RGB coordinate has a signed neural potential `z`; intensity is
`sigmoid(z)`. Zero potential decodes to neutral `0.5`. One shared recurrent
hidden population `h` couples the sensory field and optional 16 global audio
terminals. Audio decoding is linear.

States and learned connections stay anchored when gaze moves. The current
in-bounds circular patch supplies camera evidence, but every state evolves and
the complete field is decoded each interval. There is no frozen peripheral
image cache, spiking or learned constant bias drive.

Forecast and conditioned replay resolve the **same** physical interval, not two
successive intervals. Camera/audio readings are acquired after the forecast.
Missing or disabled inputs remain unclamped; real silent audio is valid
zero-valued evidence. Observe off pauses parameter learning and commits the
unforced forecast. It does not freeze neural activity or erase learned weights.

Reset clears states and evidence while retaining weights and precisions.
With zero input and feedback, activity decays toward neutral gray; a held
nonzero root or explicit source drive can still force the graph.

The model uses float64 on NumPy and CuPy. OpenGL keeps this numerical graph on
the CPU while handling image routing/rendering on the GPU. Graph storage scales
with world resolution, not only the fovea. Conditioned replay reuses energy and
gradient terms only within the same evidence/parameter state; new evidence
triggers fresh evaluation. Diagnostic energy traces retain every microstep.

## Gaussian residual objective and learned precision

The graph uses normalized diagonal Gaussian residual factors:

```text
factor_energy = sum(precision * residual**2 - log(precision)) / 2
d(factor_energy)/d(log(precision)) = (precision * residual**2 - 1) / 2
```

Only the fixed `log(2*pi)` constant is omitted. See
[Buckley et al. (2017), Section 5.1](https://arxiv.org/abs/1705.09156)
and the implementation's objective/gradient docstrings.

Hidden recurrence, sensory-potential coupling and observations each have positive
learned residual precisions. The existing learning switch/rate controls both
weights and precisions. Precision updates operate in log space with gradient
clipping, no hard precision floor/ceiling and no weight decay on precision.
Unrepresentable updates fail before either learned parameter group is modified.

Only observed coordinates contribute observation factors and observation-
precision updates. With no evidence, all parameter learning pauses. Snapshots
capture states, weights and precisions at the same **pre-learning** point.
Residual precision weights state gradients, connection gradients and covariance
curvature.

Inference gain tempers entire interaction factors, including their normalizers;
zero gain retains passive leak. Energy may be negative and is not directly
comparable across different evidence supports or with the former
squared-residual-only energy.

This is a Gaussian-factor energy evaluated at state means, not a normalized
directed generative model or posterior-expected precision learning. Parameter
updates do not include posterior residual variance or state-belief entropy.
For a fixed positive squared residual the precision optimum is finite, but
exactly zero residual permits unbounded precision. There is no precision
hyperprior or claim of calibrated noise estimation. The
[long-horizon experiment](experiments/2026-10-26-static-scene-learning.md)
records the associated unresolved failure.

## Local neural-state covariance API

```python
uncertainty = graph.state_covariance()
precision_times_vector = uncertainty.precision_product(vector)
covariance_times_vector = uncertainty.covariance_product(vector)
joint_covariance = uncertainty.marginal_covariance(
    [0, graph.hidden_slice.start],
)
```

The local Gaussian describes neural potentials `z`, then hidden states `h`,
at the captured pre-learning state, parameters and evidence support. It is not
covariance of camera readings, decoded RGB or weights.

The Gauss-Newton precision is
`H_GN = H_leak + sum(J.T @ residual_precision @ J)`. It omits nonlinear
residual-times-second-derivative terms. Positive leak anchors unconstrained
directions. Curvature products and conjugate-gradient solves avoid forming a
full state-by-state matrix; nonconvergent solves raise an error. The diagonal
preconditioner is a solver aid, not marginal covariance.

Queries return arrays on the model's NumPy/CuPy backend. Marginals select at most
64 distinct states, including paths through unselected states. Retained operators
are independent of later model steps. No separately trained covariance matrix
is introduced.

Covariance is rebuilt from the captured linearization, not propagated as temporal
observation history. Absolute variances inherit inference gain, resolution
scaling and learned residual strengths; they are not calibrated confidence.
Queries do not modify the model or observation gates.

## One-step epistemic gaze

Autonomous gaze scores a `9 x 13` candidate grid after forecasting and before
acquiring camera evidence. Each candidate selects a complete in-bounds RGB disk.
The score, in nats, is information gain about the joint sensory-potential and
hidden state in the captured local Gaussian:

```text
H = forecast Gauss-Newton precision
Q(gaze) = J_observation.T @ effective_observation_precision @ J_observation
information_gain(gaze) = (logdet(H + Q(gaze)) - logdet(H)) / 2
epistemic_EFE(gaze) = -information_gain(gaze)
```

The observation Jacobian uses forecast sigmoid slopes. Effective observation
precision is `20 * inference_rate * learned_observation_precision`, matching
the energy. Parameters and the neural linearization are fixed while scoring.
The Gaussian posterior covariance does not depend on the hypothetical observed
value, so no invented camera samples or evaluator ground truth are needed.

A diagonal sensory block and hidden Schur complement retain within-fovea
correlations and paths through unobserved states without forming a global or
fovea-by-fovea covariance. This is exact for the captured Gaussian approximation,
not exact nonlinear EFE.

```python
graph.forecast(canvas, dt)
scores = graph.candidate_information_gains(
    [[0.25, 0.5], [0.75, 0.5]], foveal_radius=32,
)
# Select a gaze from the scores, acquire evidence, then condition.
graph.condition(camera_frame, visual_mask=selected_fovea)
```

The API accepts at most 117 candidate pairs shaped `(..., 2)` and returns host
NumPy scores with the same leading shape. The singular
`candidate_information_gain(gaze, foveal_radius=...)` scores one candidate.
Queries require a pending forecast and do not modify model states, learning
counters or parameters.

Mouse mode incurs no candidate-scoring work. Observe off skips scoring and
acquisition and holds gaze. Ties retain the current gaze if it is an equal-scoring
candidate; there is no random fallback. Border disks contain fewer observations
and are scored without normalizing away the smaller support.

There are no preferred outcomes, movement costs, weight uncertainty or temporal
planning. Reobserving a patch may remain informative because covariance is
rebuilt each interval. This does not guarantee coverage, novel gaze or improved
reconstruction.

## Bench controls and measurements

```bash
.venv/bin/python -m audioviz.visualization.static_world_bench \
    --backend numpy --resolution 80 120 --foveal-radius 16 --hidden-units 6 \
    --steps 8 --rate 0.1 --dt 0.03333333333333333
```

There is one active model and no `--model` selector. The bench starts paused,
with mouse gaze unlocked, learning off, root feedback zero, motion off and no
wave dynamics.

| Control | Meaning |
| --- | --- |
| **Step / Run / Pause** | Advance using fixed model `dt`, independently of time spent paused. |
| **Observe vision** | Enable evidence acquisition; disabling it leaves state dynamics running. |
| **Lock gaze** | Hold the selected location. Otherwise mouse movement selects the next sample. |
| **Autonomous gaze** | Select patches by one-step epistemic scoring when gaze is unlocked. |
| **Model controls** | Integration, learning, leak time constant and root feedback. |
| **New trial (fresh weights)** | Restore seed-0 parameters, neural field and zero canvas; settings and gaze position remain, with mouse mode selected. |
| **Controls reset** | Clear neural states while retaining learned parameters. |
| **Save trace** | Export the latest 512 steps, not a complete replay checkpoint. |

World, pre-evidence whole-scene prediction and whole-world forecast RGB RMSE are
displayed together. Evaluator-only full-frame errors include unobserved pixels
and remain available with Observe off; they never enter inference or learning.
Observed-coordinate plots distinguish forecast error from settled decoder error
before that observation's parameter update.

Traces include settings, energy trajectories, gaze, world seed/clock, Brownian
offsets and wave enable state. Missing errors are `null`. They do not encode
every arbitrary source/wave intervention required for full replay.
Motion is opt-in via `--moving-scene`; `--memory-time-constant` sets neural leak.

## Live views, hardware and profiling

**Predictions** offers 2D comparison, prediction-only and a live continuous
**3D graph**. The graph shows reference, settled prediction, diagnostic error,
selected fovea, shared hidden units and a separate root plane. A missing camera
reference yields a gray error plane, not fabricated zero error.

The topology view samples at most 25 RGB coordinates, 12 hidden units and 4 audio
nodes, with up to 24 strongest nonzero weights per sampled block. Omitted edges
are not absent model connections. It updates at up to 10 Hz; **Live** and
**Capture frame** control the view without acquiring extra evidence or changing
learning.

Failed camera opens/reads latch until **Camera Frame** is toggled off and on.
Explicit Linux `/dev/videoN` paths use OpenCV V4L2. No per-frame retry loop is
introduced.

**Surprise: wave dynamics OFF/ON** controls optional autonomous propagation and
damping. Off holds the root while explicit drives or sensory corrections can
still update it; it does not pause model inference or scene motion. Held steps
clear wave velocity. The continuous bench remains wave-free.

```bash
.venv/bin/python scripts/profile_live.py --backend opengl \
    --warmup 10 --frames 120 --output /tmp/audioviz-live.prof
```

The profiler uses current `main.py` source/settings and reports update durations
and wall-clock update rate, saving cProfile data plus a text report. It is not a
synchronized per-kernel GPU measurement or presented-frame counter. The GUI FPS
counter measures freshly presented canvas frames; neither metric alone isolates
inference cost.

The historical `benchmark_predictive_inference.py` and
`benchmark_foveal_memory.py` scripts measure other model implementations.
Their 1080p or fovea-only memory results are not measurements of this
world-sized continuous graph.
