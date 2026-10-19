# Active vision through expected information gain

Status: research proposal — not an implementation specification.
Planning branch: `research/active-vision-eig`

## Literature and conceptual grounding for coding agents

**Read before implementing an EIG policy.** The experiment is motivated by active inference, not by generic saliency detection or reward-based tracking. These sources provide context; they are **not** a requirement to reproduce any particular published architecture.

1. **Friston, Adams, Perrinet & Breakspear (2012), [Perceptions as Hypotheses: Saccades as Experiments](https://doi.org/10.3389/fpsyg.2012.00151)** — **start here**. Directly relevant to the circular-dot thought experiment: eye movements as active tests of perceptual hypotheses, and selecting informative future observations. [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3361132/).
2. **Parr, Pezzulo & Friston (2022), [Active Inference: The Free Energy Principle in Mind, Brain, and Behavior](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)** — open-access foundational treatment. See especially the overview, generative models, and chapters on designing active inference models and discrete/continuous time.
3. **Lanillos et al. (2021), [Active Inference in Robotics and Artificial Agents: Survey and Challenges](https://arxiv.org/abs/2112.01871)** — engineering-oriented survey of state estimation, action, planning, and learning under active inference; useful for implementation alternatives and limitations.
4. **Van de Maele et al. (2021), [Active Vision for Robot Manipulators Using the Free Energy Principle](https://doi.org/10.3389/fnbot.2021.642780)** — concrete active-viewpoint selection with learned generative models and expected free energy. Includes an instrumental reaching task; **our first experiment deliberately omits instrumental utility**.
5. **Millidge et al. (2021), [Whence the Expected Free Energy?](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy)** — theoretical caution about deriving and interpreting expected free energy. Useful when deciding what objective we actually optimize.

### Interpretation guardrails

- The initial sensory-space mutual-information objective is **our deliberately simplified epistemic-only design choice**, not a claim that all active inference formulations operate only on sensory states.
- Expected information gain is an expectation over **possible future observations before observing them**, not the realized prediction error afterward.
- Confirming a hypothesis may be useful, but if every plausible hypothesis predicts the same observation, that observation has little expected information gain. Prefer measurements that discriminate uncertain alternatives.
- To allow sparse verification, define an observation budget or sampling cost, and include a **skip observation** action; otherwise an information-seeking agent may simply sample at every step.
- The dot generator's full trajectory and random seed are for the evaluator only. The agent must never use ground truth to choose its gaze.
- Agents should document which equations come from the literature and which approximations or modeling choices are specific to AudioViz.

## Motivation

AudioViz currently explores predictive coding over visual and audio inputs. We want to investigate whether a predictive model can **actively select what to observe** instead of passively consuming uniformly precise camera frames.

The biological inspiration is foveated vision: a limited sharp region of sensory input, with less precise peripheral observations and a model that predicts the broader scene. We want attention to be a *visible, measurable control variable*, not a hidden heuristic.

## Research question

Can a predictive-coding model actively choose **where and when to sample** to reduce uncertainty about **future sensory observations**? For the first experiment, the observable is a dot moving around a circle, optionally perturbed by Brownian offsets.

A core distinction to test: **large current prediction error is not equivalent to high expected information gain**. A surprising but irreducibly noisy region can have large error without being informative.

## Decisions from our discussion

**Settled for experiment 1**

- Optimize **epistemic value only** (expected information gain), without an instrumental task reward.
- Predict **future sensory states** (initially dot position / image observations), not a new semantic latent-state model.
- Attention controls **where and when to look**; continuously following the dot is *not* the objective.
- A predictable circular trajectory should allow the model to look ahead and verify its hypothesis intermittently.
- Prediction-error magnitude alone is **not** expected information gain. Prefer observations that distinguish competing predictions or reduce accumulated uncertainty.
- Start with a reproducible circular-dot stimulus, optional Brownian perturbations, and tests before integrating more complex camera scenes.

**Still open implementation/research choices**

- The probabilistic belief representation over future sensory states and how to calculate/approximate EIG with the existing predictive-coding architecture.
- The observation model: sharp fovea with blurred periphery, missing observations, or precision-weighted measurements; exact spatial/temporal sampling budget.
- Which existing modules can be reused, and the staged learning/frozen-weights schedule.
- How to quantify uncertainty calibration and separate reducible uncertainty from irreducible perturbation noise.

## Visual intuition

These are **schematic diagrams, not empirical results**. The agent must not receive the full-frame ground truth shown to evaluators.

### 1. Look ahead, not chase

```text
                    predicted next dot  ×  <- fovea samples here
                 .------------------.
              .-'                    '-.
             /                          \
            |       circular path        |
            |                            |
             \        o current dot     /
              '-.                    .-'
                 '------------------'

       Motion hypothesis -> predict future position -> sample -> update
```

### 2. A sparse observation schedule

```text
Time  -------------------------------------------------------------->
Dot     o     o     o     o     o     o     o     o     o     o
Gaze    ^     ^     ^                 ^                       ^
        learn frequently              verify                 verify

If uncertainty rises (e.g., changed dynamics): increase sampling again.
```

### 3. Experimental visualization mock-up

```text
+--------------------+  +--------------------+  +--------------------+
| FULL GROUND TRUTH  |  | ATTENDED INPUT     |  | MODEL PREDICTION   |
|   .--trajectory--. |  |   dim periphery    |  |   .--trajectory--. |
|  |   o dot        |  |   [sharp fovea]     |  |  |   x predicted    | |
|   '--------------' |  |   rest uncertain   |  |   '--------------' | |
+--------------------+  +--------------------+  +--------------------+
         evaluator             agent sees this        model output

Below: gaze location and observation events over time; prediction error;
       uncertainty estimate; EIG estimate; optional Brownian offset.
```

## Proposed computational graph (conceptual)

- Full camera frame `x_t` (available for experiment/evaluation).
- Attention/action parameters `a_t` (initially location and spatial extent; later anisotropic or multi-focus).
- Observation operator `S(x_t, a_t)`: sharp near focus, progressively lower resolution or precision in the periphery. This must be differentiable with respect to attention parameters if optimized by gradients.
- Predictive model (which may internally use existing hidden state) produces a future sensory prediction `x_hat_{t+1}`; **no new semantic latent model is required** for the first experiment.
- Inference updates to `z_t` from the *attended observation*, with a spatially varying observation precision or equivalent weighting.
- Attention policy chooses `a_t` based on predicted usefulness of the next observation.

Attention should change **what evidence the model receives**, not just add a blur effect to a displayed image.

## Information gain formulation (candidate, not yet committed)

For the first experiment, let `X_{t+1}` denote the uncertain future sensory state (dot position or full frame), `D_t` the observations so far, and `Y_a` the future attention-conditioned observation under candidate gaze action `a`. A candidate epistemic-only objective is

```math
IG(a) = E_{y_{t+1} \sim p(y|a,b_t)}
        [ D_KL( p(z|y_{t+1},a,b_t) || b_t(z) ) ].
```

Choose an action that maximizes expected information gain, subject to an attention budget and possibly action/switching costs.

Open issue: AudioViz's current predictive coding state may be point-estimate rather than a calibrated distribution. If so, a KL-based objective requires an explicit approximate belief model or a justified proxy. **Do not silently equate prediction-error magnitude with EIG.**

## Visualization and instrumentation

Show simultaneously:
1. Original full camera frame (ground truth, for development only).
2. Actual attended observation delivered to the model, with the attention field visible.
3. Model's full-frame prediction.
4. Optionally, spatial prediction error and attention/precision maps.

Log attention position/shape, reconstruction quality by region, model uncertainty/proxy, and temporal stability. Avoid evaluation leakage from the original full frame into the inference path.

## Staged experiments

### Stage A — observation operator, no learned attention
Implement and test a deterministic, adjustable spatial sampler/blur with explicit contracts. Keep predictive model weights fixed initially. Compare attended input to full-frame predictions and verify the observation operator really affects inference.

### Stage B — optimize attention with a fixed model
Use a simple, explicitly named proxy objective first (e.g. predicted reconstruction improvement under an observation budget), while separately testing whether it tracks actual information gain on synthetic controlled data. Compare against fixed-center, random, and prediction-error-greedy attention.

### Stage C — probabilistic/EIG attention
Introduce an uncertainty representation appropriate to the existing model, derive an operational EIG estimator, and validate it on a toy environment with known latent variables and distractor noise before moving to live camera scenes.

### Stage D — joint learning and dynamics
Investigate attention-model co-adaptation, temporal predictive memory, movement penalties, and whether policies revisit informative regions as beliefs change. This is exploratory and must not be conflated with Stage A.

## Workstream boundaries for CLI agents

- **WS1: architecture audit.** Read current repo; identify camera source, predictive coding interfaces, inference update, rendering and tests. Return a proposed integration seam, not code changes.
- **WS2: sensory observation operator.** Implement Stage A in a small independently testable module; include numerical tests for shape, identity/full-attention behavior, boundary behavior, differentiability if required, and precision-mask consistency.
- **WS3: visualization.** Add side-by-side observation/prediction and visible attention map with minimal coupling to the model.
- **WS4: experimental harness.** Add reproducible synthetic scenarios, metrics and baseline attention policies.
- **WS5: EIG derivation and estimator.** Research/derive probabilistic formulation grounded in the actual model representation; no implementation until approved.

Each workstream should be a separate issue or implementation branch, with tests and a small PR. No coding agent should interpret this research document as permission to implement all stages.

## Outstanding questions (not already decided)

- Which belief family can express uncertainty about **future sensory observations** with the existing model?
- What tractable estimator approximates sensory-space EIG, and how will it be validated against a controlled probabilistic toy model?
- Which observation operator and fixed sensing budget should the first implementation use?
- How should observation timing be represented (including the option **not to sample**), and what cost/budget prevents always observing?
- What should be held fixed versus learned at each stage?
- How do we distinguish reducible uncertainty from unpredictable Brownian perturbations?

The *objective* (epistemic value), *initial uncertainty target* (future sensory state), and *behavioral goal* (predictive, intermittent verification) are already decided. Do not reopen them without new evidence.

## Engineering principles

Minimal, modular, efficient, parsimonious and maintainable. Clear interfaces, reproducible configs, independent unit tests, and incremental experiments. Keep research claims separate from empirical observations and implementation choices.

## First test-driven experiment — circular moving dot

This is the **first implementation target**. Before writing new components, audit the existing repository for synthetic video sources, moving-dot generators, camera-input adapters, predictive dynamics, and test utilities; reuse existing capabilities where practical. Do not assume these features are absent or already present without checking.

### Reproducible stimulus

- Generate a deterministic, seed-controlled sequence of frames showing a single dot moving along a circular trajectory, with configurable center, radius, angular velocity, phase, dot size, frame rate, and duration.
- Add an optional stochastic perturbation: a Brownian/random-walk offset around the nominal circular path. Define its increments and scale explicitly; use a fixed seed in tests. Include a zero-noise control and multiple noise levels.
- Make the sequence available through the existing synthetic-frame interface if possible; optionally export a video file for visual inspection. Tests should consume raw frames and known ground-truth positions directly rather than relying on video codec decoding.
- Keep the generator independent of attention and inference, so ground truth cannot leak into the agent's observation or policy.

### Behavioral objective

An agent sees only its attention-conditioned observations and predicts where the dot will appear. It selects *where and when* to look to reduce uncertainty about the sensory trajectory. The goal is **epistemic value**, not tracking error or reward for keeping the dot centered. A confident circular-motion hypothesis should permit intermittent checks, while uncertain dynamics or perturbations should motivate renewed sampling. Treat this as a hypothesis to test, not guaranteed behavior.

For the simplest sensory-space formulation, the uncertain quantity may be the dot's future sensory position or the future image, rather than a new learned semantic latent state. The precise belief representation and EIG estimator remain design questions; do not label a prediction-error heuristic as EIG.

### Test-first acceptance criteria

1. **Stimulus correctness:** with zero noise, generated positions match the analytic circle within tolerance; frames and positions are reproducible for the same seed/configuration.
2. **Perturbation correctness:** zero perturbation yields the exact baseline; nonzero seeded Brownian increments follow the documented process; trajectories stay within the defined image-handling policy (e.g., clipping or allowing off-frame positions).
3. **Observation isolation:** the agent receives only the attention-conditioned frame; full-frame truth and exact dot coordinates are used only by the harness for evaluation.
4. **Prediction test:** on an analytically known circular trajectory, compare predicted future positions against ground truth over multiple horizons, including without a new observation.
5. **Attention baselines:** compare fixed fovea, random sampling, and error-greedy sampling against the proposed epistemic policy under the same observation budget.
6. **Hypothesis tests:** measure whether the policy samples less often when the trajectory becomes predictable and increases sampling when dynamics become uncertain or change. Do not hard-code these outcomes into tests; evaluate them empirically.
7. **Metrics:** log sampling frequency, selected gaze locations, uncertainty calibration or declared proxy, prediction error, and robustness across noise levels and random seeds.

### Agent handoff

Start with a repository audit and a minimal test plan. Implement the deterministic synthetic stimulus and its tests first, then add stochastic perturbations and tests. Only then integrate the attention-conditioned observation path and an explicitly defined information-seeking policy. Keep the generator, observation operator, predictive model, and policy independently testable. Do not alter the production camera pipeline unnecessarily.
