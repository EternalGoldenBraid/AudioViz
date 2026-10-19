# Active vision through expected information gain

Status: research proposal — not an implementation specification.
Planning branch: `research/active-vision-eig`

## Motivation

AudioViz currently explores predictive coding over visual and audio inputs. We want to investigate whether a predictive model can **actively select what to observe** instead of passively consuming uniformly precise camera frames.

The biological inspiration is foveated vision: a limited sharp region of sensory input, with less precise peripheral observations and a model that predicts the broader scene. We want attention to be a *visible, measurable control variable*, not a hidden heuristic.

## Research question

Can a predictive-coding model learn to allocate spatial sensory precision to observations that are expected to reduce uncertainty about its latent state and/or future observations?

A core distinction to test: **large current prediction error is not equivalent to high expected information gain**. A surprising but irreducibly noisy region can have large error without being informative.

## Proposed computational graph (conceptual)

- Full camera frame `x_t` (available for experiment/evaluation).
- Attention/action parameters `a_t` (initially location and spatial extent; later anisotropic or multi-focus).
- Observation operator `S(x_t, a_t)`: sharp near focus, progressively lower resolution or precision in the periphery. This must be differentiable with respect to attention parameters if optimized by gradients.
- Predictive model with latent state `z_t`, producing a full-frame prediction `x_hat_t`.
- Inference updates to `z_t` from the *attended observation*, with a spatially varying observation precision or equivalent weighting.
- Attention policy chooses `a_t` based on predicted usefulness of the next observation.

Attention should change **what evidence the model receives**, not just add a blur effect to a displayed image.

## Information gain formulation (candidate, not yet committed)

Let `b_t(z)` represent a belief over latent states before a candidate observation, and let `y_{t+1}` be the observation resulting from attention action `a`. One candidate objective is

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

## Decisions needed in future conversations

- What exactly is the latent random variable whose uncertainty we want to reduce?
- What is the belief family, if any, maintained by the current predictive-coding implementation?
- Is attention modeled as bandwidth/resolution, measurement precision, missingness, or a combination?
- What is the sensing budget and action timescale?
- How do we distinguish epistemic value from reconstruction improvement and curiosity for noise?
- What should be learned jointly versus held fixed for controlled experiments?

## Engineering principles

Minimal, modular, efficient, parsimonious and maintainable. Clear interfaces, reproducible configs, independent unit tests, and incremental experiments. Keep research claims separate from empirical observations and implementation choices.
