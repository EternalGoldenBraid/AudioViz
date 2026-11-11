---
layout: page
---

<script setup>
import CompletionTrace from './CompletionTrace.vue'
import baselineAtlas from './completion-baseline-atlas.png?url'
import clampedAtlas from './completion-clamped-atlas.png?url'
</script>

<div class="vp-doc experiment-page">

[All experiments](./) · [Node pulses and connection ripples](./dual)

# Scene completion from a raster-scanned fovea

A small diary entry, not a playback page: can the model complete a **static scene it has
already learned**, using only a small patch strictly smaller than the scene?

## Setup

Two 16×16 scenes, each far larger than the radius-3 fovea: vertical stripes and concentric
rings, both using the same two gray-scale levels (0.15 / 0.85, never exactly 0 — see note
below). Training scans every fixation that covers the world; evaluation fixates **only** the
center patch, which never sees the scene border. Six shared hidden units, noisy evidence
(σ=0.03) at every observation, identical seed across configurations.

Patterns avoid raw `{0, 1}` values on purpose: the network's own passive decay settles toward
exactly 0 when unobserved, so a uniform `0` patch would coincide with "no evidence at all" and
teach nothing, regardless of training. This was an early failure mode of this experiment, not
a property of the model worth fixing.

## Does the patch reach its own target?

| τ | lines RMSE inside patch | circles RMSE inside patch |
|---|---|---|
| 1/3 (default) | 0.452 | 0.126 |
| 1 | 0.428 | 0.120 |
| 3 | 0.418 | 0.117 |
| 8 | 0.415 | 0.116 |

| Clamp strength | lines RMSE inside patch | circles RMSE inside patch |
|---|---|---|
| 0 (solver's own soft pull) | 0.452 | 0.126 |
| 0.6 | 0.037 | 0.021 |
| 0.75 | 0.034 | 0.023 |
| 0.9 | 0.034 | 0.027 |
| 1.0 (hard clamp) | 0.034 | 0.030 |

Increasing τ by **24×** barely moves the observed patch toward its target. Blending observed
sensory potential toward evidence (`clamp_strength`, applied only in this experiment script,
not the shared solver) fixes it almost completely by 0.6–0.75. This confirms the soft-clamp
diagnosis directly: the fixed decay rate dominates the observation pull at τ=1/3, not a learning
or capacity problem.

## Does the rest of the scene complete correctly?

|  | outside-patch RMSE vs correct | vs *other* pattern | vs blank (0.5) |
|---|---|---|---|
| lines, τ=1/3, clamp 0 | 0.619 | 0.669 | 0.493 |
| lines, τ=1/3, clamp .75 | 0.610 | 0.681 | 0.481 |
| circles, τ=1/3, clamp 0 | 0.709 | 0.608 | 0.497 |
| circles, τ=1/3, clamp .75 | 0.709 | 0.599 | 0.485 |

Fixing in-patch reconstruction does **not** fix completion. Outside-patch error barely moves,
and the circles patch still predicts something **closer to the lines pattern than to its own**,
in both configurations. The decay/clamp balance and the completion failure are separate
problems.

<figure class="completion-atlas">
  <figcaption>τ=1/3, clamp 0 — target / prediction, lines then circles</figcaption>
  <img :src="baselineAtlas" alt="Baseline completion atlas: lines target, lines prediction, circles target, circles prediction">
</figure>
<figure class="completion-atlas">
  <figcaption>τ=1/3, clamp 0.75 — target / prediction, lines then circles</figcaption>
  <img :src="clampedAtlas" alt="Clamped completion atlas: lines target, lines prediction, circles target, circles prediction">
</figure>

Predictions stay visibly near-blank in both cases; clamping sharpens the patch itself but does
not paint structure outside it.

## Are the hidden units even separable?

<CompletionTrace />

The clamp-boosted configuration drives roughly **5–9× larger** hidden-unit magnitudes for the
lines patch than for circles, so there is *some* separation signal reaching the hidden layer.
But six units moving by a few hundredths is still far from the ±1 range `tanh` actually uses,
and separation alone did not translate into correct completion above. The next open question is
whether `Ws` (hidden→sensory) is undertrained to carry that signal outward, or whether the
patch content itself is too ambiguous — not whether the hidden layer can represent a difference
at all.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_scene_completion.py \
  --output-dir docs/continuous-inference --shape 16 16 --radius 3 --hidden 6 --epochs 80 \
  --learning-rate .1 --sigma .03 --tau 0.3333333333333333 --clamp-strength 0 \
  --label completion-baseline

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_scene_completion.py \
  --output-dir docs/continuous-inference --shape 16 16 --radius 3 --hidden 6 --epochs 80 \
  --learning-rate .1 --sigma .03 --tau 0.3333333333333333 --clamp-strength 0.75 \
  --label completion-clamped
```

Both runs are deterministic (fixed seed). The generated atlases, summaries and hidden-unit
traces stay local; this page's tables and figures were produced by exactly these two commands.

</div>

<style scoped>
.completion-atlas { margin: 16px 0; }
.completion-atlas figcaption { font-size: 12px; color: #8798ab; margin-bottom: 6px; }
.completion-atlas img { width: 100%; max-width: 640px; image-rendering: pixelated; border: 1px solid #263140; border-radius: 8px; }
</style>
