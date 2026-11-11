---
layout: page
---

<script setup>
import CompletionExplorer from './CompletionExplorer.vue'
</script>

<div class="vp-doc experiment-page">

[All experiments](./) · [Node pulses and connection ripples](./dual)

# Scene completion from a raster-scanned fovea

Can the model complete a **static scene it has already learned**, using only a small patch
strictly smaller than the scene? Training scans the whole world; the question below only ever
fixates one small center patch, holding the eye open the whole time — whatever it looks at is
the only clamp.

A curated grid of six configurations (three clamp strengths × two memory time constants)
replaces an exhaustive sweep; the combinatorics of every tunable grow too fast to show
meaningfully. Pick a configuration below — it drives every figure on this page together.

<CompletionExplorer />

## Does the patch reach its own target?

Increasing τ by 24× (1/3 → 8) barely moves the observed patch toward its target — pick the two
τ options above with clamp fixed at *soft* and compare the left canvas's sharpness. Blending
observed sensory potential toward evidence (`clamp_strength`, applied only in this experiment
script, not the shared solver) fixes it almost completely once past ~0.6. This confirms a
soft-clamp diagnosis directly: the fixed decay rate dominates the observation pull at small τ,
not a learning or capacity problem.

## Does the rest of the scene complete correctly?

Watch the right-hand playback panel scrub through settling: the patch itself sharpens under a
stronger clamp, but the area outside it stays visibly near-blank in every configuration. The
circles scene often predicts something closer to the *lines* pattern than to its own. The
decay/clamp balance and the completion failure are separate problems.

## Are the hidden units even separable?

The clamp-boosted configurations drive noticeably larger hidden-unit excursions for the lines
pattern than for circles — switch between *soft* and *hard* clamp above and compare the trace
height. There is *some* separation signal reaching the hidden layer, but it stays far from the
±1 range `tanh` actually uses, and separation alone does not translate into correct completion.
The next open question is whether `Ws` (hidden→sensory) is undertrained to carry that signal
outward, or whether the patch content itself is too ambiguous — not whether the hidden layer can
represent a difference at all.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_scene_completion.py \
  --output-dir docs/continuous-inference --shape 16 16 --radius 3 --hidden 6 --epochs 80 \
  --learning-rate .1 --sigma .03 --grid
```

Deterministic (fixed seed) across all six configurations. The generated playback atlases,
summaries and the curated grid index stay local; this page's figures were produced by exactly
this one command.

</div>
