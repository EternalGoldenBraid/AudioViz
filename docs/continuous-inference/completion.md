---
layout: page
---

<script setup>
import CompletionHighlightPlayer from './CompletionHighlightPlayer.vue'
import CompletionExplorer from './CompletionExplorer.vue'
</script>

<div class="vp-doc experiment-page">

[All experiments](./) · [Node pulses and connection ripples](./dual)

# Scene completion from a raster-scanned fovea

Can the model complete a **static scene it has already learned**, using only a small moving
fovea strictly smaller than the scene? The eye stays open throughout: whatever it looks at right
now is the only clamp. Learning stays **on** the entire time, including during the recorded
scan below — this models a continuously running system, not a frozen-weight benchmark.

The recorded scan re-visits the whole raster path twice: every region is clamped once, left
unclamped as the fovea moves on, then clamped again on the second pass. The gold ring marks the
currently clamped disk on every panel below.

## Does τ matter more than clamp strength, once learning stays on?

Both configurations below use a **hard clamp (1.0)**; only the memory time constant τ differs.
With learning left on, this turned out to be the dominant lever, not clamp strength: by the end
of the second pass, τ=3 drives the whole-scene RMSE down to **≈0.09–0.10**, close to the
σ=0.03 observation-noise floor, while τ=1/3 only reaches **≈0.41–0.46** under the same hard
clamp. Watch the right-hand panels: the τ=3 prediction visibly locks onto the target as the
second pass proceeds; the τ=1/3 prediction stays comparatively vague.

<CompletionHighlightPlayer />

The inside/outside error chart shows the same story quantitatively: both curves fall as the
fovea revisits a region, but the τ=3 (orange) curve falls much further and recovers faster after
the fovea moves away, than the τ=1/3 (blue) curve. The hidden-unit traces below show τ=3 settling
into a much larger, more structured excursion than τ=1/3 for the same lines/circles distinction.

This does not mean clamp strength stopped mattering — at fixed τ=1/3, moving from soft (0) to
hard (1.0) clamp still nearly halves the final RMSE (0.59 → 0.41 for lines). It means τ now
matters *more*, which did not hold in the earlier frozen-weight, fixed-patch version of this
experiment. Explore every clamp/τ combination below to see the full picture.

## Explore all six configurations

Three clamp strengths × two memory time constants, replacing an exhaustive sweep whose
combinatorics would grow too fast to show meaningfully. Switch the view to the error map to see
pixel-level residuals instead of the raw prediction.

<CompletionExplorer />

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_scene_completion.py \
  --output-dir docs/continuous-inference --shape 16 16 --radius 3 --hidden 6 --epochs 80 \
  --learning-rate .1 --sigma .03 --grid
```

Deterministic (fixed seed) across all six configurations. The generated playback/error atlases,
summaries and the curated grid index stay local; this page's figures were produced by exactly
this one command.

</div>
