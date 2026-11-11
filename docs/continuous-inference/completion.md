---
layout: page
---

<script setup>
import CompletionImageGrid from './CompletionImageGrid.vue'
import CompletionSeparabilityStatic from './CompletionSeparabilityStatic.vue'
import CompletionExplorer from './CompletionExplorer.vue'

import patchTargetLines from './highlight-patch-target-lines.png?url'
import patchTargetCircles from './highlight-patch-target-circles.png?url'
import patchSoftLines from './highlight-patch-completion-soft-small-lines.png?url'
import patchSoftCircles from './highlight-patch-completion-soft-small-circles.png?url'
import patchHardLines from './highlight-patch-completion-hard-small-lines.png?url'
import patchHardCircles from './highlight-patch-completion-hard-small-circles.png?url'

import sceneTargetLines from './pattern-lines.png?url'
import sceneTargetCircles from './pattern-circles.png?url'
import sceneSoftLines from './highlight-scene-completion-soft-small-lines.png?url'
import sceneSoftCircles from './highlight-scene-completion-soft-small-circles.png?url'
import sceneHardLines from './highlight-scene-completion-hard-small-lines.png?url'
import sceneHardCircles from './highlight-scene-completion-hard-small-circles.png?url'

const columns = [
  { key: "target", label: "Target" },
  { key: "soft", label: "Soft clamp (0)" },
  { key: "hard", label: "Hard clamp (1.0)" },
]
const patchImages = {
  "lines-target": patchTargetLines, "lines-soft": patchSoftLines, "lines-hard": patchHardLines,
  "circles-target": patchTargetCircles, "circles-soft": patchSoftCircles, "circles-hard": patchHardCircles,
}
const sceneImages = {
  "lines-target": sceneTargetLines, "lines-soft": sceneSoftLines, "lines-hard": sceneHardLines,
  "circles-target": sceneTargetCircles, "circles-soft": sceneSoftCircles, "circles-hard": sceneHardCircles,
}
</script>

<div class="vp-doc experiment-page">

[All experiments](./) · [Node pulses and connection ripples](./dual)

# Scene completion from a raster-scanned fovea

Can the model complete a **static scene it has already learned**, using only a small patch
strictly smaller than the scene? Training scans the whole world; evaluation only ever fixates
one small center patch, holding the eye open the whole time — whatever it looks at is the only
clamp.

Every figure below compares the same two curated configurations (τ=1/3 soft clamp vs τ=3 hard
clamp), chosen because they show the largest effect in the full six-configuration grid. A free
explorer for all six configurations together is at the bottom of the page.

## Does the patch reach its own target?

Zoomed crop around the fovea, settled final frame. Soft clamp (the solver's own decay-dominated
pull) barely reaches the target; hard clamp reconstructs it almost exactly.

<CompletionImageGrid :rows="['lines', 'circles']" :columns="columns" :images="patchImages" />

Increasing τ alone (not pictured — see reproduce command for the full sweep) moves this far less
than clamp strength does. This confirms a soft-clamp diagnosis directly: the fixed decay rate
dominates the observation pull at small τ, not a learning or capacity problem.

## Does the rest of the scene complete correctly?

Same two configurations, full scene instead of a zoomed patch.

<CompletionImageGrid :rows="['lines', 'circles']" :columns="columns" :images="sceneImages" />

Fixing in-patch reconstruction does **not** fix completion: the area outside the patch stays
visibly near-blank in both configurations, and the circles scene often predicts something closer
to the *lines* pattern than to its own. The decay/clamp balance and the completion failure are
separate problems.

## Are the hidden units even separable?

<CompletionSeparabilityStatic />

The hard-clamp configuration drives noticeably larger hidden-unit excursions for the lines
pattern than for circles. There is *some* separation signal reaching the hidden layer, but it
stays far from the ±1 range `tanh` actually uses, and separation alone does not translate into
correct completion. The next open question is whether `Ws` (hidden→sensory) is undertrained to
carry that signal outward, or whether the patch content itself is too ambiguous — not whether the
hidden layer can represent a difference at all.

## Explore all six configurations

Three clamp strengths × two memory time constants, replacing an exhaustive sweep whose
combinatorics would grow too fast to show meaningfully. This is the only interactive part of the
page; everything above already makes its point without it.

<CompletionExplorer />

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_scene_completion.py \
  --output-dir docs/continuous-inference --shape 16 16 --radius 3 --hidden 6 --epochs 80 \
  --learning-rate .1 --sigma .03 --grid
```

Deterministic (fixed seed) across all six configurations. The generated playback atlases,
summaries, section-highlight crops and the curated grid index stay local; this page's figures
were produced by exactly this one command.

</div>
