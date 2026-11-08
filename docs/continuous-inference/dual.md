---
layout: page
---

<script setup>
import Dual from './Dual.vue'
</script>

<div class="vp-doc experiment-page">

[All experiments](./) · [Learning and canvas mechanics](./review)

# Node pulses and connection ripples

Two views, one continuous-model trajectory. Drag either view to rotate both.

<Dual />

## What is measured

The left view pulses with **actual state change**. The right view carries **changes in directed messages** by default; its selector can also show message values. Node landmarks on the right stay quiet.

Below the paired views, the input preview shows the image supplied at the selected recorded boundary: red and blue blocks with a moving green circle. The initial zero-time frame is blank, before the first observation.

Canvas, sensory and evidence coordinates are grouped into nine spatial regions per layer, covering the entire RGB grid. Every hidden node is shown individually. Grouped state pulses use RMS change per coordinate. Grouped prediction messages retain the full vector when their destination is a region; messages arriving at one hidden node sum the contributing coordinates before measuring amplitude. Opposing contributions can cancel.

Predictions use weighted source activity. Feedback uses weighted residuals and the source activation derivative. The evidence connection has an identity prediction and an observation-error return. Hidden recurrence is retained, so opposite prediction directions remain independent connections.

Both panels sample the same completed simulation interval: neural inference, one learning update, then mechanical advance. Messages are evaluated with the resulting states and current weights. Message changes therefore include learning as well as state motion. Local decay, externally supplied evidence and canvas inertia also matter; these two views are not expected to be identical.

The inspector reports objective gradients, not applied updates. Evidence can have a nonzero gradient while remaining fixed by the external observation.

Color shows the sign of the mean value/change. Brightness and size use RMS magnitude, with fixed whole-recording scales for each node layer or message family/direction. These are communicated signals, **not transported energy**. Moving carriers and expanding rings are visual encodings; they do not model propagation delays.

The model remains at six shared hidden states, with the same linear output, learning rate and unit inference force as the mechanics experiment. Main's live engine and the old reference are unchanged.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_dual_review.py \
  --output-dir docs/continuous-inference --steps 600
```

Use agent-doc-review on this page or the collection index. The generated `dual-capture.json` stays local.

</div>
