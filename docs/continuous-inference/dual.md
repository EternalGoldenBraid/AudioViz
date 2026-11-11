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

**Native pixel fields** is the default view. Each spatial sheet displays all 96 × 48 RGB sites, without averaging spatial regions. Color uses the sign of each pixel's mean RGB change and opacity uses its RGB RMS change. The hidden neurons remain individual points.

The right sheets show changes in incoming prediction or feedback messages, with fixed whole-recording scales per field. Each pixel is a projected site in the 3D sheet. Focusing a hidden neuron isolates its full-resolution outgoing sensory-prediction and canvas-feedback patterns. Travelling carriers sample individual pixel paths in that focused view; the fields include every pixel. Hidden recurrent connections remain visible. The overview uses grouped carrier paths to keep the volume readable.

Prediction and feedback fields are overlaid separately, not summed into a net nudge. The Channels selector isolates either direction. Pixel fields and the input preview hold the selected recorded boundary; hidden pulses and grouped carriers interpolate between boundaries on the same playback clock.

**Grouped regions** retains the original rendering and its message-value selector. Native fields show changes, not absolute message values.

The left view pulses with **actual state change**. The right view carries **changes in directed messages** by default; the grouped view can also show message values. Node landmarks on the right stay quiet.

Below the paired views, the input preview shows the image supplied at the selected recorded boundary: red and blue blocks with a moving green circle. The initial zero-time frame is blank, before the first observation.

**Foveal observation, full-image prediction.** Only the gold disk supplies visual evidence. The radius-8 disk scans raster rows, holding each fixation for six frames; one 18.2-second pass covers the whole 96 × 48 world. Its center is rounded to world pixel coordinates, and borders clip its observed support. All sensory states and connection coordinates remain world-anchored. Predictions, recurrent dynamics and canvas feedback continue across the full field.

The input image is a scene reference, not a full-image clamp. Its gold outline shows the observed patch. The adjacent preview shows the complete decoded sensory prediction, clipped to 0–1 only for RGB display. Absent evidence is recorded as display-only zero; those coordinates are **unobserved**, not black evidence. Adding or releasing a clamp can itself pulse the evidence and feedback change fields.

The disk geometry is the world-anchored routing slice of [`SharedSensoryIO` on the active-vision branch](https://github.com/EternalGoldenBraid/AudioViz/blob/ac3d587/src/audioviz/readouts/shared_sensory_io.py). This experiment keeps its linear continuous solver, six hidden states and force-driven canvas; foveation changes observation routing, not the full-field compute budget.

In the grouped view and numeric inspector, canvas, sensory and evidence coordinates are grouped into nine spatial regions per layer, covering the entire RGB grid. Every hidden node is shown individually. Grouped state pulses use RMS change per coordinate. Grouped prediction messages retain the full vector when their destination is a region; messages arriving at one hidden node sum the contributing coordinates before measuring amplitude. Opposing contributions can cancel.

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

Use agent-doc-review on this page or the collection index. The generated JSON and PNG atlases stay local.

</div>
