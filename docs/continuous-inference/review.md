---
layout: page
---

<script setup>
import Review from "./Review.vue";
</script>

<div class="vp-doc inference-page">

# Continuous inference: follow the actual gradients

<Review />

## Energy and updates

With $a=\tanh(h)$ and $r=\tanh(c)$, the model residuals are

$$
e_h=h-W_c r-W_h a,\qquad e_z=z-W_s a,\qquad e_y=y-\sigma(z).
$$

The root aperture masks $r$ when present; only observed coordinates contribute $e_y$.
Audio predictions use the linear decoder $z$. All residual precisions are fixed at one.

$$
E=\frac{\|h\|^2+\|z\|^2}{2\tau}
+\frac{\|e_h\|^2+\|e_z\|^2+\|e_y\|^2}{2}
+\frac{\lambda}{2}\sum_W\|W\|^2.
$$

Inference uses $x\leftarrow x-\eta\,\Delta t\,\nabla_x E/M$ for $M$ microsteps.
Learning uses $W\leftarrow W-\eta_W\nabla_W E$ at the settled state.
Canvas feedback is $c\leftarrow c-\eta_c\nabla_c E$, applied to both wave states to preserve velocity.
The rates set update sizes; they do not temper the energy. The recurrent diagonal is structurally zero.
The shared legacy configuration's gradient and weight clipping controls do not apply to this solver.

Observation error can remain nonzero at an energy minimum: the observation term competes with state decay and latent coupling.
The black/gray/white and XOR results above expose this limitation, not a successful reconstruction.

## Integration boundary

This page exercises the existing research continuous solver ported into `agent/integration`.
The running application still uses its existing per-pixel readouts; the continuous solver is not yet connected
to the TUI, cochlea, local camera or LAN-camera owner.
The captured weights, residuals and gradients are evaluated before each frame's connection update.
The graph groups large populations; matrix panels explicitly sample three RGB pixels.

## Reproduce

From this worktree:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_inference_review.py \
  --output-dir docs/continuous-inference --frames 600
```

Use agent-doc-review to open this document. The generated `capture.json` and `frames/`
stay local and are ignored by Git.

</div>

<style>
.inference-page { max-width:1680px; margin:auto; padding:24px 28px 96px; }
.inference-page h1 { margin-bottom:20px; font-size:28px; line-height:1.3; }
.inference-page > h2 { margin-top:32px; }
.inference-page mjx-container { overflow-x:auto; overflow-y:hidden; padding:8px 0; }
@media (max-width:800px) {
  .inference-page { padding:20px 16px 96px; }
  .inference-page h1 { font-size:24px; }
}
</style>
