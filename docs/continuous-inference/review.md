---
layout: page
---

<script setup>
import Review from "./Review.vue";
</script>

[All experiments](./) · [Node pulses and connection ripples](./dual)

<div class="vp-doc inference-page">

# Continuous inference: follow the actual gradients

<Review />

## Paired connection updates

The old per-pixel engine and new continuous solver are recorded against the same full-frame input.
The weight comparison follows the video's frame and waves-on/held selection. Cells can show
pre-learning weights, actual applied updates, or cumulative changes from initialization through the selected frame.
Both implementations' updates are measured from their actual post-learning minus pre-learning parameters.
The old engine retains its pixel-averaged learning, gradient clipping and weight clipping.
Color limits are shared by the two implementations and fixed across the selected dynamics recording for each matrix and cell-value mode.
Scientific notation exposes changes below ordinary decimal display precision.

The architectures are not capacity matched: the old model has six hidden states per pixel and shared RGB weights;
the continuous model has six shared hidden states and dense whole-image weights.
All old RGB matrix entries are shown; continuous Wc/Ws panels sample three RGB pixels.
The old camera-only model has no same-layer recurrent matrix, so its Wh panel is explicitly unavailable.
Full-matrix applied-update norms include connections outside the sampled continuous panels.

## Energy and updates

With $a=\tanh(h)$ and $r=c$, the model residuals are

$$
e_h=h-W_c r-W_h a,\qquad e_z=z-W_s a,\qquad e_y=y-z.
$$

The root aperture masks $r$ when present; only observed coordinates contribute $e_y$.
Visual and audio predictions use the linear decoder $z$. All residual precisions are fixed at one.
Inference and learning use unclipped predictions; RGB clipping applies only to display.

$$
E=\frac{\|h\|^2+\|z\|^2}{2\tau}
+\frac{\|e_h\|^2+\|e_z\|^2+\|e_y\|^2}{2}
+\frac{\lambda}{2}\sum_W\|W\|^2.
$$

Inference uses $x\leftarrow x-\eta\,\Delta t\,\nabla_x E/M$ for $M$ microsteps.
Learning uses $W\leftarrow W-\eta_W\nabla_W E$ at the settled state.
The rates set update sizes; they do not temper the energy. The recurrent diagonal is structurally zero.
The shared legacy configuration's gradient and weight clipping controls do not apply to this solver.
The canvas readout is linear. Its unit-mass inference force is
$F_{\mathrm{PC}}=-g\nabla_c E=gW_c^\top e_h$ within the root aperture.
The signed canvas renderer's display nonlinearity does not enter the inference or force.

Sensory states update during inference even when connection learning is disabled.
Incoming predictions pull $z$ toward $W_s\tanh(h)$, observation error pulls the decoded prediction toward $y$,
and state decay pulls $z$ toward zero. Connection learning subsequently changes the predicting weights
using the settled residuals.

Observation error can remain nonzero at an energy minimum: the observation term competes with state decay and latent coupling.
With a zero incoming prediction, an isolated linear sensory coordinate settles at
$z=y/(1/\tau+2)$, or $z=y/5$ at the current $\tau=1/3$.
The black/gray/white and XOR results above expose this limitation, not a successful reconstruction.

The same solver accepts `visual_decoder="relu"` for a conventional ReLU decoder.
Its derivative is zero at zero and for negative visual states; audio stays linear.
The ReLU probe uses the same zero-root, zero-state initialization and learning settings.
It stalls for nonzero visual evidence because the initial observation drive is zero.
The default decoder is linear; the main video and synchronized graph use that default.

## Force-driven canvas and clocks

The continuous experiment uses unit mass and an explicit velocity:

$$
\ddot c=s^2\nabla^2c-\gamma\dot c-g\nabla_cE.
$$

Each mechanical substep computes acceleration from waves and inference,
then updates $v_{\mathrm{next}}=(v+\Delta t\,a)/(1+\gamma\Delta t)$
and $c_{\mathrm{next}}=c+\Delta t\,v_{\mathrm{next}}$.
Damping acts on velocity. Canvas displacement is a consequence of motion;
no correction shifts the current and previous positions together.

`--inference-force-strength` controls $g$, defaulting to one.
At zero, inference applies no canvas force; intrinsic waves still run.
Canvas-to-hidden predictions, sensory conditioning and connection learning remain enabled.
This is one-way back-action control, not a disconnection of the whole graph
or a conserved energy budget redistributed among nodes.

Every video frame advances the new neural and mechanical dynamics through the same $1/30$-second interval.
The integrator first settles neural states at the interval-start canvas, then advances mechanics
using those settled states and pre-learning weights. Force is reevaluated at each mechanical position.
Weight learning still occurs once per frame, not once per mechanical substep.
The neural energy/gradient telemetry describes the state before mechanical advance;
canvas peak and velocity describe the end of that interval.

The mechanical step bound is $2/\sqrt{8(s/\Delta x)^2+g\|W_c\|_2^2}$,
including the inference spring's restoring curvature. The recorded bound uses the full Wc matrix,
which is conservative when the root aperture masks coordinates.
Both new wave cases also respect the reference wave-only CFL limit; the interval is divided into enough equal substeps.
At the current settings this is about 31 mechanical steps per frame.
The new held case sets wave speed to zero while retaining force-driven motion and damping.
The damping rate is derived as $\gamma=-\log(0.988)/\Delta t_{\mathrm{legacy}}$;
this is velocity friction, while the unchanged old integrator multiplies its next position by 0.988.

The old reference keeps its original inference, learning, displacement corrections and one CFL wave step per video frame.
Consequently it is not clock matched to the new experiment.

## Integration boundary

This page exercises the existing research continuous solver ported into `agent/integration`.
The running application still uses its existing per-pixel readouts; the continuous solver is not yet connected
to the TUI, cochlea, local camera or LAN-camera owner.
The captured weights, residuals and gradients are evaluated before each frame's connection update.
The graph groups large populations; continuous matrix panels explicitly sample three RGB pixels.
The comparison video and graph telemetry come from the same run, captured at 30 frames per second.
The graph follows video-frame presentation callbacks; there is no independent playback timer
or interpolated neural state. Video frame $k$ displays the state after inference interval $k$.
The baseline retains its existing per-pixel update rule; the continuous solver uses the energy above.

## Reproduce

From this worktree:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/continuous_inference_review.py \
  --output-dir docs/continuous-inference --frames 600 --inference-force-strength 1
```

Use agent-doc-review to open this document. The generated `capture.json` and `comparison.mp4`
stay local and are ignored by Git. FFmpeg must be available for video encoding.

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
