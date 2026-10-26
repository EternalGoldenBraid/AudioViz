# Living Experimentation Catalogue

This directory indexes recorded evidence and quantitative milestones for the continuous whole-scene predictive coding experiments on branch `research/active-vision-eig`. Full narrative analysis and historical baselines are maintained in [`../static-world-experiment-report.md`](../static-world-experiment-report.md).

---

## 2026-10-26 — 4-Stage Static-Scene Learning & Cleared-State Associative Recall

**Model & Setup:**
- **Graph:** `ContinuousScenePredictiveCodingGraph` (`24 x 36` static geometric scene, `H = 6` dense recurrent hidden units, `16` unobserved audio nodes, foveal radius `r = 10` (`278 / 864` world pixels per patch), `dt = 1/30 s`, `tau = 1/3 s`, `M = 8` integration microsteps, `inference_rate = 0.1`, `learning_rate = 0.03`, seed `0`).
- **Controls:** Canvas input and feedback `0.0`; autonomous gaze `OFF`; unobserved ground-truth pixels masked with `NaN` prior to conditioning. Catalogue MSE values are unweighted pre-evidence RGB forecast errors on `sigmoid(z^-)`; videos also show separately labelled top-down and settled-state diagnostics.

### Recorded evidence (`docs/experiments/videos/`)

| Video Artifact | Stage & Schedule | Observed result |
| :--- | :--- | :--- |
| [Revisit learning and passive decay](videos/stage1_2_alternating_revisit_and_decay.mp4) (`1560x900`, 15 fps) | **Stages 1 & 2:** 120 alternating A/B observations, 10 steps per dwell, then 40 Observe-off steps. | At step 110, unobserved-B forecast MSE is `0.14536` frozen, `0.13222` weights-only and `0.06880` fully learned. Observe off adds no evidence or parameter updates; predictions decay toward neutral gray. |
| [Cleared-state cue recall](videos/stage3_cleared_state_cue_recall_proof.mp4) (`1560x920`, 10 fps) | **Stage 3:** 120 **full-scene** training steps, freeze learning, reset `h=z=0`, then 30 steps with only cue A. | Compares all four weight/precision pairings. Learned weights improve prediction outside the cue even with initial precisions; learned precisions alone do not reproduce this recall. |
| [Coverage and long-horizon failure](videos/stage4_revisit_coverage_and_stiffness.mp4) (`1560x880`, 20 fps) | **Stage 4:** four patches, four tours, dwell 30, 480 steps total; revisit gap 90 steps. | Full learning reaches whole-world forecast MSE `0.11266` at step 120, then degrades to `0.15941` at step 480 as precisions grow. Weights-only ends at `0.11944`; improvement is not monotonic. |

The recall recording uses **full-scene training**, unlike the alternating-A/B,
three-seed recall table in the experiment report. Its seed-0 target-B forecast
MSEs at cue step 30 are:

| Recall parameters | B forecast MSE |
| --- | ---: |
| Initial weights, initial precisions | 0.15447 |
| Learned weights, initial precisions | 0.14023 |
| Learned weights, learned precisions | 0.07547 |
| Initial weights, learned precisions | 0.15460 |

The cue covers approximately one-third of this small, previously seen scene.
These recordings support associative learning under the stated conditions,
not generalization to new scenes or long-horizon stability. The stiffness
threshold plotted in the final video is a fixed-hidden scalar reference,
not a stability certificate for the nonlinear, coupled, clipped system.
Precision growth and degradation coincide; their separate causal contributions
have not been isolated.

Videos are H.264/yuv420p; playback frame rate is distinct from model `dt`.
The generator (`render_benchmark_videos.py`), numerical runner and JSON results
remain in Math Agent session
`45344cb2-dfc9-4bd4-a39f-e4dfb2c07704/files/`, not this repository.
