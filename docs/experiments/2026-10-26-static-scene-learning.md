# E001: Static-scene learning and cleared-state recall

[Diary and next experiment](README.md) | [Full experiment report](../static-world-experiment-report.md#controlled-learning-and-cleared-state-recall-2026-10-26)

**Date:** 2026-10-26

**Status:** Completed; associative recall supported in the tested scene, long-horizon stability not established.

This is a retrospective record of the four-stage benchmark on branch
`research/active-vision-eig`, not a preregistered experiment.

## Goal and research questions

Distinguish fitting the current observation from learning a scene that can be
recalled after transient neural states have been cleared.

- **RQ1:** Do learned connections improve predictions of previously seen regions
  outside the current cue, independently of retained neural states and changes
  to precision?
- **RQ2:** Does that benefit survive longer training and revisit gaps?

**Hypothesis under examination:** Learned connections support cue-driven recall
after reset. Longer revisits test whether that benefit remains stable.

## Method and controls

- **Graph:** `ContinuousScenePredictiveCodingGraph` (`24 x 36` static geometric scene, `H = 6` dense recurrent hidden units, `16` unobserved audio nodes, foveal radius `r = 10` (`278 / 864` world pixels per patch), `dt = 1/30 s`, `tau = 1/3 s`, `M = 8` integration microsteps, `inference_rate = 0.1`, `learning_rate = 0.03`, seed `0`).
- **Controls:** Canvas input and feedback `0.0`; autonomous gaze `OFF`; unobserved ground-truth pixels masked with `NaN` prior to conditioning. MSE values here are unweighted pre-evidence RGB forecast errors on `sigmoid(z^-)`; videos also show separately labelled top-down and settled-state diagnostics.
- **Comparison:** Frozen parameters, learned weights with precisions fixed at
  initialization, and learned weights plus precisions. Recall additionally
  crosses initial/learned weights with initial/learned precisions after reset.
- **Coverage:** Recordings use seed `0`. The report's stages 1-3 aggregate seeds
  `0, 1, 2`; its long-horizon sweep uses seed `0` only.

## Results and recordings

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

## Interpretation and limits

The cue covers approximately one-third of this small, previously seen scene.
These recordings support associative learning under the stated conditions,
not generalization to new scenes or long-horizon stability. The stiffness
threshold plotted in the final video is a fixed-hidden scalar reference,
not a stability certificate for the nonlinear, coupled, clipped system.
Precision growth and degradation coincide; their separate causal contributions
have not been isolated.

## Decision and next step

Keep the cleared-state, matched-precision recall comparison as a control for
stored connection learning. Do not treat adaptive precision learning as stable
or choose a precision cap or update gate from this evidence alone.

**Next: [E002 - separate integration sensitivity from precision-learning failure](README.md#next-experiment-e002).**
Repeat the long-horizon schedule at finer integration resolutions before
choosing a numerical or statistical correction. Autonomous-gaze comparison
follows a reliable fixed-gaze baseline.

## Artifacts and reproducibility

Videos are H.264/yuv420p; playback frame rate is distinct from model `dt`.
The generator (`render_benchmark_videos.py`), numerical runner and JSON results
remain in Math Agent session
`45344cb2-dfc9-4bd4-a39f-e4dfb2c07704/files/`, not this repository.
The recordings are committed evidence; exact reruns still depend on those
session-only sources. Preserve the runner and machine-readable output in the
repository with the next experiment.
