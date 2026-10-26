# Experimentation diary

**Goal:** Learn a static scene through partial observations, recall previously
seen regions outside the current view, and eventually use gaze to acquire
evidence that improves prediction.

This page is the current decision point. Dated entries preserve what was run,
what happened and why the next experiment follows.

## Research questions and current evidence

| Question | Current answer | Evidence |
| --- | --- | --- |
| **RQ1: Is scene information stored in learned connections, not just transient activity?** | Supported for a small familiar scene: cue-driven recall survives reset and improves with learned weights at matched initial precisions. Generalization is not established. | [E001: cleared-state recall](2026-10-26-static-scene-learning.md#results-and-recordings) |
| **RQ2: Can learning remain stable over longer training and revisit gaps?** | Not established. Full learning degrades during the 480-step run; weights-only remains stable in that tested schedule. Precision growth coincides with degradation, but causality is unresolved. | [E001: limits](2026-10-26-static-scene-learning.md#interpretation-and-limits) |
| **RQ3: Does uncertainty-guided gaze improve learning over controlled gaze schedules?** | Not established by these experiments. Deferred until the fixed-gaze learning baseline is reliable. | [Detailed report](../static-world-experiment-report.md) |

## Next experiment: E002

**Status: Proposed; not run.**

**Question:** Is the long-horizon failure sensitive to numerical integration,
and what precision-learning problem remains as integration is refined?

Repeat E001's four-patch, dwell-30, 480-step schedule with:

- **One intervention:** Increase integration microsteps from `8` to `32` and
  `128`, retaining physical `dt`, model time constants, inference rate, learning
  rate and parameter-update cadence per physical step. Change no objective,
  precision bounds, update masks or gaze policy.
- **Matched controls:** Run both weights-only and weights-plus-precisions at each
  resolution, using the same initial parameters and evidence sequence within
  each of seeds `0, 1, 2`.
- **Measurements:** Whole-world and outside-current-patch pre-evidence RGB MSE,
  hidden-state norm, coupling and hidden precision distributions, and gradient
  clipping frequency. Compare trajectories and tour boundaries, not just the
  final image.

| Outcome | Decision it informs |
| --- | --- |
| Finer integration materially changes or removes degradation | Numerical sensitivity contributes. Establish convergence before judging the learning rule. Growing precisions can remain a separate unresolved problem. |
| The two finer resolutions agree closely but both degrade | Integration refinement alone has not resolved the failure. Next isolate the precision-learning objective and hidden-state contraction. |
| Results still change with resolution, or vary substantially across seeds | Inconclusive. Refine the numerical comparison or characterize variability before choosing a correction. |

**Completion record:** Set a trajectory-agreement tolerance before running.
Commit the runnable protocol and machine-readable
per-step results alongside a dated entry. Report resolution differences
explicitly; do not infer convergence from a single stable-looking run.
Link recordings where they help explain the outcome.

**After E002:** Use its result to choose the smallest justified correction,
repeat cleared-state recall to check that it survives, then compare
uncertainty-guided gaze against controlled schedules.

## Diary

| Date | Experiment | Status and decision |
| --- | --- | --- |
| 2026-10-26 | [E001: static-scene learning and cleared-state recall](2026-10-26-static-scene-learning.md) | Completed. Recall supported within the tested setup; long-horizon failure motivates E002. Includes three recordings. |

## Recording the next entry

Use `YYYY-MM-DD-short-title.md` with an experiment ID and status
(`proposed`, `running`, `completed` or `inconclusive`), followed by:

**Goal / question -> Hypothesis -> Method and controls -> Results ->
Interpretation and limits -> Decision and next step.**

Link the runner, configuration, results and relevant recordings. Separate
measurements from explanations, and identify retrospective hypotheses.
Every entry should state which decision the evidence supports, including when
the result is negative or inconclusive. Update this front page when that
decision changes; keep the dated evidence in its entry.
