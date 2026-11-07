# Automated Experiments

> **Status:** paused · **Branch:** `main` · **Updated:** 2026-11-07
> **Now:** Workstream defined; implementation deferred to an agent.
> **Next:** Inspect existing synthetic audio/visual sources and recording controls, then implement a ten-second automated run.
> **Blocked:** none
> **Resume:** `copilot --resume <session-id>` (once an agent session exists)
> **Run:** `<to be determined by implementation>`

**Essence.** Make the existing AudioViz application runnable as a configurable, unattended experiment on GitHub Actions. Reuse synthetic audio, synthetic visual input, and recording rather than building a separate simulation framework. Publish the latest recorded result on a simple page for visual inspection.

**Present.** Synthetic audio, synthetic visual input, and recording functionality are reported to exist; their current interfaces and launch-time control have not yet been verified for this workstream. No automated experiment workflow has been implemented here.

**Open.**
- Inspect how to select synthetic inputs and start/stop recording from launch configuration.
- Support a fixed-duration run without interactive input, locally and on a GitHub-hosted Ubuntu runner; use a virtual display only if rendering requires it.
- On relevant experiment-config commits, run the experiment and publish the latest video to a stable, simple GitHub Pages results page; include audio if available.
- Verify GitHub Pages publishing permissions and confirm the produced media can be viewed or retrieved.
- First acceptance test: one configuration produces a ten-second recording end-to-end, with a link to the latest result.

**Future.** Deterministic simulation time, separate input/reconstructed audio recordings, diagnostics and quantitative evaluation, multiple scenarios, and result history.

**Notes.** Keep configuration and workflow in the AudioViz repository; avoid a separate repository or persistent VM. Do not require a separate artifact page for every commit. Keep the experiment configuration runnable locally. Commit SHA can be shown as optional metadata, not the primary navigation. Keep cochlea inference-rule refactoring outside this initial workstream.
