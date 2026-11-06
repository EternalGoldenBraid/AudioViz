# Stereo occlusion: predicting one camera from the other

> **Status:** idea · **Branch:** `agent/tui` (builds on the merged `agent/multi-device`) · **Updated:** 2026-11-05
> **Now:** user is live-testing the two-camera setup (EO `/dev/video0`, IR `/dev/video2`).
> **Next:** define the occlusion readout: cover one camera, compare its prediction with what it would have seen.
> **Blocked:** none
> **Resume:** `copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e`
> **Run:** `uv run python main.py`, then check both cameras in the `camera` device knob.

**Essence.** Two simultaneous cameras give the model a stereo-like pair of observations.
After the feed has run for a while, covering one camera should leave the model still able
to predict roughly what that camera would see from the other one, the way a covered eye
keeps its stereoscopic sense of the scene.

**Present.** Two cameras open at once and each drives its own scene source (`agent/multi-device`).
The model predicts the canvas as before; nothing yet scores the prediction of a covered camera.

**Open.**
- The pair is EO and IR: different modalities and viewpoints, not a matched stereo pair. The cross-prediction may need to learn the modality mapping as well as the geometry.
- How to measure "still predicts": per-camera prediction error against the real frame while uncovered, then the same error against the held-out frame while covered.
- How to cover a camera in a repeatable way: a physical cover, or a knob that withholds one camera's observation from correction while still recording it.
- Whether the prediction needs a separate readout per camera or one shared hidden state.

**Notes.** The experiment needs the multi-device correction path (`observe_secondary_camera` in `sources/orchestrator.py`) and the per-camera error views.
