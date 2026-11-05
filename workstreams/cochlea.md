# Cochlea: raw-audio predictive oscillator bank

> **Status:** active · **Branch:** `agent/cochlea` · **Updated:** 2026-11-05
> **Now:** shared process-owned cochlea and oscillator-history views available for review.
> **Next:** review the equations and listen to input, prediction, and residual before tuning the model.
> **Blocked:** none.
> **Resume:** `copilot --resume 405c9084-e5f0-490d-a823-aaa010083ca4`
> **Run:** `PYTHONPATH=src uv run --group tui python -m audioviz.tui.app --audio synthetic --no-camera`; select **audio model: cochlea**.

**Essence.** Sixteen fixed damped oscillators generate a one-sample-ahead raw
waveform prediction by summing their positions. The scalar raw-audio residual
corrects their states. This is a generative observer, not a preprocessing filter
bank. Position and quadrature provide 32 signed graph coordinates.

**Present.** Opt-in shared graph integration; exact stable free dynamics; streaming
inference; input/prediction/error listening; waveform comparison and residual
views. Spectrograms remain diagnostics. Graph state targets are frozen after raw
inference; a saved pre-observation graph expectation corrects the next block's
starting state. Oscillator parameters stay fixed; existing graph weight learning
is optional. The spectral model remains the default. Qt and attached TUI controls
select the same process-owned bank. Both render its bounded oscillator-index/time
history; no view advances the model. Switching models resets only the audio
branch and cross-modal connections, preserving the canvas and camera branch.

**Documentation.** [Equations, timing, limitations, and diagrams](../docs/cochlea/index.md).
Editable two-page source: `docs/cochlea/cochlea.drawio`.

**Validation.** `uv run --group dev python -m pytest -q tests/audio_processing/test_cochlea.py tests/tui/test_cochlea_runtime.py`.
`PYTHONPATH=src uv run python scripts/validate_cochlea.py` writes an illustrative
two-tone figure, WAV listening taps, and measured core throughput.

**Open.** Assess non-bank-tuned tones, speech, noise, and slow graph phase
expectations. A small fixed bank is not an exact waveform representation.

**Future.** Learnable oscillator parameters and explicit temporal graph forecasting.
These require separate validation; no change to ripple
physics belongs in this stream.
