# Cochlea: resonator-bank audio frontend

> **Status:** active · **Branch:** `tui/lenovo` · **Updated:** 2026-11-05
> **Now:** design agreed; implementation not started.
> **Next:** implement and test a streaming damped-oscillator bank, then connect it to the existing audio consumers.
> **Blocked:** none.
> **Resume:** new implementation-agent session.
> **Run:** existing AudioViz entry point; add a focused test command with implementation.

**Essence.** Build a simplified artificial cochlea: N parallel damped mass–spring resonators driven by the incoming audio samples. Their continuously evolving states provide frequency-selective activity at multiple temporal scales, without requiring fixed-window spectrograms.

**Present.** Design only; no cochlea implementation or validation yet. The current audio pipeline and TUI integration should remain operational during the refactor.

**Open.** Choose initial N, frequency range, Q, output representation (oscillation, envelope/energy), and whether existing spectrogram consumers need an adapter. Discretization must remain numerically stable at audio sample rates; avoid naive explicit Euler for high-frequency resonators.

**Future.** Learnable resonator parameters, nonlinear/cochlear coupling, and predictive-coding integration.

**Notes.**
- For each node: `m*x'' + b*x' + k*x = g*s(t)`; fix `m=1`, set `f_i` logarithmically from `f_min` to `f_max`, `k_i=(2*pi*f_i)^2`, and `b_i=2*pi*f_i/Q_i`. Start with shared Q.
- Preserve phase/state continuously across audio blocks; output per-node activity and optionally a smoothed envelope. Low frequencies have longer characteristic times with fixed Q.
- Prefer a stable discrete resonator/filter realization (e.g. exact state transition or stable biquad) rather than sample-by-sample Python loops over nodes.
- Tests: pure tones peak near the intended node, mixtures activate multiple bands, silence decays, block-size independence, streaming continuity, finite/stable response across the configured range, and real-time throughput.
- Integrate behind a small audio-feature interface; keep visualization/excitation consumers working, using an adapter where needed. Do not change the ripple physics in this workstream.
