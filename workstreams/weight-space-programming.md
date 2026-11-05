# Weight-space programming

> **Status:** idea (first encounter only) · **Branch:** `agent/tui` · **Updated:** 2026-11-05
> **Now:** nothing running; note written from the blog post.
> **Next:** dump the readout weights with a layout spec, ask a frontier model what the hidden units respond to, compare with measured responses.
> **Blocked:** needs a weights export and a choice of readout and model.
> **Resume:** `copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e` (main TUI session; no own session yet)
> **Run:** none yet

Source: https://rikiyat.github.io/blog/weight-space-programming/

A frontier LLM reads, writes and edits a network's weights as text, without gradients. It names what a small net was trained for, writes game-playing nets from scratch, and fixes a failure by editing 0.016% of a language model's weights. The useful variant has the LLM interleave edits with a few forward passes. Idea for us: let an LLM interact with the computational graph.

**Good for.** Small nets whose weights have legible structure (first-layer weights show what is attended to, the output layer shows what is produced). Targeted fixes: find one failure direction, remove it. Writing a net from a rule-like spec.

**Not good for.** Weights with no structure to read. The blog's sycophancy test could not be solved from weights alone. Large dense nets that don't fit in context without tools. Anything that needs a fast loop: the frontier model is slow and costs money, and a small local model (the Ollama idea) is not in the class the results come from.

**Why it might work here.** Our readout is small and has spatial structure: a grid, a spectrogram axis, hidden units. `PredictiveCodingReadout` weights are plausibly legible, much like the blog's game networks. The TUI already exposes state and keys through the control socket, so an agent has a forward-pass loop (`state`, `screenshot`, `keys`). Seeded attractors in the hidden view are a write-from-spec task.

**Why it might not.** The model is a predictive-coding inference loop (86% of the step time is `infer_joint`). The behaviour comes from dynamics on top of the weights, not from a feed-forward map, so reading the weights may say little about it. Learning is off by default, so there may be nothing learned to read. Behaviour is subjective (what should the canvas look like?), so "fix" has no objective score yet.

**What would have to change.**
- A weights export: dump a readout's tensors as text, plus a spec of the input and output layout.
- An objective: a scalar score per scene (error energy, attractor stability, a test clip) for the agent to keep or reject an edit.
- A budgeted forward-pass tool: run N steps on a clip and return summary statistics, not a live stream.
- A sandbox for edits: apply to a copy, diff, revert.

**First encounter, cheapest.** Dump the readout weights with a spec and ask a frontier model what the hidden units respond to, then compare against measured responses to tones and camera input. This tests only the read half, with no loop and no app changes beyond the export.

**Open.** Which readout, and with learning on or off? Which frontier model and budget? Does the live-test-agent's check runner double as the forward-pass loop?
