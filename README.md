# Audio Ripple

**Audio Ripple** is an immersive audio-reactive visualizer designed for live performance, interactive installations, and creative experimentation. It transforms sound into dynamic ripple waves that animate a 2D plane, offering a perceptually compelling visual feedback loop between the sonic and the spatial.

---

## 🌊 What is Audio Ripple?

Audio Ripple generates a simulated wave propagation field in response to real-time audio input. Microphone signals are translated into visual energy ripples using a finite difference approximation of the 2D wave equation. It enables:

* Audio-driven ripple simulations
* Real-time visual feedback for musical or spoken input
* Projected visuals in installations and performances
* A canvas for future multimodal interactions (e.g., dance, pose-tracking)

---

## 🚀 Features

- Real-time wave propagation visualized on a 2D surface
- Flexible audio input: default is microphone via `sounddevice`, but any input device can be configured
- Optional synthetic tone generator for testing
- Optional GPU acceleration (via CuPy); CPU wave propagation is the default
- Adjustable wave physics: damping, decay, speed, amplitude
- Three-channel wave surface with independent persistent visual/audio hidden belief states
- Local nonlinear predictive coding: sensory surprise revises hidden states and the canvas
- Learned sensory pathways with fixed substrate conductances; the rendered image is the canvas surface
- Modular audio processor and visualizer structure

---

## 🔧 Installation

If `pixi.sh` install from `pyproject.toml`
```bash
pixi update
```
or
```bash
pip install -r requirements.txt (get from `pyproject.toml`)
```
```
python main.py
```

Adjust runtime settings in `main.py`, including resolution, plane size, and optional GPU usage.

For live streaming, the first run prompts for audio devices and stores the
machine-local selection in `config/audio_devices.json`. The file is ignored by
Git; `config/audio_devices.example.json` documents its structure and safe null
device values.

---

## 🧍 Pose Model

The pose graph demo expects a local MediaPipe Pose Landmarker `.task` model when
the installed MediaPipe version uses the Tasks API. Use the Lite bundle as the
mobile-friendly default:

```bash
mkdir -p models
curl -L -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Run the camera demo with the optional pose dependencies:

```bash
uv run --group pose-demo python scripts/pose_graph_demo.py --model-path models/pose_landmarker_lite.task
```

The downloaded `.task` bundle is ignored by git.

## Offline pose+ripple validation

Run a deterministic, offscreen end-to-end validation without live camera input:

```bash
uv run python scripts/validate_pose_ripple_render.py --frames 16 --synthetic-frequency 220 --synthetic-frequency 330 --pose-graph ring --pose-nodes 5
```

This writes PNG frames under `outputs/pose_ripple_validation/` and records an animated GIF by default. The offline dummy pose graph also carries a synthetic body mask so the same internal-boundary logic is exercised without live video. Use `--no-synthetic` to confirm the embedded pose medium stays quiescent without an external source, or pass `--video-path outputs/pose_ripple_validation/demo.mp4` when OpenCV is installed.

---

## 🎛️ Interactive Controls

Accessible from GUI sliders:

* **Damping**: Controls wave attenuation over time (physical energy loss)
* **Decay α**: Sets spatial spread using distance normalized by the field diagonal
* **Amplitude**: Controls excitation strength
* **Speed**: Wave propagation speed in meters per second
* **RGB Canvas**: Displays the three simulated canvas channels directly as red, green, and blue
* **Sensory Inference and Learning**: Shared controls for camera/audio hidden-state inference, canvas correction, and weight learning
* **Substrate Conductances (Fixed)**: Diagnostic overlay of the wave operator, not the learned visual weights
* **Show Inference Diagnostics**: Opens a toggleable live energy and learning window

## Predictive canvas

The wave surface has two learned sensory branches:

```text
                 visual hidden field <-> RGB camera evidence
                /
canvas surface
                \
                 audio hidden field  <-> spectral audio evidence
```

Each branch has persistent states and independent predictive weights. Sensory
evidence is clamped during inference; neither microphone input nor camera
frames are injected directly into the surface. Synthetic sources remain
explicit wave drives. Sound now revises belief through prediction error,
so the former frequency-to-ripple forcing pattern is not preserved.

Each physical frame:

1. Propagate the wave surface using its existing, fixed-conductance dynamics.
2. Advance persistent hidden belief toward the propagated surface's prediction,
   and save camera and audio expectations **before** acquiring their evidence.
3. Acquire available camera evidence and fresh audio evidence. At every local
   inference step, compute both branches' directions from the same canvas and
   pre-update states, sum their canvas directions, then advance all active
   states together. Weights stay fixed during this bounded loop.
4. Optionally learn each active branch's weights once from the final local errors.
5. Apply the net surface correction to both wave-state buffers, preserving wave
   velocity, and render the corrected **surface**, not the camera expectation.

Missing evidence skips inference and learning for that branch; the other
branch can still correct the shared surface. Both hidden fields continue to
advance from the wave prior. Reset clears the surface, both hidden fields,
and errors, retaining their learned weights. Weights are session-local.

### Local nonlinear updates

Following [Salvatori et al., *Learning on Arbitrary Graph Topologies via
Predictive Coding*, section 2, equations 1-4](https://arxiv.org/html/2201.13180),
with destination-by-source weight matrices:

```text
c = surface; h = hidden state; y = camera evidence
f(x) = tanh(x); f'(x) = 1 - tanh(x)^2

predicted_h = W_hc f(c)
predicted_y = W_yh f(h)
e_h = h - predicted_h
e_y = y - predicted_y
E = 1/2 sum(e_h^2 + e_y^2)

delta_h = hidden_rate [-e_h + f'(h) * (W_yh.T e_y)]
delta_c = canvas_rate [f'(c) * (W_hc.T e_h)]

delta_W_hc = learning_rate e_h f(c).T
delta_W_yh = learning_rate e_y f(h).T
```

The `*` in state updates is elementwise multiplication; weight updates are outer
products. Both state directions are computed before either state is changed.
The surface is the unpredicted top layer: there is **no additional
surface-minus-wave-prior restoring term**. The wave prior supplies its initial
state for each frame.

There are six hidden channels per pixel by default. The two weight matrices are
shared across pixels, so weight updates average their local contributions over
the image. There is no dense all-pixel connection matrix. Spatial propagation
remains the wave solver's responsibility.

Optional L2 weight decay and symmetric gradient/weight clipping are explicit
learning safeguards, not part of the unregularized equations above. Predictive
weights may be signed and have no conductance degree budget or Gaussian-bits
scaling. Hidden-state persistence and alternating wave/inference steps are this
project's online extension, not a guarantee of learned temporal forecasting.

Configure the pathway under `RIPPLE_CONFIG["transforms"]["prediction_error"]`
in `main.py`: `enabled` and `inputs` retain camera-feedback gating; `inference`
sets hidden channels, step count, hidden rate, and canvas rate; `learning`
controls both modalities' predictive weights. The **Audio** source toggle
enables audio sensory correction independently of camera availability.
The former Gaussian `predictor` and shaped-error
`output` settings no longer drive camera correction. Nonlinearity belongs in
the prediction and local inference equations, rather than an arbitrary shaping
of the camera residual.

Disabling **Pathway Learning** freezes weights, not state inference. Start with
hidden rate `0.1`, canvas rate `0.01`, eight inference steps, and weight-learning
rate `0.01`; a weight rate near `1` is an experiment, not a stability guarantee.
The pathways run in NumPy for CPU/GPU wave backends; OpenGL sensory correction
remains unsupported and is explicitly rejected. Inference cost scales with
image size and step count; reduce inference steps or resolution when exploring
the live loop at high resolutions.

### Audio observations and pooled readout

Audio uses the newest frame of the existing STFT magnitude or square-root mel
power analysis, averaged across input channels. Sixteen contiguous frequency
groups retain mean magnitudes, scaled by `2 / sum(analysis_window)` and the
**Audio Evidence Gain**. Smaller spectra are interpolated to sixteen bands.
No second FFT, per-frame level normalization, or synthesized wave target is
introduced. The optional `sources.audio.observation_gain` config key defaults
to `1.0`. Peak gates and legacy frequency mappings do not select audio evidence;
real silent blocks are valid zero-valued observations.

For an audio hidden field `a` and spectral vector `s`:

```text
predicted_a = W_ac tanh(c)
q = spatial_mean(tanh(a))
predicted_s = W_sa q
e_a = a - predicted_a
e_s = s - predicted_s

delta_a = hidden_rate [-e_a + tanh'(a) * (W_sa.T e_s)]
delta_c_audio = canvas_rate [tanh'(c) * (W_ac.T e_a)]
delta_W_ac = learning_rate spatial_mean(e_a tanh(c).T)
delta_W_sa = learning_rate e_s q.T
```

This has the same two learned projection stages as vision. Audio's sensor
nodes are global spectral bands: the fixed spatial mean pools the hidden
field, rather than inventing an audio observation at every camera pixel.
It does not encode audio spatial localization. For active branches, the
joint energy is:

```text
E_joint = 0.5 * spatial_mean(
    sum_channels(e_h_camera^2) + sum_channels(e_y_camera^2)
    + sum_channels(e_a^2)
) + 0.5 * sum_bands(e_s^2)
```

State rates multiply its gradient by the pixel
count; this cancels the pooling derivative's `1/N` and preserves the existing
resolution-independent local state-update convention. Weight updates use
the unscaled energy gradients. Both state and weight rules retain `tanh`;
only state updates contain its derivative.

Each audio analysis generation is consumed once. Waiting between blocks does
not repeat inference or weight learning on cached evidence. Diagnostics hold
the latest audio sample for at most half a second, then mark missing evidence
and clear its planes. The offline audio renderer exercises this same pathway;
its legacy frequency-mapping arguments now affect reported peak metadata only.

The visual readout also supports opt-in energy diagnostics. A node's energy is
`0.5 * prediction_error**2`; layer statistics report its mean and population
variance over pixels and channels. Diagnostics capture the fixed-weight
inference trajectory and final local errors before learning, plus the actual
weight-change norm and weights after learning. Collection is disabled by
default and does not change inference or learning.

Click **Show Inference Diagnostics** below the canvas to see hidden-layer and
camera-level energy means and variances over simulation time, and the latest
frame's energy trajectory across fixed-weight inference steps. The status line
shows whether weight learning is enabled, its rate, and the actual combined
weight-update norm. Missing camera evidence inserts a gap, not a fabricated
zero-energy sample. Reset clears the diagnostic history.

With an audio processor present, the window has **Camera** and **Audio** tabs,
each showing its own hidden/sensory means, variances, weights, and optional
2D/3D graph. Both branches collect bounded scalar history while the window is
open; spatial previews are captured only for the visible tab's enabled 3D
view. An inference trajectory includes all branches active in that inference
frame. Audio sensory statistics are over bands, not over repeated spatial
copies. Audio 3D observation/prediction planes show grayscale band columns;
rows repeat purely for display.

Enable **Show shared-channel computational graph** in that window to inspect
all connections in the selected `canvas -> hidden -> sensory` pathway. Nodes show
spatial mean states and local mean energies; signed edge colors and widths show
the shared weights. Dashed arrows indicate local error feedback. This is the
channel graph repeated at each spatial site, not a graph with one vertex for
every pixel (with spatial pooling at the audio readout). The fixed wave operator
and synthetic drives remain outside this learned-pathway graph.

History is limited to 300 entries and plots refresh at most ten times per
second. Closing or hiding the window stops its timer and telemetry collection;
the graph is separately optional. This visualization does not change topology
or prune weights. Arbitrary predictive graph topologies remain a separate
extension.

Within the graph panel, enable **3D layer view (drag to rotate)** to orbit a
stack of the corrected canvas, individual hidden-channel maps at one hidden
depth, the camera prediction made **before new evidence**, and the observed
camera image clamped during that frame's inference. Drag to rotate, use the
wheel to zoom, or Ctrl-drag to pan; **Reset 3D camera** restores the initial
view. The energy plots remain alongside it, and the divider resizes the panels.
Hidden labels include their channel's mean local energy. RGB planes use a fixed
`[0,1]` display range with clipping; hidden maps display `tanh(state)`, blue for
negative and orange for positive. The main canvas and its color controls are
unchanged. The initial 3D view omits connection lines.

Spatial previews use aspect-preserving nearest-neighbor sampling, capped at
64 pixels per side, and only the latest sampled frame is retained; images are
not stored in the 300-entry energy history. Preview requests and texture updates
are limited to 10 Hz, including requests when camera evidence is missing.
Collection occurs only while the diagnostics window, graph panel, and 3D mode
are visible. The camera is not read again for visualization. GPU canvas values
are sampled before their host transfer. Hiding, reset, and missing evidence
clear stale planes. Preview preparation and OpenGL stay on the GUI thread;
there is no worker or extra render timer, and interaction redraws cached small
planes. OpenGL is initialized lazily. Context failures are reported in the
panel and restore the 2D graph; on X11, an explicitly incompatible PyOpenGL
platform can be corrected by restarting with `PYOPENGL_PLATFORM=glx` (or `egl`
when Qt explicitly uses `QT_XCB_GL_INTEGRATION=xcb_egl`).

---

## 🖼️ Gallery (placeholder)

*Add demo visuals here.*

| Audio Stimulus | Ripple Response                   |
| -------------- | --------------------------------- |
| "S" sound      | ![img1](images/demo_s_ripple.png) |
| Percussive hit | ![img2](images/demo_hit.png)      |

---

## 🎥 Demo (placeholder)

> *Insert phone-recorded demo or YouTube link here.*

---

## ✨ Vision

This project is meant to be more than a tool—it's a medium. Imagine walking into an empty room where every footstep sends waves across a projected canvas. Every voice becomes a living waveform, every interaction ripples outward. The future includes:

* Using video/pose tracking (e.g., OpenPose) to trigger ripples from body motion
* Modeling crowd dynamics through ripple fields

> *The machine is just a medium through which waves travel between us.*

---

## 🌱 Inspiration

* Physical wave propagation dynamics
* Shared embodiment and resonance in human movement and sound
* Experiences of synesthesia

---

## 🧠 Project Structure

```
audioviz/
├── audio_processing/
│   └── audio_processor.py
├── visualization/
│   ├── ripple_wave_visualizer.py
│   └── spectrogram_visualizer.py
├── utils/
│   └── signal_processing.py
main.py
```

---

## 🤝 Contributing

PRs and feature ideas welcome! Especially contributions around:

* Pose tracking integration
* Multi-person ripple graph
* Real-time OSC/MIDI control hooks

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 📌 Footnote

Audio Ripple is part of a broader artistic and philosophical pursuit:
to build mediums where people hear and feel each other more deeply. 
