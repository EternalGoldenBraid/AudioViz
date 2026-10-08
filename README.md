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
* **Show Inference / Learning Plots**: Opens the energy / 2D weight plot catalogue
* **Show Multimodal 3D Scene**: Opens an independent, rotatable view of the shared canvas and every source

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

Each available modality has a clamped ON/OFF stream-state node derived directly
from its source toggle. Inference and enabled learning continue when sensory
samples are absent; missing image/spectrum targets remain unclamped and their
decoder weights stay frozen. Both hidden fields continue to
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

Hover a source/inference control or its label for its update semantics,
zero-value behavior, and stability caveats. **Canvas Correction Rate** is the
per-inference-step gradient rate applied to the propagated wave prior, not a
percentage blend with sensor data. At zero, sensory correction of the canvas
stops while wave propagation and hidden-state inference continue. **Hidden
State Rate** also controls the pre-evidence relaxation toward that wave prior.

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
not reuse cached evidence for inference or learning. With the optional loop on,
unclamped hidden inference continues between blocks. Diagnostics hold
the latest audio sample for at most half a second, then mark missing evidence
and clear its planes. The offline audio renderer exercises this same pathway;
its legacy frequency-mapping arguments now affect reported peak metadata only.

Audio can be much less visible than camera feedback initially: camera weights
start near identity, while both audio projections start small, and band means
dilute narrow spectral peaks. From a uniform canvas, pooled audio error can
produce a spatially uniform shift, not a localized wave pattern. **Audio
Evidence Gain** scales the actual spectral target; **Excitation Amplitude**
only scales synthetic drives. Gain zero is silent evidence, not missing input.
RGB rendering clips negative values to black and quantizes positive values to
8-bit intensity at the current display scale, so weak corrections can be
invisible. Lowering **Low-Level Reference Scale** changes visibility without
changing inference; increasing evidence/canvas rates changes the model and
can destabilize it. No automatic gain or physics changes are made to compensate.

### Optional camera/audio hidden loop

Enable **Camera / Audio Hidden Loop** in **Sensory Inference and Learning**
to add reciprocal, per-pixel projections between the two hidden fields. It is
off by default, preserving the independent-pathway behavior. Both matrices
start at zero, have destination-by-source shape `(hidden_channels, hidden_channels)`,
and share weights across pixels; there is no dense spatial graph or pruning.
Enable **Pathway Learning** with paired camera/audio evidence to learn the
connections, then freeze learning to probe them. In startup configuration,
`transforms.prediction_error.inference.cross_modal_enabled` enables the same mode.
An audio processor must exist, but its observation toggle can be off.
The shader-only backend does not support the loop.

With `h` the camera hidden field and `a` the audio hidden field:

```text
predicted_h = W_hc tanh(c) + V_ha tanh(a)
predicted_a = W_ac tanh(c) + V_ah tanh(h)
e_h = h - predicted_h
e_a = a - predicted_a

delta_h = hidden_rate [-e_h + tanh'(h) * (W_yh.T e_y + V_ah.T e_a)]
delta_a = hidden_rate [-e_a + tanh'(a) * (W_sa.T e_s + V_ha.T e_h)]
delta_V_ha = learning_rate spatial_mean(e_h tanh(a).T)
delta_V_ah = learning_rate spatial_mean(e_a tanh(h).T)
```

These are local negative energy gradients, including error feedback from
each hidden state's outgoing recurrent connection. Directions are synchronous;
all weights remain fixed throughout settling. The canvas correction still
uses both hidden errors and has no additional wave-prior restoring penalty.
The joint energy always includes both hidden residuals in this mode, and only
includes sensory residuals for observations actually available in that frame.

Missing sensors are **unclamped**, not replaced by silent audio or black images.
Every physical frame runs the configured bounded inference loop, even with
both sensors absent, so learned recurrence can affect hidden states, predictions,
and the canvas impulse response. Recurrence adds computation to independent
pathways. Stream ON/OFF state remains evidence with both captures off, so
enabled hidden and recurrent weight learning can continue in both branches.
An absent modality's sensory weights (including decay) stay frozen.
Turning the loop off or resetting states retains its learned weights.
Weight/gradient clipping uses the existing controls; it does not guarantee
stable recurrent dynamics. Reduce state rates or disable the loop if it diverges.

Audio-to-camera associations can now be tested with the camera off.
Shared per-pixel weights and global audio pooling do not provide arbitrary
spatial scene recall or audio localization, and persistence is not guaranteed.

### Stream-state context: closed is evidence

Every available modality has one clamped ON/OFF node. There is no separate
context setting: the node follows its source directly, disabled = OFF and
enabled = ON. This works with or without the camera/audio loop, including a
camera-only setup, on an array-backed CPU/GPU canvas.

The context observation comes from the actual source toggle: enabled is ON,
disabled is OFF. It describes the application's capture setting, not physical
eyelid state or device health. Waiting between audio blocks, a failed camera
read, or disabled camera-image correction leaves ON context if the source
itself remains enabled. Sample validity remains separate: no frame/spectrum
means no sensory residual, not a black image or silent spectrum.

Each context node is predicted from its own spatially pooled hidden activity:

```text
m = clamped source-enabled bit, 0 or 1
q = spatial_mean(tanh(h))
p = sigmoid(v q + b)
e_m = m - p
E_m = 0.5 e_m^2
d_m = e_m p (1 - p)

delta_h_context = hidden_rate tanh'(h) * v.T d_m
delta_v = learning_rate d_m q.T
delta_b = learning_rate d_m
```

`E_m` is counted once per modality in joint energy. The existing pixel-count
state-gradient convention cancels the pooling derivative; weight/bias updates
use the unscaled local energy gradients. Sigmoid is evaluated with the
numerically stable `0.5 * (1 + tanh(logit / 2))`. Context weights start small
and random, with zero bias; the initial ON prediction is 0.5 from zero hidden
states. Existing decay and clipping controls apply to both weights and bias.

Known OFF state is evidence, so inference and **enabled** learning continue
with both captures off. Hidden, canvas-projection, context, and active recurrent
weights may learn from the clamped context and final latent residuals. The
absent modality's image/spectrum decoder weights remain frozen, including
decay, because there is no target for them. **Pathway Learning** remains the
explicit switch to freeze all weights during an impulse or recall probe.
Resetting states retains learned context parameters.
Learning while probing can alter stored associations, so freezing remains
useful even though a closed stream now supplies a valid context observation.

### Inference and learning diagnostics

The visual readout also supports opt-in energy diagnostics. A node's energy is
`0.5 * prediction_error**2`; layer statistics report its mean and population
variance over pixels and channels. Diagnostics capture the fixed-weight
inference trajectory and final local errors before learning, plus the actual
weight-change norm and weights after learning. Collection is disabled by
default and does not change inference or learning.

Click **Show Inference / Learning Plots** below the canvas to see hidden-layer and
camera-level energy means and variances over simulation time, and the latest
frame's energy trajectory across fixed-weight inference steps. The status line
shows whether weight learning is enabled, its rate, and the actual combined
weight-update norm. Missing camera evidence inserts a gap, not a fabricated
zero-energy sample. Hidden/total inference energy continues
to be sampled without sensors; missing sensory energy remains a gap, and
sensory graph nodes show predictions. Reset clears the diagnostic history.

With an audio processor present, the catalogue has **Camera** and **Audio**
tabs. Select **Energy and inference plots**, **2D weights and hidden states**,
or both; the divider resizes the plots when both are selected. Each modality
has its own hidden/sensory means, variances, and weights. Both collect bounded
scalar history while the catalogue is open, irrespective of the selected tab.
An inference trajectory includes all branches active in that inference frame.
Audio sensory statistics are over native bands, not repeated spatial copies.
The catalogue does not retain spatial images.

Select **2D weights and hidden states** in the catalogue to inspect
all connections in the selected `canvas -> hidden -> sensory` pathway. Nodes show
spatial mean states and local mean energies; signed edge colors and widths show
the shared weights. Dashed arrows indicate local error feedback. This is the
channel graph repeated at each spatial site, not a graph with one vertex for
every pixel (with spatial pooling at the audio readout). The fixed wave operator
and synthetic drives remain outside this learned-pathway graph. With the
hidden loop enabled, extra peer nodes `P` show the other modality's hidden
means and its learned incoming recurrent weights. The other tab shows the
reciprocal projection; the update norm includes incoming recurrent weights.
Node `S` shows clamped ON/OFF, predicted `p(on)`, local
error energy, bias and incoming weights. The green **Stream state** curve
tracks its energy; population variance is zero for a single scalar node,
not a temporal-variance estimate. Total inference energy and weight-update
norms include context contributions. Errors/predictions are before learning;
displayed weights and bias are after learning.

History is limited to 300 entries and plots refresh at most ten times per
second. Closing or hiding the window stops its timer and telemetry collection;
the graph is separately optional. This visualization does not change topology
or prune weights. Arbitrary predictive graph topologies remain a separate
extension.

### Independent multimodal 3D scene

Click **Show Multimodal 3D Scene** below the canvas, or **Open multimodal 3D
scene** in the catalogue. This is a separate window: either window can remain
open after the other closes. Both sensory branches share one corrected canvas;
each shows its own hidden maps and prediction made **before new evidence**.
Camera evidence is an RGB plane. Audio prediction and evidence are signed
spectral bar histograms, with one bar per native band and a fixed `[-1,1]`
display range; values outside that range are clipped for display only. Positive
bars are blue, negative predictions orange; silence has zero-height bars.
There is no per-frame normalization or duplication into audio image pixels.

Each source declares a `scene_mapping` (`SourceSceneMapping`) defining its
identity, role, representation, and bounded sampling:

| Source | Role | 3D representation |
| --- | --- | --- |
| Camera | Learned sensory branch | RGB prediction/evidence planes and hidden-channel maps |
| Audio | Learned sensory branch | Spectral prediction/evidence bars and hidden-channel maps |
| Synthetic | Direct wave drive | Signed drive-grid plane, before the engine's amplitude scaling |
| Pose | Coupled medium | Valid mapped pose nodes and adjacency edges |

Synthetic/pose attachments do not acquire invented predictive hidden layers.
Layout links distinguish the sensory pathways from direct drive/medium
attachments; labeled hidden-to-hidden links identify the optional loop.
Individual learned connections remain in the 2D weight plots.
The scene also shows one labeled ON/OFF node per sensory branch,
linked to its hidden field, with the predicted `p(on)` **before new evidence**.
Disabled sources are identified in the status line. Missing evidence removes
that source's observed geometry; the canvas, hidden states, and predictions
continue to show the belief. Audio holds its latest genuine observation for
at most half a second between blocks, including real silent observations.

Drag to orbit, wheel to zoom, or Ctrl-drag to pan; **Reset 3D camera** restores
the initial overview. Hidden labels include mean local error energy when it
was measured. RGB planes clip to `[0,1]`; hidden and drive maps display `tanh`,
blue for negative and orange for positive. Main-canvas rendering is unchanged.

The scene samples at most ten times per second, even with missing camera
evidence. Image/hidden previews preserve aspect ratio and are capped at 64
pixels per side; pose previews show at most 64 valid nodes and their induced
edges; spectral previews accept at most 64 bands (audio currently uses 16).
Only the latest scene is retained, independently of the 300-entry scalar
histories. Hiding or minimizing the scene stops preview preparation and clears
its cached data; reset clears both windows. Scalar telemetry stays enabled
while either the catalogue or a working, non-minimized scene needs it.

Preview preparation and OpenGL stay on the GUI thread, driven by the existing
physical-frame loop; no worker or extra render timer is introduced. Sensors
are not reread for the scene, and GPU canvas values are sampled before host
transfer. OpenGL initializes lazily. Context/import failures are reported
explicitly in the scene and leave the plot catalogue usable. On X11, an
incompatible PyOpenGL platform can be corrected by restarting with
`PYOPENGL_PLATFORM=glx` (or `egl` when Qt explicitly uses
`QT_XCB_GL_INTEGRATION=xcb_egl`).

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
