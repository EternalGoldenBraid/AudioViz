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
* **Decay α**: Sets spatial spread of excitation; higher values make sources more localized
* **Amplitude**: Controls excitation strength
* **Speed**: Wave propagation speed in meters per second

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
