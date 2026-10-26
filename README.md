# AudioViz

**An experimental workspace for predictive coding, active vision and audiovisual
wave fields.**

The current application learns and predicts a whole visual scene while observing
only a movable circular patch. It combines a continuous neural field, shared
recurrent hidden units, optional audio evidence and live Qt diagnostics.
The project grew out of the Audio Ripple visualizer; wave simulation and audio
tools remain, but the current research focus is scene learning and active sensing.

[Experimentation diary](docs/experiments/README.md) ·
[Quick start](#quick-start) ·
[Model reference](docs/continuous-scene-reference.md) ·
[Results and recordings](docs/experiments/2026-10-26-static-scene-learning.md)

## Where the research stands

This README describes the continuous-scene implementation on
`research/active-vision-eig`.

| Established in the tested setup | Still open |
| --- | --- |
| Learned connections support recall of previously seen regions after neural states are cleared. | Generalization to new scenes and larger worlds. |
| NumPy and CuPy numerical backends; interactive camera and hardware-free bench applications. | Real-time performance at larger world resolutions. |
| Local state-covariance queries and one-step information-gain gaze selection. | Whether autonomous gaze improves learning over controlled schedules. |

**Long-horizon learning is not yet stable.** In the recorded 480-step experiment,
learning weights and precisions degrades as precisions grow; weights-only
learning remains stable in that schedule. The next experiment separates
integration sensitivity from the precision-learning problem.
The [diary](docs/experiments/README.md) tracks that decision and the evidence.

## Quick start

### 1. Set up the environment

The current environment is developed on Linux. The
[lockfile](uv.lock) targets Python **3.14+**; use
[uv](https://docs.astral.sh/uv/getting-started/installation/) to reproduce it:

```bash
git clone --branch research/active-vision-eig https://github.com/EternalGoldenBraid/AudioViz.git
cd AudioViz
uv sync --locked --python 3.14 --group dev
```

**Installation prerequisites:** a desktop/Qt runtime and a compatible CUDA
development environment for CuPy. The dependency manifest currently requires
`cupy`, locked as a source distribution, even when selecting NumPy at runtime.
NumPy execution is CPU-based, but this is not currently a CPU-only installation.
The OpenGL backend additionally requires OpenGL 4.3 compute support.

### 2. Try the scene bench without a camera

```bash
.venv/bin/python -m audioviz.visualization.static_world_bench --backend numpy
```

The bench starts paused with an `80 x 120` static scene, learning off and no
wave dynamics. Move the mouse over a scene panel to select gaze, then press
**Step** or **Run**. **Observe vision** gates evidence; **Model controls**
exposes inference and learning settings. **Predictions** opens the prediction
layouts and live 3D graph.

For GPU inference, use `--backend cupy`. For OpenGL rendering with CPU
inference, use `--backend opengl` (the bench default when the flag is omitted).
Run with `--help` for resolution, foveal radius, hidden-unit and timing options.

### 3. Run the live camera application

```bash
.venv/bin/python main.py
```

Startup settings live in [`main.py`](main.py), not model-selection flags:

| Setting | Current default |
| --- | --- |
| World grid / display window | `200 x 200` / `1920 x 1080` |
| Numerical backend | CuPy (`renderer.backend = "gpu"`) |
| Camera | Enabled, `/dev/video0` |
| Audio / synthetic drive | Disabled |
| Foveal radius / shared hidden units | `32` pixels / `6` units |
| Gaze | Mouse |
| Inference | `8` microsteps, rate `0.1`, leak time constant `1/3 s` |
| Learning / canvas feedback / wave propagation | Off / zero / off |

Edit `RIPPLE_CONFIG` for the model, backend and sources; edit `APP_CONFIG` for
window visibility. Changing world resolution, radius or hidden count requires
a restart. Camera-only defaults do not initialize audio devices.
The main canvas is the wave/root surface, not the predicted camera image;
open **Predictions** to inspect scene reconstruction.

## How the model works

```text
Forecast the whole scene
        |
Choose gaze: mouse or local information gain
        |
Acquire only the selected foveal evidence
        |
Condition neural states, then optionally learn weights and precisions
```

Every world RGB coordinate has an evolving neural potential `z`, decoded as
`sigmoid(z)`. A shared recurrent population `h` couples the field. Gaze selects
evidence; it does not relocate states or connections. The complete prediction
is decoded every interval, with **no frozen peripheral image cache**.

Missing input is not a black image or silent audio observation. With Observe
off, evidence acquisition and parameter learning stop while neural activity
decays toward neutral gray in an unforced model. Learned connections remain.

Autonomous gaze scores expected information gain under a **local Gaussian
approximation** before acquiring evidence. It does not use the evaluator's
ground truth, and it is not full nonlinear expected free energy or temporal
planning. See the [model reference](docs/continuous-scene-reference.md) for the
objective, covariance API, gaze semantics and controls.

## Backends and scale

| Backend | Live app setting | Bench flag | Continuous graph | Image routing / rendering |
| --- | --- | --- | --- | --- |
| NumPy | `"numpy"` | `numpy` | CPU, float64 | CPU / Qt |
| CuPy | `"gpu"` | `cupy` | GPU, float64 | GPU processing / Qt presentation |
| OpenGL | `"opengl"` | `opengl` | CPU, float64 | GPU routing / rendering |

**OpenGL does not move this continuous graph into compute shaders.** Shader
inference benchmarks for the older per-pixel model describe a different model.

The current graph guards allocations at `307,200` world pixels, `64` shared
hidden units and `4,096` foveal pixel locations. These are software limits, not
measured hardware capacity. Storage scales with the **whole world**, not just
the fovea. A 1080p window can display a smaller world; a native `1920 x 1080`
neural field exceeds the current world-size guard.

For the live application's configured workload:

```bash
.venv/bin/python scripts/profile_live.py --backend gpu \
    --warmup 10 --frames 120 --output /tmp/audioviz-live.prof
```

This uses the configured sources and measures application updates, not
synchronized GPU kernels or presented display FPS.

## Documentation

| Start here for... | Document |
| --- | --- |
| Research questions, current evidence and what to do next | [Experimentation diary](docs/experiments/README.md) |
| Recorded learning, recall and long-horizon failure | [Experiment E001](docs/experiments/2026-10-26-static-scene-learning.md) |
| Configuration, model mathematics, APIs and bench controls | [Continuous-scene reference](docs/continuous-scene-reference.md) |
| Detailed measurements and historical baselines | [Static-world experiment report](docs/static-world-experiment-report.md) |
| A visual introduction to precision-weighted predictive coding | [Interactive primer](docs/predictive-coding-precision-primer.html) and [editable diagram](docs/predictive-coding-precision-primer.drawio) |
| Earlier Gaussian active-vision experiments | [Historical active-vision bench](docs/active-vision-experiment.md) |
| Earlier per-pixel shader architecture and performance work | [Shader reference](docs/shader-inference-plan.md) and [archived per-pixel notes](https://github.com/EternalGoldenBraid/AudioViz/blob/e2b161c84b384c4fde7cf7d4ee35ac607c594858/README.md#legacy-per-pixel-predictive-canvas) |

The RC-spiking implementation and pose subsystem are retired. Their retained
explainers and historical results are not current application modes.

## Development

The development dependency group includes pytest:

```bash
.venv/bin/python -m pytest
```

Qt/OpenGL tests need a display and suitable context; headless Linux runs can
use Xvfb. CUDA cases require a working GPU environment.

| Location | Responsibility |
| --- | --- |
| `main.py` | Live application configuration and startup |
| `src/audioviz/readouts/` | Predictive models, precision and state uncertainty |
| `src/audioviz/sources/` | Camera/audio sources, gaze and evidence orchestration |
| `src/audioviz/visualization/` | Scene bench, Qt views and diagnostics |
| `src/audioviz/engine.py` and `src/audioviz/physics/` | Wave substrate and physical operators |
| `src/audioviz/audio_processing/` and `offline/` | Audio analysis and offline experiments |
| `scripts/` and `tests/` | Profiling tools and regression coverage |

## License

[MIT](LICENCE.txt).
