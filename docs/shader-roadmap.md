# Shader Backend Roadmap

The first OpenGL shader backend proves that the ripple update can run as a
local compute pass: each grid cell reads its previous state plus fixed-radius
neighbors, then writes the next state. This is the right computational shape for
future predictive-coding kernels and other local message-passing fields.

The current implementation is intentionally conservative. It still copies the
shader result back into a NumPy array because `RippleWaveVisualizer` renders via
`pyqtgraph.ImageItem`. That makes the path useful for correctness testing, but
not yet a good performance target.

## Near-Term Performance Plan

Keep the physics state on the GPU for the whole frame:

1. Compute the next ripple state into an OpenGL texture or buffer.
2. Render that same GPU object directly with an OpenGL draw pass.
3. Avoid `get_field_numpy()` in the interactive shader path.
4. Keep NumPy readback only for tests, debugging, screenshots, and CPU fallbacks.

This should be implemented as a renderer selection inside the ripple visualizer,
not by tightly coupling the visualizer to one physics engine. A useful split is:

- `NumpyImageRenderer`: current `pyqtgraph.ImageItem` path for CPU/CuPy readback.
- `OpenGLFieldRenderer`: draw the shader-owned texture or buffer directly.
- `RippleEngine`: owns simulation state and exposes either NumPy state or a GPU
  field handle depending on backend.

## Unified-Memory Devices

Jetson-class devices and many phones often use unified physical memory rather
than a discrete GPU connected over PCIe. That can remove a large copy cost, but
it does not make CPU/GPU exchange free. GPU-to-CPU readback can still introduce
synchronization stalls, cache visibility work, layout conversion, and API
barriers. The first optimization target is therefore the same on desktop,
Jetson, and mobile: avoid round-tripping field data through CPU arrays during
interactive rendering.

## Performance Tracking

Keep recording CPU and shader measurements as the architecture changes. The
initial shader path measured slower than CPU because every frame still performs
GPU upload/readback around a small stencil kernel. Future commits should include
the measured grid size, backend, and whether rendering stayed on GPU.
