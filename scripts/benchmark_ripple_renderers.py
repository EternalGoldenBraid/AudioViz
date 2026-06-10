from __future__ import annotations

import argparse
import time

import numpy as np
from PyQt5 import QtWidgets

from audioviz.engine import RippleEngine
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


def benchmark_cpu(
    *,
    resolution: tuple[int, int],
    n_sources: int,
    n_frequencies: int,
    n_steps: int,
) -> float:
    engine = RippleEngine(
        resolution=resolution,
        plane_size_m=(1.0, 1.0),
        n_sources=n_sources,
        speed=10.0,
        damping=0.99,
        amplitude=1.0,
        use_shader=False,
    )
    frequencies = np.full((n_sources, n_frequencies), 5.0, dtype=np.float32)

    for _ in range(3):
        engine.step(frequencies)

    start = time.perf_counter()
    for _ in range(n_steps):
        engine.step(frequencies)
    elapsed = time.perf_counter() - start
    return n_steps / elapsed


def benchmark_shader_readback(
    *,
    resolution: tuple[int, int],
    n_sources: int,
    n_frequencies: int,
    n_steps: int,
) -> float:
    engine = RippleEngine(
        resolution=resolution,
        plane_size_m=(1.0, 1.0),
        n_sources=n_sources,
        speed=10.0,
        damping=0.99,
        amplitude=1.0,
        use_shader=True,
        use_external_opengl_context=False,
    )
    frequencies = np.full((n_sources, n_frequencies), 5.0, dtype=np.float32)

    for _ in range(3):
        engine.step(frequencies)
        engine.get_field_numpy()

    start = time.perf_counter()
    for _ in range(n_steps):
        engine.step(frequencies)
        engine.get_field_numpy()
    elapsed = time.perf_counter() - start
    return n_steps / elapsed


def benchmark_shader_direct(
    *,
    app: QtWidgets.QApplication,
    resolution: tuple[int, int],
    n_sources: int,
    n_steps: int,
) -> float:
    widget = RippleWaveVisualizer(
        processor=None,
        use_synthetic=True,
        use_shader=True,
        resolution=resolution,
        n_sources=n_sources,
        speed=10.0,
        damping=0.99,
        amplitude=1.0,
    )
    widget.resize(320, 320)
    widget.show()
    app.processEvents()

    for _ in range(3):
        widget.update_visualization()
        app.processEvents()

    start = time.perf_counter()
    for _ in range(n_steps):
        widget.update_visualization()
        app.processEvents()
    elapsed = time.perf_counter() - start
    widget.close()
    return n_steps / elapsed


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        return int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Resolution must use WIDTHxHEIGHT format, for example 256x256."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU, shader readback, and direct OpenGL ripple rendering."
    )
    parser.add_argument(
        "--resolution",
        dest="resolutions",
        action="append",
        type=parse_resolution,
        default=None,
        help="Resolution to benchmark, e.g. 256x256. May be passed more than once.",
    )
    parser.add_argument("--sources", type=int, default=2)
    parser.add_argument("--frequencies", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    resolutions = args.resolutions or [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ]

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for resolution in resolutions:
        cpu_hz = benchmark_cpu(
            resolution=resolution,
            n_sources=args.sources,
            n_frequencies=args.frequencies,
            n_steps=args.steps,
        )
        shader_readback_hz = benchmark_shader_readback(
            resolution=resolution,
            n_sources=args.sources,
            n_frequencies=args.frequencies,
            n_steps=args.steps,
        )
        shader_direct_hz = benchmark_shader_direct(
            app=app,
            resolution=resolution,
            n_sources=args.sources,
            n_steps=args.steps,
        )
        width, height = resolution
        print(
            f"{width}x{height} "
            f"cpu={cpu_hz:.1f}Hz "
            f"shader_readback={shader_readback_hz:.1f}Hz "
            f"shader_direct={shader_direct_hz:.1f}Hz"
        )


if __name__ == "__main__":
    main()
