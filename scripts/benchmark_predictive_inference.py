"""Benchmark wave/predictive compute with synthetic sensors, without capture or GUI."""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import pstats
import time

import numpy as np

from audioviz.engine import RippleEngine
from audioviz.readouts import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.predictive_coding import PredictiveCodingHiddenCoupling, infer_joint
from audioviz.sources.camera import CameraFrameSource
from audioviz.utils.audio_features import AUDIO_SPECTRAL_BANDS


def benchmark(
    *, use_gpu: bool, resolution: tuple[int, int], config: PredictiveCodingConfig,
    frames: int, warmup: int, render: bool = False,
    profiler: cProfile.Profile | None = None,
) -> np.ndarray:
    engine = RippleEngine(
        resolution=resolution, plane_size_m=(50, 50), damping=.988,
        boundary_condition="neumann", use_gpu=use_gpu,
    )
    camera = PredictiveCodingRGBReadout(
        canvas_shape=engine.canvas_shape, config=config, backend=engine.backend,
    )
    audio = PredictiveCodingAudioReadout(
        canvas_shape=engine.canvas_shape, config=config, backend=engine.backend,
    )
    coupling = PredictiveCodingHiddenCoupling(camera, audio) if config.cross_modal_enabled else None
    rng = np.random.default_rng(12)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    spectrum = rng.random(AUDIO_SPECTRAL_BANDS).astype(np.float32)
    renderer = None
    if render:
        from PyQt5 import QtWidgets
        from audioviz.visualization.ripple_renderers import NumpyImageRenderer

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        renderer = NumpyImageRenderer(rgb_canvas_enabled=True)
        renderer.widget.resize(1920, 1080)
        renderer.widget.show()
        app.processEvents()
    samples = []
    for index in range(warmup + frames):
        if index == warmup and profiler is not None:
            profiler.enable()
        if use_gpu:
            engine.backend.cuda.get_current_stream().synchronize()
        marks = [time.perf_counter()]

        def mark():
            if use_gpu:
                engine.backend.cuda.get_current_stream().synchronize()
            marks.append(time.perf_counter())

        engine.propagate()
        mark()
        prior = engine.get_prior()
        if coupling is not None:
            coupling.predict(prior)
        else:
            audio.predict(prior)
            camera.predict(prior)
        mark()
        image = CameraFrameSource.frame_to_excitation_grid(
            frame, resolution=resolution, backend=engine.backend,
        )
        mark()
        correction = infer_joint(
            prior, ((camera, image), (audio, spectrum)), coupling=coupling,
            stream_states={camera: True, audio: True},
        )
        mark()
        engine.apply_observation_correction(correction)
        mark()
        if renderer is None:
            engine.get_field_numpy()
        else:
            renderer.render(engine.get_field())
            app.processEvents()
        mark()
        if index >= warmup:
            samples.append(np.diff(marks) * 1000)
    if profiler is not None:
        profiler.disable()
    if use_gpu:
        pool = engine.backend.get_default_memory_pool()
        print(
            f"GPU array memory: live={pool.used_bytes() / 1024**2:.1f} MiB; "
            f"pool reserved={pool.total_bytes() / 1024**2:.1f} MiB",
            flush=True,
        )
    if renderer is not None:
        renderer.widget.close()
    return np.asarray(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "gpu", "both"), default="both")
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("ROWS", "COLS"), default=(300, 400))
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--learning", action="store_true")
    parser.add_argument("--cross-modal", action="store_true")
    parser.add_argument("--profile", type=Path, help="Save a cProfile report; GPU stages still synchronize.")
    parser.add_argument("--render", action="store_true", help="Include an RGB display, automatic levels and Qt paint events.")
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0 or min(args.resolution) <= 0 or args.inference_steps <= 0:
        parser.error("frames, resolution and inference steps must be positive; warmup must be nonnegative")
    config = PredictiveCodingConfig(
        inference_steps=args.inference_steps, learning_enabled=args.learning,
        cross_modal_enabled=args.cross_modal, dtype=args.dtype,
    )
    print(
        f"Resolution {args.resolution[0]}x{args.resolution[1]}, "
        f"{config.hidden_channels} hidden channels, {config.inference_steps} inference steps, "
        f"{config.dtype} inference, learning={config.learning_enabled}, coupling={config.cross_modal_enabled}.\n"
        "Synthetic camera/audio evidence; excludes capture and audio analysis. "
        + ("Includes RGB rendering and Qt paint events. " if args.render else "Excludes GUI rendering. ")
        + "GPU timings synchronize each stage; final presentation/download is included.",
        flush=True,
    )
    backends = ("numpy", "gpu") if args.backend == "both" else (args.backend,)
    if args.render:
        from PyQt5 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for backend in backends:
        profiler = cProfile.Profile()
        try:
            timings = benchmark(
                use_gpu=backend == "gpu", resolution=tuple(args.resolution), config=config,
                frames=args.frames, warmup=args.warmup, render=args.render,
                profiler=profiler if args.profile is not None else None,
            )
        finally:
            profiler.disable()
        if args.profile is not None:
            path = args.profile.with_name(f"{args.profile.stem}-{backend}.prof")
            path.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(path))
            pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
            print(f"Saved {path}")
        print(f"\n{backend} mean stage times (ms):")
        for stage, duration in zip(
            ("wave", "prediction", "camera preprocessing", "inference", "correction",
             "render + paint" if args.render else "field download"),
            timings.mean(axis=0),
        ):
            print(f"  {stage}: {duration:.3f}")
        totals = timings.sum(axis=1)
        print(
            f"  total: {totals.mean():.3f}; p95: {np.percentile(totals, 95):.3f}; "
            f"{'rendered' if args.render else 'compute'} throughput: {1000 / totals.mean():.2f} frames/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
