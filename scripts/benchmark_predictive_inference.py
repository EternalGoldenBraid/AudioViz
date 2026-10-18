"""Benchmark wave/predictive compute with synthetic sensors, without capture or GUI."""

from __future__ import annotations

import argparse
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
    frames: int, warmup: int,
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
    samples = []
    for index in range(warmup + frames):
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
        image = CameraFrameSource.frame_to_excitation_grid(frame, resolution=resolution)
        mark()
        correction = infer_joint(
            prior, ((camera, image), (audio, spectrum)), coupling=coupling,
            stream_states={camera: True, audio: True},
        )
        mark()
        engine.apply_observation_correction(correction)
        mark()
        engine.get_field_numpy()
        mark()
        if index >= warmup:
            samples.append(np.diff(marks) * 1000)
    return np.asarray(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "gpu", "both"), default="both")
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("ROWS", "COLS"), default=(300, 400))
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--learning", action="store_true")
    parser.add_argument("--cross-modal", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0 or min(args.resolution) <= 0 or args.inference_steps <= 0:
        parser.error("frames, resolution and inference steps must be positive; warmup must be nonnegative")
    config = PredictiveCodingConfig(
        inference_steps=args.inference_steps, learning_enabled=args.learning,
        cross_modal_enabled=args.cross_modal,
    )
    print(
        f"Resolution {args.resolution[0]}x{args.resolution[1]}, "
        f"{config.hidden_channels} hidden channels, {config.inference_steps} inference steps, "
        f"float64 inference, learning={config.learning_enabled}, coupling={config.cross_modal_enabled}.\n"
        "Synthetic camera/audio evidence; excludes capture, audio analysis and GUI rendering. "
        "GPU timings synchronize each stage; final field download is included.",
        flush=True,
    )
    backends = ("numpy", "gpu") if args.backend == "both" else (args.backend,)
    for backend in backends:
        timings = benchmark(
            use_gpu=backend == "gpu", resolution=tuple(args.resolution), config=config,
            frames=args.frames, warmup=args.warmup,
        )
        print(f"\n{backend} mean stage times (ms):")
        for stage, duration in zip(
            ("wave", "prediction", "camera preprocessing", "inference", "correction", "field download"),
            timings.mean(axis=0),
        ):
            print(f"  {stage}: {duration:.3f}")
        totals = timings.sum(axis=1)
        print(
            f"  total: {totals.mean():.3f}; p95: {np.percentile(totals, 95):.3f}; "
            f"compute throughput: {1000 / totals.mean():.2f} frames/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
