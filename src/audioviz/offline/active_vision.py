from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from audioviz.engine import RippleEngine
from audioviz.readouts import GaussianSpatialPrecision, PredictiveCodingConfig
from audioviz.sources import (
    CameraFrameSourceConfig, CircularDotCapture, CircularDotConfig, CircularDotSource,
    RippleSourceOrchestrator, RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)
from audioviz.sources.gaze import GazePolicy
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.utils.opengl_compute import GLArray


@dataclass(frozen=True)
class GazeFrameMetrics:
    time: float
    gaze: tuple[float, float]
    expected_gain_nats: float
    posterior_hidden_variance_mean: float
    forecast_mse: float
    precision_mass: float


@dataclass(frozen=True)
class GazeRunResult:
    policy: str
    frames: tuple[GazeFrameMetrics, ...]
    cumulative_expected_gain_nats: float
    final_hidden_variance_mean: float
    mean_forecast_mse: float
    cumulative_precision_mass: float


def _prediction_metrics(prediction, observation, variance, *, backend):
    if isinstance(prediction, GLArray):
        device = prediction.device
        result = device.reduce(
            """layout(std430,binding=0) readonly buffer Prediction {float prediction[];};
            layout(std430,binding=1) readonly buffer Observation {float observation[];};
            layout(std430,binding=2) readonly buffer Variance {float variance[];};
            uniform int image_count,variance_count;""",
            """c==0 ? (p<image_count ? pow(prediction[p]-observation[p],2.0)/float(image_count) : 0.0)
                : (p<variance_count ? variance[p]/float(variance_count) : 0.0)""",
            [prediction, observation, variance], max(prediction.size, variance.size), 2,
            image_count=prediction.size, variance_count=variance.size,
        )
        try:
            values = result.download()
        finally:
            device.release(result)
        mse, mean_variance = map(float, values.ravel())
    else:
        mse = float(backend.mean((prediction - observation)**2))
        mean_variance = float(backend.mean(variance))
    if not np.isfinite((mse, mean_variance)).all() or min(mse, mean_variance) < 0:
        raise FloatingPointError("Active-vision evaluation produced invalid error or variance")
    return mse, mean_variance


def _precision_mass(precision, shape):
    x, y, width = precision.pixel_parameters(shape)
    rows, cols = shape
    horizontal = np.exp(-0.5*((np.arange(cols)-x)/width)**2).sum()
    vertical = np.exp(-0.5*((np.arange(rows)-y)/width)**2).sum()
    return float(precision.baseline*rows*cols + precision.peak*horizontal*vertical)


def run_gaze_comparison(
    *,
    stimulus: CircularDotConfig,
    backend: str = "numpy",
    inference_steps: int = 8,
    sigma: float = 0.15,
    baseline: float = 0.01,
    peak: float = 1.0,
    initial_variance: float = 1.0,
    process_variance: float = 0.01,
    memory_time_constant: float = 1/3,
    policy_seed: int = 0,
) -> tuple[GazeRunResult, ...]:
    """Compare policies on deterministic frames; assigned likelihood noise is not sampled.

    All runs use the same frames, observation schedule, and precision profile.
    Boundary clipping changes precision mass, which is reported independently.
    """
    if backend not in ("numpy", "cupy", "opengl"):
        raise ValueError("backend must be numpy, cupy or opengl")
    # Keep Qt's offscreen-context owner alive for all native runs; no widgets are created.
    app = None
    if backend == "opengl":
        from PyQt5 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    results = []
    for mode in ("fixed", "random", "autonomous"):
        policy = GazePolicy(mode, seed=policy_seed)
        precision = GaussianSpatialPrecision(sigma=sigma, baseline=baseline, peak=peak)
        config = PredictiveCodingConfig(
            dtype="float32", inference_steps=inference_steps, spatial_precision=precision,
        )
        engine = RippleEngine(
            resolution=stimulus.resolution, plane_size_m=(1, 1),
            use_gpu=backend == "cupy", use_shader=backend == "opengl",
        )
        source = CircularDotSource(stimulus)
        capture = CircularDotCapture(source)
        try:
            orchestrator = RippleSourceOrchestrator(
                config=RippleSourceOrchestratorConfig(
                    synthetic=SyntheticSourceConfig(enabled=False),
                    camera_frame=CameraFrameSourceConfig(enabled=True),
                    prediction_error=PredictionErrorTransformConfig(enabled=True),
                    visual_pathway=config,
                ),
                processor=None, engine=engine, resolution=stimulus.resolution,
                n_sources=1, camera_capture=capture,
            )
            uncertainty = orchestrator.enable_visual_uncertainty(
                dt=1/stimulus.frame_rate,
                initial_variance=initial_variance, process_variance=process_variance,
                memory_time_constant=memory_time_constant,
            )
            orchestrator.gaze_policy = policy
            frames = []
            for _ in range(source.frame_count):
                resolved = orchestrator.resolve()
                engine.propagate(resolved.drive_grid)
                prediction = orchestrator.predict_visual()
                orchestrator.correct_from_observations()
                if source.latest is None or orchestrator.visual_observation is None:
                    raise RuntimeError("Synthetic evaluation failed to deliver an observation")
                precision = orchestrator.visual_readout.config.spatial_precision
                scores = orchestrator.gaze_scores
                iy = round(precision.gaze[1]*(scores.shape[0]-1))
                ix = round(precision.gaze[0]*(scores.shape[1]-1))
                mse, mean_variance = _prediction_metrics(
                    prediction, orchestrator.visual_observation, uncertainty.variance,
                    backend=engine.backend,
                )
                frames.append(GazeFrameMetrics(
                    time=source.latest.time, gaze=precision.gaze,
                    expected_gain_nats=float(scores[iy, ix]),
                    posterior_hidden_variance_mean=mean_variance, forecast_mse=mse,
                    precision_mass=_precision_mass(precision, stimulus.resolution),
                ))
            results.append(GazeRunResult(
                policy=mode, frames=tuple(frames),
                cumulative_expected_gain_nats=sum(frame.expected_gain_nats for frame in frames),
                final_hidden_variance_mean=frames[-1].posterior_hidden_variance_mean,
                mean_forecast_mse=float(np.mean([frame.forecast_mse for frame in frames])),
                cumulative_precision_mass=sum(frame.precision_mass for frame in frames),
            ))
        finally:
            capture.release()
            if engine.use_shader:
                engine.propagator.close()
    return tuple(results)


def main():
    parser = argparse.ArgumentParser(description="Headless fixed/random/autonomous gaze comparison.")
    parser.add_argument("--backend", choices=("numpy", "cupy", "opengl"), default="numpy")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(120, 160))
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--brownian-scale", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--inference-steps", type=int, default=8,
        help="Legacy PC setting; unused by the single temporal Gaussian update.",
    )
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--baseline", type=float, default=0.01)
    parser.add_argument("--peak", type=float, default=1.0)
    parser.add_argument("--initial-variance", type=float, default=1.0)
    parser.add_argument(
        "--process-variance", type=float, default=0.01,
        help="Hidden-state diffusion intensity (variance per stimulus second).",
    )
    parser.add_argument("--memory-time-constant", type=float, default=1/3, help="Hidden-state memory in stimulus seconds.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stimulus = CircularDotConfig(
        resolution=tuple(args.resolution), radius=0.3*min(args.resolution),
        dot_radius=max(1, 0.02*min(args.resolution)), frame_rate=args.fps,
        duration=args.duration, brownian_scale=args.brownian_scale, seed=args.seed,
    )
    results = run_gaze_comparison(
        stimulus=stimulus, backend=args.backend, inference_steps=args.inference_steps,
        sigma=args.sigma, baseline=args.baseline, peak=args.peak,
        initial_variance=args.initial_variance, process_variance=args.process_variance,
        memory_time_constant=args.memory_time_constant,
        policy_seed=args.policy_seed,
    )
    report = {
        "stimulus": asdict(stimulus),
        "backend": args.backend,
        "policy_seed": args.policy_seed,
        "inference_steps": args.inference_steps,
        "sigma": args.sigma, "baseline": args.baseline, "peak": args.peak,
        "initial_variance": args.initial_variance, "process_variance": args.process_variance,
        "memory_time_constant": args.memory_time_constant,
        "inference_model": "Exact OU forecast and one frozen-affine Gaussian mean/covariance update; PC settling unused.",
        "canvas_feedback": "Deterministic hidden-innovation feedback for the next known input; no inferred wave covariance.",
        "measurement_model": (
            "Deterministic frames; R = I / positive precision, zero precision is unobserved. "
            "Noise is assumed, not sampled or calibrated."
        ),
        "budget": "Equal frames and observation schedules; precision mass varies with boundary clipping.",
        "runs": [asdict(result) for result in results],
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
