"""Measure shared PC graph memory across circular foveal radii and hidden sizes."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import tracemalloc

import numpy as np

from audioviz.readouts import PredictiveCodingConfig
from audioviz.readouts.shared_predictive_coding import SharedPredictiveCodingGraph


MIB = 1024**2


def disk_slots(radius: int) -> int:
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    return sum(2*math.isqrt(radius*radius-y*y)+1 for y in range(-radius, radius+1))


def parameter_bytes(slots: int, hidden: int, audio: int) -> int:
    visual = 3*slots
    sensory = visual+audio
    return 8*(hidden*visual + sensory*hidden + hidden*hidden + hidden + sensory)


def array_storage_bytes(values, array_type=np.ndarray) -> int:
    """Count retained storage once, including bases shared by array views."""
    seen = set()

    def count(value):
        if isinstance(value, array_type):
            owner = value
            while isinstance(owner.base, array_type):
                owner = owner.base
            if id(owner) in seen:
                return 0
            seen.add(id(owner))
            return int(owner.nbytes)
        if isinstance(value, (dict, list, tuple)):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            items = value.values() if isinstance(value, dict) else value
            return sum(count(item) for item in items)
        return 0

    return count(values)


def case_description(radius, hidden_nodes, audio_nodes, backend):
    slots = disk_slots(radius)
    return dict(
        radius=radius, slots=slots, visual_nodes=3*slots,
        hidden_nodes=hidden_nodes, audio_nodes=audio_nodes, backend=backend,
        parameter_bytes=parameter_bytes(slots, hidden_nodes, audio_nodes),
    )


def supported_case(slots, hidden_nodes, audio_nodes):
    return (
        slots <= SharedPredictiveCodingGraph.MAX_VISUAL_SLOTS
        and 1 <= hidden_nodes <= SharedPredictiveCodingGraph.MAX_HIDDEN_NODES
        and 0 <= audio_nodes <= SharedPredictiveCodingGraph.MAX_AUDIO_NODES
    )


def measure_case(
    *, radius, hidden_nodes, audio_nodes=16, backend="numpy", inference_steps=8,
    candidates=117, learning=True,
):
    row = case_description(radius, hidden_nodes, audio_nodes, backend)
    if not supported_case(row["slots"], hidden_nodes, audio_nodes):
        raise ValueError("case exceeds numerical core capacity")
    if backend not in ("numpy", "cupy"):
        raise ValueError("backend must be numpy or cupy")
    xp = np
    pool = None
    gpu_peak = 0
    allocator_context = nullcontext()
    if backend == "cupy":
        import cupy as xp

        xp.zeros(1)
        xp.cuda.get_current_stream().synchronize()
        pool = xp.cuda.MemoryPool()

        def allocate(size):
            nonlocal gpu_peak
            pointer = pool.malloc(size)
            gpu_peak = max(gpu_peak, pool.used_bytes())
            return pointer

        allocator_context = xp.cuda.using_allocator(allocate)
    phase_peaks = {}
    tracemalloc.start()
    try:
        with allocator_context:
            graph = SharedPredictiveCodingGraph(
                visual_shape=(row["slots"], 1), hidden_nodes=hidden_nodes,
                audio_nodes=audio_nodes, backend=xp,
                config=PredictiveCodingConfig(
                    hidden_channels=hidden_nodes, inference_steps=inference_steps,
                    dtype="float64", learning_enabled=learning, learning_rate=.03,
                ),
            )
            phase_peaks["construct"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.reset_peak()
            rng = np.random.default_rng(7)
            shape = (*graph.visual_shape, 3)
            canvas = xp.asarray(rng.normal(0, .1, shape).astype(np.float32))
            visual = xp.asarray(rng.random(shape).astype(np.float32))
            audio = xp.asarray(rng.random(audio_nodes).astype(np.float32)) if audio_nodes else None
            phase_peaks["inputs"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.reset_peak()
            graph.forecast(canvas, 1/30)
            graph.condition(visual, audio)
            if pool is not None:
                xp.cuda.get_current_stream().synchronize()
            phase_peaks["update"] = tracemalloc.get_traced_memory()[1]
            retained = array_storage_bytes(vars(graph), xp.ndarray)
            measured_parameters = sum(
                int(getattr(graph, name).nbytes) for name in (
                    "canvas_weights", "recurrent_weights", "sensory_weights",
                    "hidden_bias", "sensory_bias",
                )
            )
            if measured_parameters != row["parameter_bytes"]:
                raise AssertionError("parameter byte formula does not match allocated model")
            if candidates:
                tracemalloc.reset_peak()
                # Distinct patches exercise the same batched search as autonomous gaze.
                patches = xp.asarray(rng.normal(0, .1, (candidates, *shape)).astype(np.float32))
                scores = graph.candidate_information_gains(patches, 1/30)
                if not np.isfinite(scores).all():
                    raise FloatingPointError("nonfinite candidate information gain")
                phase_peaks["gaze"] = tracemalloc.get_traced_memory()[1]
            if not bool(xp.all(xp.isfinite(graph.mean))):
                raise FloatingPointError("nonfinite graph state")
            row.update(
                status="measured", learning=learning, inference_steps=inference_steps,
                candidates=candidates, graph_array_bytes=retained,
                host_peak_bytes=max(phase_peaks.values()),
                host_phase_peak_bytes=phase_peaks,
                gpu_live_bytes=pool.used_bytes() if pool is not None else None,
                gpu_peak_bytes=gpu_peak if pool is not None else None,
                gpu_reserved_bytes=pool.total_bytes() if pool is not None else None,
            )
    finally:
        tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row["process_peak_rss_bytes"] = int(rss*(1 if sys.platform == "darwin" else 1024))
    return row


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", nargs="+", type=positive, default=[8, 16, 24, 32, 64])
    parser.add_argument("--hidden-sizes", nargs="+", type=positive, default=[6, 16, 32])
    parser.add_argument(
        "--backend", choices=("numpy", "cupy"), default="numpy",
        help="Numerical core backend. The OpenGL runner uses the NumPy core.",
    )
    parser.add_argument("--audio-nodes", type=nonnegative, default=16)
    parser.add_argument("--inference-steps", type=positive, default=8)
    parser.add_argument("--candidates", type=nonnegative, default=117)
    parser.add_argument("--learning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--estimate-only", action="store_true", help="Only calculate parameter storage; allocate no graph.")
    parser.add_argument("--image-shape", nargs=2, type=positive, default=(1080, 1920), metavar=("ROWS", "COLS"))
    parser.add_argument("--json", action="store_true", help="Write machine-readable results to stdout.")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        print(json.dumps(measure_case(**json.loads(args.worker))))
        return
    if args.candidates > SharedPredictiveCodingGraph.MAX_CANDIDATE_BATCH:
        parser.error(f"candidates must not exceed {SharedPredictiveCodingGraph.MAX_CANDIDATE_BATCH}")
    rows = []
    for radius in args.radii:
        for hidden in args.hidden_sizes:
            row = case_description(radius, hidden, args.audio_nodes, args.backend)
            supported = supported_case(row["slots"], hidden, args.audio_nodes)
            if args.estimate_only or not supported:
                row["status"] = "parameter estimate only" if supported else "parameter estimate only: core limit"
            else:
                spec = dict(
                    radius=radius, hidden_nodes=hidden, audio_nodes=args.audio_nodes,
                    backend=args.backend, inference_steps=args.inference_steps,
                    candidates=args.candidates, learning=args.learning,
                )
                try:
                    process = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
                        capture_output=True, text=True, check=True,
                    )
                except subprocess.CalledProcessError as error:
                    parser.exit(
                        1, f"Measurement failed for radius={radius}, hidden={hidden} "
                        f"(worker exit {error.returncode}):\n{error.stderr}",
                    )
                if process.stderr:
                    print(process.stderr, end="", file=sys.stderr)
                row = json.loads(process.stdout)
            rows.append(row)
    payload = dict(
        backend=args.backend, dtype="float64", input_dtype="float32",
        gaze_backend="numpy", learning=args.learning,
        inference_steps=args.inference_steps, candidates=args.candidates,
        core_limits=dict(
            visual_slots=SharedPredictiveCodingGraph.MAX_VISUAL_SLOTS,
            hidden_nodes=SharedPredictiveCodingGraph.MAX_HIDDEN_NODES,
            audio_nodes=SharedPredictiveCodingGraph.MAX_AUDIO_NODES,
        ),
        image_shape=args.image_shape,
        image_buffer_bytes=math.prod(args.image_shape)*3*4,
        image_buffer_allocated=False,
        rows=rows,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(
        f"{args.backend} float64 core; learning={args.learning}; "
        f"{args.inference_steps} inference steps; {args.candidates} gaze candidates.\n"
        "Each measured case runs in a fresh process: construct, one forecast/evidence update, then requested gaze scoring.\n"
        "Full circular support (no edge clipping); float32 sensory/root inputs, float64 model arrays.\n"
        "Graph = unique retained array storage (includes parameter snapshots/Jacobians, excludes input fixtures).\n"
        "Host peak = tracemalloc peak including graph, inputs and temporary arrays; excludes untraced native workspaces.\n"
        "RSS = whole-process high-water mark including interpreter/libraries. GPU columns cover the private CuPy array pool.\n"
        "Gaze scoring currently runs on the host for both numerical backends. No camera, GUI, image buffers or wave simulation.\n"
        f"Core limits: {payload['core_limits']}. Above-limit rows estimate parameters ONLY; other measurements are absent.\n"
        "The current application radius limit is 16 pixels; this script also exercises larger stencils within core limits.\n"
        f"One RGB float32 image at {args.image_shape[0]}x{args.image_shape[1]}: "
        f"{payload['image_buffer_bytes']/MIB:.2f} MiB (accounting only; not allocated; not total app memory).\n"
    )
    print(f"{'radius':>6} {'pixels':>7} {'hidden':>6} {'params':>9} {'graph':>9} {'host peak':>10} {'GPU peak':>9} {'GPU pool':>9} {'RSS':>9}  status")
    print(f"{'':>6} {'':>7} {'':>6} {'MiB':>9} {'MiB':>9} {'MiB':>10} {'MiB':>9} {'MiB':>9} {'MiB':>9}")
    for row in rows:
        def mib(key):
            value = row.get(key)
            return f"{value/MIB:.3f}" if value is not None else "-"

        print(
            f"{row['radius']:6} {row['slots']:7} {row['hidden_nodes']:6} "
            f"{mib('parameter_bytes'):>9} {mib('graph_array_bytes'):>9} {mib('host_peak_bytes'):>10} "
            f"{mib('gpu_peak_bytes'):>9} {mib('gpu_reserved_bytes'):>9} {mib('process_peak_rss_bytes'):>9}  {row['status']}"
        )


if __name__ == "__main__":
    main()
