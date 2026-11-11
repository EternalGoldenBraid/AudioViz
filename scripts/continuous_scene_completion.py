"""Test whether foveal scanning learns two static scenes well enough to complete

them from one small disambiguating patch, with the fovea strictly smaller than
either scene. The eye stays open throughout: whatever the fovea currently
looks at is the only clamp. There is no separate unclamping step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audioviz.readouts.foveal_routing import WorldAnchoredFovea

from continuous_inference_review import plain_graph

PATTERNS = ("lines", "circles")


def pattern_image(name, shape, period=4, low=.15, high=.85):
    """A full scene, strictly larger than the fovea, with one global identity.

    Values avoid exact 0: the network's passive zero-decay equilibrium would
    otherwise coincide with a uniform low band and produce zero observation
    error (no hidden drive) purely from that coincidence, not from capacity.
    """
    rows, cols = shape
    y, x = np.indices(shape)
    if name == "lines":
        stripe = (x // period % 2).astype(np.float64)
    elif name == "circles":
        center_y, center_x = (rows - 1) / 2, (cols - 1) / 2
        radius = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        stripe = (radius // period % 2).astype(np.float64)
    else:
        raise ValueError("pattern name must be lines or circles")
    scaled = low + stripe * (high - low)
    return np.repeat(scaled[..., None], 3, axis=2)


def raster_fixations(shape, radius):
    """Every fixation whose disk touches a distinct world region, covering it fully."""
    axes = [
        np.linspace(0, size - 1, int(np.ceil((size - 1) / radius)) + 1)
        for size in shape
    ]
    rows, columns = axes
    return [
        (float(column / max(1, shape[1] - 1)), float(row / max(1, shape[0] - 1)))
        for row in rows for column in columns
    ]


def noisy(image, rng, sigma):
    """A single camera-like read: real sensors never return exact clean values."""
    return image + rng.normal(scale=sigma, size=image.shape)


def apply_clamp_strength(graph, observed, evidence, strength):
    """Blend the settled sensory potential toward observed evidence.

    The shared solver only pulls observed coordinates softly, through the
    decoded residual; it never overwrites ``sensory_potential`` directly.
    This blend is local to this experiment script, not a change to the
    model's energy or learned precision. strength=0 keeps the solver's own
    soft pull; strength=1 pins observed coordinates to the evidence exactly.
    """
    if strength <= 0:
        return
    mask = np.repeat(observed.reshape(-1), 3)
    visual = graph.sensory_potential[graph.visual_slice]
    visual[mask] = (1 - strength) * visual[mask] + strength * evidence.reshape(-1)[mask]
    graph.sensory_potential[graph.visual_slice] = visual


def scan_scene(graph, fovea, canvas_shape, image, fixations, *, dt, hold_frames, rng, sigma, clamp_strength):
    """Move the open fovea across every fixation; each one clamps only its disk."""
    for gaze in fixations:
        fovea.set_gaze(gaze)
        observed = fovea.observation_mask
        for _ in range(hold_frames):
            canvas = rng.normal(scale=sigma, size=(*canvas_shape, 3))
            graph.forecast(canvas, dt)
            evidence = noisy(image, rng, sigma)
            graph.condition(evidence, visual_mask=observed)
            apply_clamp_strength(graph, observed, evidence, clamp_strength)


def train(graph, fovea, shape, *, epochs, dt, hold_frames, rng, sigma, clamp_strength):
    fixations = raster_fixations(shape, fovea.radius)
    graph.set_learning(enabled=True)
    for _ in range(epochs):
        for name in PATTERNS:
            graph.reset()
            scan_scene(
                graph, fovea, shape, pattern_image(name, shape), fixations,
                dt=dt, hold_frames=hold_frames, rng=rng, sigma=sigma, clamp_strength=clamp_strength,
            )


def evaluate(graph, fovea, shape, *, patch_gaze, dt, settle_frames, rng, sigma, clamp_strength):
    """Fixate only the distinguishing patch, then report the completed full scene."""
    graph.set_learning(enabled=False)
    fovea.set_gaze(patch_gaze)
    observed = fovea.observation_mask
    results = {}
    for name in PATTERNS:
        graph.reset()
        image = pattern_image(name, shape)
        hidden_trace = []
        for _ in range(settle_frames):
            canvas = rng.normal(scale=sigma, size=(*shape, 3))
            graph.forecast(canvas, dt)
            evidence = noisy(image, rng, sigma)
            graph.condition(evidence, visual_mask=observed)
            apply_clamp_strength(graph, observed, evidence, clamp_strength)
            hidden_trace.append(graph.mean[graph.hidden_slice].copy())
        prediction = graph.visual_prediction.copy()
        blank = np.full_like(image, .5)
        mask3 = np.repeat(observed[..., None], 3, axis=2)
        results[name] = {
            "patch_pixels": int(observed.sum()),
            "prediction": prediction,
            "hidden_trace": hidden_trace,
            "rmse_correct": float(np.sqrt(np.mean((prediction - image) ** 2))),
            "rmse_other": float(np.sqrt(np.mean(
                (prediction - pattern_image(PATTERNS[1 - PATTERNS.index(name)], shape)) ** 2,
            ))),
            "rmse_blank_baseline": float(np.sqrt(np.mean((prediction - blank) ** 2))),
            "rmse_inside_patch": float(np.sqrt(np.mean((prediction[mask3] - image[mask3]) ** 2))),
            "rmse_outside_patch": float(np.sqrt(np.mean((prediction[~mask3] - image[~mask3]) ** 2))),
        }
    return results, observed


def run(
    output, *, shape=(32, 32), radius=5, hidden=4, epochs=60, hold_frames=3,
    settle_frames=30, dt=1 / 30, learning_rate=.05, sigma=.03, seed=0,
    tau=1 / 3, clamp_strength=0., label="completion",
):
    rng = np.random.default_rng(seed)
    graph = plain_graph(
        shape, hidden=hidden, inference_rate=.3, learning_enabled=True,
        learning_rate=learning_rate, tau=tau,
    )
    fovea = WorldAnchoredFovea(shape, radius=radius)
    train(
        graph, fovea, shape, epochs=epochs, dt=dt, hold_frames=hold_frames,
        rng=rng, sigma=sigma, clamp_strength=clamp_strength,
    )
    center = ((shape[0] - 1) / 2 / max(1, shape[0] - 1), (shape[1] - 1) / 2 / max(1, shape[1] - 1))
    results, observed = evaluate(
        graph, fovea, shape, patch_gaze=center, dt=dt, settle_frames=settle_frames,
        rng=rng, sigma=sigma, clamp_strength=clamp_strength,
    )

    columns = len(PATTERNS) * 2
    atlas = Image.new("RGB", (columns * shape[1], shape[0]))
    for index, name in enumerate(PATTERNS):
        atlas.paste(
            Image.fromarray(np.rint(np.clip(pattern_image(name, shape), 0, 1) * 255).astype(np.uint8)),
            (2 * index * shape[1], 0),
        )
        atlas.paste(
            Image.fromarray(np.rint(np.clip(results[name]["prediction"], 0, 1) * 255).astype(np.uint8)),
            ((2 * index + 1) * shape[1], 0),
        )

    sources = [
        Path(__file__),
        Path(__file__).resolve().parents[1] / "src/audioviz/readouts/continuous_scene_predictive_coding.py",
        Path(__file__).resolve().parents[1] / "src/audioviz/readouts/foveal_routing.py",
    ]
    summary = {
        "schema": "continuous-completion-v1",
        "settings": {
            "shape": list(shape), "foveal_radius": radius, "hidden_nodes": hidden,
            "epochs": epochs, "hold_frames": hold_frames, "settle_frames": settle_frames,
            "dt": dt, "patch_pixels": int(observed.sum()), "patch_gaze": list(center),
            "learning_rate": learning_rate, "sigma": sigma, "tau": tau,
            "clamp_strength": clamp_strength, "seed": seed,
        },
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "cases": {
            name: {key: value for key, value in case.items() if key not in ("prediction", "hidden_trace")}
            for name, case in results.items()
        },
        "atlas": {"columns": columns, "labels": ["lines target", "lines prediction", "circles target", "circles prediction"]},
    }
    trace = {
        "schema": "continuous-completion-trace-v1",
        "settings": summary["settings"],
        "hidden_nodes": hidden,
        "patterns": {
            name: [vector.tolist() for vector in case["hidden_trace"]]
            for name, case in results.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary_atlas = output / f"{label}-atlas.pending.png"
    atlas.save(temporary_atlas)
    temporary_atlas.replace(output / f"{label}-atlas.png")
    temporary = output / f"{label}-summary.pending.json"
    temporary.write_text(json.dumps(summary, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(output / f"{label}-summary.json")
    temporary_trace = output / f"{label}-trace.pending.json"
    temporary_trace.write_text(json.dumps(trace, allow_nan=False, indent=2), encoding="utf-8")
    temporary_trace.replace(output / f"{label}-trace.json")
    print(json.dumps(summary["cases"], indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--hidden", default=4, type=int)
    parser.add_argument("--radius", default=5, type=int)
    parser.add_argument("--shape", nargs=2, default=(32, 32), type=int)
    parser.add_argument("--learning-rate", default=.05, type=float)
    parser.add_argument("--sigma", default=.03, type=float)
    parser.add_argument("--tau", default=1 / 3, type=float)
    parser.add_argument("--clamp-strength", default=0., type=float)
    parser.add_argument("--label", default="completion", type=str)
    args = parser.parse_args()
    run(
        args.output_dir, shape=tuple(args.shape), radius=args.radius, hidden=args.hidden,
        epochs=args.epochs, learning_rate=args.learning_rate, sigma=args.sigma,
        tau=args.tau, clamp_strength=args.clamp_strength, label=args.label,
    )
