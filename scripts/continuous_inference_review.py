"""Capture the integrated continuous solver for the live document-review page."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audioviz.engine import RippleEngine
from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.readouts.predictive_coding import PredictiveCodingConfig


def camera_frame(index, shape=(48, 96)):
    rows, _ = shape
    y, x = np.indices(shape)
    image = np.zeros((*shape, 3))
    image[8:20, 14:32] = (.9, .15, .1)
    image[28:40, 65:84] = (.1, .2, .9)
    center = 48 + 18 * np.sin(index / 30 * .8)
    image[(x - center)**2 + (y - rows/2)**2 < 6**2] = (.1, .8, .25)
    return image


def plain_graph(shape=(48, 96), *, hidden=6, audio=0, **config):
    return ContinuousScenePredictiveCodingGraph(
        visual_shape=shape, hidden_nodes=hidden, audio_nodes=audio,
        config=PredictiveCodingConfig(hidden_channels=hidden, **config),
    )


def xor_probe():
    cases = [(a, b, float(a != b)) for a in (0., 1.) for b in (0., 1.)]
    graph = plain_graph(
        (1, 2), audio=1, inference_rate=2., inference_steps=32,
        learning_enabled=True, learning_rate=.02,
    )
    # Fixed root context permits an offset in an otherwise bias-free network.
    canvas = np.full((1, 2, 3), .2)
    for _ in range(200):
        for a, b, target in cases:
            graph.reset()
            image = np.array([[[a]*3, [b]*3]])
            for _ in range(10):
                graph.forecast(canvas, .1)
                graph.condition(image, np.array([target]))
    graph.set_learning(enabled=False)
    results = []
    for a, b, target in cases:
        graph.reset()
        image = np.array([[[a]*3, [b]*3]])
        for _ in range(100):
            graph.forecast(canvas, .1)
            graph.condition(image)
        results.append({
            "a": a, "b": b, "target": target,
            "prediction": float(graph.audio_prediction[0]),
        })
    return {
        "cases": results,
        "rmse": float(np.sqrt(np.mean([
            (case["prediction"] - case["target"])**2 for case in results
        ]))),
        "protocol": "200 epochs, fixed order, repeated inference/learning; states reset per case; "
        "weights retained; evaluation output unobserved and learning disabled; fixed root context 0.2. "
        "This is the same four-case truth table, not a held-out generalization experiment.",
    }


def black_probe():
    results = []
    for level in (0., .5, 1.):
        graph = plain_graph((1, 1), hidden=2, inference_rate=2., learning_enabled=True)
        canvas = np.zeros((1, 1, 3))
        for _ in range(900):
            graph.forecast(canvas, 1/30)
            graph.condition(np.full_like(canvas, level))
        results.append({
            "observed": level, "prediction": float(graph.visual_prediction.mean()),
            "gradient": graph.sensory_gradient.tolist(),
            "observation_drive": (
                graph.sensory_error * graph._decode_sensory(graph.sensory_potential)
                * (1 - graph._decode_sensory(graph.sensory_potential))
            ).tolist(),
        })
    return results


def capture(output, frames, rate, canvas_rate, learning_rate):
    output.mkdir(parents=True, exist_ok=True)
    images = output / "frames"
    images.mkdir(exist_ok=True)
    shape = (48, 96)
    sample_pixels = [(3, 3), (14, 23), (34, 74)]
    indices = np.array([3*(row*shape[1]+column)+channel
                        for row, column in sample_pixels for channel in range(3)])
    models = {
        name: plain_graph(shape, inference_rate=rate, learning_enabled=True,
                          learning_rate=learning_rate, canvas_rate=canvas_rate)
        for name in ("waves", "held")
    }
    engine = RippleEngine(
        resolution=shape, plane_size_m=(50, 50), speed=340,
        damping=.988, use_gpu=False, use_shader=False, boundary_condition="neumann",
    )
    held_canvas = np.zeros((*shape, 3))
    snapshots = []
    curves = {"time": [], "waves": [], "held": [], "settled_waves": [], "settled_held": []}
    for frame in range(frames):
        image = camera_frame(frame)
        for name, graph in models.items():
            if name == "waves":
                canvas = engine.propagate()
            else:
                canvas = held_canvas
            forecast, _ = graph.forecast(canvas, 1/30)
            direction = graph.condition(image)
            correction = canvas_rate * direction
            if name == "waves":
                corrected_canvas = engine.apply_observation_correction(correction)
            else:
                held_canvas += correction
                corrected_canvas = held_canvas
            curves[name].append(float(np.sqrt(np.mean((forecast - image)**2))))
            curves[f"settled_{name}"].append(
                float(np.sqrt(np.mean((graph.visual_prediction - image)**2))),
            )
            if frame % 15 == 0 or frame == frames - 1:
                snapshot = graph.belief_snapshot()
                h = snapshot["mean"][graph.hidden_slice]
                a = np.tanh(h)
                eh = snapshot["hidden_error"]
                ez = snapshot["sensory_potential_error"]
                z = snapshot["sensory_potential"]
                pred = graph._decode_sensory(z)[graph.visual_slice]
                record = {
                    "frame": frame, "time": (frame+1)/30, "mode": name,
                    "h": h.tolist(), "grad_h": snapshot["hidden_gradient"].tolist(),
                    "h_decay": (h/graph.memory_time_constant).tolist(),
                    "h_local": eh.tolist(),
                    "h_recurrent": (
                        -(snapshot["recurrent_weights"].T @ eh) * (1-a*a)
                    ).tolist(),
                    "h_sensory": (
                        -(snapshot["sensory_weights"].T @ ez) * (1-a*a)
                    ).tolist(),
                    "z": z[indices].tolist(), "grad_z": snapshot["sensory_gradient"][indices].tolist(),
                    "z_decay": (z[indices]/graph.memory_time_constant).tolist(),
                    "z_coupling": ez[indices].tolist(),
                    "z_observation": (
                        -snapshot["sensory_error"][indices] * pred[indices] * (1-pred[indices])
                    ).tolist(),
                    "root": snapshot["canvas"][indices].tolist(),
                    "grad_root": snapshot["canvas_gradient"][indices].tolist(),
                    "observed": image.ravel()[indices].tolist(),
                    "predicted": pred[indices].tolist(),
                    "Wc": snapshot["canvas_weights"][:, indices].tolist(),
                    "Wh": snapshot["recurrent_weights"].tolist(),
                    "Ws": snapshot["sensory_weights"][indices].tolist(),
                    "weight_gradients": {
                        key: (
                            value[:, indices] if key == "canvas_weights" else
                            value[indices] if key == "sensory_weights" else value
                        ).tolist()
                        for key, value in snapshot["weight_gradients"].items()
                    },
                    "weight_gradient_norms": {
                        key: float(np.linalg.norm(value))
                        for key, value in snapshot["weight_gradients"].items()
                    },
                    "energy": snapshot["energy"], "micro_energy": list(snapshot["inference_energy"]),
                    "forecast_rmse": curves[name][-1],
                    "settled_rmse": curves[f"settled_{name}"][-1],
                    "canvas_peak": float(np.max(np.abs(corrected_canvas))),
                }
                for kind, values in (
                    ("input", image), ("forecast", forecast),
                    ("settled", graph.visual_prediction),
                    ("canvas", .5 + .5*np.tanh(corrected_canvas/.15)),
                    ("error", np.repeat(
                        np.sqrt(np.mean((forecast-image)**2, axis=-1))[..., None], 3, axis=-1,
                    )),
                ):
                    filename = f"{name}-{frame:04}-{kind}.png"
                    Image.fromarray(
                        np.rint(np.clip(values, 0, 1)*255).astype(np.uint8),
                    ).save(images / filename)
                    record[kind+"_image"] = "frames/" + filename
                snapshots.append(record)
        curves["time"].append((frame+1)/30)
    result = {
        "source_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("capture_script", Path(__file__)),
                ("continuous_solver", Path(__file__).resolve().parents[1]
                 / "src/audioviz/readouts/continuous_scene_predictive_coding.py"),
            )
        },
        "settings": {
            "shape": shape, "sensory_nodes": int(np.prod(shape))*3, "hidden_nodes": 6,
            "frames": frames, "fps": 30, "rate": rate, "learning_rate": learning_rate,
            "canvas_rate": canvas_rate, "tau": 1/3, "inference_steps": 8,
            "wave_dt": engine.dt, "sample_pixels": sample_pixels, "indices": indices.tolist(),
            "unit_precision": True, "weight_decay": 0,
        },
        "snapshots": snapshots, "curves": curves,
        "black_gray_white": black_probe(), "xor": xor_probe(),
    }
    path = output / "capture.json"
    path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "capture": str(path), "black_gray_white": result["black_gray_white"],
        "xor": result["xor"], "final_forecast_rmse": {
            name: values[-1] for name, values in curves.items() if name != "time"
        },
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--inference-rate", type=float, default=.3)
    parser.add_argument("--canvas-rate", type=float, default=.3)
    parser.add_argument("--learning-rate", type=float, default=.001)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    capture(args.output_dir, args.frames, args.inference_rate, args.canvas_rate, args.learning_rate)
