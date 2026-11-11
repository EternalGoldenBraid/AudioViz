"""Capture the integrated continuous solver for the live document-review page."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
import subprocess

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from audioviz.engine import RippleEngine
from audioviz.physics.wave_propagator import ForceDrivenCanvasCPU
from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.readouts.predictive_coding import PredictiveCodingConfig, PredictiveCodingRGBReadout
from audioviz.visualization.ripple_renderers import _canvas_to_rgb


def camera_frame(index, shape=(48, 96)):
    rows, _ = shape
    y, x = np.indices(shape)
    image = np.zeros((*shape, 3))
    image[8:20, 14:32] = (.9, .15, .1)
    image[28:40, 65:84] = (.1, .2, .9)
    center = 48 + 18 * np.sin(index / 30 * .8)
    image[(x - center)**2 + (y - rows/2)**2 < 6**2] = (.1, .8, .25)
    return image


def plain_graph(shape=(48, 96), *, hidden=6, audio=0, visual_decoder="linear", tau=None, **config):
    kwargs = {} if tau is None else {"memory_time_constant": tau}
    return ContinuousScenePredictiveCodingGraph(
        visual_shape=shape, hidden_nodes=hidden, audio_nodes=audio,
        config=PredictiveCodingConfig(hidden_channels=hidden, **config),
        visual_decoder=visual_decoder, **kwargs,
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


def black_probe(visual_decoder="linear"):
    results = []
    for level in (0., .5, 1.):
        graph = plain_graph(
            (1, 1), hidden=2, inference_rate=2., learning_enabled=True, visual_decoder=visual_decoder,
        )
        canvas = np.zeros((1, 1, 3))
        for _ in range(900):
            graph.forecast(canvas, 1/30)
            graph.condition(np.full_like(canvas, level))
        results.append({
            "observed": level, "prediction": float(graph.visual_prediction.mean()),
            "gradient": graph.sensory_gradient.tolist(),
            "observation_drive": (
                graph.sensory_error * graph._sensory_slope(graph.sensory_potential)
            ).tolist(),
        })
    return results


def comparison_frame(index, frames, image, panels, curves, fonts, rate, force_strength, learning_rate):
    title, font, small, legend = fonts
    picture = Image.new("RGB", (1104, 1280), "#10141e")
    draw = ImageDraw.Draw(picture)
    draw.text((24, 16), "Current run: per-pixel vs force-driven continuous inference", font=title, fill="#f3f5fa")
    draw.text((24, 58), f"Same input | 30 fps | learning {learning_rate:g} | linear output | right precision 1",
              font=small, fill="#bac5d8")
    picture.paste(Image.fromarray(np.rint(image*255).astype(np.uint8)).resize(
        (288, 144), Image.Resampling.NEAREST,
    ), (24, 100))
    draw.text((340, 92), "Whole-frame pre-evidence forecast RMSE", font=small, fill="#f3f5fa")
    for tick in (0., .3, .6):
        yy = 238 - tick/.6*106
        draw.line((380, yy, 1060, yy), fill="#354156")
        draw.text((340, yy-9), f"{tick:g}", font=small, fill="#bac5d8")
    for seconds in (0, frames/60, frames/30):
        xx = 380 + 680*seconds/(frames/30)
        draw.text((xx-9, 242), f"{seconds:g}s", font=small, fill="#bac5d8")
    for position, (key, label, color) in enumerate((
        ("legacy_waves", "Per-pixel waves", "#6cdec0"),
        ("legacy_held", "Per-pixel held", "#e6cf68"),
        ("waves", "Continuous waves", "#7ca8ff"),
        ("held", "Continuous held", "#fa9f71"),
    )):
        legend_x = 340 + position*185
        draw.line((legend_x, 120, legend_x+12, 120), fill=color, width=3)
        draw.text((legend_x+16, 111), label, font=legend, fill=color)
        if index:
            draw.line([
                (380 + 680*i/max(1, frames-1), 238-value/.6*106)
                for i, value in enumerate(curves[key])
            ], fill=color, width=2)
    draw.text((24, 262), f"Inference: left 0.1, right {rate:g} | left correction 0.01 | right force g={force_strength:g}",
              font=small, fill="#bac5d8")
    for column, (prefix, heading) in enumerate((
        ("legacy_", "Current per-pixel inference"), ("", "Continuous: force-driven canvas"),
    )):
        draw.text((24+552*column, 290), heading, font=font, fill="#f3f5fa")
        for row, mode in enumerate(("waves", "held")):
            left, top = 24+552*column, 349+450*row
            canvas, forecast = panels[prefix+mode]
            draw.text((left, top-32), "Waves ON" if mode == "waves" else "Waves HELD",
                      font=font, fill="#8ddbbb" if mode == "waves" else "#eac896")
            draw.text((left+270, top-27), f"Canvas RMS {np.sqrt(np.mean(canvas**2)):.2e}",
                      font=small, fill="#bac5d8")
            picture.paste(Image.fromarray(_canvas_to_rgb(canvas, limit=.15)).resize(
                (504, 252), Image.Resampling.NEAREST,
            ), (left, top))
            draw.text((left, top+266), "Forecast RGB (clipped 0..1)", font=small, fill="#bac5d8")
            draw.text((left+258, top+266), f"Forecast error | RMSE {curves[prefix+mode][-1]:.3f}",
                      font=small, fill="#bac5d8")
            error = np.sqrt(np.mean((forecast-image)**2, axis=-1))
            for x, values in (
                (left, np.clip(forecast, 0, 1)),
                (left+258, colormaps["inferno"](np.clip(error, 0, 1))[..., :3]),
            ):
                picture.paste(Image.fromarray(np.rint(values*255).astype(np.uint8)).resize(
                    (246, 123), Image.Resampling.NEAREST,
                ), (x, top+292))
    draw.text((24, 1255), f"Frame {index+1}/{frames} | Video and graph: same 30 fps timeline | Canvas RGB scale 0..0.15",
              font=legend, fill="#bac5d8")
    return picture


def record_graph(graph, frame, mode, image, corrected_canvas, indices, forecast_rmse, settled_rmse):
    snapshot = graph.belief_snapshot()
    updates = {
        name: getattr(graph, name) - snapshot[name]
        for name in graph._PARAMETERS
    }

    def sampled(name, value):
        if name == "canvas_weights":
            return value[:, indices].tolist()
        if name == "sensory_weights":
            return value[indices].tolist()
        return value.tolist()

    h = snapshot["mean"][graph.hidden_slice]
    a = np.tanh(h)
    eh = snapshot["hidden_error"]
    ez = snapshot["sensory_potential_error"]
    z = snapshot["sensory_potential"]
    pred = graph._decode_sensory(z)[graph.visual_slice]
    return {
        "frame": frame, "time": frame/30, "mode": mode,
        "h": h.tolist(), "grad_h": snapshot["hidden_gradient"].tolist(),
        "h_decay": (h/graph.memory_time_constant).tolist(),
        "h_local": eh.tolist(),
        "h_recurrent": (-(snapshot["recurrent_weights"].T @ eh) * (1-a*a)).tolist(),
        "h_sensory": (-(snapshot["sensory_weights"].T @ ez) * (1-a*a)).tolist(),
        "z": z[indices].tolist(), "grad_z": snapshot["sensory_gradient"][indices].tolist(),
        "z_decay": (z[indices]/graph.memory_time_constant).tolist(),
        "z_coupling": ez[indices].tolist(),
        "z_observation": (
            -snapshot["sensory_error"][indices] * graph._sensory_slope(z)[indices]
        ).tolist(),
        "root": snapshot["canvas"][indices].tolist(),
        "grad_root": snapshot["canvas_gradient"][indices].tolist(),
        "observed": image.ravel()[indices].tolist(),
        "predicted": pred[indices].tolist(),
        "Wc": snapshot["canvas_weights"][:, indices].tolist(),
        "Wh": snapshot["recurrent_weights"].tolist(),
        "Ws": snapshot["sensory_weights"][indices].tolist(),
        "weight_gradients": {
            key: sampled(key, value)
            for key, value in snapshot["weight_gradients"].items()
        },
        "weight_updates": {key: sampled(key, value) for key, value in updates.items()},
        "weight_update_norms": {
            key: float(np.linalg.norm(value)) for key, value in updates.items()
        },
        "weight_gradient_norms": {
            key: float(np.linalg.norm(value)) for key, value in snapshot["weight_gradients"].items()
        },
        "energy": snapshot["energy"], "micro_energy": list(snapshot["inference_energy"]),
        "forecast_rmse": forecast_rmse, "settled_rmse": settled_rmse,
        "canvas_peak": float(np.max(np.abs(corrected_canvas))),
    }


def record_legacy(model, before, frame, mode):
    updates = {name: getattr(model, name) - value for name, value in before.items()}
    return {
        "frame": frame, "time": frame/30, "mode": mode,
        "Wc": before["canvas_weights"].tolist(),
        "Ws": before["visual_weights"].tolist(),
        "weight_updates": {
            "canvas_weights": updates["canvas_weights"].tolist(),
            "sensory_weights": updates["visual_weights"].tolist(),
        },
        "weight_update_norms": {
            "canvas_weights": float(np.linalg.norm(updates["canvas_weights"])),
            "sensory_weights": float(np.linalg.norm(updates["visual_weights"])),
        },
    }


def advance_canvas_force(graph, canvas, interval, force_strength, wave_step_limit):
    for name, value in (("interval", interval), ("wave_step_limit", wave_step_limit)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(force_strength) or force_strength < 0:
        raise ValueError("force_strength must be finite and non-negative")
    snapshot = graph.belief_snapshot()
    weights = snapshot["canvas_weights"]
    hidden = snapshot["mean"][graph.hidden_slice]
    offset = hidden - snapshot["recurrent_weights"] @ np.tanh(hidden)
    mask = graph.aperture_mask
    curvature = force_strength * float(np.linalg.eigvalsh(weights @ weights.T)[-1])
    limit = min(wave_step_limit, canvas.max_step(curvature))
    steps = ceil(interval / limit)
    dt = interval / steps

    def force_at_position():
        root = np.where(mask, canvas.position.ravel(), 0.)
        error = offset - weights @ root
        return (force_strength * np.where(mask, weights.T @ error, 0.)).reshape(canvas.position.shape)

    initial_force = force_at_position()
    force = initial_force
    for step in range(steps):
        if step:
            force = force_at_position()
        canvas.step(force, dt, force_curvature=curvature)
    return {
        "simulation_time": canvas.time, "wave_substeps": steps, "wave_dt": dt,
        "max_wave_dt": limit, "force_start": initial_force,
        "force_start_norm": float(np.linalg.norm(initial_force)),
        "force_last_norm": float(np.linalg.norm(force)),
        "canvas_velocity_norm": float(np.linalg.norm(canvas.velocity)),
    }


def capture(output, frames, rate, force_strength, learning_rate):
    output.mkdir(parents=True, exist_ok=True)
    shape = (48, 96)
    sample_pixels = [(3, 3), (14, 23), (34, 74)]
    indices = np.array([3*(row*shape[1]+column)+channel
                        for row, column in sample_pixels for channel in range(3)])
    models = {
        name: plain_graph(shape, inference_rate=rate, learning_enabled=True,
                          learning_rate=learning_rate)
        for name in ("waves", "held")
    }
    legacy_engine = RippleEngine(
        resolution=shape, plane_size_m=(50, 50), speed=340,
        damping=.988, use_gpu=False, use_shader=False, boundary_condition="neumann",
    )
    damping_rate = -np.log(legacy_engine.damping) / legacy_engine.dt
    mechanical = {
        name: ForceDrivenCanvasCPU(
            (*shape, 3), dx=legacy_engine.grid_spacing,
            speed=legacy_engine.speed if name == "waves" else 0.,
            damping_rate=damping_rate, boundary_condition="neumann",
        ) for name in models
    }
    legacy_held = np.zeros((*shape, 3))
    legacy = {
        name: PredictiveCodingRGBReadout(
            canvas_shape=(*shape, 3),
            config=PredictiveCodingConfig(
                hidden_channels=6, inference_steps=8, inference_rate=.1, canvas_rate=.01,
                learning_enabled=True, learning_rate=learning_rate,
            ),
        ) for name in ("waves", "held")
    }
    snapshots = []
    legacy_snapshots = []
    curves = {name: [] for name in (
        "time", "waves", "held", "settled_waves", "settled_held", "legacy_waves", "legacy_held",
    )}
    fonts = [
        ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)
        for size in (29, 24, 17, 14)
    ]
    temporary_video = output / "comparison.pending.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", "1104x1280", "-framerate", "30",
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary_video),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for frame in range(frames):
            image = camera_frame(frame)
            panels = {}
            for name, graph in models.items():
                canvas = mechanical[name].position
                forecast, _ = graph.forecast(canvas, 1/30)
                graph.condition(image)
                physics = advance_canvas_force(
                    graph, mechanical[name], 1/30, force_strength, legacy_engine.dt,
                )
                corrected_canvas = mechanical[name].position
                curves[name].append(float(np.sqrt(np.mean((forecast-image)**2))))
                curves[f"settled_{name}"].append(
                    float(np.sqrt(np.mean((graph.visual_prediction-image)**2))),
                )
                record = record_graph(
                    graph, frame, name, image, corrected_canvas, indices,
                    curves[name][-1], curves[f"settled_{name}"][-1],
                )
                record.update({
                    key: value for key, value in physics.items() if key != "force_start"
                })
                record["force_start"] = physics["force_start"].ravel()[indices].tolist()
                snapshots.append(record)
                panels[name] = corrected_canvas, forecast
            for name, model in legacy.items():
                canvas = legacy_engine.propagate() if name == "waves" else legacy_held
                forecast = model.predict(canvas)
                before = {
                    key: getattr(model, key).copy()
                    for key in ("canvas_weights", "visual_weights")
                }
                correction = model.infer(canvas, image)
                legacy_snapshots.append(record_legacy(model, before, frame, name))
                if name == "waves":
                    corrected_canvas = legacy_engine.apply_observation_correction(correction)
                else:
                    legacy_held += correction
                    corrected_canvas = legacy_held
                curves["legacy_"+name].append(float(np.sqrt(np.mean((forecast-image)**2))))
                panels["legacy_"+name] = corrected_canvas, forecast
            for name, (canvas, forecast) in panels.items():
                if not np.isfinite(canvas).all() or not np.isfinite(forecast).all():
                    raise FloatingPointError(f"{name} frame {frame} produced non-finite canvas or forecast")
            curves["time"].append(frame/30)
            process.stdin.write(comparison_frame(
                frame, frames, image, panels, curves, fonts, rate, force_strength, learning_rate,
            ).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("comparison video encoding failed")
    temporary_video.replace(output / "comparison.mp4")
    result = {
        "source_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("capture_script", Path(__file__)),
                ("continuous_solver", Path(__file__).resolve().parents[1]
                 / "src/audioviz/readouts/continuous_scene_predictive_coding.py"),
                ("per_pixel_solver", Path(__file__).resolve().parents[1]
                 / "src/audioviz/readouts/predictive_coding.py"),
                ("mechanical_canvas", Path(__file__).resolve().parents[1]
                 / "src/audioviz/physics/wave_propagator.py"),
            )
        },
        "settings": {
            "shape": shape, "sensory_nodes": int(np.prod(shape))*3, "hidden_nodes": 6,
            "frames": frames, "fps": 30, "rate": rate, "learning_rate": learning_rate,
            "inference_force_strength": force_strength, "tau": 1/3, "inference_steps": 8,
            "simulation_interval": 1/30, "damping_rate": damping_rate,
            "legacy_wave_dt": legacy_engine.dt, "sample_pixels": sample_pixels, "indices": indices.tolist(),
            "unit_precision": True, "weight_decay": 0,
            "legacy_inference_rate": .1, "legacy_canvas_rate": .01,
            "visual_decoder": models["waves"].visual_decoder,
        },
        "snapshots": snapshots, "legacy_snapshots": legacy_snapshots, "curves": curves,
        "black_gray_white": black_probe(), "xor": xor_probe(),
        "relu_black_gray_white": black_probe("relu"),
    }
    path = output / "capture.json"
    path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "capture": str(path), "black_gray_white": result["black_gray_white"],
        "relu_black_gray_white": result["relu_black_gray_white"],
        "xor": result["xor"], "final_forecast_rmse": {
            name: values[-1] for name, values in curves.items() if name != "time"
        },
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--inference-rate", type=float, default=.3)
    parser.add_argument("--inference-force-strength", type=float, default=1.)
    parser.add_argument("--learning-rate", type=float, default=.001)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not np.isfinite(args.inference_force_strength) or args.inference_force_strength < 0:
        parser.error("--inference-force-strength must be finite and non-negative")
    capture(args.output_dir, args.frames, args.inference_rate, args.inference_force_strength, args.learning_rate)
