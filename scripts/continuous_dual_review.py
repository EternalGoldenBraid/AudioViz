"""Record node motion and directed messages from one continuous inference run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audioviz.engine import RippleEngine
from audioviz.physics.wave_propagator import ForceDrivenCanvasCPU
from audioviz.readouts.foveal_routing import WorldAnchoredFovea

from continuous_inference_review import advance_canvas_force, camera_frame, plain_graph


def spatial_groups(shape):
    groups = []
    rows = np.array_split(np.arange(shape[0]), min(3, shape[0]))
    columns = np.array_split(np.arange(shape[1]), min(3, shape[1]))
    for row, ys in enumerate(rows):
        for column, xs in enumerate(columns):
            indices = (
                3 * (ys[:, None, None] * shape[1] + xs[None, :, None])
                + np.arange(3)[None, None, :]
            ).ravel()
            groups.append({
                "indices": indices, "row": row, "column": column,
                "x": (column - (len(columns)-1)/2) * 1.15,
                "z": (row - (len(rows)-1)/2) * .9,
            })
    return groups


def topology(groups, hidden):
    nodes = []
    for layer, y in (("canvas", -1.9), ("sensory", 1.3), ("evidence", 2.8)):
        for index, group in enumerate(groups):
            nodes.append({
                "id": f"{layer}-{index}", "layer": layer,
                "label": f"{layer.title()} region {group['row']+1},{group['column']+1}",
                "position": [group["x"], y, group["z"]],
                "coordinates": int(group["indices"].size),
            })
    count = len(groups)
    for index in range(hidden):
        angle = 2*np.pi*index/hidden
        nodes.append({
            "id": f"hidden-{index}", "layer": "hidden", "label": f"Hidden {index}",
            "position": [1.1*np.cos(angle), -.3 + .35*np.sin(2*angle), .8*np.sin(angle)],
            "coordinates": 1,
        })
    edges = []

    def pair(source, target, family):
        edges.extend([
            {"source": source, "target": target, "family": family, "kind": "prediction"},
            {"source": target, "target": source, "family": family, "kind": "feedback"},
        ])

    for region in range(count):
        for index in range(hidden):
            pair(region, 3*count+index, "canvas")
    for source in range(hidden):
        for target in range(hidden):
            if source != target:
                pair(3*count+source, 3*count+target, "recurrent")
    for region in range(count):
        for index in range(hidden):
            pair(3*count+index, count+region, "sensory")
    for region in range(count):
        pair(count+region, 2*count+region, "observation")
    return nodes, edges


def sample_boundary(graph, canvas, image, groups):
    """Evaluate messages together at the completed interval, with current weights."""
    h = graph.mean[graph.hidden_slice].copy()
    z = graph.sensory_potential.copy()
    c = canvas.ravel().copy()
    a = np.tanh(h)
    slope = 1-a*a
    mask = graph.aperture_mask
    root = np.where(mask, c, 0.)
    wc, wh, ws = (getattr(graph, name) for name in graph._PARAMETERS)
    eh = h - wc @ root - wh @ a
    ez = z - ws @ a
    observed = graph.clamped[:graph.visual_nodes]
    y = np.where(observed, image.ravel(), 0.)
    ey = np.where(observed, y-graph._decode_sensory(z), 0.)
    messages = []

    def pair(prediction, feedback):
        messages.extend([
            np.asarray(prediction).reshape(-1), np.asarray(feedback).reshape(-1),
        ])

    for group in groups:
        indices = group["indices"]
        for index in range(graph.hidden_nodes):
            pair(
                wc[index, indices] @ root[indices],
                mask[indices] * wc[index, indices] * eh[index],
            )
    for source in range(graph.hidden_nodes):
        for target in range(graph.hidden_nodes):
            if source != target:
                pair(wh[target, source]*a[source], wh[target, source]*eh[target]*slope[source])
    for group in groups:
        indices = group["indices"]
        for index in range(graph.hidden_nodes):
            pair(
                ws[indices, index]*a[index],
                (ws[indices, index] @ ez[indices])*slope[index],
            )
    for group in groups:
        indices = group["indices"]
        pair(graph._decode_sensory(z)[indices], ey[indices]*graph._sensory_slope(z)[indices])
    states = [
        value[group["indices"]].copy()
        for value in (c, z, y) for group in groups
    ] + [h[index:index+1].copy() for index in range(graph.hidden_nodes)]
    terms = graph._terms_for(h, z, c, mask, graph.clamped, graph.evidence)
    gradients = [
        value[group["indices"]].copy()
        for value in (terms[5], terms[4], ey) for group in groups
    ] + [terms[3][index:index+1].copy() for index in range(graph.hidden_nodes)]
    fields = {
        "state-canvas": c, "state-sensory": z, "state-evidence": y,
        "message-canvas-feedback": mask*(wc.T @ eh),
        "message-sensory-prediction": ws @ a,
        "message-sensory-feedback": ey*graph._sensory_slope(z),
        "message-evidence-prediction": graph._decode_sensory(z),
    }
    for index in range(graph.hidden_nodes):
        fields[f"message-canvas-feedback-hidden-{index}"] = mask*wc[index]*eh[index]
        fields[f"message-sensory-prediction-hidden-{index}"] = ws[:, index]*a[index]
    return {
        "states": states, "messages": messages, "gradients": gradients,
        "energy": [terms[0], terms[1], terms[2]], "fields": fields, "fovea": None,
    }


def rms(value):
    return float(np.sqrt(np.mean(value*value)))


def describe(value, previous):
    change = value-previous
    return [float(np.mean(value)), rms(value), float(np.mean(change)), rms(change)]


def record_boundary(current, previous, time):
    return {
        "time": time,
        "nodes": [describe(value, old) for value, old in zip(current["states"], previous["states"])],
        "messages": [
            describe(value, old) for value, old in zip(current["messages"], previous["messages"])
        ],
        "gradients": [rms(value) for value in current["gradients"]],
        "energy": current["energy"],
        "fovea": current["fovea"],
    }


def raster_gaze(frame, shape, radius, fixation_frames=6):
    axes = [
        np.linspace(0, size-1, int(np.ceil((size-1)/radius))+1)
        for size in shape
    ]
    rows, columns = axes
    fixation = (frame//fixation_frames) % (len(rows)*len(columns))
    row, column = divmod(fixation, len(columns))
    return float(columns[column]/max(1, shape[1]-1)), float(rows[row]/max(1, shape[0]-1))


def trajectory(shape, steps):
    graph = plain_graph(shape, inference_rate=.3, learning_enabled=True, learning_rate=.001)
    reference = RippleEngine(
        resolution=shape, plane_size_m=(50, 50), speed=340, damping=.988,
        use_gpu=False, use_shader=False, boundary_condition="neumann",
    )
    canvas = ForceDrivenCanvasCPU(
        (*shape, 3), dx=reference.grid_spacing, speed=reference.speed,
        damping_rate=-np.log(reference.damping)/reference.dt, boundary_condition="neumann",
    )
    groups = spatial_groups(shape)
    fovea = WorldAnchoredFovea(shape, radius=8)
    previous = sample_boundary(graph, canvas.position, np.zeros_like(canvas.position), groups)
    yield 0, previous, previous, np.zeros_like(canvas.position)
    for frame in range(steps):
        image = camera_frame(frame)
        fovea.set_gaze(raster_gaze(frame, shape, fovea.radius))
        observed = fovea.observation_mask
        graph.forecast(canvas.position, 1/30)
        graph.condition(image, visual_mask=observed)
        advance_canvas_force(graph, canvas, 1/30, 1., reference.dt)
        current = sample_boundary(graph, canvas.position, image, groups)
        current["fovea"] = {
            "center": fovea.center.tolist(), "radius": fovea.radius,
            "pixels": int(observed.sum()),
        }
        yield frame+1, current, previous, image
        previous = current


def field_magnitude(value, shape):
    return np.sqrt(np.mean(value.reshape(*shape, 3)**2, axis=2))


def field_image(change, scale, shape):
    values = change.reshape(*shape, 3)
    mean = np.mean(values, axis=2)
    colors = np.empty((*shape, 4), dtype=np.uint8)
    colors[..., :3] = np.where(
        (mean < 0)[..., None], [72, 211, 236],
        np.where((mean > 0)[..., None], [255, 158, 119], [185, 196, 215]),
    )
    magnitude = field_magnitude(change, shape)
    colors[..., 3] = 0 if scale == 0 else np.rint(255*np.sqrt(magnitude/scale)).astype(np.uint8)
    return Image.fromarray(colors)


def capture(output, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    shape = (48, 96)
    groups = spatial_groups(shape)
    records = []
    columns = 20
    size = (columns*shape[1], ((steps+columns)//columns)*shape[0])
    atlas = Image.new("RGB", size)
    prediction_atlas = Image.new("RGB", size)
    maxima = {}
    for frame, current, previous, image in trajectory(shape, steps):
        records.append(record_boundary(current, previous, frame/30))
        atlas.paste(
            Image.fromarray(np.rint(image*255).astype(np.uint8)),
            ((frame%columns)*shape[1], (frame//columns)*shape[0]),
        )
        prediction = current["fields"]["message-evidence-prediction"].reshape(*shape, 3)
        prediction_atlas.paste(
            Image.fromarray(np.rint(np.clip(prediction, 0, 1)*255).astype(np.uint8)),
            ((frame%columns)*shape[1], (frame//columns)*shape[0]),
        )
        for key, value in current["fields"].items():
            change = value-previous["fields"][key]
            maxima[key] = max(maxima.get(key, 0.), float(np.max(field_magnitude(change, shape))))
    final_fields = current["fields"]
    field_atlases = {key: Image.new("RGBA", size) for key in maxima}
    # Replay the same deterministic trajectory to encode fixed recording-wide scales.
    for frame, current, previous, _ in trajectory(shape, steps):
        for key, value in current["fields"].items():
            field_atlases[key].paste(
                field_image(value-previous["fields"][key], maxima[key], shape),
                ((frame%columns)*shape[1], (frame//columns)*shape[0]),
            )
    if not all(np.array_equal(value, final_fields[key]) for key, value in current["fields"].items()):
        raise RuntimeError("native field replay differs from the recorded trajectory")
    hidden_nodes = len(current["states"])-3*len(groups)
    nodes, edges = topology(groups, hidden_nodes)
    sources = [
        Path(__file__), Path(__file__).with_name("continuous_inference_review.py"),
        Path(__file__).resolve().parents[1]/"src/audioviz/readouts/continuous_scene_predictive_coding.py",
        Path(__file__).resolve().parents[1]/"src/audioviz/physics/wave_propagator.py",
        Path(__file__).resolve().parents[1]/"src/audioviz/readouts/foveal_routing.py",
    ]
    result = {
        "schema": "continuous-dual-v1",
        "settings": {
            "shape": shape, "hidden_nodes": hidden_nodes, "fps": 30,
            "duration": steps/30, "inference_rate": .3, "learning_rate": .001,
            "inference_force_strength": 1.,
            "input_atlas": {"columns": columns, "frames": steps+1},
            "prediction_atlas": {"columns": columns, "frames": steps+1},
            "fovea": {"radius": 8, "fixation_frames": 6, "policy": "raster"},
            "native_fields": maxima,
            "boundary": "After neural inference, learning and mechanical advance; current weights.",
        },
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "nodes": nodes, "edges": edges, "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary_atlas = output/"dual-input.pending.png"
    atlas.save(temporary_atlas)
    temporary_atlas.replace(output/"dual-input.png")
    temporary_prediction = output/"dual-prediction.pending.png"
    prediction_atlas.save(temporary_prediction)
    temporary_prediction.replace(output/"dual-prediction.png")
    field_directory = output/"native-fields"
    field_directory.mkdir(exist_ok=True)
    for key, image in field_atlases.items():
        temporary_field = field_directory/f"{key}.pending.png"
        image.save(temporary_field)
        temporary_field.replace(field_directory/f"{key}.png")
    temporary = output/"dual-capture.pending.json"
    temporary.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(output/"dual-capture.json")
    print(f"Recorded {steps} intervals, {len(nodes)} nodes and {len(edges)} directed streams.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", default=600, type=int)
    args = parser.parse_args()
    capture(args.output_dir, args.steps)
