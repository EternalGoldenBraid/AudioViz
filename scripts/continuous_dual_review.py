"""Record node motion and directed messages from one continuous inference run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audioviz.engine import RippleEngine
from audioviz.physics.wave_propagator import ForceDrivenCanvasCPU

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
    y = image.ravel()
    ey = np.where(graph.clamped[:graph.visual_nodes], y-graph._decode_sensory(z), 0.)
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
    return {
        "states": states, "messages": messages, "gradients": gradients,
        "energy": [terms[0], terms[1], terms[2]],
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
    }


def capture(output, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    shape = (48, 96)
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
    nodes, edges = topology(groups, graph.hidden_nodes)
    previous = sample_boundary(graph, canvas.position, np.zeros_like(canvas.position), groups)
    records = [record_boundary(previous, previous, 0.)]
    for frame in range(steps):
        image = camera_frame(frame)
        graph.forecast(canvas.position, 1/30)
        graph.condition(image)
        advance_canvas_force(graph, canvas, 1/30, 1., reference.dt)
        current = sample_boundary(graph, canvas.position, image, groups)
        records.append(record_boundary(current, previous, (frame+1)/30))
        previous = current
    sources = [
        Path(__file__), Path(__file__).with_name("continuous_inference_review.py"),
        Path(__file__).resolve().parents[1]/"src/audioviz/readouts/continuous_scene_predictive_coding.py",
        Path(__file__).resolve().parents[1]/"src/audioviz/physics/wave_propagator.py",
    ]
    result = {
        "schema": "continuous-dual-v1",
        "settings": {
            "shape": shape, "hidden_nodes": graph.hidden_nodes, "fps": 30,
            "duration": steps/30, "inference_rate": .3, "learning_rate": .001,
            "inference_force_strength": 1.,
            "boundary": "After neural inference, learning and mechanical advance; current weights.",
        },
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "nodes": nodes, "edges": edges, "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
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
