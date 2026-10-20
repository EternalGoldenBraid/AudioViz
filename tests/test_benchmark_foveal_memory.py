import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_foveal_memory.py"
spec = importlib.util.spec_from_file_location("benchmark_foveal_memory", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_circle_parameter_count_and_unique_storage_match_actual_arrays():
    assert [benchmark.disk_slots(r) for r in (1, 2, 8, 16, 32)] == [5, 13, 197, 797, 3209]
    hidden, visual, audio = 6, 591, 16
    expected = 8*(hidden*visual + (visual+audio)*hidden + hidden**2 + hidden + visual+audio)
    assert benchmark.parameter_bytes(197, hidden, audio) == expected
    base = np.ones((10, 20), dtype=np.float64)
    other = np.zeros(3, dtype=bool)
    arrays = dict(weights=base, aliases=[base, base[:, :2]], metadata=(other, other))
    assert benchmark.array_storage_bytes(arrays) == base.nbytes+other.nbytes
    for radius in (False, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            benchmark.disk_slots(radius)


@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_measured_graph_memory_includes_snapshots_and_peak_working_arrays(request, backend):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    row = benchmark.measure_case(
        radius=2, hidden_nodes=3, candidates=4, inference_steps=2, backend=backend,
    )
    assert row["status"] == "measured"
    assert row["parameter_bytes"] == 8*(3*39 + (39+16)*3 + 3**2 + 3 + 39+16)
    assert row["graph_array_bytes"] > 2*row["parameter_bytes"]
    assert set(row["host_phase_peak_bytes"]) == {"construct", "inputs", "update", "gaze"}
    assert row["host_peak_bytes"] == max(row["host_phase_peak_bytes"].values())
    assert row["process_peak_rss_bytes"] > 0
    if backend == "numpy":
        assert row["host_peak_bytes"] >= row["graph_array_bytes"]
        assert row["gpu_peak_bytes"] is None
    else:
        assert row["gpu_peak_bytes"] >= row["gpu_live_bytes"] >= row["graph_array_bytes"]
        assert row["gpu_reserved_bytes"] >= row["gpu_peak_bytes"]


def test_cli_measures_sweep_and_labels_above_limit_rows_without_allocating_images():
    limit = benchmark.SharedPredictiveCodingGraph.MAX_VISUAL_SLOTS
    over_radius = math.isqrt(limit)+1
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--radii", "1", str(over_radius), "--hidden-sizes", "2", "4",
         "--candidates", "2", "--inference-steps", "1", "--image-shape", "1080", "1920", "--json"],
        check=True, text=True, capture_output=True,
    )
    data = json.loads(process.stdout)
    assert data["image_buffer_bytes"] == 1080*1920*3*4
    assert data["image_buffer_allocated"] is False
    assert data["gaze_backend"] == "numpy" and data["input_dtype"] == "float32"
    assert data["core_limits"]["visual_slots"] == limit
    assert len(data["rows"]) == 4
    for row in data["rows"][:2]:
        assert row["slots"] == 5 and row["status"] == "measured"
        assert row["graph_array_bytes"] > row["parameter_bytes"] > 0
    for row in data["rows"][2:]:
        assert row["slots"] == benchmark.disk_slots(over_radius) > limit
        assert row["status"] == "parameter estimate only: core limit"
        assert "host_peak_bytes" not in row and "graph_array_bytes" not in row


def test_estimate_only_and_zero_gaze_candidates_have_explicit_output(capsys):
    benchmark.main([
        "--radii", "8", "--hidden-sizes", "6", "--estimate-only", "--json",
        "--no-learning", "--candidates", "0",
    ])
    data = json.loads(capsys.readouterr().out)
    assert not data["learning"] and data["candidates"] == 0
    assert data["rows"][0]["status"] == "parameter estimate only"
    row = benchmark.measure_case(radius=1, hidden_nodes=2, candidates=0, learning=False)
    assert "gaze" not in row["host_phase_peak_bytes"]


def test_worker_failure_is_reported_as_error_not_replaced_with_an_estimate(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(-9, "worker", stderr="allocation failed")

    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--radii", "1", "--hidden-sizes", "2"])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "radius=1, hidden=2 (worker exit -9)" in captured.err
    assert "allocation failed" in captured.err
