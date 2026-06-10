import time

import numpy as np
import pytest

from audioviz.engine import RippleEngine


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "small-interactive",
            "resolution": (64, 64),
            "n_sources": 2,
            "n_frequencies": 4,
            "min_hz": 60.0,
        },
        {
            "name": "medium-interactive",
            "resolution": (128, 128),
            "n_sources": 4,
            "n_frequencies": 8,
            "min_hz": 15.0,
        },
    ],
    ids=lambda case: case["name"],
)
def test_cpu_engine_update_frequency_matrix(case):
    hz = _measure_update_hz(
        resolution=case["resolution"],
        n_sources=case["n_sources"],
        n_frequencies=case["n_frequencies"],
    )
    print(f"{case['name']}: {hz:.1f} Hz")

    assert hz >= case["min_hz"], (
        f"{case['name']} processed at {hz:.1f} Hz; "
        f"expected at least {case['min_hz']:.1f} Hz"
    )


def _measure_update_hz(
    *,
    resolution: tuple[int, int],
    n_sources: int,
    n_frequencies: int,
) -> float:
    engine = RippleEngine(
        resolution=resolution,
        plane_size_m=(1.0, 1.0),
        n_sources=n_sources,
        speed=10.0,
        damping=0.99,
        amplitude=1.0,
        use_gpu=False,
    )
    frequencies = np.full(
        (n_sources, n_frequencies),
        5.0,
        dtype=np.float32,
    )

    for _ in range(3):
        engine.step(frequencies)

    n_steps = 10
    start = time.perf_counter()
    for _ in range(n_steps):
        engine.step(frequencies)
    elapsed = time.perf_counter() - start

    return n_steps / elapsed
