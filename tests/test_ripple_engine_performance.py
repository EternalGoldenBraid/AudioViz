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
            "min_hz": 60.0,
        },
        {
            "name": "medium-interactive",
            "resolution": (128, 128),
            "min_hz": 15.0,
        },
    ],
    ids=lambda case: case["name"],
)
def test_cpu_engine_update_grid_excitation(case):
    hz = _measure_update_hz(
        resolution=case["resolution"],
    )
    print(f"{case['name']}: {hz:.1f} Hz")

    assert hz >= case["min_hz"], (
        f"{case['name']} processed at {hz:.1f} Hz; "
        f"expected at least {case['min_hz']:.1f} Hz"
    )


def _measure_update_hz(
    *,
    resolution: tuple[int, int],
) -> float:
    engine = RippleEngine(
        resolution=resolution,
        plane_size_m=(1.0, 1.0),
        speed=10.0,
        damping=0.99,
        amplitude=1.0,
        use_gpu=False,
    )
    excitation = np.zeros(resolution, dtype=np.float32)
    excitation[resolution[0] // 2, resolution[1] // 2] = 1.0

    for _ in range(3):
        engine.step_grid_excitation(excitation)

    n_steps = 10
    start = time.perf_counter()
    for _ in range(n_steps):
        engine.step_grid_excitation(excitation)
    elapsed = time.perf_counter() - start

    return n_steps / elapsed
