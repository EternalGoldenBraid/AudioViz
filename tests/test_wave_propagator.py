import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.physics import BoundaryCondition, WavePropagatorCPU


def test_wave_propagator_cyclic_boundary_wraps_impulse():
    propagator = WavePropagatorCPU(
        shape=(3, 3),
        dx=1.0,
        dt=1.0,
        speed=1.0,
        damping=1.0,
        boundary_condition=BoundaryCondition.CYCLIC,
    )
    propagator.Z[1, 0] = 1.0

    propagator.step()

    assert propagator.Z[1, 2] > 0.0


def test_wave_propagator_neumann_boundary_does_not_wrap_impulse():
    propagator = WavePropagatorCPU(
        shape=(3, 3),
        dx=1.0,
        dt=1.0,
        speed=1.0,
        damping=1.0,
        boundary_condition=BoundaryCondition.NEUMANN,
    )
    propagator.Z[1, 0] = 1.0

    propagator.step()

    assert propagator.Z[1, 2] == pytest.approx(0.0)


def test_ripple_engine_coerces_boundary_condition_and_passes_it_through():
    engine = RippleEngine(
        resolution=(4, 4),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        boundary_condition="neumann",
    )

    assert engine.boundary_condition is BoundaryCondition.NEUMANN
    assert engine.propagator.boundary_condition is BoundaryCondition.NEUMANN


def test_ripple_engine_rejects_unknown_boundary_condition():
    with pytest.raises(ValueError, match="boundary_condition"):
        RippleEngine(
            resolution=(4, 4),
            plane_size_m=(1.0, 1.0),
            speed=1.0,
            damping=1.0,
            amplitude=1.0,
            use_gpu=False,
            boundary_condition="invalid",
        )


def test_neumann_boundary_uses_reduced_edge_degree():
    propagator = WavePropagatorCPU(
        shape=(3, 3),
        dx=1.0,
        dt=1.0,
        speed=1.0,
        damping=1.0,
        boundary_condition=BoundaryCondition.NEUMANN,
    )
    propagator.Z[:] = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )

    propagator.step()

    # Corner uses degree 2: (2 + 4) - 2 * 1 = 4
    assert propagator.Z[0, 0] == pytest.approx(6.0)
