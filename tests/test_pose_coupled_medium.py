import numpy as np

from audioviz.engine import RippleEngine
from audioviz.sources.pose import adjacency_from_edges


def _build_boundary_engine(*, transmission: float = 0.0, dissipation: float = 0.0) -> RippleEngine:
    engine = RippleEngine(
        resolution=(5, 5),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        body_boundary_transmission=transmission,
        body_boundary_dissipation=dissipation,
    )
    engine.update_pose_medium(
        positions=np.array([[4.0, 2.0]], dtype=np.float32),
        valid=np.array([False]),
        adjacency=adjacency_from_edges(1, []),
    )
    body_mask = np.zeros((5, 5), dtype=bool)
    body_mask[:, 2:] = True
    engine.set_body_boundary_mask(body_mask)
    engine.Z[2, 1] = 1.0
    return engine


def test_pose_coupled_medium_boundary_transmission_allows_crossing_signal():
    hard_cut = _build_boundary_engine()
    transmissive = _build_boundary_engine(transmission=0.5)

    hard_cut.step_pose_medium()
    transmissive.step_pose_medium()

    assert hard_cut.get_field_numpy()[2, 2] == 0.0
    assert transmissive.get_field_numpy()[2, 2] > 0.0


def test_pose_coupled_medium_boundary_dissipation_reduces_same_transmission_energy():
    low_loss = _build_boundary_engine(transmission=0.5, dissipation=0.0)
    high_loss = _build_boundary_engine(transmission=0.5, dissipation=1.0)

    for _ in range(3):
        low_loss.step_pose_medium()
        high_loss.step_pose_medium()

    assert np.sum(np.abs(high_loss.get_field_numpy())) < np.sum(np.abs(low_loss.get_field_numpy()))
