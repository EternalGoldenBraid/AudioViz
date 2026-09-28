import numpy as np

from audioviz.engine import RippleEngine


def _build_boundary_engine(
    *,
    transmission: float = 0.0,
    dissipation: float = 0.0,
) -> RippleEngine:
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
    body_mask = np.zeros((5, 5), dtype=bool)
    body_mask[:, 2:] = True
    engine.set_body_boundary_mask(body_mask)
    engine.Z[2, 1] = 1.0
    return engine


def test_pose_coupled_medium_boundary_transmission_allows_crossing_signal():
    hard_cut_engine = _build_boundary_engine()
    transmissive_engine = _build_boundary_engine(transmission=0.5)

    hard_cut_engine.step_without_excitation()
    transmissive_engine.step_without_excitation()

    assert np.all(hard_cut_engine.get_field_numpy()[2, 2] == 0.0)
    assert np.all(transmissive_engine.get_field_numpy()[2, 2] > 0.0)


def test_pose_coupled_medium_boundary_dissipation_reduces_same_transmission_energy():
    low_loss_engine = _build_boundary_engine(
        transmission=0.5,
        dissipation=0.0,
    )
    high_loss_engine = _build_boundary_engine(
        transmission=0.5,
        dissipation=1.0,
    )

    for _ in range(3):
        low_loss_engine.step_without_excitation()
        high_loss_engine.step_without_excitation()

    assert np.sum(np.abs(high_loss_engine.get_field_numpy())) < np.sum(
        np.abs(low_loss_engine.get_field_numpy())
    )
