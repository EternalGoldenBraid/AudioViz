import numpy as np

from audioviz.engine import RippleEngine
from audioviz.sources.pose import adjacency_from_edges


def test_pose_coupled_medium_updates_grid_state_without_exciting_pose_graph():
    engine = RippleEngine(
        resolution=(6, 6),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        pose_graph_stiffness=0.5,
    )
    adjacency = adjacency_from_edges(1, [])
    engine.set_source_positions(np.array([[2.5, 2.5]], dtype=np.float32))
    engine.update_pose_medium(
        positions=np.array([[2.5, 2.5]], dtype=np.float32),
        valid=np.array([True]),
        adjacency=adjacency,
    )

    engine.step_pose_medium(np.array([[1.0]], dtype=np.float32))

    assert np.count_nonzero(engine.get_field_numpy()) > 0
    assert engine.get_pose_medium_state()[0] == 0.0
    np.testing.assert_allclose(engine.get_pose_medium_positions(valid_only=True), [[2.5, 2.5]])


def test_pose_coupled_medium_ignores_invalid_pose_nodes():
    engine = RippleEngine(
        resolution=(6, 6),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )
    adjacency = adjacency_from_edges(2, [(0, 1)])
    engine.update_pose_medium(
        positions=np.array([[1.5, 1.5], [4.0, 4.0]], dtype=np.float32),
        valid=np.array([True, False]),
        adjacency=adjacency,
    )

    engine.step_pose_medium()

    np.testing.assert_allclose(engine.get_pose_medium_positions(valid_only=True), [[1.5, 1.5]])


def test_pose_coupled_medium_body_boundary_cuts_grid_edges():
    def build_engine() -> RippleEngine:
        engine = RippleEngine(
            resolution=(5, 5),
            plane_size_m=(1.0, 1.0),
            speed=1.0,
            damping=1.0,
            amplitude=1.0,
            use_gpu=False,
        )
        engine.update_pose_medium(
            positions=np.array([[4.0, 2.0]], dtype=np.float32),
            valid=np.array([False]),
            adjacency=adjacency_from_edges(1, []),
        )
        engine.Z[2, 1] = 1.0
        return engine

    open_engine = build_engine()
    masked_engine = build_engine()
    body_mask = np.zeros((5, 5), dtype=bool)
    body_mask[:, 2:] = True
    masked_engine.set_body_boundary_mask(body_mask)

    open_engine.step_pose_medium()
    masked_engine.step_pose_medium()

    assert open_engine.get_field_numpy()[2, 2] > 0.0
    assert masked_engine.get_field_numpy()[2, 2] == 0.0


def test_pose_coupled_medium_inner_boundary_dissipation_spares_interior_mask_nodes():
    engine = RippleEngine(
        resolution=(5, 5),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )
    engine.update_pose_medium(
        positions=np.array([[4.0, 2.0]], dtype=np.float32),
        valid=np.array([False]),
        adjacency=adjacency_from_edges(1, []),
    )
    body_mask = np.zeros((5, 5), dtype=bool)
    body_mask[1:4, 2:5] = True
    engine.set_body_boundary_mask(body_mask)
    engine.Z[2, 3] = 1.0
    engine.Z_old[2, 3] = 1.0

    engine.step_pose_medium()

    assert abs(engine.get_field_numpy()[2, 3]) > 0.0
    assert engine.Z_old[2, 3] == 1.0
    assert engine.get_field_numpy()[2, 2] == 0.0


def test_pose_coupled_medium_boundary_absorbs_more_than_hard_cut():
    def build_engine() -> RippleEngine:
        engine = RippleEngine(
            resolution=(5, 5),
            plane_size_m=(1.0, 1.0),
            speed=1.0,
            damping=1.0,
            amplitude=1.0,
            use_gpu=False,
        )
        engine.update_pose_medium(
            positions=np.array([[4.0, 2.0]], dtype=np.float32),
            valid=np.array([False]),
            adjacency=adjacency_from_edges(1, []),
        )
        return engine

    baseline = build_engine()
    masked = build_engine()
    body_mask = np.zeros((5, 5), dtype=bool)
    body_mask[:, 2:] = True
    masked.set_body_boundary_mask(body_mask)
    baseline.Z[2, 1] = 1.0
    masked.Z[2, 1] = 1.0

    for _ in range(3):
        baseline.step_pose_medium()
        masked.step_pose_medium()

    assert np.sum(np.abs(masked.get_field_numpy()[:, 2:])) < np.sum(
        np.abs(baseline.get_field_numpy()[:, 2:]),
    )


def test_pose_coupled_medium_boundary_transmission_allows_crossing_signal():
    def build_engine() -> RippleEngine:
        engine = RippleEngine(
            resolution=(5, 5),
            plane_size_m=(1.0, 1.0),
            speed=1.0,
            damping=1.0,
            amplitude=1.0,
            use_gpu=False,
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

    hard_cut = build_engine()
    transmissive = build_engine()
    transmissive.set_body_boundary_transmission(0.5)
    transmissive.set_body_boundary_dissipation(0.0)

    hard_cut.step_pose_medium()
    transmissive.step_pose_medium()

    assert hard_cut.get_field_numpy()[2, 2] == 0.0
    assert transmissive.get_field_numpy()[2, 2] > 0.0


def test_pose_coupled_medium_boundary_dissipation_reduces_same_transmission_energy():
    def build_engine() -> RippleEngine:
        engine = RippleEngine(
            resolution=(5, 5),
            plane_size_m=(1.0, 1.0),
            speed=1.0,
            damping=1.0,
            amplitude=1.0,
            use_gpu=False,
            body_boundary_transmission=0.5,
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

    low_loss = build_engine()
    high_loss = build_engine()
    low_loss.set_body_boundary_dissipation(0.0)
    high_loss.set_body_boundary_dissipation(1.0)

    for _ in range(3):
        low_loss.step_pose_medium()
        high_loss.step_pose_medium()

    assert np.sum(np.abs(high_loss.get_field_numpy())) < np.sum(
        np.abs(low_loss.get_field_numpy())
    )


def test_pose_coupled_medium_boundary_dissipation_targets_only_inner_mask_ring():
    engine = RippleEngine(
        resolution=(5, 5),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        body_boundary_transmission=0.0,
        body_boundary_dissipation=1.0,
    )
    engine.update_pose_medium(
        positions=np.array([[4.0, 2.0]], dtype=np.float32),
        valid=np.array([False]),
        adjacency=adjacency_from_edges(1, []),
    )
    body_mask = np.zeros((5, 5), dtype=bool)
    body_mask[1:4, 2:5] = True
    engine.set_body_boundary_mask(body_mask)
    engine.Z[2, 2] = 1.0
    engine.Z[2, 3] = 1.0
    engine.Z_old[2, 2] = 1.0
    engine.Z_old[2, 3] = 1.0

    engine.step_pose_medium()

    assert engine.get_field_numpy()[2, 2] == 0.0
    assert abs(engine.get_field_numpy()[2, 3]) > 0.0


def test_pose_coupled_medium_keeps_pose_graph_quiescent_under_segmentation():
    engine = RippleEngine(
        resolution=(6, 6),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        pose_graph_stiffness=0.5,
    )
    engine.set_source_positions(np.array([[2.5, 2.5]], dtype=np.float32))
    engine.update_pose_medium(
        positions=np.array([[2.5, 2.5]], dtype=np.float32),
        valid=np.array([True]),
        adjacency=adjacency_from_edges(1, []),
    )
    body_mask = np.zeros((6, 6), dtype=bool)
    body_mask[1:5, 1:5] = True
    engine.set_body_boundary_mask(body_mask)

    engine.step_pose_medium(np.array([[1.0]], dtype=np.float32))

    assert np.count_nonzero(engine.get_field_numpy()) > 0
    assert engine.get_pose_medium_state()[0] == 0.0
