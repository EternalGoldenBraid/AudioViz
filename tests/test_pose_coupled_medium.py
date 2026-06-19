import numpy as np

from audioviz.engine import RippleEngine
from audioviz.sources.pose import adjacency_from_edges


def test_pose_coupled_medium_updates_grid_and_pose_states():
    engine = RippleEngine(
        resolution=(6, 6),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        pose_graph_stiffness=0.5,
        pose_coupling_strength=1.0,
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
    assert engine.get_pose_medium_state()[0] > 0.0
    np.testing.assert_allclose(engine.get_pose_medium_positions(valid_only=True), [[2.5, 2.5]])


def test_pose_coupled_medium_ignores_invalid_pose_nodes():
    engine = RippleEngine(
        resolution=(6, 6),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        pose_coupling_strength=1.0,
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
            pose_coupling_strength=0.0,
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


def test_pose_coupled_medium_body_boundary_absorbs_only_at_boundary_edges():
    engine = RippleEngine(
        resolution=(5, 5),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        pose_coupling_strength=0.0,
    )
    engine.update_pose_medium(
        positions=np.array([[4.0, 2.0]], dtype=np.float32),
        valid=np.array([False]),
        adjacency=adjacency_from_edges(1, []),
    )
    body_mask = np.zeros((5, 5), dtype=bool)
    body_mask[1:4, 2:5] = True
    engine.set_body_boundary_mask(body_mask)
    engine.Z[2, 4] = 1.0
    engine.Z_old[2, 4] = 1.0

    engine.step_pose_medium()

    assert abs(engine.get_field_numpy()[2, 4]) > 0.0
    assert engine.Z_old[2, 4] == 1.0
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
            pose_coupling_strength=0.0,
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


def test_pose_coupled_medium_segmentation_disables_overlap_coupling_without_bridge():
    engine = RippleEngine(
        resolution=(6, 6),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
        pose_graph_stiffness=0.5,
        pose_coupling_strength=1.0,
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
