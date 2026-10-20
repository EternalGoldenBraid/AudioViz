import numpy as np
import pytest

from audioviz.engine import RippleEngine
from audioviz.readouts import IdentityRGBReadout


def _build_engine() -> RippleEngine:
    return RippleEngine(
        resolution=(2, 2),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )


def _rgb_field(values: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(values, dtype=np.float32)[..., None], 3, axis=2)


def _prepare_stationary_prior(
    engine: RippleEngine,
    field: np.ndarray,
) -> np.ndarray:
    engine.Z[:] = field
    engine.Z_old[:] = field
    engine.propagator.Z[:] = field
    engine.propagator.Z_old[:] = field
    wave_scale = engine.propagator.c2_dt2
    engine.propagator.c2_dt2 = 0.0
    try:
        engine.propagate()
    finally:
        engine.propagator.c2_dt2 = wave_scale
    return engine.get_prior_numpy().copy()


def test_identity_readout_reads_prior_without_graph_aggregation():
    engine = _build_engine()
    prior = _prepare_stationary_prior(
        engine,
        _rgb_field([[1.0, 2.0], [3.0, 4.0]]),
    )
    engine.prediction_horizontal_edge_weights = np.array(
        [[1.5], [2.0]],
        dtype=np.float32,
    )
    engine.prediction_vertical_edge_weights = np.array(
        [[1.25, 1.75]],
        dtype=np.float32,
    )

    prediction = IdentityRGBReadout(
        canvas_shape=engine.canvas_shape,
    ).predict(engine.get_prior_numpy())

    np.testing.assert_allclose(prediction, prior, rtol=1e-6, atol=1e-6)


def test_ripple_engine_requires_propagation_before_reading_prior():
    engine = _build_engine()

    with np.testing.assert_raises_regex(
        RuntimeError,
        "propagate must be called",
    ):
        engine.get_prior_numpy()


def test_ripple_engine_correction_preserves_prior_and_wave_velocity():
    engine = _build_engine()
    engine.propagate()
    prior = engine.get_prior_numpy().copy()
    velocity_before = engine.Z - engine.Z_old
    correction = np.full(engine.canvas_shape, 0.25, dtype=np.float32)
    engine.amplitude = 100.0

    posterior = engine.apply_observation_correction(correction)

    np.testing.assert_allclose(posterior, prior + correction)
    np.testing.assert_allclose(engine.Z - engine.Z_old, velocity_before)
    np.testing.assert_array_equal(engine.get_prior_numpy(), prior)
    np.testing.assert_array_equal(engine.propagator.Z, engine.Z)
    np.testing.assert_array_equal(engine.propagator.Z_old, engine.Z_old)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "opengl"))
@pytest.mark.parametrize("channels", (1, 3))
def test_stationary_root_keeps_corrections_and_drives_and_restarts_waves_without_old_velocity(
    qapp, request, backend, channels,
):
    if backend == "cupy":
        request.getfixturevalue("cupy_backend")
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native OpenGL context required")
    engine = RippleEngine(
        resolution=(9, 11), plane_size_m=(1, 1), speed=1, damping=.9,
        amplitude=.5, use_shader=backend == "opengl", use_gpu=backend == "cupy",
        canvas_channels=channels,
    )

    def field():
        return engine.get_field_numpy().copy()

    try:
        drive = np.zeros(engine.canvas_shape, dtype=np.float32)
        drive[4, 5] = np.array((1, -.5, .25))[:channels]
        engine.propagate(drive)
        held = field()
        engine.set_wave_propagation(False)
        before = engine.time
        engine.propagate()
        assert engine.time == before+engine.dt
        np.testing.assert_array_equal(field(), held)
        prior = engine.get_prior_numpy().copy()
        correction = np.full_like(held, .125)
        if backend == "opengl":
            from audioviz.utils.opengl_compute import GLArray
            native = GLArray(engine.propagator.compute, correction.shape, correction)
        else:
            native = engine.backend.asarray(correction)
        engine.apply_observation_correction(native)
        np.testing.assert_array_equal(field(), held+correction)
        np.testing.assert_array_equal(engine.get_prior_numpy(), prior)
        engine.propagate(drive)
        held = held+correction+.5*drive
        np.testing.assert_array_equal(field(), held)
        engine.propagate()
        np.testing.assert_array_equal(field(), held)
        np.testing.assert_array_equal(engine.get_prior_numpy(), held)
        laplacian = sum(
            np.roll(held, direction, axis=axis) for axis in (0, 1) for direction in (-1, 1)
        )-4*held
        expected = (held+engine.propagator.c2_dt2*laplacian)*engine.damping
        engine.set_wave_propagation(True)
        engine.propagate()
        np.testing.assert_allclose(field(), expected, atol=2e-7, rtol=2e-6)
        engine.set_wave_propagation(False)
        engine.propagate()
        stopped = field()
        engine.propagate()
        np.testing.assert_array_equal(field(), stopped)
        engine.reset()
        assert not engine.wave_propagation_enabled and engine.time == 0
        engine.propagate()
        np.testing.assert_array_equal(field(), 0)
    finally:
        if backend == "opengl":
            engine.propagator.close()


def test_ripple_engine_normalizes_prediction_conductances_to_stable_degree_budget():
    engine = RippleEngine(
        resolution=(3, 3),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )
    engine.prediction_horizontal_edge_weights = np.full((3, 2), 4.0, dtype=np.float32)
    engine.prediction_vertical_edge_weights = np.full((2, 3), 4.0, dtype=np.float32)

    horizontal = engine.prediction_horizontal_edge_weights
    vertical = engine.prediction_vertical_edge_weights
    degree = np.zeros((3, 3, 3), dtype=np.float32)
    degree[:, :-1] += horizontal
    degree[:, 1:] += horizontal
    degree[:-1, :] += vertical
    degree[1:, :] += vertical

    assert np.all(horizontal >= 0.0)
    assert np.all(vertical >= 0.0)
    assert np.max(degree) <= 4.0 + 1e-6


def test_ripple_engine_aggregates_patch_edge_strengths_from_symmetric_operator():
    engine = RippleEngine(
        resolution=(4, 4),
        plane_size_m=(1.0, 1.0),
        speed=1.0,
        damping=1.0,
        amplitude=1.0,
        use_gpu=False,
    )
    engine.prediction_vertical_edge_weights = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.8, 0.8, 0.8, 0.8],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    engine.prediction_horizontal_edge_weights = np.array(
        [
            [1.0, 1.05, 1.0],
            [1.0, 1.05, 1.0],
            [1.0, 1.025, 1.0],
            [1.0, 1.025, 1.0],
        ],
        dtype=np.float32,
    )

    segments, strengths = engine.get_prediction_coupling_overlay_edges(
        stride=2,
        threshold=0.0,
    )

    np.testing.assert_allclose(
        segments,
        np.array(
            [
                [[1.0, 1.0], [3.0, 1.0]],
                [[1.0, 3.0], [3.0, 3.0]],
                [[1.0, 1.0], [1.0, 3.0]],
                [[3.0, 1.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        strengths,
        np.array([1.05, 1.025, 0.8, 0.8], dtype=np.float32),
        atol=1e-6,
    )
