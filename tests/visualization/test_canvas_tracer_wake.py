import numpy as np
import pytest

from audioviz.visualization.canvas_tracer_wake import CanvasTracerWake


def test_canvas_components_use_backdrop_coordinates_and_are_extruded_through_depth():
    wake = CanvasTracerWake()
    field = np.array([
        [[1., 0., 0.], [0., 1., 0.]],
        [[0., 0., 1.], [-1., -1., -1.]],
    ])
    original = field.copy()
    wake.set_field(field)
    points = np.array([
        [-7.5, 14., 0.], [7.5, 14., 0.],
        [-7.5, -1., 0.], [7.5, -1., 0.],
        [-3.75, 10.25, -10.], [-3.75, 10.25, 10.], [8., 0., 0.],
    ])
    velocity = wake.velocity(points)
    np.testing.assert_allclose(velocity[:4], 3 * np.tanh(field.reshape(-1, 3)))
    np.testing.assert_allclose(velocity[4:6], np.tile(3 * np.tanh([.5, .125, .125]), (2, 1)))
    np.testing.assert_array_equal(velocity[6], 0)
    field.fill(0)
    np.testing.assert_allclose(wake.velocity(points)[:4], 3 * np.tanh(original.reshape(-1, 3)))
    assert not wake.field.flags.writeable
    wake.set_field(np.ones((256, 512, 3)))
    assert wake.field.shape == (32, 64, 3)
    wake.set_field(None)
    np.testing.assert_array_equal(wake.velocity(points), 0)
    wake.set_field(np.zeros((8, 12, 3)))
    np.testing.assert_array_equal(wake.velocity(points), 0)


def test_existing_wake_turns_with_the_field_and_continuous_release_is_anisotropic():
    wake = CanvasTracerWake()
    field = np.zeros((8, 12, 3))
    field[:, :, 0] = 1
    wake.set_field(field)
    positions = np.tile([-3.6, 2.8, .9], (24, 1))
    phase = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    positions += .25 * np.column_stack((np.cos(phase), np.sin(phase), np.cos(2 * phase)))
    source = positions.copy()
    for frame in range(20):
        wake.advance(frame / 10)
        wake.emit(positions, .5)
    active = wake.positions[wake.active].copy()
    assert np.ptp(active[:, 0]) > 4
    variance = np.var(active, axis=0)
    assert variance[0] > 10 * np.max(variance[1:])
    np.testing.assert_array_equal(active[:, 1:], np.tile(source[:, 1:], (20, 1)))
    assert len(active) == 480
    field[:, :, 0] = 0
    field[:, :, 1] = 1
    wake.set_field(field)
    wake.advance(1.95)
    np.testing.assert_allclose(wake.positions[:480, 0], active[:, 0])
    np.testing.assert_allclose(wake.positions[:480, 1] - active[:, 1], .15 * np.tanh(1))
    np.testing.assert_allclose(wake.trails[:480, 1], active)
    np.testing.assert_array_equal(field[:, :, 1], 1)
    np.testing.assert_array_equal(positions, source)


def test_zero_field_has_no_artificial_drift_and_lifetime_capacity_and_stalls_are_bounded():
    wake = CanvasTracerWake()
    positions = np.tile([-3.6, 2.8, .9], (24, 1))
    wake.advance(0)
    wake.emit(positions, 0)
    wake.advance(.1)
    np.testing.assert_array_equal(wake.positions[:24], positions)
    wake.set_field(np.full((8, 12, 3), 100.))
    before = wake.positions[:24].copy()
    wake.advance(1)
    np.testing.assert_allclose(wake.positions[:24] - before, .6)
    for _ in range(100):
        wake.emit(positions, 1)
    assert np.count_nonzero(wake.active) == wake.MAX_PARTICLES
    assert wake.positions.shape == (768, 3)
    assert wake.trails.shape == (768, 8, 3)
    wake.advance(3)
    assert not wake.active.any()
    wake.clear()
    wake.advance(0)
    with pytest.raises(ValueError, match="monotonic"):
        wake.advance(-1)
    with pytest.raises(ValueError, match="three-channel"):
        wake.set_field(np.zeros((8, 12)))
    with pytest.raises(ValueError, match="finite"):
        wake.set_field(np.full((8, 12, 3), np.nan))
