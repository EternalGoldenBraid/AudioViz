import numpy as np
import pytest

from audioviz.physics.wave_propagator import ForceDrivenCanvasCPU


def test_force_accelerates_velocity_before_moving_position():
    canvas = ForceDrivenCanvasCPU((2, 2, 3), dx=1., speed=0., damping_rate=0.)
    force = np.full_like(canvas.position, 2.)
    canvas.step(force, .1, force_curvature=0.)
    np.testing.assert_allclose(canvas.velocity, .2)
    np.testing.assert_allclose(canvas.position, .02)
    canvas.step(force, .1, force_curvature=0.)
    np.testing.assert_allclose(canvas.velocity, .4)
    np.testing.assert_allclose(canvas.position, .06)
    assert canvas.time == pytest.approx(.2)


def test_damping_reduces_velocity_without_attenuating_uniform_position():
    canvas = ForceDrivenCanvasCPU((2, 2), dx=1., speed=1., damping_rate=2.)
    canvas.position.fill(1.)
    canvas.step(np.zeros_like(canvas.position), .1, force_curvature=0.)
    np.testing.assert_array_equal(canvas.position, 1.)
    canvas.velocity.fill(3.)
    canvas.step(np.zeros_like(canvas.position), .1, force_curvature=0.)
    np.testing.assert_allclose(canvas.velocity, 2.5)
    np.testing.assert_allclose(canvas.position, 1.25)


@pytest.mark.parametrize("boundary,corner_acceleration,wrapped_acceleration", [
    ("cyclic", -4., 1.),
    ("neumann", -2., 0.),
])
def test_intrinsic_waves_continue_without_inference_force(boundary, corner_acceleration, wrapped_acceleration):
    canvas = ForceDrivenCanvasCPU((3, 3), dx=1., speed=1., damping_rate=0., boundary_condition=boundary)
    canvas.position[0, 0] = 1.
    canvas.step(np.zeros_like(canvas.position), .1, force_curvature=0.)
    assert canvas.velocity[0, 0] == pytest.approx(.1 * corner_acceleration)
    assert canvas.velocity[0, 2] == pytest.approx(.1 * wrapped_acceleration)
    assert canvas.velocity[0, 1] == pytest.approx(.1)
    assert canvas.position.sum() == pytest.approx(1.)


def test_gradient_force_and_damping_settle_a_displaced_spring():
    canvas = ForceDrivenCanvasCPU((2, 2), dx=1., speed=0., damping_rate=1.)
    canvas.position.fill(1.)
    for _ in range(100):
        canvas.step(-4 * canvas.position, .05, force_curvature=4.)
    energy = .5 * canvas.velocity**2 + 2 * canvas.position**2
    assert np.max(energy) < .02


def test_step_bound_accounts_for_extra_restoring_stiffness():
    canvas = ForceDrivenCanvasCPU((2, 2), dx=1., speed=1., damping_rate=1.)
    assert canvas.max_step(4.) < .6 < canvas.max_step(0.)
    with pytest.raises(ValueError, match="combined wave and force"):
        canvas.step(np.zeros_like(canvas.position), .6, force_curvature=4.)
    np.testing.assert_array_equal(canvas.position, 0.)
    assert canvas.time == 0.
    with pytest.raises(FloatingPointError, match="force became non-finite"):
        canvas.step(np.full_like(canvas.position, np.nan), .1, force_curvature=0.)
