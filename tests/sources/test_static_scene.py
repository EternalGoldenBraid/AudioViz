import numpy as np
import pytest

from audioviz.sources.camera import CameraFrameSource
from audioviz.sources.static_scene import BrownianScene, geometric_scene, StaticSceneCapture


def test_static_scene_repeats_exact_rgb_evidence_without_caller_mutation_or_hardware():
    scene = geometric_scene((40, 60))
    np.testing.assert_array_equal(scene, geometric_scene((40, 60)))
    assert scene.dtype == np.uint8 and not scene.flags.writeable
    assert len(np.unique(scene.reshape(-1, 3), axis=0)) == 4
    supplied = scene.copy()
    capture = StaticSceneCapture(supplied)
    supplied[:] = 255
    for _ in range(3):
        ok, frame = capture.read()
        assert ok and not frame.flags.writeable and frame.flags.c_contiguous
        np.testing.assert_array_equal(frame, scene[..., ::-1])
        actual = CameraFrameSource.frame_to_excitation_grid(frame, resolution=(40, 60), gain=1)
        np.testing.assert_array_equal(actual, scene.astype(np.float32)/255)
    assert capture.read_count == 3
    capture.release()
    assert capture.read() == (False, None)
    assert capture.read_count == 3


def test_static_scene_rejects_invalid_dimensions_and_non_rgb_data():
    for resolution in ((0, 10), (True, 10), (1.5, 10)):
        with pytest.raises(ValueError, match="resolution"):
            geometric_scene(resolution)
    for image in (np.zeros((3, 4)), np.zeros((3, 4, 3)), np.zeros((0, 4, 3), np.uint8)):
        with pytest.raises(ValueError, match="uint8 RGB"):
            StaticSceneCapture(image)


def test_brownian_scene_uses_seeded_sqrt_dt_increments_and_exact_cropped_translation():
    world = BrownianScene((40, 60), seed=7)
    base = world.frame.copy()
    random = np.random.default_rng(7)
    expected_offset = np.zeros(2)
    first = None
    for dt, scale in ((.25, 4.), (.1, 8.), (.5, 0.), (.2, 10.)):
        expected_offset += random.normal(scale=scale*np.sqrt(dt), size=2)
        frame = world.advance(dt, scale=scale)
        np.testing.assert_array_equal(world.offset, expected_offset)
        dx, dy = world.pixel_offset
        assert (dx, dy) == tuple(int(np.rint(v)) for v in expected_offset)
        y, x = np.indices(base.shape[:2])
        valid = (x-dx >= 0) & (x-dx < 60) & (y-dy >= 0) & (y-dy < 40)
        expected = np.full_like(base, 20)
        expected[valid] = base[y[valid]-dy, x[valid]-dx]
        np.testing.assert_array_equal(frame, expected)
        assert not frame.flags.writeable
        if first is None:
            first = frame.copy()
    assert world.time == pytest.approx(1.05)
    assert not np.array_equal(frame, base)
    world.reset()
    assert world.time == 0 and world.pixel_offset == (0, 0)
    np.testing.assert_array_equal(world.frame, base)
    np.testing.assert_array_equal(world.advance(.25, scale=4.), first)


def test_scene_clock_and_capture_are_independent_and_offscreen_motion_does_not_wrap():
    world = BrownianScene((40, 60))
    capture = StaticSceneCapture(world.frame)
    world.advance(1., scale=1e5)
    np.testing.assert_array_equal(world.frame, np.full((40, 60, 3), 20, dtype=np.uint8))
    capture.set_frame(world.frame)
    for _ in range(3):
        ok, bgr = capture.read()
        assert ok and world.time == 1.
        np.testing.assert_array_equal(bgr[..., ::-1], world.frame)
    assert capture.read_count == 3
    offset = world.offset.copy()
    world.advance(.5, scale=0.)
    np.testing.assert_array_equal(world.offset, offset)
    assert world.time == 1.5 and capture.read_count == 3
    capture.release()
    with pytest.raises(RuntimeError, match="released"):
        capture.set_frame(world.frame)


def test_brownian_scene_rejects_invalid_seed_or_step_without_advancing_world():
    for seed in (-1, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            BrownianScene(seed=seed)
    world = BrownianScene()
    for dt, scale in ((0, 4), (True, 4), (np.nan, 4), (1, -1), (1, np.inf), (1, True)):
        with pytest.raises(ValueError):
            world.advance(dt, scale=scale)
    assert world.time == 0 and world.pixel_offset == (0, 0)
