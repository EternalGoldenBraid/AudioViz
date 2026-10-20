from dataclasses import replace

import numpy as np
import pytest

from audioviz.sources.camera import CameraFrameSource
from audioviz.sources.circular_dot import CircularDotCapture, CircularDotConfig, CircularDotSource


def test_circle_matches_analytic_positions_and_camera_receives_only_frames():
    config = CircularDotConfig(
        resolution=(25, 35), center=(17, 12), radius=8, dot_radius=1,
        frame_rate=4, angular_velocity=2 * np.pi, duration=1,
    )
    source = CircularDotSource(config)
    camera = CameraFrameSource(
        resolution=config.resolution, enabled=True, capture=CircularDotCapture(source),
    )
    for index in range(4):
        image = camera.excitation()
        expected = np.array((17, 12)) + 8 * np.array((
            np.cos(index * np.pi / 2), np.sin(index * np.pi / 2),
        ))
        np.testing.assert_allclose(source.latest.position, expected, atol=1e-12)
        assert source.latest.time == index / 4
        x, y = np.rint(expected).astype(int)
        np.testing.assert_array_equal(image[y, x], 1)
        assert image.shape == (25, 35, 3) and image.dtype == np.float32
        assert not source.latest.frame.flags.writeable
    assert camera.excitation() is None
    camera.close()


def test_seeded_brownian_increments_replay_and_zero_noise_is_exact():
    config = CircularDotConfig(frame_rate=8, duration=1, brownian_scale=2, seed=7)
    noisy, replay = CircularDotSource(config), CircularDotSource(config)
    baseline = CircularDotSource(replace(config, brownian_scale=0))
    other_seed = CircularDotSource(replace(config, brownian_scale=0, seed=42))
    rng = np.random.default_rng(7)
    offset = np.zeros(2)
    for index in range(8):
        if index:
            offset += rng.normal(scale=2 / np.sqrt(8), size=2)
        sample, repeated = noisy.next_sample(), replay.next_sample()
        clean, clean_other = baseline.next_sample(), other_seed.next_sample()
        np.testing.assert_allclose(sample.offset, offset)
        np.testing.assert_allclose(np.array(sample.position) - clean.position, offset)
        np.testing.assert_array_equal(sample.frame, repeated.frame)
        np.testing.assert_array_equal(clean.frame, clean_other.frame)
        assert sample.position == repeated.position


def test_off_frame_trajectory_is_not_clamped_and_closed_capture_stops():
    source = CircularDotSource(CircularDotConfig(center=(-100, -100), radius=0))
    capture = CircularDotCapture(source)
    ok, frame = capture.read()
    assert ok and not frame.any()
    assert source.latest.position == (-100, -100)
    capture.release()
    assert capture.read() == (False, None)


@pytest.mark.parametrize("changes", [
    {"resolution": (0, 10)}, {"frame_rate": 0}, {"duration": np.nan},
    {"brownian_scale": -1}, {"center": (np.inf, 0)}, {"seed": -1},
])
def test_stimulus_rejects_invalid_configuration(changes):
    with pytest.raises(ValueError):
        CircularDotConfig(**changes)
