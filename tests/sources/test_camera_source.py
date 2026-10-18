import numpy as np
import pytest

from audioviz.sources import CameraFrameSource


class _FailedCameraCapture:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return False

    def release(self):
        self.released = True


class _FakeCv2:
    def __init__(self):
        self.capture = _FailedCameraCapture()
        self.requested_device = None

    def VideoCapture(self, camera_index):
        self.requested_device = camera_index
        return self.capture


def test_camera_frame_source_maps_frame_to_excitation_grid():
    frame = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    grid = CameraFrameSource.frame_to_excitation_grid(
        frame,
        resolution=(2, 3),
        gain=2.0,
    )

    expected = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(grid, expected, atol=1e-6)


def test_camera_frame_source_releases_failed_capture():
    fake_cv2 = _FakeCv2()
    source = CameraFrameSource(
        resolution=(6, 8),
        camera_index=0,
        cv2_loader=lambda: fake_cv2,
    )

    with pytest.raises(RuntimeError, match="Failed to open camera source index 0"):
        source.ensure_open()

    assert fake_cv2.capture.released
    assert source.capture is None


def test_camera_frame_source_accepts_explicit_device_path():
    fake_cv2 = _FakeCv2()
    source = CameraFrameSource(
        resolution=(6, 8),
        camera_index="/dev/video0",
        cv2_loader=lambda: fake_cv2,
    )

    with pytest.raises(RuntimeError, match="/dev/video0"):
        source.ensure_open()

    assert fake_cv2.requested_device == "/dev/video0"


@pytest.mark.parametrize("kind", ("rgb", "rgba", "gray", "float"))
def test_gpu_camera_preprocessing_matches_cpu(cupy_backend, kind):
    cp = cupy_backend
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (9, 13, 4), dtype=np.uint8)
    if kind == "rgb":
        frame = frame[..., :3]
    elif kind == "gray":
        frame = frame[..., 0]
    elif kind == "float":
        frame = frame.astype(np.float32)
        frame[0, 0] = [np.nan, np.inf, -np.inf, 0]
    expected = CameraFrameSource.frame_to_excitation_grid(frame, resolution=(17, 25), gain=.7)
    actual = CameraFrameSource.frame_to_excitation_grid(
        frame, resolution=(17, 25), gain=.7, backend=cp,
    )
    assert isinstance(actual, cp.ndarray)
    assert actual.dtype == cp.float32
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-7)
