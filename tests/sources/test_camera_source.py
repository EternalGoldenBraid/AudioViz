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
    CAP_V4L2 = 200

    def __init__(self):
        self.capture = _FailedCameraCapture()
        self.requested_device = None
        self.requested_backend = None

    def VideoCapture(self, camera_index, backend=None):
        self.requested_device = camera_index
        self.requested_backend = backend
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


@pytest.mark.parametrize("device,backend", (
    ("/dev/video0", _FakeCv2.CAP_V4L2),
    ("/dev/video12", _FakeCv2.CAP_V4L2),
    (0, None),
    ("clip.mp4", None),
))
def test_camera_frame_source_selects_device_backend_without_forcing_video_files(device, backend):
    fake_cv2 = _FakeCv2()
    source = CameraFrameSource(
        resolution=(6, 8),
        camera_index=device,
        cv2_loader=lambda: fake_cv2,
    )

    with pytest.raises(RuntimeError, match="Failed to open camera source"):
        source.ensure_open()

    assert fake_cv2.requested_device == device
    assert fake_cv2.requested_backend == backend


def test_camera_failure_latches_until_explicit_retry_and_recovers(monkeypatch):
    from types import SimpleNamespace
    from audioviz.sources import camera as module

    warnings = []
    monkeypatch.setattr(module.logger, "warning", warnings.append)

    class Capture:
        def __init__(self, *, opened=True):
            self.opened = opened
            self.reads = 0
            self.released = False

        def isOpened(self):
            return self.opened

        def read(self):
            self.reads += 1
            return (True, np.full((2, 3, 3), 255, dtype=np.uint8)) if self.reads == 1 else (False, None)

        def release(self):
            self.released = True

    failed_open, disconnected, recovered = Capture(opened=False), Capture(), Capture()
    available = iter((failed_open, disconnected, recovered))
    opened = []

    def open_capture(index, backend):
        assert backend == _FakeCv2.CAP_V4L2
        opened.append(index)
        return next(available)

    source = CameraFrameSource(
        resolution=(2, 3), camera_index="/dev/video2", enabled=True,
        cv2_loader=lambda: SimpleNamespace(VideoCapture=open_capture, CAP_V4L2=_FakeCv2.CAP_V4L2),
    )
    assert source.excitation() is None
    assert failed_open.released and "cannot open" in source.status
    for _ in range(5):
        assert source.excitation() is None
    assert opened == ["/dev/video2"] and len(warnings) == 1

    source.set_enabled(False)
    assert source.status == "disabled"
    source.set_enabled(True)
    np.testing.assert_array_equal(source.excitation(), np.ones((2, 3, 3)))
    assert source.status == "receiving"
    assert source.excitation() is None
    assert disconnected.released and "off/on to retry" in source.status
    for _ in range(5):
        assert source.excitation() is None
    assert disconnected.reads == 2 and len(warnings) == 2

    source.set_enabled(False)
    source.set_enabled(True)
    np.testing.assert_array_equal(source.excitation(), np.ones((2, 3, 3)))
    assert recovered.reads == 1 and source.error is None
    source.close()
    assert recovered.released


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
