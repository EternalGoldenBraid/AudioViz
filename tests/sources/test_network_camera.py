import queue
import threading
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from audioviz.sources import network_camera
from audioviz.sources.network_camera import NetworkCapture, validate_camera_url

URL = "http://192.168.1.20:8080/video"


class Capture:
    def __init__(self, frames=()):
        self.frames = queue.Queue(maxsize=4)
        for value in frames:
            self.frames.put((True, np.full((2, 2, 3), value, dtype=np.uint8)))
        self.released = threading.Event()
        self.condition = threading.Condition()
        self.reads = 0

    def isOpened(self):
        return True

    def read(self):
        with self.condition:
            self.reads += 1
            self.condition.notify_all()
        return self.frames.get(timeout=3)

    def wait_reads(self, count):
        with self.condition:
            assert self.condition.wait_for(lambda: self.reads >= count, timeout=3)

    def release(self):
        self.released.set()


def backend(factory):
    return SimpleNamespace(
        CAP_FFMPEG=cv2.CAP_FFMPEG,
        CAP_PROP_OPEN_TIMEOUT_MSEC=cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        CAP_PROP_READ_TIMEOUT_MSEC=cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        VideoCapture=factory,
    )


def test_slow_open_cannot_block_read_or_release_and_the_worker_owns_cleanup():
    opening, finish_open = threading.Event(), threading.Event()
    capture = Capture()

    def open_stream(*args):
        opening.set()
        assert finish_open.wait(3)
        return capture

    receiver = NetworkCapture(URL, cv2_loader=lambda: backend(open_stream))
    try:
        assert opening.wait(3)
        assert receiver.status == "connecting"
        started = monotonic()
        assert receiver.read() == (False, None)
        receiver.release()
        assert monotonic() - started < 0.1
        assert receiver.status == "off"
        finish_open.set()
        assert receiver.wait_closed(3)
        assert capture.released.is_set()
    finally:
        receiver.release()
        finish_open.set()
        assert receiver.wait_closed(3)


def test_only_latest_fresh_frame_is_consumed_and_disconnects_reconnect(monkeypatch):
    monkeypatch.setattr(network_camera, "RECONNECT_SECONDS", 0.01)
    clock = [0.0]
    monkeypatch.setattr(network_camera, "monotonic", lambda: clock[0])
    first, second = Capture((1, 2, 3)), Capture((9,))
    factory = Mock(side_effect=(first, second))
    receiver = NetworkCapture(URL, cv2_loader=lambda: backend(factory))
    try:
        first.wait_reads(4)  # Entering the next read means all three frames were published.
        ok, frame = receiver.read()
        assert ok
        np.testing.assert_array_equal(frame, np.full((2, 2, 3), 3, dtype=np.uint8))
        assert receiver.read() == (False, None)
        first.frames.put((True, np.full((2, 2, 3), 4, dtype=np.uint8)))
        first.wait_reads(5)
        clock[0] += network_camera.MAX_FRAME_AGE_SECONDS + 0.1
        assert receiver.read() == (False, None)
        first.frames.put((False, None))
        second.wait_reads(2)
        assert first.released.is_set()
        assert receiver.status == "live"
        assert receiver.error is None
        ok, frame = receiver.read()
        assert ok and np.all(frame == 9)
        assert factory.call_count == 2
        assert factory.call_args.args == (URL, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, network_camera.OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, network_camera.READ_TIMEOUT_MS,
        ])
    finally:
        receiver.release()
        first.frames.put((False, None))
        second.frames.put((False, None))
        assert receiver.wait_closed(3)
        assert second.released.is_set()


@pytest.mark.parametrize("value", [
    "file:///tmp/video", "ftp://phone/video", "http://", "http://phone:0/video",
    "http://phone:99999/video", "http://phone/video#fragment",
    "http://user:password@phone/video", "http://phone/with space",
])
def test_invalid_or_credential_bearing_camera_urls_are_rejected(value):
    with pytest.raises(ValueError):
        validate_camera_url(value)


@pytest.mark.parametrize("value", [URL, "rtsp://phone.local:8554/live", "https://phone.local/video"])
def test_stream_addresses_are_preserved(value):
    assert validate_camera_url(value) == value
