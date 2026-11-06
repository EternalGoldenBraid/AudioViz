import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic, sleep

import cv2
import numpy as np
import pytest

from audioviz.sources.camera import CameraFrameSource
from audioviz.sources import network_camera


@pytest.fixture
def phone_stream():
    stop = threading.Event()
    available = threading.Event()
    available.set()
    image = np.full((32, 48, 3), (10, 120, 240), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    packet = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
    packet += b"\r\n\r\n" + jpeg.tobytes() + b"\r\n"

    class Phone(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if not available.is_set():
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            if self.path == "/stall":
                stop.wait(5)
                return
            while not stop.is_set() and available.is_set():
                try:
                    self.wfile.write(packet)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                stop.wait(0.02)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Phone)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", available
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()


def wait_for(predicate, timeout=5):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    assert predicate(), "camera did not reach the expected state"


def test_real_mjpeg_stream_reaches_camera_observations_and_recovers_after_disconnect(phone_stream, monkeypatch):
    base, available = phone_stream
    monkeypatch.setattr(network_camera, "RECONNECT_SECONDS", 0.05)
    source = CameraFrameSource(resolution=(8, 8), camera_index=base + "/video", enabled=True)
    source.ensure_open()
    receiver = source.capture
    try:
        def observed():
            source.observe()
            return source.observation is not None

        wait_for(observed)
        assert source.capture_status == "live"
        assert source.observation.shape == (8, 8, 3)
        assert np.mean(source.observation[..., 0]) > 0.9
        assert 0.4 < np.mean(source.observation[..., 1]) < 0.55
        assert np.mean(source.observation[..., 2]) < 0.1
        available.clear()
        wait_for(lambda: receiver.status == "reconnecting")
        source.observe()
        assert source.observation is None
        assert receiver.error
        available.set()
        wait_for(observed)
        assert source.capture_status == "live"
    finally:
        source.close()
        assert source.observation is None
        assert receiver.wait_closed(3)


def test_ffmpeg_open_timeout_releases_a_server_that_never_delivers_video(phone_stream, monkeypatch):
    base, _ = phone_stream
    monkeypatch.setattr(network_camera, "OPEN_TIMEOUT_MS", 200)
    receiver = network_camera.NetworkCapture(base + "/stall")
    try:
        wait_for(lambda: receiver.status == "reconnecting", timeout=3)
        assert receiver.read() == (False, None)
        assert "Cannot open" in receiver.error
    finally:
        started = monotonic()
        receiver.release()
        assert monotonic() - started < 0.1
        assert receiver.wait_closed(3)
