"""LAN video capture: decode on a worker and publish one fresh frame."""

import threading
from time import monotonic
from urllib.parse import urlsplit
from typing import TypeGuard

import cv2
import numpy as np
from loguru import logger

OPEN_TIMEOUT_MS = 2000
READ_TIMEOUT_MS = 1000
RECONNECT_SECONDS = 1.0
MAX_FRAME_AGE_SECONDS = 1.0
NETWORK_SCHEMES = ("http", "https", "rtsp")


def validate_camera_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in NETWORK_SCHEMES or not parts.hostname:
        raise ValueError("Camera URL must be an http://, https:// or rtsp:// stream address.")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Camera URL credentials are not supported; use a stream on a trusted LAN.")
    if parts.fragment or any(char.isspace() for char in value):
        raise ValueError("Camera URL cannot contain whitespace or a fragment.")
    if parts.port is not None and parts.port == 0:
        raise ValueError("Camera URL port must be between 1 and 65535.")
    return value


def is_network_camera(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and "://" in value


class NetworkCapture:
    """Only the worker touches VideoCapture; read/release never wait for the network."""

    def __init__(self, url: str, *, cv2_loader=lambda: cv2) -> None:
        self.url = validate_camera_url(url)
        self._cv2_loader = cv2_loader
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._received_at = 0.0
        self._status = "connecting"
        self._error: str | None = None
        self._thread = threading.Thread(target=self._run, name="network-camera", daemon=True)
        self._thread.start()

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def isOpened(self) -> bool:
        return self.status == "live"

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            frame, self._frame = self._frame, None
            fresh = monotonic() - self._received_at <= MAX_FRAME_AGE_SECONDS
            return (True, frame) if frame is not None and fresh else (False, None)

    def release(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame = None
            self._status = "off"

    def wait_closed(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        backend = self._cv2_loader()
        try:
            while not self._stop.is_set():
                capture = None
                try:
                    capture = backend.VideoCapture(self.url, backend.CAP_FFMPEG, [
                        backend.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                        backend.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS,
                    ])
                    if not capture.isOpened():
                        raise RuntimeError("Cannot open video stream; check the phone server, URL and Wi-Fi.")
                    while not self._stop.is_set():
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            raise RuntimeError("Video stream stopped delivering frames.")
                        with self._lock:
                            if self._stop.is_set():
                                break
                            connected = self._status != "live"
                            self._frame = frame
                            self._received_at = monotonic()
                            self._status, self._error = "live", None
                        if connected:
                            logger.info("LAN camera connected: {}", self.url)
                except (cv2.error, OSError, RuntimeError) as error:
                    with self._lock:
                        changed = self._error != str(error)
                        self._frame = None
                        if not self._stop.is_set():
                            self._status, self._error = "reconnecting", str(error)
                    if changed and not self._stop.is_set():
                        logger.warning("LAN camera {}: {} Retrying.", self.url, error)
                finally:
                    if capture is not None:
                        capture.release()
                self._stop.wait(RECONNECT_SECONDS)
        finally:
            with self._lock:
                self._frame = None
                self._status = "off"
