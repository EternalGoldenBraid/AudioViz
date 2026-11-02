"""Camera capture on a background thread; read() never blocks the graph step."""

import threading

import numpy as np


class CameraFeed:
    """Quacks like cv2.VideoCapture for the orchestrator's CameraFrameSource and also serves the camera tile."""

    def __init__(self, index: int | str = 0) -> None:
        self.index = index
        self.error: str | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._capture = None
        self._thread: threading.Thread | None = None

    def open(self) -> bool:
        """Start capturing; False (with .error set) when no camera can be opened."""
        try:
            import cv2

            capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"cannot open camera {self.index}")
        except Exception as error:  # no camera must leave the rest of the app running
            self.error = str(error)
            return False
        self._capture = capture
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def isOpened(self) -> bool:
        return self._capture is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            return (self._frame is not None), self._frame

    def rgb(self) -> np.ndarray | None:
        ok, frame = self.read()
        return frame[..., ::-1] if ok else None

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
