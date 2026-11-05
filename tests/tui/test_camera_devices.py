import numpy as np

from audioviz.sources import CameraFrameSource
from audioviz.tui.deck_devices import CameraDevices

CHOICES = ("0: EO", "2: IR")


class _Capture:
    def __init__(self, index):
        self.index = index

    def isOpened(self):
        return True

    def read(self):
        return True, np.full((4, 4, 3), 255 if self.index == 0 else 128, dtype=np.uint8)

    def release(self):
        _Cv2.open.discard(self.index)


class _Cv2:
    open: set[int] = set()

    def VideoCapture(self, index):
        assert index not in self.open, "device is exclusive"
        self.open.add(index)
        return _Capture(index)


def _sources():
    _Cv2.open = set()
    make = lambda **kw: CameraFrameSource(resolution=(2, 2), cv2_loader=_Cv2, **kw)
    return make(camera_index=0, enabled=True), make(camera_index=2)


def test_two_cameras_open_at_once_and_each_reads_its_own_frames():
    primary, secondary = _sources()
    control = CameraDevices(CHOICES, primary, secondary)
    control.select(None, ["0: EO", "2: IR"])
    assert control.selected() == ["0: EO", "2: IR"]
    assert float(primary.excitation().max()) == 1.0
    assert 0.4 < float(secondary.excitation().max()) < 0.6
    assert _Cv2.open == {0, 2}


def test_one_camera_frees_the_second_and_a_swap_does_not_collide():
    primary, secondary = _sources()
    control = CameraDevices(CHOICES, primary, secondary)
    control.select(None, ["0: EO", "2: IR"])
    control.select(None, ["2: IR"])  # the IR camera becomes primary while the secondary held it
    assert control.selected() == ["2: IR"]
    assert secondary.excitation() is None
    assert primary.camera_index == 2
