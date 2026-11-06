import numpy as np
import pytest

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


def test_selecting_no_camera_closes_both_and_choosing_one_again_reopens_it():
    primary, secondary = _sources()
    control = CameraDevices(CHOICES, primary, secondary)
    control.select(None, ["0: EO", "2: IR"])
    control.select(None, [])
    assert control.selected() == [] and _Cv2.open == set()  # devices are freed, not just ignored
    assert primary.excitation() is None and secondary.excitation() is None
    control.select(None, ["2: IR"])
    assert control.selected() == ["2: IR"] and _Cv2.open == {2}


class _UnpluggableCv2(_Cv2):
    missing: set[int] = set()

    def VideoCapture(self, index):
        if index in self.missing:
            return type("Dead", (), {"isOpened": lambda s: False, "release": lambda s: None})()
        return super().VideoCapture(index)


def test_a_failed_open_leaves_the_previous_cameras_running():
    primary, secondary = _sources()
    primary.cv2_loader = secondary.cv2_loader = _UnpluggableCv2
    _UnpluggableCv2.missing = {5}
    control = CameraDevices(CHOICES + ("5: Trust",), primary, secondary)
    control.select(None, ["0: EO", "2: IR"])
    with pytest.raises(RuntimeError):
        control.select(None, ["0: EO", "5: Trust"])
    assert control.selected() == ["0: EO", "2: IR"]
    assert float(primary.excitation().max()) == 1.0


def test_choices_are_read_fresh_and_a_renumbered_device_is_matched_by_name():
    primary, secondary = _sources()
    listing = {"now": ("0: EO", "2: IR", "5: Trust")}
    control = CameraDevices(lambda: listing["now"], primary, secondary)
    listing["now"] = ("0: EO", "2: IR", "4: Trust")  # replugged under another number
    assert control.choices[-1] == "4: Trust"
    control.select(None, ["0: EO", "5: Trust"])  # a stale client still names the old index
    assert control.selected() == ["0: EO", "4: Trust"]


def test_config_roundtrip_retains_both_camera_selections_and_matches_new_indices(tmp_path):
    import json
    from audioviz.tui.deck_server import DeckServer

    primary, secondary = _sources()
    listing = {"now": CHOICES}
    camera = CameraDevices(lambda: listing["now"], primary, secondary)
    camera.select(None, list(CHOICES))
    server = object.__new__(DeckServer)
    server.target = None
    server.controls = {"camera": camera}
    server.config_path = tmp_path / "settings.json"
    server.save({"backend": "window"})
    assert json.loads(server.config_path.read_text())["mixer"]["camera"] == list(CHOICES)
    camera.select(None, [])
    listing["now"] = ("4: EO", "5: IR")
    server.load(strict=True)
    assert camera.selected() == ["4: EO", "5: IR"]
    camera.select(None, [])
