import numpy as np
import pytest

from audioviz.sources import CameraFrameSource
from audioviz.tui.deck_devices import CameraDevices

CHOICES = ("0: EO", "2: IR")
PHONE = "http://192.168.1.20:8080/video"


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


def test_network_camera_identity_status_and_preset_roundtrip_without_local_devices(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from audioviz.sources import camera as camera_module
    from audioviz.tui import deck_devices
    from audioviz.tui.deck_server import DeckServer, describe_control

    class Network:
        def __init__(self, url, **kwargs):
            self.url, self.status = url, "connecting"

        def read(self):
            return True, np.full((2, 2, 3), 255, dtype=np.uint8)

        def release(self):
            self.status = "off"

    monkeypatch.setattr(camera_module, "NetworkCapture", Network)
    primary, secondary = _sources()
    camera = CameraDevices(CHOICES, primary, secondary)
    try:
        camera.select(None, [CHOICES[0], PHONE])
        assert camera.selected() == [CHOICES[0], f"net: {PHONE}"]
        assert secondary.camera_index == PHONE
        assert camera.format() == "1 live | 2 connecting"
        assert describe_control(camera, None)["live"] == [CHOICES[0]]
        secondary.capture.status = "live"
        assert describe_control(camera, None)["live"] == camera.selected()
        assert camera.format() == "1 live | 2 live"
        np.testing.assert_array_equal(secondary.excitation(), np.ones((2, 2, 3)))
        server = object.__new__(DeckServer)
        server.target = None
        server.controls = {"camera": camera}
        server.config_path = tmp_path / "phone.json"
        server.save({"backend": "window"})
        assert json.loads(server.config_path.read_text())["mixer"]["camera"] == [CHOICES[0], f"net: {PHONE}"]
        camera.select(None, [])
        primary, secondary = _sources()
        camera = CameraDevices(("4: EO",), primary, secondary)
        assert f"net: {PHONE}" not in camera.choices
        server.controls["camera"] = camera
        server.load(strict=True)
        assert camera.selected() == ["4: EO", f"net: {PHONE}"]
        assert secondary.camera_index == PHONE
        camera.select(None, [])
        monkeypatch.setattr(deck_devices, "camera_choices", lambda: ())
        window = SimpleNamespace(
            processor=SimpleNamespace(input_channels=1, output_channels=2),
            source_orchestrator=SimpleNamespace(camera_source=primary, secondary_camera_source=secondary),
        )
        camera = deck_devices.device_controls(window, [])["camera"]
        camera.select(None, [PHONE])
        assert camera.selected() == [f"net: {PHONE}"]
    finally:
        camera.select(None, [])


def test_invalid_network_selection_leaves_the_existing_local_camera_open():
    primary, secondary = _sources()
    camera = CameraDevices(CHOICES, primary, secondary)
    camera.select(None, [CHOICES[0]])
    capture = primary.capture
    try:
        with pytest.raises(ValueError, match="credentials"):
            camera.select(None, ["http://user:password@phone/video"])
        assert camera.selected() == [CHOICES[0]]
        assert primary.capture is capture
        assert _Cv2.open == {0}
    finally:
        camera.select(None, [])
