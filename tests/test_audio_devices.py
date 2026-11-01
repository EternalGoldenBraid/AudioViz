from audioviz.utils.audio_devices import list_devices, prompt_for_device


def test_null_device_is_available_without_hardware(monkeypatch, capsys):
    monkeypatch.setattr("audioviz.utils.audio_devices.sd.query_devices", lambda: [])
    assert list_devices() == []
    assert "[-1] Null Device" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda _: "-1")
    index, device = prompt_for_device([])
    assert index == -1
    assert device["max_input_channels"] == 0
    assert device["max_output_channels"] == 0


def test_invalid_negative_and_old_null_indices_are_rejected(monkeypatch):
    devices = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    choices = iter(["-2", "1", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(choices))
    assert prompt_for_device(devices) == (0, devices[0])
