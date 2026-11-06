from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import pytest

from audioviz.tui import launcher
from audioviz.utils import audio_devices


def _load_main_module():
    module_path = Path(__file__).resolve().parents[2] / "main.py"
    spec = importlib.util.spec_from_file_location("audioviz_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_ripple_visualizer_config_flattens_nested_source_config():
    main = _load_main_module()
    config = deepcopy(main.RIPPLE_CONFIG)
    config["renderer"]["backend"] = "opengl"
    config["sources"]["audio"]["enabled"] = False
    config["sources"]["camera_frame"]["enabled"] = True
    config["sources"]["camera_frame"]["camera_index"] = 2
    config["transforms"]["prediction_error"]["enabled"] = True
    config["transforms"]["prediction_error"]["inference"]["hidden_channels"] = 4
    config["transforms"]["prediction_error"]["inference"]["steps"] = 12
    config["transforms"]["prediction_error"]["inference"]["hidden_rate"] = 0.2
    config["transforms"]["prediction_error"]["inference"]["canvas_rate"] = 0.03
    config["transforms"]["prediction_error"]["learning"]["enabled"] = True
    config["transforms"]["prediction_error"]["learning"]["learning_rate"] = 0.01

    flattened = main.build_ripple_visualizer_config(config)

    assert flattened["use_shader"] is True
    assert flattened["use_gpu"] is False
    orchestrator_config = flattened["source_orchestrator_config"]

    assert orchestrator_config.audio.enabled is False
    assert orchestrator_config.camera_frame.enabled is True
    assert orchestrator_config.camera_frame.camera_index == 2
    assert orchestrator_config.prediction_error.enabled is True
    assert orchestrator_config.prediction_error.inputs == ("camera_frame",)
    assert orchestrator_config.visual_pathway.hidden_channels == 4
    assert orchestrator_config.visual_pathway.inference_steps == 12
    assert orchestrator_config.visual_pathway.inference_rate == 0.2
    assert orchestrator_config.visual_pathway.canvas_rate == 0.03
    assert orchestrator_config.visual_pathway.cross_modal_enabled is False
    assert orchestrator_config.visual_pathway.learning_enabled is True
    assert orchestrator_config.visual_pathway.learning_rate == 0.01
    config["transforms"]["prediction_error"]["inference"]["cross_modal_enabled"] = True
    assert main.build_ripple_visualizer_config(config)["source_orchestrator_config"].visual_pathway.cross_modal_enabled


@pytest.mark.parametrize("configured", [False, True], ids=["first-run", "saved-devices"])
def test_entrypoint_leaves_device_setup_to_tui(tmp_path, monkeypatch, configured):
    main_path = Path(__file__).resolve().parents[2] / "main.py"
    config_path = tmp_path / "config/audio_devices.json"
    expected = {
        "input_device_index": 0,
        "output_device_index": 0,
        "input_channels": 1,
        "output_channels": 2,
        "samplerate": 44100,
    }
    if configured:
        config_path.parent.mkdir()
        config_path.write_text(json.dumps(expected))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(main_path)])
    def unexpected_prompt(*args, **kwargs):
        pytest.fail("Startup must reach the TUI before querying or prompting for devices")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    monkeypatch.setattr(audio_devices.sd, "query_devices", unexpected_prompt)

    def launch_window(command, *, config_file):
        assert command == [sys.executable, str(main_path), "--no-tui"]
        assert config_file == Path("config/audio_devices.json")
        assert config_file.exists() == configured
        return 0

    monkeypatch.setattr(launcher, "run_with_tui", launch_window)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(main_path), run_name="__main__")
    assert exited.value.code == 0
    assert config_path.exists() == configured


@pytest.mark.parametrize("setup", ["missing", "saved", "stale"])
def test_window_only_uses_saved_setup_without_prompting(tmp_path, monkeypatch, setup):
    from PyQt5 import QtWidgets

    class StopBeforeStreams(Exception):
        pass

    def stop_before_streams(*args, **kwargs):
        raise StopBeforeStreams

    devices = [{
        "name": "Test audio", "max_input_channels": 1, "max_output_channels": 2,
        "default_samplerate": 48000,
    }]
    main_path = Path(__file__).resolve().parents[2] / "main.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(main_path), "--no-tui"])
    monkeypatch.setattr(audio_devices.sd, "query_devices", lambda: devices)
    monkeypatch.setattr(audio_devices.sd, "check_input_settings", lambda **_: None)
    monkeypatch.setattr(audio_devices.sd, "check_output_settings", lambda **_: None)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Window-only startup must not prompt"))
    errors = []
    monkeypatch.setattr("loguru.logger.error", lambda message, *args: errors.append(message.format(*args)))
    monkeypatch.setattr(QtWidgets, "QApplication", stop_before_streams)
    config_path = tmp_path / "config/audio_devices.json"
    if setup != "missing":
        config_path.parent.mkdir()
        config = audio_devices.make_device_config(devices, 0, 0)
        if setup == "stale":
            config["input_device_index"] = 9
        config_path.write_text(json.dumps(config))
    before = config_path.read_bytes() if config_path.exists() else None
    if setup == "saved":
        with pytest.raises(StopBeforeStreams):
            runpy.run_path(str(main_path), run_name="__main__")
    else:
        with pytest.raises(SystemExit) as exited:
            runpy.run_path(str(main_path), run_name="__main__")
        assert exited.value.code == 1
        assert any("Run without --no-tui" in message for message in errors)
    after = config_path.read_bytes() if config_path.exists() else None
    assert after == before


def test_window_only_applies_the_runtime_preset_before_starting_audio(tmp_path, monkeypatch, qapp):
    from types import SimpleNamespace
    from audioviz.tui.deck_server import DeckServer, saved_grid

    main = _load_main_module()
    monkeypatch.chdir(tmp_path)
    devices = [{"name": "Test audio", "max_input_channels": 1, "max_output_channels": 2}]
    monkeypatch.setattr(audio_devices.sd, "query_devices", lambda: devices)
    monkeypatch.setattr(audio_devices.sd, "check_input_settings", lambda **_: None)
    monkeypatch.setattr(audio_devices.sd, "check_output_settings", lambda **_: None)

    def fake_stream(**options):
        return SimpleNamespace(
            device=options["device"], start=lambda: None, stop=lambda: None, close=lambda: None,
        )

    monkeypatch.setattr(audio_devices.sd, "InputStream", fake_stream)
    monkeypatch.setattr(audio_devices.sd, "OutputStream", fake_stream)
    monkeypatch.setattr("cv2.VideoCapture", lambda *_args, **_kw: pytest.fail("No real camera"))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "audio_devices.json").write_text(json.dumps({
        "input_device_index": 0, "output_device_index": 0,
        "input_channels": 1, "output_channels": 2, "samplerate": 44100,
    }))
    preset = config_dir / "deck.json"
    preset.write_text(json.dumps({
        "backend": "window",
        "mixer": {"grid at launch": "8x8", "speed": 123, "damping": 0.7,
                  "inference steps": 3, "learning": False},
    }))
    main.APP_CONFIG["windows"]["spectrogram"]["enabled"] = False
    main.APP_CONFIG["windows"]["helix"]["enabled"] = False
    main.RIPPLE_CONFIG["sources"]["camera_frame"]["enabled"] = False
    monkeypatch.setattr(main, "QtWidgets", SimpleNamespace(QApplication=lambda _: qapp))
    monkeypatch.setattr(qapp, "exec", lambda: 0)
    monkeypatch.setattr(main.signal, "signal", lambda *_: None)
    monkeypatch.setattr(main, "saved_grid", lambda: saved_grid(preset))
    servers = []

    def create_deck(window, *, windows):
        server = DeckServer(window, windows=windows, path=tmp_path / "deck.sock", config_path=preset)
        servers.append(server)
        return server

    def start_audio(_processor):
        window = servers[0].target.window
        assert window.engine.canvas_shape == (8, 8, 3)
        assert window.engine.speed == window.speed == 123
        assert window.engine.damping == 0.7
        assert window.source_orchestrator.visual_readout.config.inference_steps == 3
        assert not window.source_orchestrator.visual_readout.config.learning_enabled
        return True

    monkeypatch.setattr(main, "DeckServer", create_deck)
    monkeypatch.setattr(main.AudioProcessor, "start", start_audio)
    stylesheet = qapp.styleSheet()
    try:
        with pytest.raises(SystemExit) as exited:
            main.main()
        assert exited.value.code == 0
    finally:
        for server in servers:
            server.server.close()
            server.target.window.close()
        qapp.setStyleSheet(stylesheet)
