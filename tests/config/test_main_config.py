from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest


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
    config["transforms"]["prediction_error"]["inference"]["dtype"] = "float32"
    config["transforms"]["prediction_error"]["inference"]["steps"] = 12
    config["transforms"]["prediction_error"]["inference"]["hidden_rate"] = 0.2
    config["transforms"]["prediction_error"]["inference"]["canvas_rate"] = 0.03
    config["transforms"]["prediction_error"]["learning"]["enabled"] = True
    config["transforms"]["prediction_error"]["learning"]["learning_rate"] = 0.01

    flattened = main.build_ripple_visualizer_config(config)

    assert flattened["use_shader"] is True
    assert flattened["use_gpu"] is False
    orchestrator_config = flattened["source_orchestrator_config"]
    pose_config = flattened["pose_config"]

    assert orchestrator_config.audio.enabled is False
    assert orchestrator_config.camera_frame.enabled is True
    assert orchestrator_config.camera_frame.camera_index == 2
    assert orchestrator_config.prediction_error.enabled is True
    assert orchestrator_config.prediction_error.inputs == ("camera_frame",)
    assert orchestrator_config.visual_pathway.hidden_channels == 4
    assert orchestrator_config.visual_pathway.dtype == "float32"
    assert orchestrator_config.visual_pathway.inference_steps == 12
    assert orchestrator_config.visual_pathway.inference_rate == 0.2
    assert orchestrator_config.visual_pathway.canvas_rate == 0.03
    assert orchestrator_config.visual_pathway.cross_modal_enabled is False
    assert orchestrator_config.visual_pathway.learning_enabled is True
    assert orchestrator_config.visual_pathway.learning_rate == 0.01
    assert pose_config.camera_index == config["sources"]["pose"]["camera_index"]
    assert (
        pose_config.body_boundary_transmission
        == config["sources"]["pose"]["boundary"]["transmission"]
    )
    config["transforms"]["prediction_error"]["inference"]["cross_modal_enabled"] = True
    assert main.build_ripple_visualizer_config(config)["source_orchestrator_config"].visual_pathway.cross_modal_enabled


def test_active_vision_config_requires_explicit_camera_only_ripple():
    main = _load_main_module()
    config = deepcopy(main.RIPPLE_CONFIG)
    assert main.build_ripple_visualizer_config(config)["active_vision"] is None
    config["active_vision"].update(enabled=True, mode="autonomous", memory_time_constant=.7)
    with pytest.raises(ValueError, match="audio"):
        main.build_ripple_visualizer_config(config)
    config["active_vision"]["audio_precision"] = 1.4
    av_with_audio = main.build_ripple_visualizer_config(config)["active_vision"]
    assert av_with_audio.audio_precision == pytest.approx(1.4)
    config["active_vision"]["audio_precision"] = None
    config["sources"]["audio"]["enabled"] = False
    cfg = main.build_ripple_visualizer_config(config)["active_vision"]
    assert cfg.mode == "autonomous" and cfg.memory_time_constant == .7
    assert cfg.precision.gaze == (.5, .5)
    assert cfg.audio_precision is None


@pytest.mark.parametrize("mode,shared", (
    (None, False), ("mouse", False), ("autonomous", False), ("full", False), ("autonomous", True),
))
def test_main_camera_profile_skips_audio_and_preserves_default_startup(qapp, monkeypatch, mode, shared):
    main = _load_main_module()
    original_app, original_ripple = deepcopy(main.APP_CONFIG), deepcopy(main.RIPPLE_CONFIG)
    created, starts = [], []

    class Processor:
        def process_pending_audio(self):
            pass

        def start(self):
            starts.append(True)
            return True

        def stop(self):
            pass

    processor = Processor()

    def create_audio(config):
        assert mode is None, "camera-only startup must not initialize audio or select devices"
        return processor, dict(cmap=None, norm=None, waveform_plot_duration=.5), 50

    class Window:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def setWindowTitle(self, title):
            pass

        def resize(self, *size):
            pass

        def show(self):
            pass

    monkeypatch.setattr(main, "_create_audio_processor", create_audio)
    monkeypatch.setattr(main, "RippleWaveVisualizer", Window)
    monkeypatch.setattr(main, "SpectrogramVisualizer", Window)
    monkeypatch.setattr(main.QtWidgets.QApplication, "exec", lambda self: 0)
    monkeypatch.setattr(main.signal, "signal", lambda *args: None)
    old_style = qapp.styleSheet()
    try:
        with pytest.raises(SystemExit) as stopped:
            main.main(([] if mode is None else ["--active-vision", mode]) + (["--shared-sensory-graph"] if shared else []))
        assert stopped.value.code == 0
        ripple = next(value for value in created if "source_orchestrator_config" in value)
        assert ripple["processor"] is (processor if mode is None else None)
        assert starts == ([True] if mode is None else [])
        if mode is not None:
            assert len(created) == 1
            assert ripple["active_vision"].mode == mode
            assert ripple["active_vision"].shared_sensory_graph is shared
            assert ripple["source_orchestrator_config"].camera_frame.enabled
            assert not ripple["source_orchestrator_config"].audio.enabled
        assert main.APP_CONFIG == original_app and main.RIPPLE_CONFIG == original_ripple
    finally:
        qapp.setStyleSheet(old_style)


@pytest.mark.parametrize("shared", (False, True))
def test_main_active_audio_requires_active_vision_and_routes_processor_to_ripple(qapp, monkeypatch, shared):
    main = _load_main_module()
    with pytest.raises(SystemExit):
        main.main(["--active-audio"])
    with pytest.raises(SystemExit):
        main.main(["--shared-sensory-graph"])
    with pytest.raises(SystemExit):
        main.main(["--active-vision", "full", "--shared-sensory-graph"])

    created, starts = [], []

    class Processor:
        def process_pending_audio(self):
            pass

        def start(self):
            starts.append(True)
            return True

        def stop(self):
            pass

    processor = Processor()
    monkeypatch.setattr(
        main, "_create_audio_processor",
        lambda config: (processor, dict(cmap=None, norm=None, waveform_plot_duration=.5), 50),
    )

    class Window:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def setWindowTitle(self, title):
            pass

        def resize(self, *size):
            pass

        def show(self):
            pass

    monkeypatch.setattr(main, "RippleWaveVisualizer", Window)
    monkeypatch.setattr(main, "SpectrogramVisualizer", Window)
    monkeypatch.setattr(main.QtWidgets.QApplication, "exec", lambda self: 0)
    monkeypatch.setattr(main.signal, "signal", lambda *args: None)
    old_style = qapp.styleSheet()
    try:
        with pytest.raises(SystemExit) as stopped:
            main.main(["--active-vision", "autonomous", "--active-audio"] + (["--shared-sensory-graph"] if shared else []))
        assert stopped.value.code == 0
        assert len(created) == 1
        ripple = created[0]
        assert ripple["processor"] is processor
        assert starts == [True]
        assert ripple["active_vision"].mode == "autonomous"
        assert ripple["active_vision"].audio_precision == pytest.approx(1.0)
        assert ripple["active_vision"].shared_sensory_graph is shared
        assert ripple["active_vision"].foveal_radius == 8
        assert ripple["source_orchestrator_config"].audio.enabled is True
        assert ripple["source_orchestrator_config"].camera_frame.enabled is True
    finally:
        qapp.setStyleSheet(old_style)
