from copy import deepcopy
import importlib.util
from pathlib import Path
import signal
import sys

import numpy as np
import pytest

from audioviz.sources import CircularDotCapture, CircularDotConfig, CircularDotSource


def _load_main_module():
    module_path = Path(__file__).resolve().parents[2] / "main.py"
    spec = importlib.util.spec_from_file_location("audioviz_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_configuration_wires_total_hidden_units_and_foveal_radius():
    main = _load_main_module()
    config = deepcopy(main.RIPPLE_CONFIG)
    config["model"].update(foveal_radius=32, hidden_units=16, memory_time_constant=.7)
    config["gaze"].update(mode="autonomous", position=(.25, .75))
    config["renderer"]["backend"] = "opengl"
    config["sources"]["audio"]["enabled"] = False
    config["sources"]["camera_frame"]["camera_index"] = 2
    inference = config["transforms"]["prediction_error"]["inference"]
    inference.update(steps=12, hidden_rate=.2, canvas_rate=.03)
    config["transforms"]["prediction_error"]["learning"].update(enabled=True, learning_rate=.01)

    flattened = main.build_ripple_visualizer_config(config)
    assert flattened["use_shader"] and not flattened["use_gpu"]
    source = flattened["source_orchestrator_config"]
    assert not source.audio.enabled
    assert source.camera_frame.enabled and source.camera_frame.camera_index == 2
    assert source.prediction_error.enabled and source.prediction_error.inputs == ("camera_frame",)
    assert source.visual_pathway.hidden_channels == 16
    assert source.visual_pathway.dtype == "float32"
    assert source.visual_pathway.inference_steps == 12
    assert source.visual_pathway.inference_rate == .2
    assert source.visual_pathway.canvas_rate == .03
    assert not source.visual_pathway.cross_modal_enabled
    assert source.visual_pathway.learning_enabled
    assert source.visual_pathway.learning_rate == .01
    gaze = flattened["active_vision"]
    assert gaze.shared_sensory_graph and gaze.foveal_radius == 32
    assert gaze.mode == "autonomous" and gaze.precision.gaze == (.25, .75)
    assert gaze.memory_time_constant == .7 and gaze.audio_precision is None
    assert flattened["pose_config"].camera_index == config["sources"]["pose"]["camera_index"]


@pytest.mark.parametrize("field,value,message", (
    ("hidden_units", 65, "hidden_units"), ("hidden_units", 0, "hidden_units"),
    ("hidden_units", True, "hidden_units"),
    ("foveal_radius", 37, "foveal_radius"), ("foveal_radius", True, "foveal_radius"),
))
def test_invalid_model_sizes_fail_before_hardware_initialization(field, value, message):
    main = _load_main_module()
    config = deepcopy(main.RIPPLE_CONFIG)
    config["model"][field] = value
    with pytest.raises(ValueError, match=message):
        main.build_ripple_visualizer_config(config)


@pytest.mark.parametrize("audio,spectrogram", ((None, None), (False, False), (True, False), (False, True)))
def test_unflagged_startup_uses_foveal_model_and_explicit_audio_settings(qapp, monkeypatch, audio, spectrogram):
    main = _load_main_module()
    if audio is None:
        assert not main.RIPPLE_CONFIG["sources"]["audio"]["enabled"]
        assert not main.APP_CONFIG["windows"]["spectrogram"]["enabled"]
        assert not main.APP_CONFIG["windows"]["helix"]["enabled"]
        audio = spectrogram = False
    else:
        main.RIPPLE_CONFIG["sources"]["audio"]["enabled"] = audio
        main.APP_CONFIG["windows"]["spectrogram"]["enabled"] = spectrogram
    main.APP_CONFIG["windows"]["helix"]["enabled"] = False
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
        assert audio or spectrogram, "camera-only configuration must not initialize audio"
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
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    old_style = qapp.styleSheet()
    try:
        with pytest.raises(SystemExit) as stopped:
            main.main([])
        assert stopped.value.code == 0
        assert len(created) == 1+spectrogram
        ripple = next(value for value in created if "source_orchestrator_config" in value)
        assert ripple["processor"] is (processor if audio else None)
        assert starts == ([True] if audio or spectrogram else [])
        assert ripple["active_vision"].shared_sensory_graph
        assert ripple["active_vision"].foveal_radius == 32
        assert ripple["active_vision"].audio_precision == (1.0 if audio else None)
        assert ripple["source_orchestrator_config"].visual_pathway.hidden_channels == 6
        assert ripple["source_orchestrator_config"].audio.enabled is audio
        assert main.APP_CONFIG == original_app and main.RIPPLE_CONFIG == original_ripple
    finally:
        qapp.setStyleSheet(old_style)


@pytest.mark.parametrize("backend,hidden_units", (("numpy", 6), ("opengl", 16)))
def test_normal_main_constructs_and_updates_the_configured_foveal_graph(qapp, monkeypatch, backend, hidden_units):
    if backend == "opengl" and qapp.platformName() == "offscreen":
        pytest.skip("Native widget requires a window-system context")
    main = _load_main_module()
    main.RIPPLE_CONFIG["model"]["hidden_units"] = hidden_units
    main.RIPPLE_CONFIG["sources"]["audio"]["enabled"] = False
    main.RIPPLE_CONFIG["renderer"]["backend"] = backend
    main.RIPPLE_CONFIG["field"]["resolution"] = (80, 96)
    main.APP_CONFIG["windows"]["spectrogram"]["enabled"] = False
    main.APP_CONFIG["windows"]["helix"]["enabled"] = False
    main.APP_CONFIG["windows"]["ripples"]["size"] = (800, 600)
    stimulus = CircularDotSource(CircularDotConfig(resolution=(80, 96), radius=20))
    created, errors = [], []

    class Window(main.RippleWaveVisualizer):
        def __init__(self, **kwargs):
            super().__init__(camera_capture=CircularDotCapture(stimulus), **kwargs)
            self.timer.stop()
            created.append(self)

    def run(app):
        qapp.processEvents()
        window = created[0]
        window.update_visualization()
        source = window.source_orchestrator
        graph = source.sensory_graph
        assert source.graph_io.radius == 32 and source.graph_io.visual_shape == (3209, 1)
        assert graph.hidden_nodes == hidden_units and graph.canvas_weights.shape == (hidden_units, 9627)
        assert graph.mean.dtype == np.dtype("float64")
        assert graph.clamped[:graph.visual_nodes].all()
        assert np.isfinite(graph.belief_snapshot()["variance"]).all()
        assert window.visual_prediction.shape == (80, 96, 3)
        assert not errors, "\n".join(map(str, errors))
        return 0

    monkeypatch.setattr(main, "RippleWaveVisualizer", Window)
    monkeypatch.setattr(main.QtWidgets.QApplication, "exec", run)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    old_style = qapp.styleSheet()
    try:
        with pytest.raises(SystemExit) as stopped:
            main.main([])
        assert stopped.value.code == 0
    finally:
        for window in created:
            window.close()
        qapp.setStyleSheet(old_style)


def test_model_selection_flags_are_not_an_alternative_application_entrypoint():
    main = _load_main_module()
    for args in (["--shared-sensory-graph"], ["--active-vision", "mouse"], ["--active-audio"]):
        with pytest.raises(SystemExit) as stopped:
            main.main(args)
        assert stopped.value.code == 2
    config = deepcopy(main.RIPPLE_CONFIG)
    config["gaze"]["mode"] = "full"
    with pytest.raises(ValueError, match="full-image"):
        main.build_ripple_visualizer_config(config)
