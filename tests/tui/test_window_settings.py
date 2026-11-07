import json

from audioviz.tui.deck_server import DeckServer, WindowTarget
from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.window_settings import ALIASES, window_settings
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


def test_every_writable_qt_control_has_a_live_adapter_and_typed_config_roundtrip(tmp_path, qapp):
    runtime = GraphRuntime(resolution=16, camera=False)
    window = RippleWaveVisualizer(
        processor=runtime.processor, resolution=(16, 16),
        source_orchestrator_config=runtime.orchestrator.config,
    )
    try:
        controls = window_settings(window)
        expected = {
            ALIASES.get((section.key, c.key), f"{section.key} {c.key}")
            for section in window.source_control_binding.build_sections()
            for c in section.controls if c.kind != "text"
        }
        assert expected <= controls.keys()
        controls["inference steps"].set(None, 12)
        controls["learning"].set(None, True)
        controls["weight decay"].set(None, 0.02)
        controls["speed"].set(None, 123)
        controls["excitation amplitude"].set(None, 2.5)
        controls["excitation falloff"].set(None, 3.5)
        assert controls["wave propagation"].get(None) is True
        controls["wave propagation"].set(None, False)
        config = window.source_orchestrator.visual_readout.config
        assert config.inference_steps == 12 and config.learning_enabled
        assert config.weight_decay == 0.02
        assert controls["inference steps"].get(None) == 12
        assert window.speed == window.engine.speed == 123
        assert window.dt == window.engine.dt
        assert window.base_amplitude == 2.5
        assert window.engine.decay_alpha == 3.5
        controls["learning-overlay overlay_stride"].set(None, 10)
        assert window.learning_overlay_stride == 10
        controls["synthetic-source-0 frequency_hz"].set(None, 99)
        assert controls["synthetic-source-0 frequency_hz"].get(None) == 99
        server = object.__new__(DeckServer)
        server.target = WindowTarget(window)
        server.controls = controls
        server.config_path = tmp_path / "config" / "window.json"
        server.save({"backend": "window"})
        saved = json.loads(server.config_path.read_text())["mixer"]
        assert saved["learning"] is True and saved["inference steps"] == 12
        assert saved["wave propagation"] is False
        controls["inference steps"].set(None, 4)
        controls["speed"].set(None, 300)
        controls["wave propagation"].set(None, True)
        server.load(strict=True)
        assert window.source_orchestrator.visual_readout.config.inference_steps == 12
        assert window.engine.speed == window.speed == 123
        assert window.engine.wave_propagation_enabled is False
        assert window.control_panel is None
    finally:
        window.close()
        runtime.stop()
