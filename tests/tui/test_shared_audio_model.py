from types import SimpleNamespace

import numpy as np

from audioviz.tui.deck_server import DeckServer, WindowTarget
from audioviz.tui.graph_runtime import GraphRuntime


def test_remote_model_control_changes_the_window_process_not_a_second_runtime(tmp_path, monkeypatch):
    from audioviz.tui import deck_server

    runtime = GraphRuntime(resolution=8, camera=False)
    window = SimpleNamespace(
        engine=runtime.engine, processor=runtime.processor,
        source_orchestrator=runtime.orchestrator,
    )
    monkeypatch.setattr(deck_server, "device_controls", lambda *_: {})
    monkeypatch.setattr(deck_server, "query_audio_devices", lambda: [])
    server = DeckServer(window, tmp_path / "audio.sock", tmp_path / "deck.json")
    try:
        camera_readout = runtime.orchestrator.visual_readout
        camera_weights = camera_readout.visual_weights.copy()
        reply = server.handle({"op": "set", "name": "audio model", "value": "cochlea"})
        assert next(c for c in reply["controls"] if c["name"] == "audio model")["text"] == "cochlea"
        assert WindowTarget(window).processor.cochlea is runtime.processor.cochlea
        assert runtime.orchestrator.visual_readout is camera_readout
        np.testing.assert_array_equal(camera_readout.visual_weights, camera_weights)
        runtime.step()
        bank = runtime.processor.cochlea
        target = bank.last_block.state.copy()
        expected = (1 - bank.config.graph_rate) * target + bank.config.graph_rate * runtime.orchestrator.audio_prediction
        np.testing.assert_allclose(bank.state.ravel(), expected)
        state = bank.state.copy()
        runtime.orchestrator.correct_from_observations()
        np.testing.assert_array_equal(bank.state, state)
        server.handle({"op": "set", "name": "audio model", "value": "spectral"})
        assert runtime.processor.cochlea is None
        assert runtime.orchestrator.audio_source.recent_observation() is None
    finally:
        server.server.close()
        runtime.stop()
