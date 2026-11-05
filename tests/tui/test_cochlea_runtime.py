import numpy as np
import pytest

from audioviz.tui.audio_feed import BLOCK_SIZE
from audioviz.tui.audio_out import AudioOut
from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.views import VIEWS


@pytest.fixture
def runtime():
    runtime = GraphRuntime(resolution=8, camera=False, audio_model="cochlea")
    runtime.feed.use_tone(800)
    yield runtime
    runtime.stop()


def test_raw_inference_couples_both_ways_with_pre_observation_graph_prediction(runtime, monkeypatch):
    runtime.orchestrator.update_pathway_config(learning_enabled=True)
    source = runtime.orchestrator.audio_source
    source.observation_gain = 2
    pump = runtime.feed.pump
    priors = []

    def capture():
        priors.append(runtime.orchestrator.audio_prediction.copy())
        return pump()

    monkeypatch.setattr(runtime.feed, "pump", capture)
    weights = runtime.audio_readout.visual_weights.copy()
    runtime.step()
    bank = runtime.processor.cochlea
    target = bank.last_block.state
    np.testing.assert_array_equal(source.recent_observation(), 2 * target)
    np.testing.assert_array_equal(runtime.orchestrator.audio_prediction, priors[0])
    np.testing.assert_allclose(bank.state.ravel(), (1 - bank.config.graph_rate) * target + bank.config.graph_rate * priors[0] / 2)
    assert runtime.audio_readout.visual_weights.shape == (32, 6)
    assert np.linalg.norm(runtime.audio_readout.hidden) > 0
    assert np.linalg.norm(runtime.engine.get_field_numpy()) > 0
    assert not np.array_equal(runtime.audio_readout.visual_weights, weights)
    assert runtime.audio_readout.diagnostics.observation_present


@pytest.mark.parametrize("mode", ["input", "prediction", "error", "evidence"])
def test_listening_taps_are_signed_sample_aligned_waveforms(runtime, mode):
    runtime.out.set_mode(mode, speakers=False)
    runtime.step()
    block = runtime.processor.cochlea.last_block
    expected = {
        "input": block.prediction + block.error,
        "prediction": block.prediction,
        "error": block.error,
        "evidence": block.prediction + block.error,
    }[mode]
    np.testing.assert_allclose(runtime.out.last_block, expected * runtime.out.volume, atol=3e-8)
    assert runtime.out.last_block.min() < 0 < runtime.out.last_block.max()


def test_multiple_input_blocks_play_once_in_order_and_no_input_does_not_replay(runtime):
    class Stream:
        def stop(self):
            pass
        def close(self):
            pass

    runtime.feed._stream = Stream()
    for amplitude in (0.1, -0.2, 0.3):
        runtime.feed._on_block(np.full((BLOCK_SIZE, 1), amplitude), BLOCK_SIZE, None, None)
    runtime.out.set_mode("input", speakers=False)
    runtime.step()
    assert len(runtime.out._queue) == 3
    for amplitude in (0.1, -0.2, 0.3):
        out = np.empty((BLOCK_SIZE, 1))
        runtime.out._fill(out, BLOCK_SIZE, None, None)
        np.testing.assert_allclose(out, amplitude * runtime.out.volume)
    state = runtime.processor.cochlea.state.copy()
    runtime.step()
    np.testing.assert_array_equal(runtime.processor.cochlea.state, state)
    assert not runtime.out._queue
    assert not runtime.audio_readout.diagnostics.observation_present


def test_zero_gain_disables_graph_feedback_without_muting_raw_audio(runtime):
    runtime.orchestrator.audio_source.observation_gain = 0
    runtime.out.set_mode("input", speakers=False)
    runtime.step()
    bank = runtime.processor.cochlea
    np.testing.assert_array_equal(bank.state.ravel(), bank.last_block.state)
    np.testing.assert_array_equal(runtime.orchestrator.audio_source.recent_observation(), 0)
    assert np.max(np.abs(runtime.out.last_block)) > 0.1


def test_cochlea_views_are_waveforms_and_spectrogram_remains_diagnostic(runtime):
    assert VIEWS["prediction"](runtime, 30, 80) is None
    runtime.step()
    for name in ("prediction", "error", "spectrogram", "waveform", "hidden"):
        image = VIEWS[name](runtime, 30, 80)
        assert image.dtype == np.uint8 and image.shape[-1] == 3
    assert VIEWS["prediction"](runtime, 30, 80).shape == (30, 80, 3)


def test_model_switch_rebuilds_state_shape_and_mutes_output(runtime):
    runtime.out.set_mode("prediction", speakers=False)
    runtime.step()
    runtime.set_audio_model("spectral")
    assert runtime.processor.cochlea is None
    assert runtime.audio_readout.visual_weights.shape == (16, 6)
    assert runtime.out.mode == "off" and not runtime.out._queue
    runtime.step()
    assert runtime.orchestrator.audio_source.recent_observation().shape == (16,)
    runtime.set_audio_model("cochlea")
    assert runtime.processor.cochlea.last_block is None
    runtime.step()
    assert runtime.orchestrator.audio_source.recent_observation().shape == (32,)
    with pytest.raises(ValueError):
        runtime.set_audio_model("unknown")


def test_settings_control_selects_the_model_and_keeps_mixer_values(runtime):
    from types import SimpleNamespace
    from audioviz.tui.settings import settings_controls

    control = next(c for c in settings_controls() if c.name == "audio model")
    app = SimpleNamespace(runtime=runtime)
    runtime.orchestrator.audio_source.observation_gain = 3
    control.nudge(app, -1)
    assert control.get(app) == "spectral"
    control.nudge(app, 1)
    assert control.get(app) == "cochlea"
    assert runtime.orchestrator.audio_source.observation_gain == 3


def test_input_source_change_resets_cochlea_history(runtime):
    runtime.step()
    runtime.feed.use_tone(440)
    assert runtime.processor.cochlea.last_block is None
    np.testing.assert_array_equal(runtime.processor.cochlea.state, 0)


def test_output_mode_switch_discards_old_audio_and_callback_keeps_partial_block():
    out = AudioOut()
    out.set_mode("input", speakers=False)
    out.submit_waveform(np.linspace(-0.5, 0.5, BLOCK_SIZE))
    head = np.empty((128, 1))
    out._fill(head, 128, None, None)
    tail = np.empty((BLOCK_SIZE - 128, 1))
    out._fill(tail, BLOCK_SIZE - 128, None, None)
    np.testing.assert_allclose(np.concatenate((head[:, 0], tail[:, 0])), out.last_block)
    out.submit_waveform(np.zeros(BLOCK_SIZE))
    out.set_mode("error", speakers=False)
    assert not out._queue and out.last_block is None
    with pytest.raises(ValueError):
        out.submit_waveform(np.array([np.nan]))
