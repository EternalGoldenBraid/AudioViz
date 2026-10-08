import audioviz

from audioviz.sources import AudioRippleSource, SyntheticRippleSource
from audioviz.source_controls import (
    AudioSourceControls,
    CameraFrameSourceControls,
    PredictionLearningControls,
    PredictionOverlayControls,
)


def test_audio_source_controls_expose_gate_mapping_and_readout_controls():
    controls = AudioSourceControls().get_controls()
    keys = {control.key: control for control in controls}

    assert audioviz.AudioRippleSource is AudioRippleSource
    assert audioviz.SyntheticRippleSource is SyntheticRippleSource
    assert keys["signal_level"].kind == "text"
    assert keys["mapping_mode"].kind == "choice"
    assert keys["mapping_mode"].choices == ("legacy", "linear")
    assert keys["signal_gate_threshold"].kind == "number"
    assert keys["drive_amplitude"].default == 1.0


def test_camera_frame_source_controls_expose_gain_control():
    controls = CameraFrameSourceControls(gain=2.0).get_controls()

    assert controls[0].key == "gain"
    assert controls[0].label == "Camera Evidence Gain"
    assert controls[0].default == 2.0
    assert controls[0].minimum == 0.0


def test_prediction_learning_controls_expose_toggle_and_learning_parameters():
    controls = PredictionLearningControls(
        enabled=True,
        learning_rate=0.01,
        weight_decay=0.02,
        weight_clip=3.0,
        gradient_clip=4.0,
    ).get_controls()
    keys = {control.key: control for control in controls}

    assert keys["learning_enabled"].kind == "toggle"
    assert keys["learning_enabled"].default is True
    assert keys["learning_rate"].default == 0.01
    assert keys["learning_weight_decay"].default == 0.02
    assert keys["learning_weight_clip"].default == 3.0
    assert keys["learning_gradient_clip"].default == 4.0
    assert keys["inference_steps"].default == 8
    assert keys["inference_rate"].default == 0.1
    assert keys["canvas_rate"].default == 0.01


def test_prediction_overlay_controls_expose_overlay_parameters():
    controls = PredictionOverlayControls(
        enabled=True,
        stride=6,
        threshold=0.2,
        scale=8.0,
    ).get_controls()
    keys = {control.key: control for control in controls}

    assert keys["show_learning_overlay"].kind == "toggle"
    assert keys["show_learning_overlay"].default is True
    assert keys["overlay_stride"].default == 6
    assert keys["overlay_threshold"].default == 0.2
    assert keys["overlay_scale"].default == 8.0
