import audioviz

from audioviz.sources import AudioRippleSource, SyntheticRippleSource
from audioviz.source_controls import (
    AudioSourceControls,
    CameraFrameSourceControls,
    PredictionLearningControls,
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
    assert controls[0].label == "Camera Excitation Gain"
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
