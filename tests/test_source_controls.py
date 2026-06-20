import numpy as np

import audioviz
from audioviz.source_controls import (
    AudioSourceControls,
    SourceControl,
    SourceControlProvider,
    SyntheticFrequencySource,
)


def test_source_control_provider_defaults_to_no_controls():
    assert SourceControlProvider().get_controls() == ()
    assert audioviz.SourceControlProvider().get_controls() == ()


def test_source_control_metadata_is_lightweight():
    control = SourceControl(
        key="gain",
        label="Gain",
        default=1.0,
        minimum=0.0,
        maximum=2.0,
        step=0.1,
        unit="x",
    )

    assert control.key == "gain"
    assert control.kind == "number"
    assert control.default == 1.0
    assert control.unit == "x"


def test_synthetic_frequency_source_exposes_frequency_control_and_matrix():
    source = SyntheticFrequencySource(frequency_hz=440.0, n_sources=3)

    controls = source.get_controls()
    freqs = source.frequencies()

    assert controls[0].key == "frequency_hz"
    assert controls[0].unit == "Hz"
    assert freqs.dtype == np.float32
    np.testing.assert_array_equal(freqs, np.full((3, 1), 440.0, dtype=np.float32))


def test_audio_source_controls_expose_gate_mapping_and_readout_controls():
    controls = AudioSourceControls().get_controls()
    keys = {control.key: control for control in controls}

    assert keys["signal_level"].kind == "text"
    assert keys["mapping_mode"].kind == "choice"
    assert keys["mapping_mode"].choices == ("legacy", "linear")
    assert keys["signal_gate_threshold"].kind == "number"
    assert keys["drive_amplitude"].default == 1.0
