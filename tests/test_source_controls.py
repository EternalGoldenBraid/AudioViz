import numpy as np

import audioviz
from audioviz.sources import AudioRippleSource, SyntheticRippleSource
from audioviz.source_controls import (
    AudioSourceControls,
    CameraFrameSourceControls,
    SourceControl,
    SourceControlProvider,
    SyntheticFrequencySource,
)


def test_source_control_provider_defaults_to_no_controls():
    assert SourceControlProvider().get_controls() == ()
    assert audioviz.SourceControlProvider().get_controls() == ()
    assert audioviz.AudioRippleSource is AudioRippleSource
    assert audioviz.SyntheticRippleSource is SyntheticRippleSource


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


def test_synthetic_ripple_source_owns_enabled_frequencies_and_controls():
    source = SyntheticRippleSource(
        frequency=(220.0, 330.0),
        n_sources=2,
        enabled=True,
    )

    np.testing.assert_array_equal(
        source.frequencies(),
        np.array([[220.0], [330.0]], dtype=np.float32),
    )
    controls = source.controls_for_index(1)
    assert controls[0].key == "frequency_hz"
    assert controls[0].default == 330.0

    source.set_frequency(1, 440.0)
    assert source.frequency == 220.0
    np.testing.assert_array_equal(
        source.frequencies(),
        np.array([[220.0], [440.0]], dtype=np.float32),
    )

    source.enabled = False
    assert source.frequencies() is None


def test_audio_source_controls_expose_gate_mapping_and_readout_controls():
    controls = AudioSourceControls().get_controls()
    keys = {control.key: control for control in controls}

    assert keys["signal_level"].kind == "text"
    assert keys["mapping_mode"].kind == "choice"
    assert keys["mapping_mode"].choices == ("legacy", "linear")
    assert keys["signal_gate_threshold"].kind == "number"
    assert keys["drive_amplitude"].default == 1.0


def test_audio_ripple_source_maps_frequencies_and_scales_amplitude():
    class Processor:
        current_top_k_frequencies = [440.0, 880.0, None]
        current_signal_level = 0.25
        minimum_frequency_peak_magnitude = 0.1
        minimum_frequency_peak_to_median_ratio = 5.0
        num_top_frequencies = 3

        def set_num_top_frequencies(self, count):
            self.num_top_frequencies = count

    processor = Processor()
    source = AudioRippleSource(
        processor=processor,
        enabled=True,
        signal_gate_threshold=0.1,
        drive_amplitude=2.0,
        mapping_mode="linear",
        linear_scale=0.1,
    )

    frequencies = source.frequencies(n_sources=2)

    assert frequencies is not None
    np.testing.assert_allclose(
        frequencies,
        np.array([[44.0, 88.0], [44.0, 88.0]], dtype=np.float32),
    )
    assert source.excitation_amplitude(
        base_amplitude=4.0,
        frequencies=frequencies,
        synthetic_enabled=False,
    ) == 2.0

    source.update_control("top_k_count", 2)
    assert processor.num_top_frequencies == 2


def test_camera_frame_source_controls_expose_gain_control():
    controls = CameraFrameSourceControls(gain=2.0).get_controls()

    assert controls[0].key == "gain"
    assert controls[0].label == "Camera Excitation Gain"
    assert controls[0].default == 2.0
    assert controls[0].minimum == 0.0
