import numpy as np

from audioviz.sources import AudioRippleSource


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
    grid = source.excitation_grid(
        t=0.01,
        n_sources=2,
        frequencies=frequencies,
        source_positions=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        resolution=(4, 5),
        grid_spacing=0.1,
        speed=1.0,
        max_frequency=10.0,
        decay_alpha=0.0,
    )
    assert grid is not None
    assert grid.shape == (4, 5)
    assert grid.dtype == np.float32

    source.update_control("top_k_count", 2)
    assert processor.num_top_frequencies == 2
