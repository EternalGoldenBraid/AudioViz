import numpy as np

from audioviz.sources import SyntheticRippleSource
from audioviz.sources.ripple_grid import normalized_radial_decay
from audioviz.source_controls import SyntheticFrequencySource


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
    grid = source.excitation_grid(
        t=0.01,
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

    source.enabled = False
    assert source.frequencies() is None
    assert source.excitation_grid(
        t=0.01,
        source_positions=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        resolution=(4, 5),
        grid_spacing=0.1,
        speed=1.0,
        max_frequency=10.0,
        decay_alpha=0.0,
    ) is None


def test_normalized_radial_decay_changes_gradually_across_field_diagonal():
    distances = np.array([0.0, 50.0, 100.0], dtype=np.float32)

    subtle = normalized_radial_decay(
        distances,
        resolution=(1, 101),
        decay_alpha=0.1,
    )
    moderate = normalized_radial_decay(
        distances,
        resolution=(1, 101),
        decay_alpha=1.0,
    )

    np.testing.assert_allclose(subtle, [1.0, np.exp(-0.05), np.exp(-0.1)])
    np.testing.assert_allclose(moderate, [1.0, np.exp(-0.5), np.exp(-1.0)])
