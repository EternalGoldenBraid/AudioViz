from dataclasses import replace

import numpy as np

from audioviz.audio_processing.history import AudioHistory, TimedHistory
from audioviz.visualization.audio_history_view import spectrum_db, waveform_envelope


def test_spectral_display_distinguishes_amplitude_and_power_without_auto_scaling():
    history = AudioHistory(
        sample_rate=8, sample_end=8, waveform=np.zeros(8),
        spectrum=TimedHistory(np.array([[.1, .01], [.01, .001]]), np.array([0, 4])),
        frequencies=np.array([100, 1000]), mel_power=False, cochlea=None,
    )
    np.testing.assert_allclose(spectrum_db(history, 2), [[-20, -40], [-40, -60]])
    np.testing.assert_allclose(spectrum_db(replace(history, mel_power=True), 2), [[-10, -20], [-20, -30]])
    quieter = replace(history, spectrum=TimedHistory(history.spectrum.values / 10, history.spectrum.samples))
    np.testing.assert_allclose(spectrum_db(quieter, 2), spectrum_db(history, 2) - 20)


def test_waveform_envelope_preserves_impulses_and_absolute_amplitude_without_startup_padding():
    samples = np.array([0, 0, 0, .8, -.6, .1, .1, .1])
    history = AudioHistory(8, 5, samples, TimedHistory(np.zeros((1, 1)), np.array([0])), np.array([0]), False, None)
    times, low, high = waveform_envelope(history, 4)
    np.testing.assert_allclose(times, [-.875, -.625, -.375, -.125])
    np.testing.assert_allclose(low, [np.nan, .8, -.6, .1])
    np.testing.assert_allclose(high, [np.nan, .8, .1, .1])
    quieter = replace(history, waveform=samples / 2)
    _, quiet_low, quiet_high = waveform_envelope(quieter, 4)
    np.testing.assert_allclose(quiet_low, low / 2)
    np.testing.assert_allclose(quiet_high, high / 2)
