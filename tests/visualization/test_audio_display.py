from dataclasses import replace

import numpy as np

from audioviz.audio_processing.history import AudioHistory, TimedHistory
from audioviz.visualization.audio_history_view import spectrum_db


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
