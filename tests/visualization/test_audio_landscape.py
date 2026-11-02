from types import SimpleNamespace

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PyQt5 import QtCore, QtWidgets

from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer


def test_landscape_geometry_is_bounded_and_new_frames_have_real_history(qapp, monkeypatch):
    from audioviz.visualization.audio_landscape_view import AudioLandscapeView

    view = AudioLandscapeView(cmap=colormaps["viridis"], history_seconds=5)
    surface = {}
    set_data = view.surface.setData

    def record_surface(**data):
        surface.update(data)
        set_data(**data)

    monkeypatch.setattr(view.surface, "setData", record_surface)
    waveform = np.linspace(-2, 2, 2000)
    levels = np.broadcast_to(np.linspace(0, 1, 200), (300, 200))
    for frame in range(60):
        view.set_audio(waveform, levels, timestamp=frame / 20, new_frame=True)
    assert len(view.waveforms) == view.MAX_WAVEFORMS
    assert surface["z"].shape == (128, 96)
    assert surface["colors"].shape == (128 * 96, 4)
    view.surface.parseMeshData()
    np.testing.assert_array_equal(surface["z"][:, 0], 2.5)
    np.testing.assert_array_equal(surface["z"][:, -1], 0)
    assert view.wave_lines[-1].pos.shape == (256, 3)
    np.testing.assert_allclose(view.wave_lines[-1].pos[:, 1], 0)
    assert np.max(view.wave_lines[-1].pos[:, 2]) == 1.3
    view.set_audio(waveform, levels, timestamp=3, new_frame=False)
    assert len(view.waveforms) == 48
    view.set_audio(waveform, levels, timestamp=9, new_frame=False)
    assert not view.waveforms
    assert not any(line.visible() for line in view.wave_lines)
    view.set_canvas(np.ones((8, 12, 3)))
    assert view.planes["canvas"].visible()
    view.set_canvas(None)
    assert not view.planes["canvas"].visible()
    view.close()


def test_landscape_toggle_uses_fixed_db_scale_new_generations_and_read_only_canvas(
    qapp, monkeypatch,
):
    import audioviz.visualization.spectrogram_visualizer as module
    import audioviz.visualization.audio_landscape_view as landscape_module

    class LandscapeStub(QtWidgets.QWidget):
        rendering_failed = QtCore.pyqtSignal(str)

        def __init__(self, **kwargs):
            super().__init__(kwargs["parent"])
            self.waveforms = []
            self.frames = []
            self.resets = 0

        def set_audio(self, waveform, levels, **kwargs):
            self.frames.append((waveform, levels, kwargs))

        def set_canvas(self, canvas):
            self.canvas = canvas

        def reset_view(self):
            self.resets += 1

    monkeypatch.setattr(landscape_module, "AudioLandscapeView", LandscapeStub)
    clock = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    audio = np.full((1000, 2), .2)
    spectrum = np.full((16, 20), .1)
    processor = SimpleNamespace(
        sr=1000, n_spec_bins=16, freq_bins=np.arange(16), n_mels=None,
        num_samples_in_buffer=1000, frame_counter=1,
        get_latest_snapshot=lambda: (audio, [spectrum, spectrum]),
    )
    view = SpectrogramVisualizer(
        processor, colormaps["viridis"], Normalize(-80, 0), .1,
    )
    view.timer.stop()
    view.show()
    canvas = np.ones((6, 8, 3))
    reads = []
    view.set_canvas_provider(lambda: reads.append(True) or canvas)
    view.landscape_toggle.setChecked(True)
    view.update_visualization()
    waveform, levels, options = view.landscape.frames[-1]
    assert waveform.shape == (100,)
    np.testing.assert_allclose(levels, .75)  # Magnitude .1 is -20 dB on [-80, 0].
    assert options["new_frame"]
    assert view.landscape.canvas is canvas
    assert not view.waveform_plot.isVisible()
    view.update_visualization()
    assert len(reads) == 1
    clock[0] = .2
    view.update_visualization()
    assert not view.landscape.frames[-1][2]["new_frame"]
    processor.frame_counter += 1
    processor.n_mels = 16
    clock[0] = .4
    view.update_visualization()
    np.testing.assert_allclose(view.landscape.frames[-1][1], .875)
    assert view.landscape.frames[-1][2]["new_frame"]
    view._reset_landscape()
    assert view.landscape.resets == 1
    view.hide()
    clock[0] = .6
    view.update_visualization()
    assert len(reads) == 3
    view.show()
    processor.get_latest_snapshot = lambda: None
    clock[0] = .8
    view.update_visualization()
    assert len(reads) == 4
    assert len(view.landscape.frames) == 3
    view.landscape.rendering_failed.emit("test context failure")
    assert not view.landscape_toggle.isChecked()
    assert not view.landscape_toggle.isEnabled()
    assert view.waveform_plot.isVisible()
    assert "test context failure" in view.landscape_error.text()
    processor.get_latest_snapshot = lambda: (audio, [spectrum, spectrum])
    view.update_visualization()
    np.testing.assert_array_equal(view.waveform_curve.getData()[1], .2)
    np.testing.assert_array_equal(canvas, 1)
    np.testing.assert_array_equal(audio, .2)
    np.testing.assert_array_equal(spectrum, .1)
    view.close()
