import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PyQt5 import QtWidgets

from audioviz.tui.deck_server import WindowTarget, render_view
from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.views import VIEWS
from audioviz.visualization.audio_history_view import (
    cochlea_db, level_rgb, COCHLEA_COLORMAP, COCHLEA_DB_RANGE,
    spectrum_db, SPECTRUM_COLORMAP, SPECTRUM_DB_RANGE,
)
from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_qt_and_attached_thumbnail_project_the_same_process_without_advancing_it(qapp):
    runtime = GraphRuntime(resolution=8, camera=False)
    view = SpectrogramVisualizer(runtime.processor, colormaps["viridis"], Normalize(-80, 0), .1)
    view.timer.stop()
    view.set_audio_model_controller(runtime.orchestrator)
    try:
        view.audio_model_selector.activated[str].emit("cochlea")
        runtime.step()
        bank = runtime.processor.cochlea
        state = bank.state.copy()
        sample_end = runtime.processor.samples_processed
        target = WindowTarget(SimpleNamespace(
            processor=runtime.processor, source_orchestrator=runtime.orchestrator,
            engine=runtime.engine,
        ))
        history = runtime.processor.get_audio_history()
        for width in (500, 900):
            view.resize(width, 700)
            view.show()
            qapp.processEvents()
            view.update_visualization()
            columns = view.cochlea_image.image.shape[1]
            np.testing.assert_array_equal(view.cochlea_image.image, cochlea_db(history, columns))
            np.testing.assert_array_equal(
                VIEWS["cochlea"](target, 16, columns),
                level_rgb(view.cochlea_image.image[::-1], COCHLEA_DB_RANGE, COCHLEA_COLORMAP),
            )
            spectrum = view.spectrogram_image_item.image
            np.testing.assert_array_equal(spectrum, spectrum_db(history, spectrum.shape[1]))
            np.testing.assert_array_equal(
                VIEWS["spectrogram"](target, 16, spectrum.shape[1]),
                level_rgb(spectrum[::-1], SPECTRUM_DB_RANGE, SPECTRUM_COLORMAP),
            )
            assert view.spectrogram_plot_item.getAxis("left").labelText == "Hz (mel)"
            assert render_view(target, "cochlea", 16, 80)["shape"] == [16, 80, 3]
            view.hide()
        np.testing.assert_array_equal(bank.state, state)
        assert runtime.processor.samples_processed == sample_end
        runtime.orchestrator.set_audio_model("spectral")
        view.update_visualization()
        assert view.audio_model_selector.currentText() == "spectral"
        assert view.cochlea_image.image is None
        assert VIEWS["cochlea"](target, 16, 80) is None
    finally:
        view.close()
        runtime.stop()
