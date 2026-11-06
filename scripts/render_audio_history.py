"""Render the real Qt audio monitor using synthetic input, without audio devices."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PyQt5 import QtWidgets

from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/cochlea/audio-history.png"))
    args = parser.parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    pg.setConfigOptions(background="#111827", foreground="#cbd5e1")
    runtime = GraphRuntime(resolution=8, camera=False, audio_model="cochlea")
    processor = runtime.processor
    view = SpectrogramVisualizer(processor, colormaps["viridis"], Normalize(-80, 0), .1)
    view.timer.stop()
    try:
        t = np.arange(91 * 1024) / processor.sr
        frequencies = processor.cochlea.frequencies
        samples = np.zeros_like(t)
        for start, end, frequency in ((.4, 1.3, frequencies[4]), (1.5, 2.4, frequencies[8]), (2.6, 3.9, frequencies[12])):
            mask = (t >= start) & (t < end)
            envelope = np.minimum(1, (t[mask] - start) / .04) * np.minimum(1, (end - t[mask]) / .08)
            samples[mask] += .3 * envelope * np.sin(2 * np.pi * frequency * t[mask])
        for block in samples.reshape(-1, 1024):
            processor.process_audio(block[:, None])
        view.resize(1250, 940)
        view.show()
        app.processEvents()
        view.update_visualization()
        app.processEvents()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not view.grab().save(str(args.output)):
            raise RuntimeError(f"Could not save {args.output}")
    finally:
        view.close()
        runtime.stop()


if __name__ == "__main__":
    main()
