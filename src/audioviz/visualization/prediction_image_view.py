from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from audioviz.utils.source_preview import SourceScenePreview
from audioviz.visualization.ripple_renderers import _canvas_to_rgb


class PredictionImageView(QtWidgets.QWidget):
    """Host-only, fixed-scale views of the pre-evidence prediction and its target."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.panels = {}
        self.images = {}
        for key in ("canvas", "prediction", "observation"):
            plot = pg.PlotWidget()
            plot.setAspectLocked(True)
            plot.invertY(True)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            self.panels[key] = plot
            self.images[key] = image
        audio = pg.PlotWidget(title="Audio: prediction / evidence")
        audio.addLegend()
        audio.setLabel("bottom", "Spectral band")
        self.audio_prediction = audio.plot(name="Prediction", pen=(85, 200, 255))
        self.audio_observation = audio.plot(name="Evidence", pen=(255, 185, 75))
        self.panels["audio"] = audio
        self._has_audio = False
        self._layout = "comparison"
        self.set_layout("comparison")
        self.clear_preview()

    def set_layout(self, name: str) -> None:
        if name not in ("comparison", "prediction"):
            raise ValueError(f"Unsupported image layout: {name}")
        self._layout = name
        for panel in self.panels.values():
            self.grid.removeWidget(panel)
            panel.hide()
        keys = ("canvas", "prediction", "observation", "audio") if name == "comparison" else ("prediction",)
        for index, key in enumerate(keys):
            panel = self.panels[key]
            self.grid.addWidget(panel, index // 2, index % 2)
            panel.setVisible(key != "audio" or self._has_audio)
        for index in range(2):
            self.grid.setColumnStretch(index, 1 if name == "comparison" or index == 0 else 0)
            self.grid.setRowStretch(index, 1 if name == "comparison" or index == 0 else 0)

    def set_scene(self, canvas: np.ndarray, sources: tuple[SourceScenePreview, ...]) -> None:
        self._set_image("canvas", canvas, "Canvas (corrected)")
        camera = next((source for source in sources if source.mapping.key == "camera"), None)
        self._set_image(
            "prediction", camera.prediction if camera is not None else None,
            "Camera prediction (pre-evidence)",
        )
        self._set_image(
            "observation", camera.observation if camera is not None else None,
            "Camera evidence" if camera is not None and camera.enabled else "Camera evidence (disabled)",
        )
        audio = next((source for source in sources if source.mapping.key == "audio"), None)
        self._has_audio = audio is not None and audio.prediction is not None
        self.panels["audio"].setVisible(self._layout == "comparison" and self._has_audio)
        for curve, values in (
            (self.audio_prediction, audio.prediction if audio is not None else None),
            (self.audio_observation, audio.observation if audio is not None else None),
        ):
            if values is None:
                curve.clear()
            else:
                curve.setData(np.arange(len(values)), values)

    def _set_image(self, key: str, values: np.ndarray | None, title: str) -> None:
        self.panels[key].setTitle(title + (" - no current sample" if values is None else ""))
        if values is None:
            self.images[key].clear()
        else:
            self.images[key].setImage(_canvas_to_rgb(values, limit=1.0), autoLevels=False, levels=(0, 255))

    def clear_preview(self) -> None:
        for key in self.images:
            self._set_image(key, None, {
                "canvas": "Canvas (corrected)",
                "prediction": "Camera prediction (pre-evidence)",
                "observation": "Camera evidence",
            }[key])
        self.audio_prediction.clear()
        self.audio_observation.clear()
        self._has_audio = False
        self.panels["audio"].hide()
