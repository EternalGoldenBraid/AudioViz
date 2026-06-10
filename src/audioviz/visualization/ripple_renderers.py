from __future__ import annotations

from typing import Protocol

import matplotlib.cm as cm
import numpy as np
import pyqtgraph as pg


class RippleFieldSource(Protocol):
    def get_field_numpy(self) -> np.ndarray:
        """Return the current ripple field as a NumPy array."""


class NumpyImageRenderer:
    """Render ripple fields through PyQtGraph from a NumPy array."""

    def __init__(
        self,
        *,
        title: str = "Ripple Simulation",
        colormap_name: str = "inferno",
    ):
        self.image_item = pg.ImageItem()
        colormap = cm.get_cmap(colormap_name)
        lookup_table = (colormap(np.linspace(0, 1, 256))[:, :3] * 255).astype(
            np.uint8
        )
        self.image_item.setLookupTable(lookup_table)

        self.plot = pg.PlotItem()
        self.plot.setTitle(title)
        self.plot.addItem(self.image_item)

        self.widget = pg.GraphicsLayoutWidget()
        self.widget.addItem(self.plot)

    def render(self, field_source: RippleFieldSource) -> None:
        field = field_source.get_field_numpy()
        max_abs = np.max(np.abs(field))
        self.image_item.setLevels([-max_abs, max_abs])
        self.image_item.setImage(field, autoLevels=False)
