from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from audioviz.sources.pose import pose_coords_in_image_support


MASK_COLOR = (255.0, 64.0, 208.0)
MASK_ALPHA = 0.35
MASK_OUTLINE_COLOR = (255, 255, 255)


class PoseDebugView:
    def __init__(self) -> None:
        self.frame_count = 0
        self.widget = pg.GraphicsLayoutWidget()
        plot = self.widget.addPlot(row=0, col=0)
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.invertY(True)

        self.image = pg.ImageItem(axisOrder="row-major")
        self.edges = pg.PlotDataItem(
            pen=pg.mkPen((255, 180, 40), width=2),
            connect="finite",
        )
        self.points = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(40, 180, 255),
            pen=pg.mkPen(255, 255, 255, width=1),
        )
        plot.addItem(self.image)
        plot.addItem(self.edges)
        plot.addItem(self.points)

    def update(
        self,
        frame: np.ndarray,
        pose,
        *,
        segmentation_mask: np.ndarray | None = None,
    ) -> None:
        rgb_frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
        if segmentation_mask is None:
            segmentation_mask = pose.segmentation_mask
        if segmentation_mask is not None:
            rgb_frame = overlay_pose_debug_mask(rgb_frame, segmentation_mask)
        self.image.setImage(rgb_frame, autoLevels=False)
        self.frame_count += 1

        height, width = frame.shape[:2]
        if not pose.coords.size:
            self._clear_pose_graph()
            return

        valid = pose_coords_in_image_support(pose.coords)
        if not np.any(valid):
            self._clear_pose_graph()
            return

        coords_px = pose.coords * np.array([width - 1, height - 1], dtype=np.float32)
        coords_px[:, 0] = (width - 1) - coords_px[:, 0]
        edge_xs = []
        edge_ys = []
        for i, j in np.argwhere(np.triu(pose.adjacency, k=1) > 0):
            if not (valid[i] and valid[j]):
                continue
            edge_xs.extend([coords_px[i, 0], coords_px[j, 0], np.nan])
            edge_ys.extend([coords_px[i, 1], coords_px[j, 1], np.nan])

        self.edges.setData(edge_xs, edge_ys)
        self.points.setData(coords_px[valid, 0], coords_px[valid, 1])

    def _clear_pose_graph(self) -> None:
        self.edges.setData([], [])
        self.points.setData([], [])


def overlay_pose_debug_mask(
    rgb_frame: np.ndarray,
    segmentation_mask: np.ndarray,
) -> np.ndarray:
    height, width = rgb_frame.shape[:2]
    mirrored_mask = mirrored_pose_debug_mask(
        segmentation_mask,
        height=height,
        width=width,
    )
    if not np.any(mirrored_mask):
        return rgb_frame

    overlay = rgb_frame.astype(np.float32, copy=True)
    tint = np.asarray(MASK_COLOR, dtype=np.float32)
    alpha = np.float32(MASK_ALPHA)
    overlay[mirrored_mask] = (
        overlay[mirrored_mask] * (np.float32(1.0) - alpha)
        + tint * alpha
    )
    outline = pose_debug_mask_outline(mirrored_mask)
    overlay[outline] = np.asarray(MASK_OUTLINE_COLOR, dtype=np.float32)
    return np.ascontiguousarray(np.rint(overlay).astype(np.uint8))


def mirrored_pose_debug_mask(
    segmentation_mask: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.asarray(segmentation_mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError("segmentation_mask must have shape (rows, cols)")

    row_index = np.rint(
        np.linspace(0, mask.shape[0] - 1, height, dtype=np.float32)
    ).astype(np.int32)
    col_index = np.rint(
        np.linspace(0, mask.shape[1] - 1, width, dtype=np.float32)
    ).astype(np.int32)
    return mask[row_index][:, col_index][:, ::-1] >= np.float32(0.5)


def pose_debug_mask_outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors_same = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~neighbors_same
