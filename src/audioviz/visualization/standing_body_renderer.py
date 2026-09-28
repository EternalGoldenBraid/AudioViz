from __future__ import annotations

import numpy as np


SILHOUETTE_COLOR = np.array([220, 220, 230], dtype=np.uint8)
GRAPH_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def lookup_table_from_renderer(renderer) -> np.ndarray | None:
    renderer_lookup = getattr(renderer, "lookup_table", None)
    if renderer_lookup is not None:
        lookup = np.asarray(renderer_lookup, dtype=np.uint8)
        if lookup.ndim == 2 and lookup.shape[0] > 0:
            return lookup
    image_item = getattr(renderer, "image_item", None)
    if image_item is None:
        return None
    lookup = getattr(image_item, "lut", None)
    if callable(lookup):
        try:
            lookup = lookup()
        except AttributeError:
            return None
    if lookup is None:
        return None
    lookup = np.asarray(lookup, dtype=np.uint8)
    if lookup.ndim != 2 or lookup.shape[0] == 0:
        return None
    return lookup


class StandingBodyRenderer:
    def render(
        self,
        *,
        field: np.ndarray,
        lookup_table: np.ndarray | None,
        pose_coords: np.ndarray,
        pose_adjacency: np.ndarray,
        segmentation_mask: np.ndarray | None,
    ) -> np.ndarray:
        floor_rgb = self._field_to_floor_rgb(field, lookup_table=lookup_table)
        frame = self._project_floor_to_perspective(floor_rgb)
        if pose_coords.size == 0:
            return frame
        silhouette_mask, bbox = self._silhouette_mask_and_bbox(segmentation_mask)
        if silhouette_mask is None or bbox is None or segmentation_mask is None:
            return frame
        return self._overlay_body(
            frame,
            silhouette_mask=silhouette_mask,
            bbox=bbox,
            coords=pose_coords,
            adjacency=pose_adjacency,
            segmentation_shape=segmentation_mask.shape,
        )

    @staticmethod
    def _field_to_floor_rgb(
        field: np.ndarray,
        *,
        lookup_table: np.ndarray | None,
    ) -> np.ndarray:
        values = np.asarray(field, dtype=np.float32)
        limit = np.max(np.abs(values))
        limit = max(float(limit), 1e-6)
        if values.ndim == 3:
            if values.shape[2] != 3:
                raise ValueError("canvas field must have exactly three channels")
            normalized = np.clip(values / limit, 0.0, 1.0)
            return np.ascontiguousarray(np.rint(normalized * 255.0).astype(np.uint8))
        normalized = np.clip((values / limit + 1.0) * 0.5, 0.0, 1.0)
        if lookup_table is None:
            gray = np.rint(normalized * 255.0).astype(np.uint8)
            return np.repeat(gray[..., None], 3, axis=2)
        lookup = np.asarray(lookup_table, dtype=np.uint8)
        index = np.rint(normalized * (len(lookup) - 1)).astype(np.int32)
        return np.ascontiguousarray(lookup[index])

    @staticmethod
    def _project_floor_to_perspective(floor_rgb: np.ndarray) -> np.ndarray:
        rows, cols = floor_rgb.shape[:2]
        output = np.zeros_like(floor_rgb)
        horizon = max(1, int(round(rows * 0.18)))
        center_x = (cols - 1) * 0.5
        height = max(rows - 1 - horizon, 1)
        ys, xs = np.indices((rows, cols), dtype=np.float32)
        depth = np.clip((ys - horizon) / height, 0.0, 1.0)
        src_y = np.rint((depth ** 1.2) * (rows - 1)).astype(np.int32)
        width_scale = 0.35 + 0.65 * (depth ** 1.2)
        src_x = ((xs - center_x) / np.maximum(width_scale, 1e-6)) + center_x
        src_x = np.rint(src_x).astype(np.int32)
        valid = (ys >= horizon) & (src_x >= 0) & (src_x < cols)
        output[valid] = floor_rgb[src_y[valid], src_x[valid]]
        return output

    @staticmethod
    def _silhouette_mask_and_bbox(
        segmentation_mask: np.ndarray | None,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        if segmentation_mask is None:
            return None, None
        mask = np.asarray(segmentation_mask, dtype=np.float32) >= 0.5
        if not np.any(mask):
            return None, None
        mask = mask[:, ::-1]
        ys, xs = np.nonzero(mask)
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        left = int(xs.min())
        right = int(xs.max()) + 1
        cropped = mask[top:bottom, left:right]
        if cropped.size == 0:
            return None, None
        return cropped, (top, left, bottom, right)

    def _overlay_body(
        self,
        frame: np.ndarray,
        *,
        silhouette_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        coords: np.ndarray,
        adjacency: np.ndarray,
        segmentation_shape: tuple[int, ...],
    ) -> np.ndarray:
        output = frame.copy()
        rows, cols = output.shape[:2]
        top, left, bottom, right = bbox
        mask_height = max(bottom - top, 1)
        mask_width = max(right - left, 1)
        mirrored_coords = np.asarray(coords, dtype=np.float32).copy()
        mirrored_coords[:, 0] = 1.0 - mirrored_coords[:, 0]
        base_y_norm = float(np.clip(np.max(mirrored_coords[:, 1]), 0.0, 1.0))
        anchor_x_norm = float(np.clip(np.mean(mirrored_coords[:, 0]), 0.0, 1.0))
        anchor_screen_x, anchor_screen_y = self._project_floor_point(
            anchor_x_norm * max(cols - 1, 1),
            base_y_norm * max(rows - 1, 1),
            rows=rows,
            cols=cols,
        )
        scale = 0.35 + 0.65 * base_y_norm
        body_height = max(16, int(round(mask_height * scale * 1.2)))
        body_width = max(8, int(round(body_height * (mask_width / max(mask_height, 1)))))
        body_height = min(body_height, rows)
        body_width = min(body_width, cols)
        top_screen = max(0, min(anchor_screen_y - body_height, rows - body_height))
        left_screen = int(np.clip(anchor_screen_x - body_width // 2, 0, max(cols - body_width, 0)))
        silhouette = self._resize_bool_mask(silhouette_mask, body_height, body_width)
        body_slice = output[top_screen:top_screen + body_height, left_screen:left_screen + body_width]
        body_slice[silhouette] = self._blend_rgb(
            body_slice[silhouette],
            SILHOUETTE_COLOR,
            alpha=0.55,
        )

        bbox_width_norm = max((right - left) / max(segmentation_shape[1], 1), 1e-6)
        bbox_height_norm = max((bottom - top) / max(segmentation_shape[0], 1), 1e-6)
        left_norm = left / max(segmentation_shape[1], 1)
        top_norm = top / max(segmentation_shape[0], 1)
        projected_points: list[tuple[int, int] | None] = []
        for x_norm, y_norm in mirrored_coords:
            if not np.isfinite(x_norm) or not np.isfinite(y_norm):
                projected_points.append(None)
                continue
            local_x = (x_norm - left_norm) / bbox_width_norm
            local_y = (y_norm - top_norm) / bbox_height_norm
            if not (0.0 <= local_x <= 1.0 and 0.0 <= local_y <= 1.0):
                projected_points.append(None)
                continue
            point_x = left_screen + int(round(local_x * max(body_width - 1, 0)))
            point_y = top_screen + int(round(local_y * max(body_height - 1, 0)))
            projected_points.append((point_x, point_y))

        for start, end in np.argwhere(np.triu(np.asarray(adjacency, dtype=np.float32), k=1) > 0):
            point_a = projected_points[int(start)]
            point_b = projected_points[int(end)]
            if point_a is None or point_b is None:
                continue
            self._draw_line(output, point_a, point_b, GRAPH_COLOR)
        for point in projected_points:
            if point is None:
                continue
            self._draw_disk(output, point, radius=2, color=GRAPH_COLOR)
        return output

    @staticmethod
    def _project_floor_point(
        source_x: float,
        source_y: float,
        *,
        rows: int,
        cols: int,
    ) -> tuple[int, int]:
        horizon = max(1, int(round(rows * 0.18)))
        depth = np.clip(float(source_y) / max(rows - 1, 1), 0.0, 1.0)
        screen_y = horizon + int(round((depth ** (1.0 / 1.2)) * max(rows - 1 - horizon, 1)))
        width_scale = 0.35 + 0.65 * depth
        center_x = (cols - 1) * 0.5
        screen_x = center_x + (float(source_x) - center_x) * width_scale
        return int(round(screen_x)), int(round(screen_y))

    @staticmethod
    def _resize_bool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        row_index = np.rint(
            np.linspace(0, mask.shape[0] - 1, height, dtype=np.float32)
        ).astype(np.int32)
        col_index = np.rint(
            np.linspace(0, mask.shape[1] - 1, width, dtype=np.float32)
        ).astype(np.int32)
        return mask[row_index][:, col_index]

    @staticmethod
    def _blend_rgb(base: np.ndarray, color: np.ndarray, *, alpha: float) -> np.ndarray:
        blended = (1.0 - alpha) * base.astype(np.float32) + alpha * color.astype(np.float32)
        return np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)

    @staticmethod
    def _draw_line(
        frame: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: np.ndarray,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
        ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)
        rows, cols = frame.shape[:2]
        valid = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
        frame[ys[valid], xs[valid]] = color

    @staticmethod
    def _draw_disk(
        frame: np.ndarray,
        center: tuple[int, int],
        *,
        radius: int,
        color: np.ndarray,
    ) -> None:
        cx, cy = center
        rows, cols = frame.shape[:2]
        y0 = max(cy - radius, 0)
        y1 = min(cy + radius + 1, rows)
        x0 = max(cx - radius, 0)
        x1 = min(cx + radius + 1, cols)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        frame[y0:y1, x0:x1][mask] = color
