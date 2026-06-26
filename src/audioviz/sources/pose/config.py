from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RipplePoseConfig:
    enabled: bool = False
    render_mode: str = "overlay"
    model_path: str | None = None
    camera_index: int = 0
    debug_view: bool = False
    graph_stiffness: float = 0.25
    field_width_fraction: float = 1.0
    field_height_fraction: float = 1.0
    body_boundary_transmission: float = 0.0
    body_boundary_dissipation: float = 1.0
