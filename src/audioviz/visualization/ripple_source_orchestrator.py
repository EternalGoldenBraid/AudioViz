from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResolvedSourceFrame:
    audio_frequencies: np.ndarray | None
    grid_excitation: np.ndarray | None
    amplitude: float


class RippleSourceOrchestrator:
    def __init__(
        self,
        *,
        audio_source,
        synthetic_source,
        camera_source,
        prediction_error_transform,
        engine,
        resolution: tuple[int, int],
        n_sources: int,
    ) -> None:
        self.audio_source = audio_source
        self.synthetic_source = synthetic_source
        self.camera_source = camera_source
        self.prediction_error_transform = prediction_error_transform
        self.engine = engine
        self.resolution = resolution
        self.n_sources = int(n_sources)

    def resolve(self, *, base_amplitude: float) -> ResolvedSourceFrame:
        audio_frequencies = self.audio_source.frequencies(n_sources=self.n_sources)
        grid_excitation = self.resolve_excitation_grid(
            audio_frequencies=audio_frequencies,
        )
        amplitude = self.audio_source.excitation_amplitude(
            base_amplitude=base_amplitude,
            frequencies=audio_frequencies,
            synthetic_enabled=self.synthetic_source.enabled,
        )
        return ResolvedSourceFrame(
            audio_frequencies=audio_frequencies,
            grid_excitation=grid_excitation,
            amplitude=amplitude,
        )

    def resolve_excitation_grid(
        self,
        *,
        audio_frequencies: np.ndarray | None,
    ) -> np.ndarray | None:
        next_time = self.engine.time + self.engine.dt
        grids = [
            self.synthetic_source.excitation_grid(
                t=next_time,
                source_positions=self.engine.source_positions,
                resolution=self.resolution,
                grid_spacing=self.engine.grid_spacing,
                speed=self.engine.speed,
                max_frequency=self.engine.max_frequency,
                decay_alpha=self.engine.decay_alpha,
            ),
            self.audio_source.excitation_grid(
                t=next_time,
                n_sources=self.n_sources,
                frequencies=audio_frequencies,
                source_positions=self.engine.source_positions,
                resolution=self.resolution,
                grid_spacing=self.engine.grid_spacing,
                speed=self.engine.speed,
                max_frequency=self.engine.max_frequency,
                decay_alpha=self.engine.decay_alpha,
            ),
            self.resolve_camera_frame_excitation(),
        ]
        active = [grid for grid in grids if grid is not None]
        if not active:
            return None
        return np.sum(np.stack(active, axis=0), axis=0, dtype=np.float32)

    def resolve_camera_frame_excitation(self) -> np.ndarray | None:
        excitation = self.camera_source.excitation()
        if excitation is None:
            return None
        return self.apply_source_transforms("camera_frame", excitation)

    def apply_source_transforms(
        self,
        source_key: str,
        excitation: np.ndarray,
    ) -> np.ndarray:
        return self.prediction_error_transform.apply(
            source_key=source_key,
            observation=excitation,
            ripple_state=self.engine.get_field_numpy(),
        )
