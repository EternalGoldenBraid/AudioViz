from __future__ import annotations

import numpy as np


def frequency_excitation_grid(
    *,
    t: float,
    frequencies: np.ndarray,
    source_positions: np.ndarray,
    resolution: tuple[int, int],
    grid_spacing: float,
    speed: float,
    max_frequency: float,
    decay_alpha: float,
) -> np.ndarray:
    values = np.asarray(frequencies, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("frequencies must have shape (n_sources, n_frequencies)")
    n_sources, _ = values.shape

    positions = np.asarray(source_positions, dtype=np.float32)
    if positions.shape != (n_sources, 2):
        raise ValueError(
            f"Expected source_positions shape {(n_sources, 2)}, got {positions.shape}."
        )

    rows, cols = resolution
    x0 = positions[:, 0].reshape(n_sources, 1, 1)
    y0 = positions[:, 1].reshape(n_sources, 1, 1)
    xs, ys = np.meshgrid(
        np.arange(cols, dtype=np.float32),
        np.arange(rows, dtype=np.float32),
    )

    r_pixels = np.sqrt((xs[None, :, :] - x0) ** 2 + (ys[None, :, :] - y0) ** 2)
    r_meters = r_pixels * np.float32(grid_spacing)
    decay = np.exp(-np.float32(decay_alpha) * r_meters)

    clipped = np.clip(values, np.float32(1e-3), np.float32(max_frequency))
    wavelengths = np.float32(speed) / clipped
    phases = np.float32(2.0 * np.pi * t) * clipped

    r = r_meters[:, None, :, :]
    decay = decay[:, None, :, :]
    wavelengths = wavelengths[:, :, None, None]
    phases = phases[:, :, None, None]
    ripple = decay * np.sin(phases - np.float32(2.0 * np.pi) * r / wavelengths)
    return np.asarray(ripple.sum(axis=(0, 1)), dtype=np.float32)
