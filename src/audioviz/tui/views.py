"""Preset subgraph views: pure projections of a GraphRuntime onto RGB images.

Each view takes (runtime, rows, cols) in pixel units and returns an (h, w, 3) uint8
image or None while its node has no data. Tiles resample the result to their cells.
"""

from typing import Callable

import numpy as np

from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.halfblock import field_to_rgb, waveform_to_rgb
from audioviz.visualization.audio_history_view import (
    COCHLEA_COLORMAP, COCHLEA_DB_RANGE, cochlea_db, level_rgb,
    SPECTRUM_COLORMAP, SPECTRUM_DB_RANGE, spectrum_db,
    waveform_envelope,
)

View = Callable[[GraphRuntime, int, int], "np.ndarray | None"]

WAVEFORM_PEAK_FLOOR = 0.02  # keeps room noise from filling the tile
BACKGROUND = np.array([12, 12, 16], dtype=np.uint8)
OBSERVED = np.array([40, 90, 120], dtype=np.uint8)
PREDICTED = np.array([250, 150, 30], dtype=np.uint8)
COCHLEA_PREVIEW_SAMPLES = 128


def cochlea(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    history = runtime.processor.get_audio_history()
    if history is None:
        return None
    levels = cochlea_db(history, cols)
    if levels is None or history.cochlea.samples.size == 0:
        return None
    return level_rgb(levels[::-1], COCHLEA_DB_RANGE, COCHLEA_COLORMAP)


def waveform(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    history = runtime.processor.get_audio_history()
    if history is None:
        return None
    _, low, high = waveform_envelope(history, cols)
    bottom = (1 - np.clip(low, -1, 1)) / 2 * (rows - 1)
    top = (1 - np.clip(high, -1, 1)) / 2 * (rows - 1)
    height = np.arange(rows)[:, None]
    image = np.tile(BACKGROUND, (rows, len(low), 1))
    image[rows // 2] = (50, 60, 70)
    image[(height >= np.floor(top)) & (height <= np.floor(bottom))] = (90, 220, 140)
    return image


def spectrogram(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    history = runtime.processor.get_audio_history()
    if history is None:
        return None
    return level_rgb(spectrum_db(history, cols)[::-1], SPECTRUM_DB_RANGE, SPECTRUM_COLORMAP)


def camera(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    return runtime.camera.rgb() if runtime.show_camera else None  # None while the camera is off or has not delivered a frame


def sensor_view(sensor_id: str) -> View:
    """The latest observation of one orchestrator sensor, as the image the graph received (gain included)."""
    def view(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
        sensor = runtime.orchestrator.sensors.get(sensor_id)
        observation = None if sensor is None else sensor.observation
        return None if observation is None else (np.clip(observation, 0.0, 1.0) * 255).astype(np.uint8)
    return view


def canvas(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray:
    """The corrected canvas: signed RGB field, auto-scaled so faint activity stays visible."""
    field = runtime.engine.get_field_numpy()
    scale = max(float(np.abs(field).max()), 1e-6)
    return ((0.5 + 0.5 * np.clip(field / scale, -1.0, 1.0)) * 255).astype(np.uint8)


def _band_index(bands: int, cols: int) -> np.ndarray:
    return np.minimum(np.arange(cols) * bands // cols, bands - 1)


def prediction(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    """Spectral bars, or raw audio (blue) and its causal cochlea prediction (orange)."""
    cochlea = runtime.processor.cochlea
    if cochlea is not None:
        block = cochlea.last_block
        if block is None or not runtime.orchestrator.audio_source.has_recent_observation():
            return None
        predicted = block.prediction[-COCHLEA_PREVIEW_SAMPLES:]
        error = block.error[-COCHLEA_PREVIEW_SAMPLES:]
        observed = predicted + error
        scale = max(float(np.abs(observed).max()), float(np.abs(predicted).max()),
                    float(np.abs(error).max()), WAVEFORM_PEAK_FLOOR)
        image = np.tile(BACKGROUND, (rows, cols, 1))
        image[waveform_to_rgb(observed / scale, rows, cols).any(axis=2)] = OBSERVED
        image[waveform_to_rgb(predicted / scale, rows, cols).any(axis=2)] = PREDICTED
        return image
    observed = runtime.orchestrator.audio_source.recent_observation()
    predicted = runtime.orchestrator.audio_prediction
    if observed is None or predicted is None:
        return None
    top = max(float(max(observed.max(), predicted.max())), 1e-6)
    index = _band_index(len(observed), cols)
    height = np.arange(rows)[:, None]
    obs_rows = ((1 - np.clip(observed[index] / top, 0, 1)) * (rows - 1)).astype(int)
    pred_rows = ((1 - np.clip(predicted[index] / top, 0, 1)) * (rows - 1)).astype(int)
    image = np.tile(BACKGROUND, (rows, cols, 1))
    image[height >= obs_rows[None, :]] = OBSERVED
    image[np.abs(height - pred_rows[None, :]) <= 0] = PREDICTED
    return image


def prediction_error(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    """Signed spectral error, or the cochlea's sample-aligned raw waveform residual."""
    cochlea = runtime.processor.cochlea
    if cochlea is not None:
        block = cochlea.last_block
        if block is None or not runtime.orchestrator.audio_source.has_recent_observation():
            return None
        predicted = block.prediction[-COCHLEA_PREVIEW_SAMPLES:]
        error = block.error[-COCHLEA_PREVIEW_SAMPLES:]
        raw = predicted + error
        scale = max(float(np.abs(raw).max()), float(np.abs(predicted).max()),
                    float(np.abs(error).max()), WAVEFORM_PEAK_FLOOR)
        image = np.tile(BACKGROUND, (rows, cols, 1))
        image[waveform_to_rgb(error / scale, rows, cols).any(axis=2)] = (220, 90, 100)
        image[rows // 2] = (70, 70, 80)
        return image
    observed = runtime.orchestrator.audio_source.recent_observation()
    predicted = runtime.orchestrator.audio_prediction
    if observed is None or predicted is None:
        return None
    error = (observed - predicted)[_band_index(len(observed), cols)]
    top = max(float(np.abs(error).max()), 1e-6)
    middle = (rows - 1) / 2
    reach = np.round(error / top * middle).astype(int)
    height = np.arange(rows)[:, None]
    image = np.tile(BACKGROUND, (rows, cols, 1))
    up = (height <= middle) & (height >= middle - reach[None, :]) & (reach[None, :] > 0)
    down = (height >= middle) & (height <= middle - reach[None, :]) & (reach[None, :] < 0)
    image[up] = (255, 140, 40)
    image[down] = (50, 120, 255)
    image[int(middle), :] = (70, 70, 80)
    return image


def _signed_rgb(field: np.ndarray) -> np.ndarray:
    scale = max(float(np.abs(field).max()), 1e-6)
    return ((0.5 + 0.5 * np.clip(field / scale, -1.0, 1.0)) * 255).astype(np.uint8)


def visual_prediction(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    """The canvas the model expects, as signed RGB auto-scaled like the canvas tile."""
    predicted = runtime.visual_prediction
    return None if predicted is None else _signed_rgb(predicted)


def visual_error(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    """Camera observation minus visual prediction; mid-grey while the canvas has no camera feed."""
    observed = runtime.orchestrator.visual_observation
    predicted = runtime.visual_prediction
    if predicted is None:
        return None
    return _signed_rgb(np.zeros_like(predicted) if observed is None else np.asarray(observed) - predicted)


def hidden(runtime: GraphRuntime, rows: int, cols: int) -> np.ndarray | None:
    """Hidden units as rows: evidence->hidden weights | activity | hidden->canvas weights."""
    readout = runtime.audio_readout
    diagnostics = None if readout is None else readout.diagnostics
    if diagnostics is None:
        return None
    activity = np.asarray(diagnostics.hidden_means, dtype=np.float32)[:, None]
    panels = (np.asarray(diagnostics.visual_weights).T, activity, np.asarray(diagnostics.canvas_weights))
    gap = np.zeros((activity.shape[0], 1, 3), dtype=np.uint8)
    parts = []
    for panel in panels:
        scale = max(float(np.abs(panel).max()), 1e-6)
        parts += [field_to_rgb(panel, scale), gap]
    return np.concatenate(parts[:-1], axis=1)


VIEWS: dict[str, View] = {
    "cochlea": cochlea,
    "waveform": waveform,
    "spectrogram": spectrogram,
    "canvas": canvas,
    "camera_1": camera,
    "camera_2": sensor_view("camera 2"),
    "prediction": prediction,
    "error": prediction_error,
    "visual_prediction": visual_prediction,
    "visual_error": visual_error,
    "hidden": hidden,
}
