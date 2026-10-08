"""Sensor-specific predictions derived from the shared canvas."""

from audioviz.readouts.visual import IdentityRGBReadout, VisualReadout
from audioviz.readouts.predictive_coding import (
    PredictiveCodingConfig,
    PredictiveCodingRGBReadout,
)
from audioviz.readouts.audio import PredictiveCodingAudioReadout

__all__ = [
    "IdentityRGBReadout",
    "VisualReadout",
    "PredictiveCodingConfig",
    "PredictiveCodingRGBReadout",
    "PredictiveCodingAudioReadout",
]
