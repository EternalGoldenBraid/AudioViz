"""Sensor-specific predictions derived from the shared canvas."""

from audioviz.readouts.visual import IdentityRGBReadout, VisualReadout
from audioviz.readouts.predictive_coding import (
    PredictiveCodingConfig,
    PredictiveCodingRGBReadout,
)
from audioviz.readouts.audio import PredictiveCodingAudioReadout
from audioviz.readouts.gaussian_audio_uncertainty import AudioBeliefSnapshot, GaussianAudioUncertainty
from audioviz.readouts.gaussian_uncertainty import GaussianSensoryUncertainty, VisualBeliefSnapshot
from audioviz.readouts.spatial_precision import GaussianSpatialPrecision

__all__ = [
    "IdentityRGBReadout",
    "VisualReadout",
    "PredictiveCodingConfig",
    "PredictiveCodingRGBReadout",
    "PredictiveCodingAudioReadout",
    "AudioBeliefSnapshot",
    "GaussianAudioUncertainty",
    "GaussianSensoryUncertainty",
    "VisualBeliefSnapshot",
    "GaussianSpatialPrecision",
]
