"""Input source helpers."""

from audioviz.sources.camera import CameraFrameSource, load_cv2
from audioviz.sources.audio import AudioRippleSource
from audioviz.sources.synthetic import SyntheticRippleSource

__all__ = [
    "AudioRippleSource",
    "CameraFrameSource",
    "SyntheticRippleSource",
    "load_cv2",
]
