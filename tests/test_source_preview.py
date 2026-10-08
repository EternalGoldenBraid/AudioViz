import numpy as np
import pytest

from audioviz.sources.audio import AudioRippleSource
from audioviz.sources.camera import CameraFrameSource
from audioviz.sources.pose import PoseFrameSource
from audioviz.sources.synthetic import SyntheticRippleSource


def test_sources_declare_distinct_roles_and_bounded_native_display_mappings():
    camera = CameraFrameSource.scene_mapping
    audio = AudioRippleSource.scene_mapping
    synthetic = SyntheticRippleSource.scene_mapping
    pose = PoseFrameSource.scene_mapping
    assert (camera.representation, camera.role) == ("rgb", "sensory")
    assert (audio.representation, audio.role) == ("spectrum", "sensory")
    assert (synthetic.representation, synthetic.role) == ("signed", "drive")
    assert (pose.representation, pose.role) == ("graph", "medium")
    for mapping in (camera, synthetic):
        field = np.arange(100 * 200 * 3, dtype=np.float32).reshape(100, 200, 3)
        sampled = mapping.sample(field)
        assert sampled.shape == (32, 64, 3)
        np.testing.assert_array_equal(sampled[-1, -1], field[-1, -1])
        assert not sampled.flags.writeable
    bands = np.linspace(0, 1, 16)
    sampled = audio.sample(bands)
    assert sampled.shape == (16,)
    np.testing.assert_array_equal(sampled, bands.astype(np.float32))
    assert not sampled.flags.writeable
    assert pose.sample(np.zeros((100, 2))).shape == (64, 2)
    with pytest.raises(ValueError, match="1-64 bands"):
        audio.sample(np.ones(100))
