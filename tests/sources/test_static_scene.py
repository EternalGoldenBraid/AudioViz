import numpy as np
import pytest

from audioviz.sources.camera import CameraFrameSource
from audioviz.sources.static_scene import geometric_scene, StaticSceneCapture


def test_static_scene_repeats_exact_rgb_evidence_without_caller_mutation_or_hardware():
    scene = geometric_scene((40, 60))
    np.testing.assert_array_equal(scene, geometric_scene((40, 60)))
    assert scene.dtype == np.uint8 and not scene.flags.writeable
    assert len(np.unique(scene.reshape(-1, 3), axis=0)) == 4
    supplied = scene.copy()
    capture = StaticSceneCapture(supplied)
    supplied[:] = 255
    for _ in range(3):
        ok, frame = capture.read()
        assert ok and not frame.flags.writeable and frame.flags.c_contiguous
        np.testing.assert_array_equal(frame, scene[..., ::-1])
        actual = CameraFrameSource.frame_to_excitation_grid(frame, resolution=(40, 60), gain=1)
        np.testing.assert_array_equal(actual, scene.astype(np.float32)/255)
    assert capture.read_count == 3
    capture.release()
    assert capture.read() == (False, None)
    assert capture.read_count == 3


def test_static_scene_rejects_invalid_dimensions_and_non_rgb_data():
    for resolution in ((0, 10), (True, 10), (1.5, 10)):
        with pytest.raises(ValueError, match="resolution"):
            geometric_scene(resolution)
    for image in (np.zeros((3, 4)), np.zeros((3, 4, 3)), np.zeros((0, 4, 3), np.uint8)):
        with pytest.raises(ValueError, match="uint8 RGB"):
            StaticSceneCapture(image)
