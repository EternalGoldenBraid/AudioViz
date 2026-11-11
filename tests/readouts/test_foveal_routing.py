import numpy as np
import pytest

from audioviz.readouts.foveal_routing import WorldAnchoredFovea


def test_moving_disk_keeps_world_addresses_and_clips_at_borders():
    fovea = WorldAnchoredFovea((15, 21), radius=2)
    positions = fovea.positions.copy()
    assert fovea.observation_mask.sum() == 13
    np.testing.assert_array_equal(fovea.center, [7, 10])
    for gaze, center in (((0., 0.), [0, 0]), ((1., 1.), [14, 20])):
        fovea.set_gaze(gaze)
        np.testing.assert_array_equal(fovea.positions, positions)
        np.testing.assert_array_equal(fovea.center, center)
        assert fovea.observation_mask.sum() == 6
        assert fovea.observation_mask[tuple(center)]
    for gaze in ((np.nan, .5), (-.1, .5), (.5,), (.5, .5, .5)):
        with pytest.raises(ValueError, match="gaze"):
            fovea.set_gaze(gaze)
    for radius in (True, 0, 37, 1.5):
        with pytest.raises(ValueError, match="radius"):
            WorldAnchoredFovea((15, 21), radius=radius)
