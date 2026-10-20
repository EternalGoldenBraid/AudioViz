import numpy as np
import pytest

from audioviz.readouts.shared_sensory_io import SharedSensoryIO
from audioviz.readouts.shared_predictive_coding import SharedPredictiveCodingGraph
from audioviz.utils.array_backend import to_numpy


@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_circle_routing_retains_world_predictions_and_ignores_outside_evidence(request, backend):
    xp = np if backend == "numpy" else request.getfixturevalue("cupy_backend")
    io = SharedSensoryIO((15, 21, 3), radius=2, backend=xp)
    assert len(io.offsets) == 13
    assert np.all(np.sum(io.offsets**2, axis=1) <= 4)
    world = xp.arange(15*21*3, dtype=xp.float32).reshape(15, 21, 3)
    io.set_gaze((.5, .5))
    samples = io.sample(world)
    outside_changed = world.copy()
    rows, cols = io.positions.T
    outside_changed[:] = -999
    outside_changed[xp.asarray(rows), xp.asarray(cols)] = world[xp.asarray(rows), xp.asarray(cols)]
    np.testing.assert_array_equal(to_numpy(io.sample(outside_changed)), to_numpy(samples))
    io.render(samples)
    prediction = to_numpy(io._prediction).copy()
    io.set_gaze((0, 0))
    assert io.valid.sum() == 6
    clipped = to_numpy(io.sample(world)).reshape(-1, 3)
    np.testing.assert_array_equal(clipped[~io.valid], 0)
    np.testing.assert_array_equal(clipped[io.valid], to_numpy(world)[tuple(io.positions[io.valid].T)])
    io.render(xp.zeros_like(samples))
    prediction[tuple(io.positions[io.valid].T)] = 0
    np.testing.assert_array_equal(to_numpy(io._prediction), prediction)

    gazes = [(0., 0.), (.5, .5), (1., 1.)]
    before = io.positions.copy()
    patches, masks = io.sample_candidates(world, gazes)
    np.testing.assert_array_equal(io.positions, before)
    for index, gaze in enumerate(gazes):
        positions, valid = io.route(gaze)
        np.testing.assert_array_equal(masks[index], valid)
        np.testing.assert_array_equal(to_numpy(patches[index])[valid], to_numpy(world)[tuple(positions[valid].T)])
    io.reset()
    np.testing.assert_array_equal(to_numpy(io._prediction), .5)
    np.testing.assert_array_equal(to_numpy(io._correction), 0)


def test_foveal_parameters_do_not_scale_with_world_dimensions():
    small = SharedSensoryIO((20, 30, 3), radius=8)
    large = SharedSensoryIO((1080, 1920, 3), radius=8)
    assert small.visual_shape == large.visual_shape == (197, 1)
    np.testing.assert_array_equal(small.offsets, large.offsets)
    models = [SharedPredictiveCodingGraph(visual_shape=io.visual_shape) for io in (small, large)]
    assert models[0].canvas_weights.shape == models[1].canvas_weights.shape == (8, 591)
    assert models[0].sensory_weights.shape == models[1].sensory_weights.shape == (607, 8)
    assert large._prediction.nbytes == 1080*1920*3*4
    for radius in (True, 0, 17, 1.5):
        with pytest.raises(ValueError, match="radius"):
            SharedSensoryIO((10, 10, 3), radius=radius)
