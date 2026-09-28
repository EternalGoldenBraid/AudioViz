import numpy as np
import pytest

from audioviz.readouts import IdentityRGBReadout


def test_identity_rgb_readout_copies_prediction_and_backprojection():
    readout = IdentityRGBReadout(canvas_shape=(2, 3, 3))
    prior = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    visual_correction = np.full((2, 3, 3), 0.25, dtype=np.float32)

    prediction = readout.predict(prior)
    canvas_correction = readout.backproject(visual_correction)

    np.testing.assert_array_equal(prediction, prior)
    np.testing.assert_array_equal(canvas_correction, visual_correction)
    assert not np.shares_memory(prediction, prior)
    assert not np.shares_memory(canvas_correction, visual_correction)


def test_identity_rgb_readout_rejects_non_rgb_canvas():
    with pytest.raises(ValueError, match="three-channel canvas"):
        IdentityRGBReadout(canvas_shape=(2, 3, 1))


def test_identity_rgb_readout_rejects_wrong_prediction_shape():
    readout = IdentityRGBReadout(canvas_shape=(2, 3, 3))

    with pytest.raises(ValueError, match="canvas prior"):
        readout.predict(np.zeros((2, 3), dtype=np.float32))
