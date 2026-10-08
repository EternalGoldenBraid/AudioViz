import numpy as np

from audioviz.engine import RippleEngine
from audioviz.utils.spatial_preview import preview_indices, spatial_preview


def test_preview_samples_edges_preserves_aspect_and_owns_readonly_pixels():
    field = np.arange(100 * 200 * 3, dtype=np.float32).reshape(100, 200, 3)
    preview = spatial_preview(field)

    assert preview.shape == (32, 64, 3)
    np.testing.assert_array_equal(preview[0, 0], field[0, 0])
    np.testing.assert_array_equal(preview[-1, -1], field[-1, -1])
    rows, columns = preview_indices(field.shape)
    np.testing.assert_array_equal(preview, field[rows[:, None], columns[None, :]])
    assert not preview.flags.writeable
    assert not np.shares_memory(preview, field)
    small = spatial_preview(field[:1, :3])
    assert small.shape == (1, 3, 3)
    assert not np.shares_memory(small, field)


def test_engine_preview_samples_corrected_field_before_gpu_host_transfer():
    engine = RippleEngine(resolution=(100, 200), plane_size_m=(1, 1), speed=1)
    engine.Z[:] = np.random.default_rng(3).normal(size=engine.Z.shape)
    expected = spatial_preview(engine.get_field_numpy())
    transfers = []

    class ArrayBackend:
        @staticmethod
        def asnumpy(values):
            transfers.append(values.shape)
            return np.asarray(values)

    engine.use_gpu = True
    engine.backend = ArrayBackend()
    preview = engine.get_field_preview()

    np.testing.assert_array_equal(preview, expected)
    assert transfers == [(32, 64, 3)]
    assert not preview.flags.writeable
    engine.Z.fill(0)
    assert np.any(preview != 0)
