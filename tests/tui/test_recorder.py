import numpy as np

from audioviz.tui.recorder import CELL_PX, RecordedTile, Recorder, compose


def solid(value, shape=(4, 4)):
    return np.full((*shape, 3), value, dtype=np.uint8)


def test_tiles_land_where_the_cells_say_and_later_ones_cover_earlier():
    cell_h, cell_w = CELL_PX
    tiles = [RecordedTile(2, 1, 4, 2, solid(10)), RecordedTile(4, 1, 4, 2, solid(200)), RecordedTile(2, 3, 2, 1, solid(90))]
    frame = compose(tiles)
    assert frame.shape == (3 * cell_h, 6 * cell_w, 3)
    assert frame[0, 0, 0] == 10 and frame[0, 2 * cell_w, 0] == 200  # the second covers the first from column 4
    assert frame[2 * cell_h, 0, 0] == 90


def test_a_recording_keeps_its_first_size_when_the_layout_changes(tmp_path):
    import cv2

    recorder = Recorder(tmp_path / "r.mp4", 10)
    for width in (6, 4, 8):
        recorder.capture([RecordedTile(0, 0, width, 3, solid(120))])
    recorder.stop()
    video = cv2.VideoCapture(str(tmp_path / "r.mp4"))
    assert recorder.frames == 3 and int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) == 6 * CELL_PX[1]


def test_a_named_tile_gets_a_label_over_its_corner():
    plain = compose([RecordedTile(0, 0, 10, 4, solid(200))])
    labelled = compose([RecordedTile(0, 0, 10, 4, solid(200), name="canvas")])
    assert (labelled[:8, :8] != plain[:8, :8]).any() and (labelled[-8:, -8:] == plain[-8:, -8:]).all()
