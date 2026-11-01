import numpy as np
from PyQt5 import QtCore

from audioviz.visualization import frame_rate_display
from audioviz.visualization.ripple_renderers import NumpyImageRenderer


def test_fps_uses_wall_time_and_resets_on_toggle_hide_and_idle(qapp, monkeypatch):
    now = [100.]
    monkeypatch.setattr(frame_rate_display, "monotonic", lambda: now[0])
    display = frame_rate_display.FrameRateDisplay()
    display.show()
    try:
        for _ in range(10):
            display.record_frame()
        now[0] += .5
        display._refresh()
        assert display.readout.text() == "FPS: 20.0 | 50.0 ms/frame"

        now[0] += 1
        display._refresh()
        assert display.readout.text() == "FPS: 0.0 | -- ms/frame"
        display.toggle.setChecked(False)
        assert display.readout.isHidden() and not display._timer.isActive()
        display.record_frame()
        now[0] += 30
        display.toggle.setChecked(True)
        assert display.readout.text() == "FPS: -- | -- ms/frame"
        display.record_frame()
        now[0] += .25
        display._refresh()
        assert display.readout.text() == "FPS: 4.0 | 250.0 ms/frame"

        display.hide()
        assert not display._timer.isActive()
        display.record_frame()
        now[0] += 60
        display.show()
        display.record_frame()
        now[0] += .5
        display._refresh()
        assert display.readout.text() == "FPS: 2.0 | 500.0 ms/frame"
    finally:
        display.close()
        display.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_image_renderer_counts_fresh_paints_not_updates_or_unchanged_redraws(qapp):
    from PyQt5.QtTest import QSignalSpy

    renderer = NumpyImageRenderer(rgb_canvas_enabled=True)
    renderer.widget.resize(640, 480)
    renderer.widget.show()
    qapp.processEvents()
    presentations = QSignalSpy(renderer.frame_presented)
    try:
        # Multiple submissions before a paint produce just one displayed frame.
        renderer.render(np.ones((8, 12, 3), dtype=np.float32))
        renderer.render(np.zeros((8, 12, 3), dtype=np.float32))
        assert len(presentations) == 0
        renderer.widget.viewport().repaint()
        assert len(presentations) == 1
        renderer.widget.viewport().repaint()
        assert len(presentations) == 1
        renderer.render_rgb_frame(np.ones((8, 12, 3), dtype=np.uint8))
        renderer.widget.viewport().repaint()
        assert len(presentations) == 2
    finally:
        renderer.widget.close()
        renderer.widget.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
