import numpy as np
from PyQt5 import QtCore, QtWidgets

from audioviz.visualization.attention_overlay import fit_image_rect, preferred_gain_position
from audioviz.visualization.belief_diagnostics_view import BeliefDiagnosticsFrame, BeliefDiagnosticsView


def test_gain_advice_handles_ties_flat_scores_and_letterboxing():
    scores = np.zeros((3, 5), dtype=np.float32)
    scores[0, 0] = scores[2, 4] = 2
    assert preferred_gain_position(scores, (.75, .5)) == (1, 1)
    assert preferred_gain_position(scores, (1, 1)) is None
    assert preferred_gain_position(np.ones_like(scores), (.75, .5)) is None
    scores.fill(1)
    scores[0, 0] = np.nextafter(np.float32(1), np.float32(2))
    assert preferred_gain_position(scores, (1, 1)) is None
    assert fit_image_rect(800, 800, (12, 16)) == QtCore.QRectF(0, 100, 800, 600)
    assert fit_image_rect(1000, 600, (12, 16)) == QtCore.QRectF(100, 0, 800, 600)


def test_belief_view_shows_marginals_covariance_and_visible_only_gain_preview(qapp):
    scores = np.arange(15, dtype=np.float32).reshape(3, 5)
    visual = dict(
        mean=np.array([1., -2.]), variance=np.array([.25, 4.]),
        sensory_mean=np.array([.1, .2, .3]), sensory_covariance=np.eye(3)*.5,
    )
    audio = dict(
        mean=np.zeros(2), variance=np.array([.3, .6]),
        sensory_mean=np.zeros(16), sensory_covariance=np.eye(16),
    )
    frames = [BeliefDiagnosticsFrame(
        scores=scores, gaze=(.5, .5), dt=.025, sampling=False, visual=visual,
        audio=audio, audio_gain=.123, visual_location="Vision pixel (3, 4) (posterior)",
    )]
    calls = []

    def snapshot():
        calls.append(True)
        return frames[0]

    view = BeliefDiagnosticsView(snapshot)
    try:
        view.refresh()
        assert not calls
        view.show()
        qapp.processEvents()
        view.refresh_timer.stop()
        assert view.latest is frames[0]
        assert "sampling off" in view.status.text()
        assert "global sound potential EIG: 0.123 nats" in view.status.text()
        assert "posterior" in view.status.text()
        assert "7 nats" in view.gain_status.text()
        assert "grid margin: 7" in view.gain_status.text()
        assert view.tabs.isTabEnabled(2)
        marginals, hidden, sensory, mean = view.belief_tables["visual"]
        assert marginals.item(0, 1).text() == "0.5"
        assert marginals.item(1, 2).text() == "4"
        assert hidden.item(0, 1).text() == "0"
        assert sensory.rowCount() == 3 and sensory.columnCount() == 3
        assert view.belief_tables["audio"][2].rowCount() == 16
        assert view.best_marker.getData()[0].tolist() == [1.]
        frames[0] = BeliefDiagnosticsFrame(
            scores=np.ones_like(scores), gaze=(.5, .5), dt=.1, sampling=True, visual=visual,
        )
        view.refresh()
        assert not view.tabs.isTabEnabled(2)
        assert len(view.best_marker.getData()[0]) == 0
        assert "No higher-scoring" in view.candidate_status.text()
        view.hide()
        count = len(calls)
        view.refresh()
        assert len(calls) == count and not view.refresh_timer.isActive()
        assert view.latest is None and marginals.rowCount() == 0
    finally:
        view.close()
        view.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
