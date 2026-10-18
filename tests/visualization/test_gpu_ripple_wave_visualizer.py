from types import SimpleNamespace

import numpy as np
import pytest

from audioviz.readouts import PredictiveCodingConfig
from audioviz.sources import (
    AudioSourceConfig,
    CameraFrameSourceConfig,
    RippleSourceOrchestratorConfig,
)
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig


@pytest.mark.parametrize("with_audio", (False, True))
def test_gpu_visualizer_renders_host_images_and_previews_then_resets(
    cupy_backend, layer_view_stub, with_audio
):
    from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

    cp = cupy_backend

    class Capture:
        released = False

        def read(self):
            return True, np.full((10, 12, 3), 180, dtype=np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    processor = SimpleNamespace(
        frame_counter=1, n_mels=None, stft_window=np.ones(16),
        spectrogram_buffers=[np.ones((16, 1), dtype=np.float32)],
    ) if with_audio else None
    widget = RippleWaveVisualizer(
        processor=processor, use_gpu=True, resolution=(8, 12),
        plane_size_m=(1, 1), rgb_canvas_enabled=True,
        source_orchestrator_config=RippleSourceOrchestratorConfig(
            audio=AudioSourceConfig(enabled=with_audio),
            camera_frame=CameraFrameSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(
                learning_enabled=True, cross_modal_enabled=with_audio,
            ),
        ),
        camera_capture=capture,
    )
    widget.timer.stop()
    try:
        widget.set_prediction_diagnostics_visible(True)
        widget.set_prediction_scene_visible(True)
        widget.update_visualization()
        widget.prediction_diagnostics.refresh()
        assert isinstance(widget.visual_prediction, cp.ndarray)
        assert isinstance(widget.renderer.image_item.image, np.ndarray)
        assert widget.renderer.image_item.image.dtype == np.uint8
        assert widget.prediction_diagnostics.latest is not None
        canvas, sources = widget.prediction_scene.layer_view.preview
        assert isinstance(canvas, np.ndarray)
        for source in sources:
            for value in (source.observation, source.prediction, source.hidden):
                if value is not None:
                    assert isinstance(value, np.ndarray)
                    assert not value.flags.writeable
        if with_audio:
            widget.prediction_diagnostics.audio_view.refresh()
            assert widget.prediction_diagnostics.audio_view.latest is not None
        widget.engine.reset()
        widget._sync_after_reset()
        assert isinstance(widget.visual_prediction, cp.ndarray)
        assert bool(cp.all(widget.visual_prediction == 0))
        assert np.all(widget.engine.get_field_numpy() == 0)
        assert widget.prediction_scene.canvas is None
    finally:
        widget.close()
    assert capture.released
