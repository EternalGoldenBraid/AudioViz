import numpy as np

from audioviz.engine import RippleEngine
from audioviz.sources import (
    AudioSourceConfig,
    CameraFrameSourceConfig,
    RippleSourceOrchestrator,
    RippleSourceOrchestratorConfig,
    SyntheticSourceConfig,
)
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig


def _orchestrator() -> RippleSourceOrchestrator:
    engine = RippleEngine(
        resolution=(12, 16), plane_size_m=(1.0, 1.0), speed=1.0, damping=1.0, use_gpu=False,
    )
    orchestrator = RippleSourceOrchestrator(
        config=RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False),
            audio=AudioSourceConfig(enabled=False),
            camera_frame=CameraFrameSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True, inputs=("camera_frame",)),
        ),
        processor=None, engine=engine, resolution=engine.resolution, n_sources=1,
    )
    orchestrator.camera_source.capture = object()  # keeps ensure_open from touching a device
    orchestrator.camera_source.excitation = lambda: None  # camera 1 is covered: no evidence
    return orchestrator


def _step(orchestrator: RippleSourceOrchestrator) -> np.ndarray:
    orchestrator.engine.propagate(None)
    orchestrator.predict_visual()
    return orchestrator.correct_from_observations()


def test_a_second_camera_corrects_the_canvas_when_the_first_gives_no_evidence():
    image = np.zeros((12, 16, 3), dtype=np.float32)
    image[:, 8:] = 1.0

    alone = _orchestrator()
    baseline = _step(alone)

    both = _orchestrator()
    camera_2 = both.sensors["camera 2"]
    camera_2.enabled = True
    camera_2.excitation = lambda: image
    correction = _step(both)

    assert both.sensor_predictions["camera 2"].shape == both.canvas_shape
    assert both.sensor_readouts["camera 2"].visual_error.any()  # camera 2's own readout saw the frame
    assert not np.array_equal(correction, baseline)
    assert np.abs(correction).max() > 0


def test_a_disabled_or_covered_second_camera_adds_no_branch():
    orchestrator = _orchestrator()
    assert orchestrator._extra_camera_branches() == []
    camera_2 = orchestrator.sensors["camera 2"]
    camera_2.enabled = True
    camera_2.excitation = lambda: None
    _step(orchestrator)
    assert orchestrator._extra_camera_branches() == []
    assert not orchestrator.sensor_readouts["camera 2"].visual_error.any()


def test_each_camera_learns_its_own_readout():
    orchestrator = _orchestrator()
    orchestrator.update_pathway_config(learning_enabled=True)  # also reaches camera 2's readout
    camera_2 = orchestrator.sensors["camera 2"]
    camera_2.enabled = True
    camera_2.excitation = lambda: np.ones((12, 16, 3), dtype=np.float32)
    first = orchestrator.camera_source
    first.excitation = lambda: np.zeros((12, 16, 3), dtype=np.float32)
    before = orchestrator.sensor_readouts["camera 2"].visual_weights.copy()
    for _ in range(3):
        _step(orchestrator)
    assert orchestrator.sensor_readouts["camera 2"] is not orchestrator.visual_readout
    assert not np.array_equal(orchestrator.sensor_readouts["camera 2"].visual_weights, before)
