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


def test_branch_flows_show_each_camera_error_and_push_direction():
    orchestrator = _orchestrator()
    orchestrator.set_diagnostics_enabled(True)
    orchestrator.camera_source.excitation = lambda: np.zeros((12, 16, 3), dtype=np.float32)
    camera_2 = orchestrator.sensors["camera 2"]
    camera_2.enabled = True
    camera_2.excitation = lambda: np.ones((12, 16, 3), dtype=np.float32)
    _step(orchestrator)

    flows = orchestrator.branch_flows
    assert set(flows) == {"camera 1", "camera 2"}
    assert flows["camera 2"].error_direction > 0 and flows["camera 2"].push_intensity > 0
    assert flows["camera 2"].push_direction > 0 > flows["camera 1"].push_direction


def test_branch_flows_are_not_collected_without_diagnostics():
    orchestrator = _orchestrator()
    orchestrator.sensors["camera 2"].enabled = True
    orchestrator.sensors["camera 2"].excitation = lambda: np.ones((12, 16, 3), dtype=np.float32)
    _step(orchestrator)
    assert orchestrator.branch_flows == {}


def test_virtual_cameras_drive_both_branches_through_the_real_capture_path():
    from audioviz.sources.virtual_camera import VIRTUAL_BASE

    orchestrator = _orchestrator()
    orchestrator.set_diagnostics_enabled(True)
    first = orchestrator.camera_source
    first.capture = None
    first.camera_index = VIRTUAL_BASE + 1  # dark
    camera_2 = orchestrator.sensors["camera 2"]
    camera_2.enabled = True
    camera_2.camera_index = VIRTUAL_BASE + 2  # bright
    _step(orchestrator)

    flows = orchestrator.branch_flows
    assert set(flows) == {"camera 1", "camera 2"}
    assert flows["camera 2"].push_direction > 0 > flows["camera 1"].push_direction


def test_flows_exist_for_a_single_camera_and_the_view_never_goes_blank():
    from audioviz.tui.views import flows

    orchestrator = _orchestrator()
    orchestrator.set_diagnostics_enabled(True)
    orchestrator.camera_source.excitation = lambda: np.ones((12, 16, 3), dtype=np.float32)
    runtime = type("Runtime", (), {"orchestrator": orchestrator})()
    assert flows(runtime, 12, 20).shape == (12, 20, 3)  # before any inference

    _step(orchestrator)

    assert set(orchestrator.branch_flows) == {"camera 1"}
    assert orchestrator.branch_flows["camera 1"].error_intensity > 0
    assert flows(runtime, 12, 20).any()


def test_two_virtual_cameras_see_one_object_at_parallax_offset():
    import numpy as np

    from audioviz.sources.virtual_camera import VirtualCapture, virtual_choices

    names = list(virtual_choices())
    left, right = (VirtualCapture(100 + names.index(n)) for n in [c for c in names if "object" in c])
    centre = lambda frame: np.nonzero(frame[..., 0].any(axis=0))[0].mean()
    assert abs(centre(left.read()[1]) - centre(right.read()[1])) > 20
