from copy import deepcopy
import importlib.util
from pathlib import Path


def _load_main_module():
    module_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("audioviz_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_ripple_visualizer_config_flattens_nested_source_config():
    main = _load_main_module()
    config = deepcopy(main.RIPPLE_CONFIG)
    config["renderer"]["backend"] = "opengl"
    config["sources"]["audio"]["enabled"] = False
    config["sources"]["camera_frame"]["enabled"] = True
    config["sources"]["camera_frame"]["camera_index"] = 2
    config["transforms"]["prediction_error"]["enabled"] = True
    config["transforms"]["prediction_error"]["predictor"]["sigma"] = 0.25
    config["transforms"]["prediction_error"]["output"]["activation"]["function"] = "tanh"
    config["transforms"]["prediction_error"]["output"]["activation"]["scale"] = 2.0
    config["transforms"]["prediction_error"]["learning"]["enabled"] = True
    config["transforms"]["prediction_error"]["learning"]["learning_rate"] = 0.01

    flattened = main.build_ripple_visualizer_config(config)

    assert flattened["use_shader"] is True
    assert flattened["use_gpu"] is False
    orchestrator_config = flattened["source_orchestrator_config"]
    pose_config = flattened["pose_config"]

    assert orchestrator_config.audio.enabled is False
    assert orchestrator_config.camera_frame.enabled is True
    assert orchestrator_config.camera_frame.camera_index == 2
    assert orchestrator_config.prediction_error.enabled is True
    assert orchestrator_config.prediction_error.inputs == ("camera_frame",)
    assert orchestrator_config.prediction_error.sigma == 0.25
    assert orchestrator_config.prediction_error.activation_function == "tanh"
    assert orchestrator_config.prediction_error.activation_scale == 2.0
    assert orchestrator_config.prediction_error.learning_enabled is True
    assert orchestrator_config.prediction_error.learning_rate == 0.01
    assert pose_config.camera_index == config["sources"]["pose"]["camera_index"]
    assert (
        pose_config.body_boundary_transmission
        == config["sources"]["pose"]["boundary"]["transmission"]
    )
