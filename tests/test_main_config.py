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

    flattened = main.build_ripple_visualizer_config(config)

    assert flattened["use_shader"] is True
    assert flattened["use_gpu"] is False
    assert flattened["use_audio_source"] is False
    assert flattened["use_camera_source"] is True
    assert flattened["camera_source_index"] == 2
    assert flattened["prediction_error_transform_enabled"] is True
    assert flattened["prediction_error_inputs"] == ("camera_frame",)
    assert flattened["prediction_error_sigma"] == 0.25
    assert flattened["pose_camera_index"] == config["sources"]["pose"]["camera_index"]
    assert (
        flattened["body_boundary_transmission"]
        == config["sources"]["pose"]["boundary"]["transmission"]
    )
