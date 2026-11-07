"""Adapt the window's existing control metadata and callbacks to terminal knobs."""

from audioviz.tui.mixer import Control

SECTION_CATEGORIES = {
    "audio-source": "audio",
    "camera-source": "sources",
    "learning-dynamics": "inference",
    "learning-overlay": "interface",
}
ALIASES = {
    ("audio-source", "observation_gain"): "evidence gain",
    ("camera-source", "gain"): "camera gain",
    ("learning-dynamics", "inference_steps"): "inference steps",
    ("learning-dynamics", "inference_rate"): "inference rate",
    ("learning-dynamics", "canvas_rate"): "canvas rate",
    ("learning-dynamics", "learning_enabled"): "learning",
    ("learning-dynamics", "learning_rate"): "learning rate",
    ("learning-dynamics", "learning_weight_decay"): "weight decay",
    ("learning-dynamics", "learning_weight_clip"): "weight clip",
    ("learning-dynamics", "learning_gradient_clip"): "gradient clip",
    ("learning-dynamics", "cross_modal_enabled"): "hidden loop",
}


def window_settings(window) -> dict[str, Control]:
    binding = window.source_control_binding
    result = {}
    for section in binding.build_sections():
        for metadata in section.controls:
            if metadata.kind == "text":
                continue
            name = ALIASES.get((section.key, metadata.key), f"{section.key} {metadata.key}")
            category = SECTION_CATEGORIES.get(section.key, "sources")
            if section.key == "learning-dynamics" and metadata.key.startswith("learning_"):
                category = "learning"
            result[name] = Control(
                name, metadata.minimum or 0, metadata.maximum or 1,
                lambda _target, s=section.key, k=metadata.key: binding.control_value(s, k),
                lambda _target, value, s=section.key, k=metadata.key: binding.update_control(s, k, value),
                choices=metadata.choices or None, toggle=metadata.kind == "toggle",
                category=category, description=metadata.tooltip,
                label=(f"{section.title}: {metadata.label}" if section.key.startswith("synthetic-source-") else metadata.label),
                step=metadata.step,
            )
    for toggle in binding.build_toggles():
        if toggle.available:
            name = f"{toggle.key} evidence" if toggle.key != "synthetic" else "synthetic source"
            result[name] = Control(
                name, 0, 1,
                lambda _target, key=toggle.key: next(t.enabled for t in binding.build_toggles() if t.key == key),
                lambda _target, value, key=toggle.key: binding.update_toggle(key, bool(value)),
                toggle=True, category="audio" if toggle.key == "audio" else "sources",
                label=toggle.label + " enabled",
            )

    def set_physics(value, setter, callback):
        setter(value)
        callback(value)

    for name, attribute, low, high, step, setter, callback in (
        ("excitation amplitude", "base_amplitude", 0, max(10, window.base_amplitude), 0.01, lambda v: setattr(window.engine, "amplitude", v), window._update_amplitude),
        ("excitation falloff", "decay_alpha", 0, 10, 0.01, lambda v: setattr(window.engine, "decay_alpha", v), window._update_decay_alpha),
        ("damping", "damping", 0, 1, 0.001, window.engine.set_damping, window._update_damping),
        ("speed", "speed", 10, 3000, 1, window.engine.set_speed, window._update_speed),
    ):
        result[name] = Control(
            name, low, high, lambda _target, attr=attribute: getattr(window, attr),
            lambda _target, value, apply=setter, sync=callback: set_physics(value, apply, sync),
            step=step, category="wave",
        )
    result["wave propagation"] = Control(
        "wave propagation", 0, 1, lambda _target: window.engine.wave_propagation_enabled,
        lambda _target, value: window.engine.set_wave_propagation(bool(value)),
        toggle=True, category="wave",
        description="Wave equation dynamics. Off holds the canvas between sensory inference updates.",
    )
    for name, attribute, low, high, step, setter in (
        ("auto color threshold", "auto_color_activation_threshold", 0, 2, 0.01, window._update_auto_color_activation_threshold),
        ("reference scale", "auto_color_floor", 0, 2, 0.01, window._update_auto_color_floor),
    ):
        result[name] = Control(
            name, low, high, lambda _target, attr=attribute: getattr(window, attr),
            lambda _target, value, apply=setter: apply(value),
            step=step, category="interface",
        )
    for name, attribute, setter in (
        ("RGB canvas", "rgb_canvas_enabled", window._update_rgb_canvas),
        ("auto color scaling", "auto_color_levels_enabled", window._update_auto_color_levels),
    ):
        if name == "RGB canvas" and not callable(getattr(window.renderer, "set_rgb_canvas_enabled", None)):
            continue
        result[name] = Control(
            name, 0, 1, lambda _target, attr=attribute: getattr(window, attr),
            lambda _target, value, apply=setter: apply(bool(value)), toggle=True, category="interface",
        )
    if not window.use_shader:
        for name, attribute, setter in (
            ("inference plots", "prediction_diagnostics", window.set_prediction_diagnostics_visible),
            ("3D scene", "prediction_scene", window.set_prediction_scene_visible),
        ):
            result[name] = Control(
                name, 0, 1,
                lambda _target, attr=attribute: bool(getattr(window, attr) and getattr(window, attr).isVisible()),
                lambda _target, value, apply=setter: apply(bool(value)), toggle=True, category="inference",
            )
    return result
