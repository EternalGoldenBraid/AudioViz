"""Mixer controls: named, bounded parameters of a GraphRuntime."""

import math
from dataclasses import dataclass
from typing import Callable

from audioviz.tui.graph_runtime import GraphRuntime

STEPS = 50  # slider resolution: one h/l press moves 1/STEPS of the range


@dataclass(frozen=True)
class Control:
    name: str
    low: float
    high: float
    get: Callable[[GraphRuntime], float]
    set: Callable[[GraphRuntime, float], None]
    log: bool = False
    toggle: bool = False

    def position(self, runtime: GraphRuntime) -> float:
        """Current value as a slider position in [0, 1]."""
        value = self.get(runtime)
        if self.toggle:
            return float(bool(value))
        if self.log:
            low, high, value = math.log10(self.low), math.log10(self.high), math.log10(max(value, self.low))
        else:
            low, high = self.low, self.high
        return min(max((value - low) / (high - low), 0.0), 1.0)

    def nudge(self, runtime: GraphRuntime, steps: int) -> None:
        if self.toggle:
            self.set(runtime, 1.0 if steps > 0 else 0.0)
            return
        position = min(max(self.position(runtime) + steps / STEPS, 0.0), 1.0)
        if self.log:
            value = 10 ** (math.log10(self.low) + position * (math.log10(self.high) - math.log10(self.low)))
        else:
            value = self.low + position * (self.high - self.low)
        self.set(runtime, value)

    def format(self, runtime: GraphRuntime) -> str:
        value = self.get(runtime)
        return ("on" if value else "off") if self.toggle else f"{value:.4g}"


def _pathway(key: str) -> Callable[[GraphRuntime, float], None]:
    return lambda runtime, value: runtime.orchestrator.update_pathway_config(**{key: value})


def _config(key: str) -> Callable[[GraphRuntime], float]:
    return lambda runtime: getattr(runtime.orchestrator.visual_readout.config, key)


def _set_gain(runtime: GraphRuntime, value: float) -> None:
    runtime.orchestrator.audio_source.observation_gain = value


CONTROLS: tuple[Control, ...] = (
    Control("evidence gain", 0.1, 10000.0, lambda r: r.orchestrator.audio_source.observation_gain, _set_gain, log=True),
    Control("damping", 0.9, 1.0, lambda r: r.engine.damping, lambda r, v: r.engine.set_damping(v)),
    Control("speed", 0.1, 3.0, lambda r: r.engine.speed, lambda r, v: r.engine.set_speed(v)),
    Control("inference rate", 0.001, 1.0, _config("inference_rate"), _pathway("inference_rate"), log=True),
    Control("canvas rate", 0.0001, 1.0, _config("canvas_rate"), _pathway("canvas_rate"), log=True),
    Control("learning", 0, 1, _config("learning_enabled"), lambda r, v: _pathway("learning_enabled")(r, bool(v)), toggle=True),
    Control("learning rate", 0.0001, 1.0, _config("learning_rate"), _pathway("learning_rate"), log=True),
)
