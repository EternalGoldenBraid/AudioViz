from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PredictionErrorTransformConfig:
    enabled: bool = False
    inputs: tuple[str, ...] = ("camera_frame",)
    predictor_source: str = "ripple_state"
    sigma: float = 0.1
    output_mode: str = "bits"
    activation_function: str = "softsign"
    activation_scale: float = 1.0
    gain: float = 1.0
    prediction_clip: float = 10.0
    max_output: float = 10.0
    learning_enabled: bool = False
    learning_rate: float = 1e-4
    learning_weight_decay: float = 1e-4
    learning_weight_clip: float = 1.0
    learning_gradient_clip: float = 1.0


class PredictionErrorTransform:
    def __init__(
        self,
        *,
        config: PredictionErrorTransformConfig,
        resolution: tuple[int, int],
    ):
        self.config = config
        self.resolution = resolution
        self.edge_weights: np.ndarray | None = None
        self._validate_config()

    def update_config(self, **changes) -> None:
        next_config = replace(self.config, **changes)
        previous_config = self.config
        self.config = next_config
        try:
            self._validate_config()
        except Exception:
            self.config = previous_config
            raise

    def applies_to(self, source_key: str) -> bool:
        return self.config.enabled and source_key in self.config.inputs

    def apply(
        self,
        *,
        source_key: str,
        observation: np.ndarray,
        ripple_state: np.ndarray,
    ) -> np.ndarray:
        if not self.applies_to(source_key):
            return observation

        prediction = self.predict(ripple_state)
        observed = np.nan_to_num(
            np.asarray(observation, dtype=np.float64),
            nan=0.0,
            posinf=self.config.prediction_clip,
            neginf=-self.config.prediction_clip,
        )
        error = observed - prediction.astype(np.float64, copy=False)
        self._update_edge_weights(error, ripple_state)
        transformed = self._error_to_output(error)
        transformed = self._activate_output(
            transformed,
            signed=self.config.output_mode == "raw_error",
        )
        transformed = transformed * np.float64(self.config.gain)
        max_output = np.float64(self.config.max_output)
        transformed = np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=max_output,
            neginf=-max_output,
        )
        if self.config.output_mode == "raw_error":
            transformed = np.clip(transformed, -max_output, max_output)
        else:
            transformed = np.clip(transformed, 0.0, max_output)
        return transformed.astype(np.float32, copy=False)

    def predict(self, ripple_state: np.ndarray) -> np.ndarray:
        field = self._coerce_ripple_state(ripple_state)
        if not self.config.learning_enabled:
            return field
        self._ensure_edge_weights()
        assert self.edge_weights is not None
        neighbors = self.neighbor_fields(field)
        prediction = field.astype(np.float64) + np.sum(
            self.edge_weights.astype(np.float64) * neighbors,
            axis=0,
        )
        clip = np.float32(self.config.prediction_clip)
        return np.clip(prediction, -clip, clip).astype(np.float32, copy=False)

    @staticmethod
    def neighbor_fields(field: np.ndarray) -> np.ndarray:
        padded = np.pad(np.asarray(field, dtype=np.float64), 1, mode="edge")
        return np.stack(
            (
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ),
            axis=0,
        )

    def _error_to_output(self, error: np.ndarray) -> np.ndarray:
        mode = self.config.output_mode
        if mode == "raw_error":
            return error
        if mode == "abs_error":
            return np.abs(error)
        if mode == "squared_error":
            return error * error
        if mode == "bits":
            sigma = np.float64(self.config.sigma)
            return (error * error) / (
                np.float64(2.0) * sigma * sigma * np.float64(np.log(2.0))
            )
        raise ValueError(f"Unknown prediction error output mode: {mode}")

    def _activate_output(self, values: np.ndarray, *, signed: bool) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        scale = np.float64(self.config.activation_scale)
        max_output = np.float64(self.config.max_output)
        function = self.config.activation_function
        if function == "linear_clipped":
            activated = values
        elif function == "tanh":
            activated = max_output * np.tanh(values / scale)
        elif function == "softsign":
            activated = max_output * values / (scale + np.abs(values))
        elif function == "log1p":
            activated = scale * np.sign(values) * np.log1p(np.abs(values) / scale)
        else:
            raise ValueError(f"Unknown prediction error activation: {function}")
        if not signed:
            activated = np.maximum(activated, 0.0)
        return activated

    def _update_edge_weights(self, error: np.ndarray, ripple_state: np.ndarray) -> None:
        if not self.config.learning_enabled or self.config.learning_rate == 0.0:
            return
        self._ensure_edge_weights()
        assert self.edge_weights is not None
        field = self._coerce_ripple_state(ripple_state)
        neighbors = self.neighbor_fields(field)
        denom = (
            np.float64(self.config.sigma)
            * np.float64(self.config.sigma)
            * np.float64(np.log(2.0))
        )
        gradient = error[None, :, :] * neighbors / denom
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        gradient = np.clip(
            gradient,
            -self.config.learning_gradient_clip,
            self.config.learning_gradient_clip,
        )
        weights = self.edge_weights.astype(np.float64, copy=False)
        if self.config.learning_weight_decay:
            weights *= np.float64(1.0 - self.config.learning_weight_decay)
        weights += np.float64(self.config.learning_rate) * gradient
        weights = np.clip(
            weights,
            -self.config.learning_weight_clip,
            self.config.learning_weight_clip,
        )
        self.edge_weights = weights.astype(np.float32, copy=False)

    def _coerce_ripple_state(self, ripple_state: np.ndarray) -> np.ndarray:
        field = np.asarray(ripple_state, dtype=np.float32)
        if field.shape != self.resolution:
            raise ValueError("ripple-state prediction must match engine resolution")
        clip = np.float32(self.config.prediction_clip)
        field = np.nan_to_num(field, nan=0.0, posinf=clip, neginf=-clip)
        return np.clip(field, -clip, clip).astype(np.float32, copy=False)

    def _ensure_edge_weights(self) -> None:
        if self.edge_weights is None or self.edge_weights.shape != (4, *self.resolution):
            self.edge_weights = np.zeros((4, *self.resolution), dtype=np.float32)

    def _validate_config(self) -> None:
        if self.config.sigma <= 0.0:
            raise ValueError("prediction_error_sigma must be positive")
        if self.config.prediction_clip <= 0.0:
            raise ValueError("prediction_error_prediction_clip must be positive")
        if self.config.max_output <= 0.0:
            raise ValueError("prediction_error_max_output must be positive")
        if self.config.activation_scale <= 0.0:
            raise ValueError("prediction_error_activation_scale must be positive")
        if self.config.learning_rate < 0.0:
            raise ValueError("prediction_error_learning_rate must be non-negative")
        if self.config.learning_weight_decay < 0.0 or self.config.learning_weight_decay > 1.0:
            raise ValueError(
                "prediction_error_learning_weight_decay must be between 0 and 1"
            )
        if self.config.learning_weight_clip <= 0.0:
            raise ValueError("prediction_error_learning_weight_clip must be positive")
        if self.config.learning_gradient_clip <= 0.0:
            raise ValueError("prediction_error_learning_gradient_clip must be positive")
        if self.config.predictor_source != "ripple_state":
            raise ValueError("prediction_error_predictor_source must be 'ripple_state'")
        valid_output_modes = {"raw_error", "abs_error", "squared_error", "bits"}
        if self.config.output_mode not in valid_output_modes:
            raise ValueError(
                "prediction_error_output_mode must be one of "
                f"{sorted(valid_output_modes)}"
            )
        valid_activation_functions = {
            "linear_clipped",
            "tanh",
            "softsign",
            "log1p",
        }
        if self.config.activation_function not in valid_activation_functions:
            raise ValueError(
                "prediction_error_activation_function must be one of "
                f"{sorted(valid_activation_functions)}"
            )
