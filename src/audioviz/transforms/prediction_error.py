from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PredictionErrorTransformConfig:
    enabled: bool = False
    inputs: tuple[str, ...] = ("camera_frame",)
    sigma: float = 0.1
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
        resolution: tuple[int, ...],
    ):
        self.config = config
        self.resolution = resolution
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
        prediction: np.ndarray,
    ) -> np.ndarray:
        error = self.compute_error(
            source_key=source_key,
            observation=observation,
            prediction=prediction,
        )
        if error is None:
            return observation
        return self.shape_error(error)

    def compute_error(
        self,
        *,
        source_key: str,
        observation: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray | None:
        if not self.applies_to(source_key):
            return None

        predicted = self._coerce_field(prediction)
        observed = self._coerce_field(observation)
        return observed.astype(np.float64, copy=False) - predicted.astype(
            np.float64,
            copy=False,
        )

    @staticmethod
    def neighbor_fields(field: np.ndarray) -> np.ndarray:
        values = np.asarray(field, dtype=np.float64)
        if values.ndim < 2:
            raise ValueError("prediction field must have at least two spatial axes")
        pad_width = ((1, 1), (1, 1)) + ((0, 0),) * (values.ndim - 2)
        padded = np.pad(values, pad_width, mode="edge")
        return np.stack(
            (
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ),
            axis=0,
        )

    def _activate_output(self, values: np.ndarray) -> np.ndarray:
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
        return activated

    def shape_error(self, error: np.ndarray) -> np.ndarray:
        transformed = self._activate_output(error)
        transformed = transformed * np.float64(self.config.gain)
        max_output = np.float64(self.config.max_output)
        transformed = np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=max_output,
            neginf=-max_output,
        )
        transformed = np.clip(transformed, -max_output, max_output)
        return transformed.astype(np.float32, copy=False)

    def _coerce_field(self, values: np.ndarray) -> np.ndarray:
        field = np.asarray(values, dtype=np.float32)
        if field.shape != self.resolution:
            raise ValueError("prediction must match the configured observation shape")
        clip = np.float32(self.config.prediction_clip)
        field = np.nan_to_num(field, nan=0.0, posinf=clip, neginf=-clip)
        return np.clip(field, -clip, clip).astype(np.float32, copy=False)

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
        if (
            self.config.learning_weight_decay < 0.0
            or self.config.learning_weight_decay > 1.0
        ):
            raise ValueError(
                "prediction_error_learning_weight_decay must be between 0 and 1"
            )
        if self.config.learning_weight_clip <= 0.0:
            raise ValueError("prediction_error_learning_weight_clip must be positive")
        if self.config.learning_gradient_clip <= 0.0:
            raise ValueError("prediction_error_learning_gradient_clip must be positive")
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
