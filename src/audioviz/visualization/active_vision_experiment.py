from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from audioviz.readouts import GaussianSpatialPrecision, PredictiveCodingConfig
from audioviz.sources import (
    CameraFrameSourceConfig, CircularDotCapture, CircularDotConfig, CircularDotSource,
    RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


class ActiveVisionExperiment(QtWidgets.QWidget):
    """Stage A evaluator: fixed weights, manually selected gaze, no policy."""

    PREVIEW_MAX_SIDE = 256

    def __init__(
        self, *, stimulus: CircularDotConfig, precision: GaussianSpatialPrecision,
        backend: str = "opengl", inference_steps: int = 8,
    ):
        super().__init__()
        if backend not in ("numpy", "cupy", "opengl"):
            raise ValueError("backend must be numpy, cupy or opengl")
        self.stimulus = CircularDotSource(stimulus)
        self.precision = precision
        self.setWindowTitle("Active Vision - Gaussian Precision Experiment")
        self.resize(1180, 860)
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Stage A: fixed model weights and manual gaze. Ground truth and previews are evaluator-only. "
            "This is not a learned motion predictor or an EIG policy."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        controls = QtWidgets.QHBoxLayout()
        self.gaze_x = self._gaze_control(precision.gaze[0])
        self.gaze_y = self._gaze_control(precision.gaze[1])
        controls.addWidget(QtWidgets.QLabel("Gaze X"))
        controls.addWidget(self.gaze_x)
        controls.addWidget(QtWidgets.QLabel("Gaze Y"))
        controls.addWidget(self.gaze_y)
        self.observe = QtWidgets.QCheckBox("Observe")
        self.observe.setChecked(precision.sample)
        controls.addWidget(self.observe)
        controls.addWidget(QtWidgets.QLabel(
            f"Width: {precision.sigma:g} x shorter side | "
            f"Precision: {precision.baseline:g} + {precision.peak:g} x Gaussian"
        ))
        controls.addStretch()
        self.progress = QtWidgets.QLabel("Waiting for first frame")
        controls.addWidget(self.progress)
        layout.addLayout(controls)

        self.previews = pg.GraphicsLayoutWidget()
        self.previews.setMinimumHeight(220)
        self.truth_image, _ = self._image_panel("Ground truth (evaluator only)", 0)
        self.precision_background, field_plot = self._image_panel("Observation precision and gaze", 1)
        self.precision_image = pg.ImageItem(axisOrder="row-major")
        self.precision_image.setLookupTable(pg.colormap.get("viridis").getLookupTable(nPts=256))
        self.precision_image.setOpacity(0.7)
        field_plot.addItem(self.precision_image)
        self.gaze_marker = pg.ScatterPlotItem(symbol="+", size=18, pen=pg.mkPen("w", width=2))
        field_plot.addItem(self.gaze_marker)
        self.prediction_image, _ = self._image_panel("Prediction before current evidence", 2)
        layout.addWidget(self.previews, 1)

        self.visualizer = RippleWaveVisualizer(
            resolution=stimulus.resolution, plane_size_m=(1, 1),
            update_interval_ms=max(1, round(1000 / stimulus.frame_rate)),
            use_shader=backend == "opengl", use_gpu=backend == "cupy",
            rgb_canvas_enabled=True,
            camera_capture=CircularDotCapture(self.stimulus),
            source_orchestrator_config=RippleSourceOrchestratorConfig(
                synthetic=SyntheticSourceConfig(enabled=False),
                camera_frame=CameraFrameSourceConfig(enabled=True),
                prediction_error=PredictionErrorTransformConfig(enabled=True),
                visual_pathway=PredictiveCodingConfig(
                    dtype="float32", inference_steps=inference_steps, spatial_precision=precision,
                ),
            ),
        )
        layout.addWidget(self.visualizer, 2)
        self.visualizer.timer.timeout.connect(self._after_frame)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self.refresh_previews)
        self.preview_timer.start()
        self.gaze_x.valueChanged.connect(self._update_precision)
        self.gaze_y.valueChanged.connect(self._update_precision)
        self.observe.toggled.connect(self._update_precision)
        self._update_precision()

    @staticmethod
    def _gaze_control(value):
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(0, 1)
        control.setDecimals(3)
        control.setSingleStep(0.05)
        control.setValue(value)
        return control

    def _image_panel(self, title, column):
        plot = self.previews.addPlot(row=0, col=column, title=title)
        plot.setAspectLocked()
        plot.invertY(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        image = pg.ImageItem(axisOrder="row-major")
        plot.addItem(image)
        return image, plot

    def _update_precision(self):
        self.precision = replace(
            self.precision, gaze=(self.gaze_x.value(), self.gaze_y.value()),
            sample=self.observe.isChecked(),
        )
        self.visualizer.source_orchestrator.update_pathway_config(spatial_precision=self.precision)
        # This CPU field is evaluator-only; OpenGL inference evaluates the Gaussian natively.
        field = spatial_preview(
            self.precision.field(self.stimulus.config.resolution), max_side=self.PREVIEW_MAX_SIDE,
        )
        self.precision_image.setImage(
            field, autoLevels=False, levels=(0, max(self.precision.baseline + self.precision.peak, 1e-6)),
        )
        self.gaze_marker.setData(
            [0.5 + self.precision.gaze[0] * (field.shape[1] - 1)],
            [0.5 + self.precision.gaze[1] * (field.shape[0] - 1)],
        )

    def _after_frame(self):
        sample = self.stimulus.latest
        if sample is not None:
            self.progress.setText(f"Stimulus time: {sample.time:.2f} s")
            if self.stimulus.exhausted:
                self.visualizer.timer.stop()
                self.refresh_previews()
                self.preview_timer.stop()
                self.progress.setText(f"Finished: {self.stimulus.frame_count} frames")

    def refresh_previews(self):
        sample = self.stimulus.latest
        prediction = self.visualizer.visual_prediction
        if sample is None or prediction is None:
            return
        if not self.visualizer.renderer.prepare_frame():
            return
        truth = spatial_preview(sample.frame, max_side=self.PREVIEW_MAX_SIDE)
        self.truth_image.setImage(truth, autoLevels=False, levels=(0, 255))
        self.precision_background.setImage(truth, autoLevels=False, levels=(0, 255))
        preview = spatial_preview(prediction, max_side=self.PREVIEW_MAX_SIDE)
        self.prediction_image.setImage(np.clip(preview, 0, 1), autoLevels=False, levels=(0, 1))

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.visualizer.timer.stop()
        self.visualizer.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Fixed-gaze Gaussian precision experiment; no EIG policy.")
    parser.add_argument("--backend", choices=("numpy", "cupy", "opengl"), default="opengl")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(120, 160))
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--brownian-scale", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gaze", type=float, nargs=2, metavar=("X", "Y"), default=(0.5, 0.5))
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--baseline", type=float, default=0.01)
    parser.add_argument("--peak", type=float, default=1)
    args = parser.parse_args()
    stimulus = CircularDotConfig(
        resolution=tuple(args.resolution), radius=0.3 * min(args.resolution),
        dot_radius=max(1, 0.02 * min(args.resolution)), frame_rate=args.fps,
        duration=args.duration, brownian_scale=args.brownian_scale, seed=args.seed,
    )
    precision = GaussianSpatialPrecision(
        gaze=tuple(args.gaze), sigma=args.sigma, baseline=args.baseline, peak=args.peak,
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = ActiveVisionExperiment(stimulus=stimulus, precision=precision, backend=args.backend)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
