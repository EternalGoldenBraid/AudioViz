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
    """Manual gaze with model-linked conditional Gaussian information advice."""

    PREVIEW_MAX_SIDE = 256

    def __init__(
        self, *, stimulus: CircularDotConfig, precision: GaussianSpatialPrecision,
        backend: str = "opengl", inference_steps: int = 8,
        initial_variance: float = 1.0, process_variance: float = 0.01,
    ):
        super().__init__()
        if backend not in ("numpy", "cupy", "opengl"):
            raise ValueError("backend must be numpy, cupy or opengl")
        self.stimulus = CircularDotSource(stimulus)
        self.precision = precision
        self.information_scores = None
        self._score_dirty = True
        self.setWindowTitle("Active Vision - Gaussian Precision Experiment")
        self.resize(1180, 860)
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Manual gaze; fixed model weights. The heatmap advises where to look within this image. "
            "It never moves gaze automatically. Ground truth is evaluator-only."
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
        self.follow_mouse = QtWidgets.QCheckBox("Follow mouse")
        self.follow_mouse.setChecked(True)
        self.follow_mouse.setToolTip("Move over any preview image to position gaze. Turn off to freeze it.")
        controls.addWidget(self.follow_mouse)
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
        advice = QtWidgets.QHBoxLayout()
        self.information_toggle = QtWidgets.QCheckBox("Information gain")
        self.information_toggle.setChecked(True)
        self.information_toggle.setToolTip(
            "Brighter means greater expected gain. The white arrow points from gaze to the best candidate."
        )
        advice.addWidget(self.information_toggle)
        self.information_status = QtWidgets.QLabel("Waiting for uncertainty estimate")
        advice.addWidget(self.information_status)
        advice.addStretch()
        layout.addLayout(advice)
        assumptions = QtWidgets.QLabel(
            f"Conditional Gaussian: initial hidden variance {initial_variance:g}; "
            f"process variance {process_variance:g}/s. Brighter = greater expected information gain. "
            "Diagonal hidden covariance; known wave canvas and fixed weights. Not calibrated confidence."
        )
        assumptions.setWordWrap(True)
        layout.addWidget(assumptions)

        self.previews = pg.GraphicsLayoutWidget()
        self.previews.setMinimumHeight(220)
        self.truth_image, self.truth_plot = self._image_panel("Ground truth + advisory gain", 0)
        self.information_image = pg.ImageItem(axisOrder="row-major")
        self.information_image.setLookupTable(pg.colormap.get("magma").getLookupTable(nPts=256))
        self.information_image.setOpacity(0.65)
        self.truth_plot.addItem(self.information_image)
        self.direction_line = pg.PlotDataItem(pen=pg.mkPen("w", width=2))
        self.truth_plot.addItem(self.direction_line)
        self.direction_arrow = pg.ArrowItem(headLen=12, pen="w", brush="w")
        self.truth_plot.addItem(self.direction_arrow)
        self.recommended_marker = pg.ScatterPlotItem(
            symbol="o", size=10, pen=pg.mkPen("w", width=2), brush=None,
        )
        self.truth_plot.addItem(self.recommended_marker)
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
        self.uncertainty = self.visualizer.source_orchestrator.enable_visual_uncertainty(
            dt=1 / stimulus.frame_rate,
            initial_variance=initial_variance, process_variance=process_variance,
        )
        self.visualizer.timer.timeout.connect(self._after_frame)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self.refresh_previews)
        self.preview_timer.start()
        self.gaze_x.valueChanged.connect(self._update_precision)
        self.gaze_y.valueChanged.connect(self._update_precision)
        self.observe.toggled.connect(self._update_precision)
        self.information_toggle.toggled.connect(self._update_advisory_marker)
        self._update_precision()
        self.previews.setMouseTracking(True)
        self.previews.viewport().setMouseTracking(True)
        self.previews.scene().sigMouseMoved.connect(self._move_gaze_from_mouse)

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
        self._update_advisory_marker()

    def _update_advisory_marker(self):
        scores = self.information_scores
        available = scores is not None and float(scores.max()) > 0
        visible = available and self.information_toggle.isChecked()
        self.truth_plot.setTitle(
            "Ground truth + advisory gain" if visible else "Ground truth (evaluator only)"
        )
        for item in (
            self.information_image, self.recommended_marker, self.direction_line, self.direction_arrow,
        ):
            item.setVisible(visible)
        if scores is None:
            return
        if not available:
            self.information_status.setText("No expected gain with the configured observation precision")
            return
        if np.allclose(scores, scores.flat[0], rtol=1e-6, atol=0):
            self.information_status.setText(
                f"Next observation: {scores.flat[0]:.1f} nats | no gaze preference"
                + (" | sampling paused" if not self.precision.sample else "")
            )
            for item in (self.recommended_marker, self.direction_line, self.direction_arrow):
                item.hide()
            return
        iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
        target = np.array((ix / (scores.shape[1] - 1), iy / (scores.shape[0] - 1)))
        self.information_status.setText(
            f"Next observation: {scores[iy, ix]:.1f} nats | range {scores.min():.1f}-{scores.max():.1f} | "
            f"look at ({target[0]:.2f}, {target[1]:.2f})"
            + (" | sampling paused" if not self.precision.sample else "")
        )
        bounds = self.truth_image.boundingRect()
        extent = np.array((bounds.width() - 1, bounds.height() - 1))
        start = 0.5 + np.asarray(self.precision.gaze) * extent
        end = 0.5 + target * extent
        self.recommended_marker.setData([end[0]], [end[1]])
        self.direction_line.setData([start[0], end[0]], [start[1], end[1]])
        self.direction_arrow.setPos(*end)
        delta = end - start
        self.direction_arrow.setStyle(angle=float(np.degrees(np.arctan2(delta[1], delta[0])) + 180))
        directional = visible and bool(np.linalg.norm(target - self.precision.gaze) > 0.02)
        self.direction_line.setVisible(directional)
        self.direction_arrow.setVisible(directional)

    def _move_gaze_from_mouse(self, scene_position):
        if not self.follow_mouse.isChecked():
            return
        for image in (self.truth_image, self.precision_image, self.prediction_image):
            position = image.mapFromScene(scene_position)
            bounds = image.boundingRect()
            if not bounds.contains(position):
                continue
            x = float(np.clip((position.x() - 0.5) / max(bounds.width() - 1, 1), 0, 1))
            y = float(np.clip((position.y() - 0.5) / max(bounds.height() - 1, 1), 0, 1))
            with QtCore.QSignalBlocker(self.gaze_x), QtCore.QSignalBlocker(self.gaze_y):
                self.gaze_x.setValue(x)
                self.gaze_y.setValue(y)
            if (self.gaze_x.value(), self.gaze_y.value()) != self.precision.gaze:
                self._update_precision()
            return

    def _after_frame(self):
        self._score_dirty = True
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
        if self._score_dirty:
            # Advice evaluates a future observation even while actual sampling is paused.
            self.information_scores = self.uncertainty.expected_information_gain(
                replace(self.precision, sample=True), dt=1 / self.stimulus.config.frame_rate,
                prior=self.visualizer.engine.get_prior(),
            )
            self._score_dirty = False
            scores = self.information_scores
            rows = np.rint(np.linspace(0, scores.shape[0] - 1, truth.shape[0])).astype(np.intp)
            cols = np.rint(np.linspace(0, scores.shape[1] - 1, truth.shape[1])).astype(np.intp)
            self.information_image.setImage(
                scores[rows[:, None], cols[None, :]], autoLevels=False,
                levels=(float(scores.min()), max(float(scores.max()), float(scores.min()) + 1e-6)),
            )
        self._update_advisory_marker()

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.visualizer.timer.stop()
        self.visualizer.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Manual gaze with advisory Gaussian information gain.")
    parser.add_argument("--backend", choices=("numpy", "cupy", "opengl"), default="opengl")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(120, 160))
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--brownian-scale", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gaze", type=float, nargs=2, metavar=("X", "Y"), default=(0.5, 0.5))
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--baseline", type=float, default=0.01)
    parser.add_argument("--peak", type=float, default=1.0)
    parser.add_argument("--initial-variance", type=float, default=1.0)
    parser.add_argument("--process-variance", type=float, default=0.01)
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
    window = ActiveVisionExperiment(
        stimulus=stimulus, precision=precision, backend=args.backend,
        initial_variance=args.initial_variance, process_variance=args.process_variance,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
