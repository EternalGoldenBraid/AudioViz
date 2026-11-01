from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PyQt5 import QtCore, QtWidgets

from audioviz.readouts import GaussianSpatialPrecision, PredictiveCodingConfig
from audioviz.sources import (
    CameraFrameSourceConfig, CircularDotCapture, CircularDotConfig, CircularDotSource,
    RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)
from audioviz.sources.gaze import GazePolicy, has_gaze_preference, information_gaze
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.ui.gaze_controls import GazeControls
from audioviz.utils.spatial_preview import preview_indices, spatial_preview
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


class ActiveVisionExperiment(QtWidgets.QWidget):
    """Mouse, autonomous, or full-canvas observation with temporal Gaussian inference."""

    PREVIEW_MAX_SIDE = 256

    def __init__(
        self, *, stimulus: CircularDotConfig, precision: GaussianSpatialPrecision,
        backend: str = "opengl", inference_steps: int = 8,
        initial_variance: float = 1.0, process_variance: float = 0.01,
        memory_time_constant: float = 1/3,
        gaze_mode: str = "mouse",
    ):
        super().__init__()
        if backend not in ("numpy", "cupy", "opengl"):
            raise ValueError("backend must be numpy, cupy or opengl")
        if gaze_mode not in ("mouse", "autonomous", "full"):
            raise ValueError("gaze mode must be mouse, autonomous or full")
        self.stimulus = CircularDotSource(stimulus)
        self.precision = precision
        self._gaze_precision = precision
        self.information_scores = None
        self._score_dirty = True
        self.setWindowTitle("Active Vision - Gaussian Precision Experiment")
        self.resize(1180, 860)
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Choose mouse gaze, autonomous information-seeking gaze, or ungated full-canvas observation. "
            "Fixed model weights; ground truth is evaluator-only. Eye tracking is not connected."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.gaze_controls = GazeControls(precision, mode=gaze_mode)
        self.gaze_x, self.gaze_y = self.gaze_controls.gaze_x, self.gaze_controls.gaze_y
        self.observe = self.gaze_controls.observe
        self.gaze_modes = self.gaze_controls.gaze_modes
        self.autonomous_gaze = self.gaze_controls.autonomous_gaze
        self.follow_mouse = self.gaze_controls.follow_mouse
        self.full_canvas = self.gaze_controls.full_canvas
        controls = self.gaze_controls.coordinates
        self.restart_button = QtWidgets.QPushButton("Restart")
        self.restart_button.setToolTip("Replay the same seeded stimulus and reset beliefs; keep the selected gaze mode.")
        self.restart_button.setEnabled(False)
        self.restart_button.clicked.connect(self._restart)
        controls.insertWidget(controls.count()-1, self.restart_button)
        self.progress = QtWidgets.QLabel("Waiting for first frame")
        self.progress.setWordWrap(True)
        controls.addWidget(self.progress)
        layout.addWidget(self.gaze_controls)
        advice = QtWidgets.QHBoxLayout()
        self.information_toggle = QtWidgets.QCheckBox("Information gain")
        self.information_toggle.setChecked(True)
        self.information_toggle.setToolTip(
            "Brighter means greater expected gain. The white arrow points from gaze to the best candidate."
        )
        advice.addWidget(self.information_toggle)
        self.information_status = QtWidgets.QLabel("Waiting for uncertainty estimate")
        self.information_status.setWordWrap(True)
        advice.addWidget(self.information_status)
        advice.addStretch()
        layout.addLayout(advice)
        self.assumptions = QtWidgets.QLabel(
            f"Temporal Gaussian: memory {memory_time_constant:g} s; initial hidden variance {initial_variance:g}; "
            f"process diffusion {process_variance:g}/s. One shared mean/covariance update per frame. "
            "Assumed R = I / precision where precision > 0 (full canvas: R = I). "
            "Stimulus pixels are deterministic; no measurement noise is sampled. "
            "Diagonal hidden covariance; known wave canvas and fixed weights. Not calibrated confidence."
        )
        self.assumptions.setWordWrap(True)
        layout.addWidget(self.assumptions)

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
        self.prediction_image, prediction_plot = self._image_panel("Prediction before current evidence", 2)
        self._preview_plots = (self.truth_plot, field_plot, prediction_plot)
        self._preview_columns = 3
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
            memory_time_constant=memory_time_constant,
        )
        self.visualizer.timer.timeout.connect(self._after_frame)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self.refresh_previews)
        self.preview_timer.start()
        self.gaze_controls.precision_changed.connect(self._update_precision)
        self.information_toggle.toggled.connect(self._update_advisory_marker)
        self.gaze_controls.mode_changed.connect(self._change_gaze_mode)
        self._change_gaze_mode()
        self.previews.setMouseTracking(True)
        self.previews.viewport().setMouseTracking(True)
        self.previews.scene().sigMouseMoved.connect(self._move_gaze_from_mouse)

    def _image_panel(self, title, column):
        plot = self.previews.addPlot(row=0, col=column, title=title)
        plot.setAspectLocked()
        plot.invertY(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        image = pg.ImageItem(axisOrder="row-major")
        plot.addItem(image)
        return image, plot

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "_preview_plots"):
            return
        columns = 3 if self.width() >= 1000 else 1
        if columns == self._preview_columns:
            return
        for plot in self._preview_plots:
            self.previews.removeItem(plot)
        for index, plot in enumerate(self._preview_plots):
            self.previews.addItem(plot, row=index // columns, col=index % columns)
        self.previews.setMinimumHeight(220 if columns == 3 else 450)
        self._preview_columns = columns

    def _change_gaze_mode(self):
        self.information_scores = None
        self.gaze_x.setEnabled(self.follow_mouse.isChecked())
        self.gaze_y.setEnabled(self.follow_mouse.isChecked())
        self.visualizer.source_orchestrator.gaze_policy = (
            GazePolicy() if self.autonomous_gaze.isChecked() else None
        )
        self._update_precision()

    def _update_precision(self):
        self._gaze_precision = self.gaze_controls.gaze_precision(self._gaze_precision)
        self.precision = self.gaze_controls.observation_precision(self._gaze_precision)
        self.visualizer.source_orchestrator.update_pathway_config(spatial_precision=self.precision)
        self._score_dirty = True
        self._refresh_precision_preview()

    def _refresh_precision_preview(self):
        # This CPU field is evaluator-only; OpenGL inference evaluates the Gaussian natively.
        shape = self.stimulus.config.resolution
        field = self.precision.field(
            shape, indices=preview_indices(shape, max_side=self.PREVIEW_MAX_SIDE),
        )
        self.precision_image.setImage(
            field, autoLevels=False, levels=(0, max(self.precision.baseline + self.precision.peak, 1e-6)),
        )
        self.gaze_marker.setData(
            [0.5 + self.precision.gaze[0] * (field.shape[1] - 1)],
            [0.5 + self.precision.gaze[1] * (field.shape[0] - 1)],
        )
        self.gaze_marker.setVisible(not self.full_canvas.isChecked())
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
            self.information_status.setText(
                "Finished - click Restart to run this gaze mode"
                if self.stimulus.exhausted else "Waiting for uncertainty estimate"
            )
            return
        if not available:
            self.information_status.setText("No expected gain with the configured observation precision")
            return
        if self.full_canvas.isChecked() or not has_gaze_preference(scores):
            self.information_status.setText(
                f"Next observation: {scores.flat[0]:.1f} nats | no gaze preference"
                + (" | full canvas: uniform unit precision" if self.full_canvas.isChecked() else "")
                + (" | sampling paused" if not self.precision.sample else "")
            )
            for item in (self.recommended_marker, self.direction_line, self.direction_arrow):
                item.hide()
            return
        target = np.asarray(information_gaze(scores, self.precision.gaze))
        iy, ix = round(target[1]*(scores.shape[0]-1)), round(target[0]*(scores.shape[1]-1))
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
        if self.autonomous_gaze.isChecked():
            self.precision = self.visualizer.source_orchestrator.visual_readout.config.spatial_precision
            self._gaze_precision = self.precision
            with QtCore.QSignalBlocker(self.gaze_x), QtCore.QSignalBlocker(self.gaze_y):
                self.gaze_x.setValue(self.precision.gaze[0])
                self.gaze_y.setValue(self.precision.gaze[1])
        sample = self.stimulus.latest
        if sample is not None:
            self.restart_button.setEnabled(True)
            self.progress.setText(
                f"Stimulus time: {sample.time:.2f} / {self.stimulus.config.duration:g} s",
            )
            if self.stimulus.exhausted:
                self.visualizer.timer.stop()
                self.refresh_previews()
                self.preview_timer.stop()
                self.progress.setText(f"Finished: {self.stimulus.frame_count} frames - Restart to continue")

    def refresh_previews(self):
        sample = self.stimulus.latest
        prediction = self.visualizer.visual_prediction
        if sample is None or prediction is None:
            return
        if not self.visualizer.renderer.prepare_frame():
            return
        self._refresh_precision_preview()
        truth = spatial_preview(sample.frame, max_side=self.PREVIEW_MAX_SIDE)
        self.truth_image.setImage(truth, autoLevels=False, levels=(0, 255))
        self.precision_background.setImage(truth, autoLevels=False, levels=(0, 255))
        preview = spatial_preview(prediction, max_side=self.PREVIEW_MAX_SIDE)
        self.prediction_image.setImage(np.clip(preview, 0, 1), autoLevels=False, levels=(0, 1))
        if self._score_dirty:
            # Advice evaluates a future observation even while actual sampling is paused.
            self.information_scores = self.uncertainty.expected_information_gain(
                replace(self.precision, sample=True), dt=1 / self.stimulus.config.frame_rate,
                prior=self.visualizer.engine.get_field(),
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

    def _restart(self):
        if not self.visualizer.renderer.prepare_frame():
            logger.warning("Cannot restart active vision before the canvas renderer is ready")
            return
        self.visualizer.timer.stop()
        self.preview_timer.stop()
        self.visualizer.engine.reset()
        self.visualizer._sync_after_reset()
        self.visualizer.camera_source.close()
        self.stimulus = CircularDotSource(self.stimulus.config)
        self.visualizer.camera_source.capture = CircularDotCapture(self.stimulus)
        self.visualizer.visual_prediction = None
        for image in (self.truth_image, self.precision_background, self.prediction_image, self.information_image):
            image.clear()
        self.information_scores = None
        self._update_precision()
        self.progress.setText("Restarting stimulus")
        self.restart_button.setEnabled(False)
        self.visualizer.timer.start()
        self.preview_timer.start()

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.visualizer.timer.stop()
        self.visualizer.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Mouse, autonomous, or full-canvas active vision.")
    parser.add_argument("--backend", choices=("numpy", "cupy", "opengl"), default="opengl")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(120, 160))
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--brownian-scale", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gaze", type=float, nargs=2, metavar=("X", "Y"), default=(0.5, 0.5))
    parser.add_argument("--gaze-mode", choices=("mouse", "autonomous", "full"), default="mouse")
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--baseline", type=float, default=0.01)
    parser.add_argument("--peak", type=float, default=1.0)
    parser.add_argument("--initial-variance", type=float, default=1.0)
    parser.add_argument(
        "--process-variance", type=float, default=0.01,
        help="Hidden-state diffusion intensity (variance per stimulus second).",
    )
    parser.add_argument("--memory-time-constant", type=float, default=1/3, help="Hidden-state memory in stimulus seconds.")
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
        memory_time_constant=args.memory_time_constant,
        gaze_mode=args.gaze_mode,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
