"""Static-scene experiments using the feature application's foveal model."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PyQt5 import QtCore, QtWidgets
import qdarkstyle

from audioviz.offline.static_world_metrics import StaticWorldMetrics, measure_static_world_step
from audioviz.readouts import PredictiveCodingConfig, SharedPredictiveCodingGraph
from audioviz.sources import (
    AudioSourceConfig, CameraFrameSourceConfig, RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)
from audioviz.sources.gaze import ActiveVisionConfig
from audioviz.sources.static_scene import geometric_scene, StaticSceneCapture
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.utils.array_backend import to_numpy
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


class _BenchVisualizer(RippleWaveVisualizer):
    def __init__(self, **kwargs):
        self.gaze_locked = True
        super().__init__(**kwargs)

    def _set_mouse_gaze(self, x, y):
        if not self.gaze_locked:
            super()._set_mouse_gaze(x, y)

    def _refresh_attention_preview(self):
        diagnostics_visible = self.belief_diagnostics is not None and self.belief_diagnostics.isVisible()
        if not self.gaze_locked or diagnostics_visible:
            super()._refresh_attention_preview()


class StaticWorldBench(QtWidgets.QWidget):
    """One observation per step, with bounded evaluator-only traces."""

    HISTORY_LIMIT = 512
    PREVIEW_MAX_SIDE = 256

    def __init__(
        self, *, resolution=(80, 120), foveal_radius=32, hidden_units=6,
        backend="opengl", dt=1/30, inference_steps=8, inference_rate=.1,
    ):
        super().__init__()
        if backend not in ("numpy", "cupy", "opengl"):
            raise ValueError("backend must be numpy, cupy or opengl")
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (
            isinstance(hidden_units, bool) or not isinstance(hidden_units, int)
            or not 1 <= hidden_units <= SharedPredictiveCodingGraph.MAX_HIDDEN_NODES
        ):
            raise ValueError(f"hidden_units must be in 1..{SharedPredictiveCodingGraph.MAX_HIDDEN_NODES}")
        self.scene = geometric_scene(resolution)
        self.backend, self.dt = backend, float(dt)
        self.steps = 0
        self.history: deque[StaticWorldMetrics] = deque(maxlen=self.HISTORY_LIMIT)
        self.source_config = RippleSourceOrchestratorConfig(
            synthetic=SyntheticSourceConfig(enabled=False),
            audio=AudioSourceConfig(enabled=False),
            camera_frame=CameraFrameSourceConfig(enabled=True),
            prediction_error=PredictionErrorTransformConfig(enabled=True),
            visual_pathway=PredictiveCodingConfig(
                dtype="float32" if backend == "opengl" else "float64",
                hidden_channels=hidden_units, inference_steps=inference_steps,
                inference_rate=inference_rate, canvas_rate=0,
                learning_enabled=False, learning_rate=.03, weight_decay=0,
            ),
        )
        self.vision_config = ActiveVisionConfig(shared_sensory_graph=True, foveal_radius=foveal_radius)
        self.setWindowTitle("Static World - Foveal Predictive Coding Bench")
        self.resize(1440, 1000)
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setCheckable(True)
        self.run_button.toggled.connect(self._set_running)
        self.step_button = QtWidgets.QPushButton("Step")
        self.step_button.clicked.connect(self._advance)
        self.new_trial_button = QtWidgets.QPushButton("New trial (fresh weights)")
        self.new_trial_button.clicked.connect(self.new_trial)
        self.controls_button = QtWidgets.QPushButton("Model controls")
        self.controls_button.clicked.connect(self._show_model_controls)
        self.lock_gaze = QtWidgets.QCheckBox("Lock gaze")
        self.lock_gaze.setChecked(True)
        self.lock_gaze.toggled.connect(self._set_gaze_lock)
        self.export_button = QtWidgets.QPushButton("Save trace")
        self.export_button.clicked.connect(self._save_trace)
        for widget in (
            self.run_button, self.step_button, self.new_trial_button, self.controls_button,
            self.lock_gaze, self.export_button,
        ):
            toolbar.addWidget(widget)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        description = QtWidgets.QLabel(
            "Static world; no camera or microphone. Model controls change settling steps/rate, learning, and canvas feedback. "
            "New trial keeps model settings/gaze but restores seed-0 weights, hidden state and zero canvas; "
            "other source/wave settings return to bench defaults. "
            f"Each step uses forecast dt={self.dt:g} s; playback speed does not change this clock."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(self.splitter, 1)
        self.images = pg.GraphicsLayoutWidget()
        self.outlines = []
        self.world_image = self._image_panel("World (evaluator only)", 0)
        self.forecast_image = self._image_panel("Forecast (outside fovea retained)", 1)
        self.error_image = self._image_panel("Forecast RGB RMSE (0-1; gray = unobserved)", 2)
        self.error_colormap = pg.colormap.get("magma")
        self.splitter.addWidget(self.images)
        self.plots = pg.GraphicsLayoutWidget()
        self.mse_plot = self.plots.addPlot(row=0, col=0, title="Across observations: observed RGB MSE")
        self.mse_plot.addLegend()
        self.mse_plot.setLabel("bottom", "Observation step")
        self.mse_plot.setLabel("left", "MSE per observed RGB value")
        self.forecast_curve = self.mse_plot.plot(pen=pg.mkPen("#4ddde0", width=2), name="Pre-evidence forecast")
        self.settled_curve = self.mse_plot.plot(pen=pg.mkPen("#ffb454", width=2), name="Settled decoder (before learning)")
        self.energy_plot = self.plots.addPlot(row=0, col=1, title="Within latest observation: conditioning energy")
        self.energy_plot.addLegend()
        self.energy_plot.setLabel("bottom", "Accepted hidden updates")
        self.energy_plot.setLabel("left", "Residual energy (sum)")
        for plot in (self.mse_plot, self.energy_plot):
            plot.getAxis("left").enableAutoSIPrefix(False)
        self.energy_curve = self.energy_plot.plot(pen=pg.mkPen("#f6df75", width=2), symbol="o", symbolSize=5, name="Total energy")
        self.hidden_final = self.energy_plot.plot(pen=None, symbol="s", symbolBrush="#4ddde0", name="Final hidden energy")
        self.sensory_final = self.energy_plot.plot(pen=None, symbol="t", symbolBrush="#ffb454", name="Final sensory energy")
        self.splitter.addWidget(self.plots)
        footer = QtWidgets.QLabel(
            "Only observed foveal RGB values enter MSE; no-evidence steps are gaps, not zero error. "
            "Settled error uses decoder residuals, not clamped sensory states. "
            "Total energy decreases within a fixed conditioning objective; visual MSE and next-step forecast need not. "
            "Defaults hold weights and root fixed. Canvas feedback enables live wave dynamics. "
            f"Plots/export retain the latest {self.HISTORY_LIMIT} steps."
        )
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(max(1, round(1000*self.dt)))
        self.timer.timeout.connect(self._advance)
        self.visualizer = None
        self.new_trial()
        self.splitter.setSizes((270, 220, 330))

    def _image_panel(self, title, column):
        plot = self.images.addPlot(row=0, col=column, title=title)
        plot.setAspectLocked()
        plot.invertY(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMouseEnabled(x=False, y=False)
        rows, cols = self.scene.shape[:2]
        plot.setRange(xRange=(0, cols), yRange=(0, rows), padding=0)
        image = pg.ImageItem(axisOrder="row-major")
        plot.addItem(image)
        outline = pg.PlotCurveItem(pen=pg.mkPen("#ff65df", width=1.5))
        plot.addItem(outline)
        self.outlines.append(outline)
        return image

    def _set_running(self, running):
        self.run_button.setText("Pause" if running else "Run")
        self.step_button.setEnabled(not running)
        self.timer.start() if running else self.timer.stop()

    def _set_gaze_lock(self, locked):
        self.visualizer.gaze_locked = locked
        controls = self.visualizer.gaze_controls
        if locked:
            controls.follow_mouse.setChecked(True)
        controls.follow_mouse.setEnabled(not locked)
        controls.autonomous_gaze.setEnabled(not locked)
        controls.gaze_x.setEnabled(not locked and controls.follow_mouse.isChecked())
        controls.gaze_y.setEnabled(not locked and controls.follow_mouse.isChecked())

    def _show_model_controls(self):
        first_open = self.visualizer.control_panel is None
        self.visualizer.controls_button.click()
        if first_open:
            panel = self.visualizer.control_panel
            panel.section_widgets["gaze"].set_expanded(False)
            panel.section_widgets["camera-source"].set_expanded(False)

    def new_trial(self):
        self.run_button.setChecked(False)
        if self.visualizer is not None:
            self.source_config = replace(
                self.source_config,
                visual_pathway=replace(
                    self.visualizer.source_orchestrator.visual_readout.config,
                    dtype=self.source_config.visual_pathway.dtype,
                ),
            )
            self.vision_config = replace(
                self.vision_config,
                mode="autonomous" if self.visualizer.gaze_controls.autonomous_gaze.isChecked() else "mouse",
                precision=self.visualizer.source_orchestrator.visual_readout.config.spatial_precision,
            )
            self.visualizer.close()
            self.visualizer.setParent(None)
            self.visualizer.deleteLater()
        self.capture = StaticSceneCapture(self.scene)
        self.visualizer = _BenchVisualizer(
            resolution=self.scene.shape[:2], plane_size_m=(1, 1),
            use_shader=self.backend == "opengl", use_gpu=self.backend == "cupy",
            rgb_canvas_enabled=True, camera_capture=self.capture,
            source_orchestrator_config=self.source_config, active_vision=self.vision_config,
            update_interval_ms=self.timer.interval(), forecast_dt=self.dt,
        )
        self.visualizer.timer.stop()
        self.splitter.addWidget(self.visualizer)
        self.visualizer.show()
        self._set_gaze_lock(self.lock_gaze.isChecked())
        self.history.clear()
        self.steps = 0
        for curve in (self.forecast_curve, self.settled_curve, self.energy_curve, self.hidden_final, self.sensory_final):
            curve.clear()
        self.status.setText("Paused at step 0. Fresh seed-0 weights; no evidence consumed. Step or Run to begin.")
        self._refresh_images()

    def step_once(self) -> StaticWorldMetrics:
        before = self.visualizer.engine.time
        self.visualizer.update_visualization()
        if self.visualizer.engine.time == before:
            raise RuntimeError("Renderer is not ready; show the bench before stepping.")
        self.steps += 1
        source = self.visualizer.source_orchestrator
        metrics = measure_static_world_step(
            source.sensory_graph, step=self.steps, dt=self.dt, gaze=source.graph_io.gaze,
        )
        self.history.append(metrics)
        x = [frame.step for frame in self.history]
        self.forecast_curve.setData(x, [np.nan if frame.forecast_mse is None else frame.forecast_mse for frame in self.history], connect="finite")
        self.settled_curve.setData(x, [np.nan if frame.settled_mse is None else frame.settled_mse for frame in self.history], connect="finite")
        energy = metrics.inference_energy
        self.energy_curve.setData(np.arange(len(energy)), energy)
        self.hidden_final.setData([len(energy)-1], [metrics.hidden_energy])
        self.sensory_final.setData([len(energy)-1], [metrics.sensory_energy])
        errors = (
            "no visual evidence; MSE unavailable" if metrics.forecast_mse is None else
            f"forecast MSE {metrics.forecast_mse:.5f}; settled MSE {metrics.settled_mse:.5f}"
        )
        self.status.setText(
            f"Step {self.steps} | {metrics.observed_rgb} observed RGB values | {errors} | "
            f"E {metrics.energy_initial:.5g} -> {metrics.energy_final:.5g} | "
            f"steps {metrics.inference_steps}, rate {metrics.inference_rate:g}, "
            f"learning {'on' if metrics.learning_enabled else 'off'}, updates {metrics.learning_updates}, "
            f"canvas rate {metrics.canvas_rate:g}"
        )
        self._refresh_images()
        return metrics

    def _advance(self):
        try:
            self.step_once()
        except (ValueError, RuntimeError, FloatingPointError) as error:
            self.run_button.setChecked(False)
            logger.exception("Static-world step failed")
            self.status.setText(f"Experiment stopped: {error}")

    def _refresh_images(self):
        rows, cols = self.scene.shape[:2]
        bounds = QtCore.QRectF(0, 0, cols, rows)
        self.world_image.setImage(self.scene, autoLevels=False, levels=(0, 255))
        prediction = self.visualizer.visual_prediction
        preview = (
            np.full((*self.scene.shape[:2], 3), .5, dtype=np.float32) if prediction is None
            else spatial_preview(prediction, max_side=self.PREVIEW_MAX_SIDE)
        )
        self.forecast_image.setImage(preview, autoLevels=False, levels=(0, 1))
        error_image = np.full((rows, cols, 4), (65, 65, 65, 255), dtype=np.uint8)
        source = self.visualizer.source_orchestrator
        graph, io = source.sensory_graph, source.graph_io
        observed = to_numpy(graph.clamped[graph.visual_slice]).reshape(-1, 3).all(axis=1) & io.valid
        if observed.any():
            evidence = to_numpy(graph.evidence[graph.visual_slice]).reshape(-1, 3)[observed]
            forecast = to_numpy(graph.prior_mean[graph.visual_slice]).reshape(-1, 3)[observed]
            rmse = np.sqrt(np.mean((evidence-forecast)**2, axis=1))
            error_image[tuple(io.positions[observed].T)] = self.error_colormap.map(rmse, mode="byte")
        self.error_image.setImage(error_image, autoLevels=False)
        for image in (self.world_image, self.forecast_image, self.error_image):
            image.setRect(bounds)
        theta = np.linspace(0, 2*np.pi, 97)
        center = np.rint(np.asarray(io.gaze)*(cols-1, rows-1))+.5
        for outline in self.outlines:
            outline.setData(center[0]+io.radius*np.cos(theta), center[1]+io.radius*np.sin(theta))

    def save_trace(self, path):
        graph = self.visualizer.source_orchestrator.sensory_graph
        payload = dict(
            scene="static-geometric-v1", resolution=self.scene.shape[:2],
            backend=self.backend, foveal_radius=self.vision_config.foveal_radius,
            hidden_units=graph.hidden_nodes, seed=0, forecast_dt=self.dt,
            wave_propagation=True, history_limit=self.HISTORY_LIMIT, total_steps=self.steps,
            current_config=asdict(graph.config), frames=[asdict(frame) for frame in self.history],
        )
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")

    def _save_trace(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save experiment trace", "static-world.json", "JSON (*.json)")
        if path:
            try:
                self.save_trace(path)
            except OSError as error:
                logger.exception("Could not save static-world trace")
                QtWidgets.QMessageBox.critical(self, "Trace save failed", str(error))

    def closeEvent(self, event):
        self.timer.stop()
        self.visualizer.close()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "cupy", "opengl"), default="opengl")
    parser.add_argument("--resolution", nargs=2, type=int, default=(80, 120), metavar=("ROWS", "COLS"))
    parser.add_argument("--foveal-radius", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8, help="Maximum hidden settling updates per forecast and conditioning.")
    parser.add_argument("--rate", type=float, default=.1, help="Initial hidden relaxation step size (backtracking may reduce it).")
    parser.add_argument("--dt", type=float, default=1/30, help="Fixed model forecast interval per observation, in seconds.")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = StaticWorldBench(
        resolution=tuple(args.resolution), foveal_radius=args.foveal_radius,
        hidden_units=args.hidden_units, backend=args.backend,
        dt=args.dt, inference_steps=args.steps, inference_rate=args.rate,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
