"""Continuous whole-scene neural beliefs with foveal visual evidence."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PyQt5 import QtCore, QtWidgets
import qdarkstyle

from audioviz.offline.continuous_scene_metrics import (
    ContinuousSceneWorldMetrics, measure_continuous_scene_step,
)
from audioviz.readouts import PredictiveCodingConfig
from audioviz.readouts.continuous_scene_predictive_coding import ContinuousScenePredictiveCodingGraph
from audioviz.sources import (
    AudioSourceConfig, CameraFrameSourceConfig, RippleSourceOrchestratorConfig, SyntheticSourceConfig,
)
from audioviz.sources.gaze import ActiveVisionConfig
from audioviz.sources.static_scene import BrownianScene, StaticSceneCapture
from audioviz.transforms.prediction_error import PredictionErrorTransformConfig
from audioviz.utils.array_backend import to_numpy
from audioviz.utils.spatial_preview import spatial_preview
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer


class _BenchVisualizer(RippleWaveVisualizer):
    gaze_changed = QtCore.pyqtSignal()

    def __init__(self, *, diagnostic_world, **kwargs):
        self.gaze_locked = True
        self._diagnostic_world = diagnostic_world
        super().__init__(**kwargs)

    def _capture_continuous_graph_preview(self):
        from audioviz.utils.continuous_graph_preview import continuous_graph_preview

        source = self.source_orchestrator
        return continuous_graph_preview(
            source.sensory_graph, self._diagnostic_world.frame.astype(np.float32) / 255,
            foveal_mask=source.graph_io.observation_mask, independent_reference=True,
        )

    def _set_mouse_gaze(self, x, y):
        if not self.gaze_locked:
            super()._set_mouse_gaze(x, y)
            self.gaze_changed.emit()

    def toggle_controls(self):
        super().toggle_controls()
        self.control_panel.beliefs_button.setEnabled(False)
        self.control_panel.beliefs_button.setToolTip(
            "Use the continuous-state bench plots; legacy Gaussian belief diagnostics are unavailable."
        )
        self.control_panel.wave_dynamics_button.setEnabled(False)
        self.control_panel.wave_dynamics_button.setToolTip("This continuous-scene baseline is wave-free.")


@dataclass(frozen=True)
class WorldStepMetadata:
    wave_propagation: bool
    brownian_enabled: bool
    brownian_scale: float
    world_offset: tuple[float, float]
    world_pixel_offset: tuple[int, int]


@dataclass(frozen=True)
class SceneStepMetrics(ContinuousSceneWorldMetrics, WorldStepMetadata):
    pass


class StaticWorldBench(QtWidgets.QWidget):
    """Advance the world/model per step, with bounded evaluator-only traces."""

    HISTORY_LIMIT = 512
    PREVIEW_MAX_SIDE = 256

    def __init__(
        self, *, resolution=(80, 120), foveal_radius=32, hidden_units=6,
        backend="opengl", dt=1/30, inference_steps=8, inference_rate=.1,
        motion_enabled=False, brownian_scale=4., memory_time_constant=1/3,
    ):
        super().__init__()
        if backend not in ("numpy", "cupy", "opengl"):
            raise ValueError("backend must be numpy, cupy or opengl")
        if isinstance(memory_time_constant, bool) or not np.isfinite(memory_time_constant) or memory_time_constant <= 0:
            raise ValueError("memory_time_constant must be finite and positive")
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (
            isinstance(hidden_units, bool) or not isinstance(hidden_units, int)
            or not 1 <= hidden_units <= ContinuousScenePredictiveCodingGraph.MAX_HIDDEN_NODES
        ):
            raise ValueError(f"hidden_units must be in 1..{ContinuousScenePredictiveCodingGraph.MAX_HIDDEN_NODES}")
        if not isinstance(motion_enabled, bool):
            raise ValueError("motion_enabled must be a boolean")
        if isinstance(brownian_scale, bool) or not np.isfinite(brownian_scale) or brownian_scale < 0:
            raise ValueError("brownian_scale must be finite and non-negative")
        self.world = BrownianScene(resolution)
        self.backend, self.dt = backend, float(dt)
        self.steps = 0
        self.history: deque[SceneStepMetrics] = deque(maxlen=self.HISTORY_LIMIT)
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
        self.vision_config = ActiveVisionConfig(
            shared_sensory_graph=True, world_anchored=True,
            foveal_radius=foveal_radius, memory_time_constant=memory_time_constant,
        )
        self.setWindowTitle("Test World - Continuous Scene Beliefs")
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
        self.lock_gaze.setToolTip("Hold the selected gaze for a fixed-patch trial.")
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
        world_controls = QtWidgets.QHBoxLayout()
        self.motion_enabled = QtWidgets.QCheckBox("Brownian scene motion")
        self.motion_enabled.setChecked(motion_enabled)
        self.motion_enabled.setToolTip(
            "Move the world each model step, even with Observe off. Uncheck to freeze its current position."
        )
        self.motion_scale = pg.SpinBox(value=brownian_scale, bounds=(0, None), step=.5)
        self.motion_scale.setEnabled(motion_enabled)
        self.motion_scale.setToolTip("Per-axis displacement standard deviation = scale * sqrt(dt).")
        self.motion_enabled.toggled.connect(self.motion_scale.setEnabled)
        world_controls.addWidget(self.motion_enabled)
        world_controls.addWidget(QtWidgets.QLabel("Scale (pixels/sqrt(s))"))
        world_controls.addWidget(self.motion_scale)
        world_controls.addWidget(QtWidgets.QLabel("World seed: 0; New trial resets motion."))
        world_controls.addStretch()
        layout.addLayout(world_controls)
        state_controls = QtWidgets.QHBoxLayout()
        self.memory_tau = pg.SpinBox(value=memory_time_constant, bounds=(1e-6, None), step=.05)
        self.memory_tau.setMaximumWidth(130)
        self.memory_tau.sigValueChanged.connect(self._update_memory_time_constant)
        state_controls.addWidget(QtWidgets.QLabel("Continuous leak time constant (seconds)"))
        state_controls.addWidget(self.memory_tau)
        state_controls.addWidget(QtWidgets.QLabel("One evolving scene field; no spikes or last-visit image cache."))
        state_controls.addStretch()
        layout.addLayout(state_controls)
        self.description = description = QtWidgets.QLabel(
            "Static test world; no camera or microphone. Every image location has a continuous neural state; "
            "the fovea gates evidence, not dynamics. Model controls change integration and weight/precision learning. "
            "Move the mouse over any image to select gaze; Step/Run samples it. "
            "For fixed-patch trials, lock gaze and turn scene motion off. "
            "New trial keeps model settings/gaze but restores seed-0 weights, initial precisions, states and zero canvas. "
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
        self.world_image = self._image_panel("World (next gaze; evaluator only)", 0)
        self.forecast_image = self._image_panel("Whole-scene forecast (live neural state)", 1)
        self.error_image = self._image_panel("Whole-world forecast RGB RMSE (0-1; evaluator only)", 2)
        self.images.scene().sigMouseMoved.connect(self._move_image_gaze)
        self.error_colormap = pg.colormap.get("magma")
        self.splitter.addWidget(self.images)
        self.plots = pg.GraphicsLayoutWidget()
        self.mse_plot = self.plots.addPlot(row=0, col=0, title="Across steps: observed RGB MSE")
        self.mse_plot.addLegend(offset=(10, 0))
        self.mse_plot.setLabel("bottom", "Model/world step")
        self.mse_plot.setLabel("left", "Observed RGB MSE")
        self.forecast_curve = self.mse_plot.plot(pen=pg.mkPen("#4ddde0", width=2), name="Pre-evidence forecast")
        self.settled_curve = self.mse_plot.plot(pen=pg.mkPen("#ffb454", width=2), name="Post-evidence decoder (before learning)")
        self.energy_plot = self.plots.addPlot(row=0, col=1, title="Within latest interval: continuous-state energy")
        self.energy_plot.addLegend(offset=(10, 10), colCount=3)
        self.energy_plot.setLabel("bottom", "Integration microsteps")
        self.energy_plot.setLabel("left", "Gaussian-factor energy (sum)")
        for plot in (self.mse_plot, self.energy_plot):
            plot.getAxis("left").enableAutoSIPrefix(False)
        self.energy_curve = self.energy_plot.plot(pen=pg.mkPen("#f6df75", width=2), symbol="o", symbolSize=5, name="Total energy")
        self.hidden_final = self.energy_plot.plot(pen=None, symbol="s", symbolBrush="#4ddde0", name="Final hidden energy")
        self.sensory_final = self.energy_plot.plot(pen=None, symbol="t", symbolBrush="#ffb454", name="Final sensory energy")
        self.splitter.addWidget(self.plots)
        self.state_plots = pg.GraphicsLayoutWidget()
        potential = self.state_plots.addPlot(row=0, col=0, title="Whole-scene neural state")
        potential.setLabel("left", "Logit-field L2 norm")
        potential.setLabel("bottom", "Model/world step")
        self.visual_norm_curve = potential.plot(pen=pg.mkPen("#4ddde0", width=2))
        hidden = self.state_plots.addPlot(row=0, col=1, title="Shared recurrent hidden state")
        hidden.setLabel("left", "|h|")
        hidden.getAxis("left").enableAutoSIPrefix(False)
        hidden.setLabel("bottom", "Model/world step")
        self.hidden_norm_curve = hidden.plot(pen=pg.mkPen("#ffb454", width=2))
        neutral = self.state_plots.addPlot(row=0, col=2, title="Whole-scene forecast distance from neutral")
        neutral.setLabel("left", "Mean |RGB - 0.5|")
        neutral.getAxis("left").enableAutoSIPrefix(False)
        neutral.setLabel("bottom", "Model/world step")
        self.neutral_curve = neutral.plot(pen=pg.mkPen("#4ddde0", width=2))
        self.splitter.addWidget(self.state_plots)
        self.plots.setMinimumHeight(160)
        self.state_plots.setMinimumHeight(160)
        self.footer = footer = QtWidgets.QLabel()
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(max(1, round(1000*self.dt)))
        self.timer.timeout.connect(self._advance)
        self.visualizer = None
        self.new_trial()

    @property
    def scene(self):
        return self.world.frame

    def _update_memory_time_constant(self):
        if self.visualizer is not None:
            graph = self.visualizer.source_orchestrator.sensory_graph
            graph.set_memory_time_constant(float(self.memory_tau.value()))

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

    def _move_image_gaze(self, scene_position):
        rows, cols = self.scene.shape[:2]
        for image in (self.world_image, self.forecast_image, self.error_image):
            if image.contains(image.mapFromScene(scene_position)):
                point = image.getViewBox().mapSceneToView(scene_position)
                self.visualizer._set_mouse_gaze(
                    (point.x()-.5)/max(cols-1, 1),
                    (point.y()-.5)/max(rows-1, 1),
                )
                return

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
                mode="mouse",
                precision=self.visualizer.source_orchestrator.visual_readout.config.spatial_precision,
            )
            self.visualizer.close()
            self.visualizer.setParent(None)
            self.visualizer.deleteLater()
        self.vision_config = replace(
            self.vision_config,
            memory_time_constant=float(self.memory_tau.value()),
        )
        self.world.reset()
        self.capture = StaticSceneCapture(self.scene)
        self.visualizer = _BenchVisualizer(
            diagnostic_world=self.world,
            resolution=self.scene.shape[:2], plane_size_m=(1, 1),
            use_shader=self.backend == "opengl", use_gpu=self.backend == "cupy",
            rgb_canvas_enabled=True, camera_capture=self.capture,
            source_orchestrator_config=self.source_config, active_vision=self.vision_config,
            update_interval_ms=self.timer.interval(), forecast_dt=self.dt,
            wave_propagation_enabled=False,
            shared_graph_factory=ContinuousScenePredictiveCodingGraph,
        )
        self.visualizer.timer.stop()
        self.visualizer.setMinimumHeight(190)
        self._update_memory_time_constant()
        self.visualizer.gaze_controls.autonomous_gaze.setToolTip(
            "Maximize one-step expected joint-state information gain; no outcome preferences or temporal planning.",
        )
        self.visualizer.diagnostics_button.setEnabled(False)
        self.visualizer.diagnostics_button.setToolTip("Use the whole-scene state plots in the bench.")
        self.visualizer.scene_button.setEnabled(True)
        self.visualizer.scene_button.setToolTip("Open image layouts or the live continuous graph; uncheck Live to freeze it.")
        self.visualizer.gaze_description = (
            "Bias-free continuous scene graph: one world-anchored neural state per RGB coordinate, "
            "coupled to one shared recurrent hidden population. Every state evolves every step. "
            "The circular fovea gates camera evidence only. Predicted RGB is sigmoid(neural state); "
            "there is no last-visit image cache. Autonomous gaze uses one-step local Gaussian information gain, "
            "not whole-world evaluator truth, outcome preferences or temporal planning."
        )
        self.visualizer.gaze_status.setToolTip(self.visualizer.gaze_description)
        self.footer.setText(
            "Evidence is foveal; neural dynamics and decoding cover the whole scene. "
            "Learned connections persist while unforced state returns toward neutral gray. "
            "Observed MSE is unavailable without evidence; whole-world error is evaluator-only. "
            "No spiking or wave dynamics. "
            f"Plots/export retain the latest {self.HISTORY_LIMIT} steps."
        )
        self.visualizer.gaze_changed.connect(self._refresh_gaze_outlines)
        self.visualizer.gaze_controls.precision_changed.connect(self._refresh_gaze_outlines)
        self.splitter.addWidget(self.visualizer)
        self.visualizer.show()
        self._set_gaze_lock(self.lock_gaze.isChecked())
        self.history.clear()
        self.steps = 0
        for curve in (
            self.forecast_curve, self.settled_curve, self.energy_curve, self.hidden_final,
            self.sensory_final, self.neutral_curve, self.hidden_norm_curve, self.visual_norm_curve,
        ):
            curve.clear()
        self.status.setText("Paused at step 0. Fresh seed-0 weights; no evidence consumed. Step or Run to begin.")
        self._refresh_images()
        self.splitter.setSizes((200, 160, 160, 240))

    def step_once(self) -> SceneStepMetrics:
        if not self.visualizer.renderer.prepare_frame():
            raise RuntimeError("Renderer is not ready; show the bench before stepping.")
        moving = self.motion_enabled.isChecked()
        scale = float(self.motion_scale.value())
        self.world.advance(self.dt, scale=scale if moving else 0.)
        self.capture.set_frame(self.scene)
        before = self.visualizer.engine.time
        self.visualizer.update_visualization()
        if self.visualizer.engine.time == before:
            raise RuntimeError("Renderer is not ready; show the bench before stepping.")
        self.steps += 1
        source = self.visualizer.source_orchestrator
        measured = measure_continuous_scene_step(
            source.sensory_graph, step=self.steps, dt=self.dt, gaze=source.graph_io.gaze,
        )
        metrics = SceneStepMetrics(
            **asdict(measured), brownian_enabled=moving, brownian_scale=scale,
            wave_propagation=self.visualizer.engine.wave_propagation_enabled,
            world_offset=tuple(float(value) for value in self.world.offset),
            world_pixel_offset=self.world.pixel_offset,
        )
        self.history.append(metrics)
        x = [frame.step for frame in self.history]
        self.forecast_curve.setData(x, [np.nan if frame.forecast_mse is None else frame.forecast_mse for frame in self.history], connect="finite")
        self.settled_curve.setData(x, [np.nan if frame.settled_mse is None else frame.settled_mse for frame in self.history], connect="finite")
        energy = metrics.inference_energy
        self.energy_curve.setData(np.arange(len(energy)), energy)
        self.hidden_final.setData([len(energy)-1], [metrics.hidden_energy])
        self.sensory_final.setData([len(energy)-1], [metrics.sensory_energy])
        self.neutral_curve.setData(x, [frame.neutral_visual_mae for frame in self.history])
        self.hidden_norm_curve.setData(x, [frame.decoded_hidden_norm for frame in self.history])
        self.visual_norm_curve.setData(x, [frame.visual_potential_norm for frame in self.history])
        errors = (
            "no visual evidence; MSE unavailable" if metrics.forecast_mse is None else
            f"forecast MSE {metrics.forecast_mse:.5f}; settled MSE {metrics.settled_mse:.5f}"
        )
        self.status.setText(
            f"Step {self.steps} | sampled gaze ({metrics.gaze[0]:.3f}, {metrics.gaze[1]:.3f}) | "
            f"scene offset {metrics.world_pixel_offset} px | "
            f"{metrics.observed_rgb} observed RGB values | {errors} | "
            f"E {metrics.energy_initial:.5g} -> {metrics.energy_final:.5g} | "
            f"steps {metrics.inference_steps}, rate {metrics.inference_rate:g}, "
            f"learning {'on' if metrics.learning_enabled else 'off'}, updates {metrics.learning_updates}, "
            f"canvas rate {metrics.canvas_rate:g}; waves {'on' if metrics.wave_propagation else 'off'}"
        )
        self.status.setText(
            self.status.text()+f" | tau {metrics.memory_time_constant:g}s"
            +f" | |h| {metrics.decoded_hidden_norm:.4g}, |z| {metrics.visual_potential_norm:.4g}"
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
        reference = spatial_preview(self.scene, max_side=self.PREVIEW_MAX_SIDE).astype(np.float32) / 255
        preview = (
            np.full_like(reference, .5) if prediction is None
            else spatial_preview(prediction, max_side=self.PREVIEW_MAX_SIDE)
        )
        self.forecast_image.setImage(preview, autoLevels=False, levels=(0, 1))
        rmse = np.sqrt(np.mean((reference - preview)**2, axis=-1))
        error_image = self.error_colormap.map(rmse, mode="byte")
        self.error_image.setImage(error_image, autoLevels=False)
        for image in (self.world_image, self.forecast_image, self.error_image):
            image.setRect(bounds)
        self._refresh_gaze_outlines()

    def _refresh_gaze_outlines(self):
        rows, cols = self.scene.shape[:2]
        source = self.visualizer.source_orchestrator
        io = source.graph_io
        selected_gaze = source.visual_readout.config.spatial_precision.gaze
        theta = np.linspace(0, 2*np.pi, 97)
        for index, outline in enumerate(self.outlines):
            gaze = selected_gaze if index == 0 else io.gaze
            center = np.rint(np.asarray(gaze)*(cols-1, rows-1))+.5
            outline.setData(center[0]+io.radius*np.cos(theta), center[1]+io.radius*np.sin(theta))

    def save_trace(self, path):
        graph = self.visualizer.source_orchestrator.sensory_graph
        payload = dict(
            model="continuous-scene",
            scene="brownian-geometric-v1", resolution=self.scene.shape[:2],
            world_seed=self.world.seed, world_time=self.world.time,
            motion_enabled=self.motion_enabled.isChecked(), brownian_scale=float(self.motion_scale.value()),
            backend=self.backend, foveal_radius=self.vision_config.foveal_radius,
            hidden_units=graph.hidden_nodes, seed=0, forecast_dt=self.dt,
            wave_propagation=self.visualizer.engine.wave_propagation_enabled,
            history_limit=self.HISTORY_LIMIT, total_steps=self.steps,
            current_config=asdict(graph.config), frames=[asdict(frame) for frame in self.history],
        )
        payload["memory_time_constant"] = graph.memory_time_constant
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
    parser.add_argument("--memory-time-constant", type=float, default=1/3, help="Continuous neural leak time constant, in seconds.")
    parser.add_argument("--resolution", nargs=2, type=int, default=(80, 120), metavar=("ROWS", "COLS"))
    parser.add_argument("--foveal-radius", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8, help="Integration microsteps per physical interval.")
    parser.add_argument("--rate", type=float, default=.1, help="Continuous inference gain.")
    parser.add_argument("--dt", type=float, default=1/30, help="Fixed model/world interval per step, in seconds.")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--static-scene", dest="motion_enabled", action="store_false", help="Static world (the default).")
    motion.add_argument("--moving-scene", dest="motion_enabled", action="store_true", help="Opt into the existing Brownian scene-motion experiment.")
    parser.set_defaults(motion_enabled=False)
    parser.add_argument("--brownian-scale", type=float, default=4., help="Brownian displacement scale, pixels/sqrt(second).")
    args = parser.parse_args(argv)
    if QtWidgets.QApplication.instance() is None:
        # Graph windows reuse pyqtgraph's cached shader programs.
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = StaticWorldBench(
        resolution=tuple(args.resolution), foveal_radius=args.foveal_radius,
        hidden_units=args.hidden_units, backend=args.backend,
        dt=args.dt, inference_steps=args.steps, inference_rate=args.rate,
        motion_enabled=args.motion_enabled, brownian_scale=args.brownian_scale,
        memory_time_constant=args.memory_time_constant,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
