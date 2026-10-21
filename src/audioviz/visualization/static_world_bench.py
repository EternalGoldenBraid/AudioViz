"""PC control and RC-spiking prototype with static or Brownian scene motion."""
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

from audioviz.offline.static_world_metrics import (
    StaticWorldMetrics, RCSpikingWorldMetrics, measure_static_world_step, measure_rc_spiking_step,
)
from audioviz.readouts import PredictiveCodingConfig, SharedPredictiveCodingGraph, RCSpikingPredictiveCodingGraph
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

    def __init__(self, **kwargs):
        self.gaze_locked = True
        super().__init__(**kwargs)

    def _set_mouse_gaze(self, x, y):
        if not self.gaze_locked:
            super()._set_mouse_gaze(x, y)
            self.gaze_changed.emit()

    def _refresh_attention_preview(self):
        if isinstance(self.source_orchestrator.sensory_graph, RCSpikingPredictiveCodingGraph):
            self.attention_scores = None
            return
        diagnostics_visible = self.belief_diagnostics is not None and self.belief_diagnostics.isVisible()
        if self.gaze_controls.autonomous_gaze.isChecked() or diagnostics_visible:
            super()._refresh_attention_preview()
        else:
            self.attention_scores = None

    def toggle_controls(self):
        super().toggle_controls()
        if isinstance(self.source_orchestrator.sensory_graph, RCSpikingPredictiveCodingGraph):
            self.control_panel.beliefs_button.setEnabled(False)
            self.control_panel.beliefs_button.setToolTip("RC prototype: use the bench spike and decay plots; Gaussian diagnostics are unavailable.")


@dataclass(frozen=True)
class WorldStepMetadata:
    wave_propagation: bool
    brownian_enabled: bool
    brownian_scale: float
    world_offset: tuple[float, float]
    world_pixel_offset: tuple[int, int]


@dataclass(frozen=True)
class SceneStepMetrics(StaticWorldMetrics, WorldStepMetadata):
    pass


@dataclass(frozen=True)
class RCSceneStepMetrics(RCSpikingWorldMetrics, WorldStepMetadata):
    pass


class StaticWorldBench(QtWidgets.QWidget):
    """Advance the world/model per step, with bounded evaluator-only traces."""

    HISTORY_LIMIT = 512
    PREVIEW_MAX_SIDE = 256

    def __init__(
        self, *, resolution=(80, 120), foveal_radius=32, hidden_units=6,
        backend="opengl", dt=1/30, inference_steps=8, inference_rate=.1,
        motion_enabled=True, brownian_scale=4.,
        model="pc", memory_time_constant=1/3, spike_threshold=.02, membrane_capacitance=.02,
    ):
        super().__init__()
        if backend not in ("numpy", "cupy", "opengl"):
            raise ValueError("backend must be numpy, cupy or opengl")
        if model not in ("pc", "rc-spiking"):
            raise ValueError("model must be pc or rc-spiking")
        for name, value in (
            ("memory_time_constant", memory_time_constant), ("spike_threshold", spike_threshold),
            ("membrane_capacitance", membrane_capacitance),
        ):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (
            isinstance(hidden_units, bool) or not isinstance(hidden_units, int)
            or not 1 <= hidden_units <= SharedPredictiveCodingGraph.MAX_HIDDEN_NODES
        ):
            raise ValueError(f"hidden_units must be in 1..{SharedPredictiveCodingGraph.MAX_HIDDEN_NODES}")
        if not isinstance(motion_enabled, bool):
            raise ValueError("motion_enabled must be a boolean")
        if isinstance(brownian_scale, bool) or not np.isfinite(brownian_scale) or brownian_scale < 0:
            raise ValueError("brownian_scale must be finite and non-negative")
        self.world = BrownianScene(resolution)
        self.backend, self.dt = backend, float(dt)
        self.steps = 0
        self.history: deque[SceneStepMetrics | RCSceneStepMetrics] = deque(maxlen=self.HISTORY_LIMIT)
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
        self.setWindowTitle("Test World - Foveal Predictive Coding Bench")
        self.resize(1440, 1000)
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItem("Predictive coding (control)", "pc")
        self.model_selector.addItem("RC-spiking (prototype)", "rc-spiking")
        self.model_selector.setCurrentIndex(self.model_selector.findData(model))
        self.model_selector.setToolTip("Changing model starts a fresh seed-0 trial; production main is unchanged.")
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
            self.model_selector, self.run_button, self.step_button, self.new_trial_button, self.controls_button,
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
        self.rc_controls = QtWidgets.QWidget()
        rc_layout = QtWidgets.QHBoxLayout(self.rc_controls)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        self.rc_tau = pg.SpinBox(value=memory_time_constant, bounds=(1e-6, None), step=.05)
        self.rc_threshold = pg.SpinBox(value=spike_threshold, bounds=(1e-6, None), step=.005)
        self.rc_capacitance = pg.SpinBox(value=membrane_capacitance, bounds=(1e-6, None), step=.005)
        self.rc_resistance = QtWidgets.QLabel()
        for label, widget in (
            ("RC tau = R*C (seconds)", self.rc_tau), ("Capacitance C (model units)", self.rc_capacitance),
            ("Ternary spike threshold", self.rc_threshold),
        ):
            rc_layout.addWidget(QtWidgets.QLabel(label))
            rc_layout.addWidget(widget)
            widget.setMaximumWidth(130)
            widget.sigValueChanged.connect(self._update_rc_parameters)
        rc_layout.addWidget(self.rc_resistance)
        rc_layout.addWidget(QtWidgets.QLabel("Mouse/locked gaze only; Gaussian EIG unavailable."))
        rc_layout.addStretch()
        layout.addWidget(self.rc_controls)
        self.description = description = QtWidgets.QLabel(
            "Test world; no camera or microphone. Model controls change PC settling or RC integration, learning, and canvas feedback. "
            "Move the mouse over any image to select gaze; Step/Run samples it. "
            "For fixed-patch trials, lock gaze and turn scene motion off. "
            "New trial keeps model settings/gaze and the wave switch but restores seed-0 weights, hidden state and zero canvas; "
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
        self.world_image = self._image_panel("World (next gaze; evaluator only)", 0)
        self.forecast_image = self._image_panel("Forecast cache (outside fovea retained)", 1)
        self.error_image = self._image_panel("Forecast RGB RMSE (0-1; gray = unobserved)", 2)
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
        self.energy_plot = self.plots.addPlot(row=0, col=1, title="Within latest step: conditioning energy")
        self.energy_plot.addLegend(offset=(10, 10), colCount=3)
        self.energy_plot.setLabel("bottom", "Accepted hidden updates")
        self.energy_plot.setLabel("left", "Residual energy (sum)")
        for plot in (self.mse_plot, self.energy_plot):
            plot.getAxis("left").enableAutoSIPrefix(False)
        self.energy_curve = self.energy_plot.plot(pen=pg.mkPen("#f6df75", width=2), symbol="o", symbolSize=5, name="Total energy")
        self.hidden_final = self.energy_plot.plot(pen=None, symbol="s", symbolBrush="#4ddde0", name="Final hidden energy")
        self.sensory_final = self.energy_plot.plot(pen=None, symbol="t", symbolBrush="#ffb454", name="Final sensory energy")
        self.splitter.addWidget(self.plots)
        self.rc_plots = pg.GraphicsLayoutWidget()
        spikes = self.rc_plots.addPlot(row=0, col=0, title="RC hidden spikes per model step")
        spikes.addLegend(offset=(10, 0))
        spikes.setLabel("left", "Spike count")
        spikes.setLabel("bottom", "Model/world step")
        self.spike_curves = [
            spikes.plot(pen=pg.mkPen(color, width=2), name=name)
            for name, color in (
                ("Forecast hidden (counterfactual)", "#4ddde0"),
                ("Committed hidden", "#ffb454"),
            )
        ]
        hidden = self.rc_plots.addPlot(row=0, col=1, title="Committed decoded RC state")
        hidden.setLabel("left", "Decoded |h|")
        hidden.getAxis("left").enableAutoSIPrefix(False)
        hidden.setLabel("bottom", "Model/world step")
        self.hidden_norm_curve = hidden.plot(pen=pg.mkPen("#ffb454", width=2))
        neutral = self.rc_plots.addPlot(row=0, col=2, title="Current foveal forecast distance from neutral")
        neutral.setLabel("left", "Mean |RGB - 0.5|")
        neutral.getAxis("left").enableAutoSIPrefix(False)
        neutral.setLabel("bottom", "Model/world step")
        self.neutral_curve = neutral.plot(pen=pg.mkPen("#4ddde0", width=2))
        self.splitter.addWidget(self.rc_plots)
        self.plots.setMinimumHeight(160)
        self.rc_plots.setMinimumHeight(160)
        self.footer = footer = QtWidgets.QLabel(
            "Only observed foveal RGB values enter MSE; no-evidence steps are gaps, not zero error. "
            "Settled error uses decoder residuals, not clamped sensory states. "
            "Total energy decreases within a fixed conditioning objective; visual MSE and next-step forecast need not. "
            "Motion changes the prediction target; defaults hold weights and root fixed. "
            "Canvas feedback updates the root; Surprise optionally enables wave dynamics. "
            f"Plots/export retain the latest {self.HISTORY_LIMIT} steps."
        )
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(max(1, round(1000*self.dt)))
        self.timer.timeout.connect(self._advance)
        self.visualizer = None
        self.model_selector.currentIndexChanged.connect(self.new_trial)
        self.new_trial()

    @property
    def scene(self):
        return self.world.frame

    def _update_rc_parameters(self):
        if self.visualizer is not None and isinstance(
            self.visualizer.source_orchestrator.sensory_graph, RCSpikingPredictiveCodingGraph,
        ):
            graph = self.visualizer.source_orchestrator.sensory_graph
            graph.set_memory_time_constant(float(self.rc_tau.value()))
            graph.set_spike_threshold(float(self.rc_threshold.value()))
            graph.set_membrane_capacitance(float(self.rc_capacitance.value()))
            self.rc_resistance.setText(f"R = tau/C = {graph.membrane_resistance:g}")

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
        controls.autonomous_gaze.setEnabled(
            not locked and not isinstance(self.visualizer.source_orchestrator.sensory_graph, RCSpikingPredictiveCodingGraph)
        )
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
        wave_enabled = False if self.visualizer is None else self.visualizer.engine.wave_propagation_enabled
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
        rc_mode = self.model_selector.currentData() == "rc-spiking"
        self.vision_config = replace(
            self.vision_config,
            mode="mouse" if rc_mode else self.vision_config.mode,
            memory_time_constant=float(self.rc_tau.value()) if rc_mode else 1/3,
        )
        self.world.reset()
        self.capture = StaticSceneCapture(self.scene)
        self.visualizer = _BenchVisualizer(
            resolution=self.scene.shape[:2], plane_size_m=(1, 1),
            use_shader=self.backend == "opengl", use_gpu=self.backend == "cupy",
            rgb_canvas_enabled=True, camera_capture=self.capture,
            source_orchestrator_config=self.source_config, active_vision=self.vision_config,
            update_interval_ms=self.timer.interval(), forecast_dt=self.dt,
            wave_propagation_enabled=wave_enabled,
            shared_graph_factory=RCSpikingPredictiveCodingGraph if rc_mode else SharedPredictiveCodingGraph,
        )
        self.visualizer.timer.stop()
        self.visualizer.setMinimumHeight(190)
        self.rc_controls.setVisible(rc_mode)
        self.rc_plots.setVisible(rc_mode)
        self._update_rc_parameters()
        if rc_mode:
            unavailable = "RC-spiking prototype has no Gaussian covariance or information-gain policy."
            self.visualizer.gaze_controls.autonomous_gaze.setEnabled(False)
            self.visualizer.gaze_controls.autonomous_gaze.setToolTip(unavailable)
            self.visualizer.diagnostics_button.setEnabled(False)
            self.visualizer.diagnostics_button.setToolTip(unavailable)
            self.visualizer.scene_button.setEnabled(False)
            self.visualizer.scene_button.setToolTip("Use the bench image views; the RC graph popup is not supported.")
            self.visualizer.gaze_description = (
                "Experimental bias-free continuous RC-spiking graph, not DiffPC. "
                "Voltage, ternary spikes and decoded hidden state are distinct. "
                "Mouse/locked gaze only; no Gaussian covariance or EIG. "
                "Outside-fovea display values remain cached; the neutral-distance plot uses current foveal forecasts."
            )
            self.visualizer.gaze_status.setToolTip(self.visualizer.gaze_description)
        self.energy_plot.setTitle("RC residual diagnostic (not monotone)" if rc_mode else "Within latest step: conditioning energy")
        self.energy_plot.setLabel("bottom", "RC microsteps" if rc_mode else "Accepted hidden updates")
        self.footer.setText(
            (
                "Bias-free RC: learned connections persist; unforced activity decays toward neutral gray. "
                "Observe off stops evidence, not held-root currents or residual/recurrent spikes. "
                "Neutral distance covers the current fovea; outside pixels stay cached. "
                "Residual energy is not monotone. "
                if rc_mode else
                "PC control: held-objective total energy is non-increasing during conditioning; MSE need not decrease. "
            )
            + "No-evidence MSE is unavailable, not zero. World motion and Surprise wave dynamics are independent. "
            + f"Plots/export retain the latest {self.HISTORY_LIMIT} steps."
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
            self.sensory_final, self.neutral_curve, self.hidden_norm_curve, *self.spike_curves,
        ):
            curve.clear()
        self.status.setText("Paused at step 0. Fresh seed-0 weights; no evidence consumed. Step or Run to begin.")
        self._refresh_images()
        self.splitter.setSizes((200, 160, 160 if rc_mode else 0, 240))

    def step_once(self) -> SceneStepMetrics | RCSceneStepMetrics:
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
        rc_mode = isinstance(source.sensory_graph, RCSpikingPredictiveCodingGraph)
        measure = measure_rc_spiking_step if rc_mode else measure_static_world_step
        measured = measure(
            source.sensory_graph, step=self.steps, dt=self.dt, gaze=source.graph_io.gaze,
        )
        metric_type = RCSceneStepMetrics if rc_mode else SceneStepMetrics
        metrics = metric_type(
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
        if isinstance(metrics, RCSceneStepMetrics):
            frames = [frame for frame in self.history if isinstance(frame, RCSceneStepMetrics)]
            for curve, name in zip(self.spike_curves, (
                "forecast_hidden_spikes", "committed_hidden_spikes",
            )):
                curve.setData([frame.step for frame in frames], [getattr(frame, name) for frame in frames])
            self.neutral_curve.setData([frame.step for frame in frames], [frame.neutral_visual_mae for frame in frames])
            self.hidden_norm_curve.setData([frame.step for frame in frames], [frame.decoded_hidden_norm for frame in frames])
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
        if isinstance(metrics, RCSceneStepMetrics):
            self.status.setText(
                self.status.text()+f" | RC tau {metrics.memory_time_constant:g}s, threshold {metrics.spike_threshold:g}"
                +f" | spikes forecast/condition h/error {metrics.forecast_hidden_spikes}/{metrics.condition_hidden_spikes}/{metrics.condition_error_spikes}"
                +f" | |h| {metrics.decoded_hidden_norm:.4g}, |Vh| {metrics.hidden_potential_norm:.4g}"
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
            model=self.model_selector.currentData(),
            scene="brownian-geometric-v1", resolution=self.scene.shape[:2],
            world_seed=self.world.seed, world_time=self.world.time,
            motion_enabled=self.motion_enabled.isChecked(), brownian_scale=float(self.motion_scale.value()),
            backend=self.backend, foveal_radius=self.vision_config.foveal_radius,
            hidden_units=graph.hidden_nodes, seed=0, forecast_dt=self.dt,
            wave_propagation=self.visualizer.engine.wave_propagation_enabled,
            history_limit=self.HISTORY_LIMIT, total_steps=self.steps,
            current_config=asdict(graph.config), frames=[asdict(frame) for frame in self.history],
        )
        if isinstance(graph, RCSpikingPredictiveCodingGraph):
            payload.update(
                memory_time_constant=graph.memory_time_constant, spike_threshold=graph.spike_threshold,
                membrane_capacitance=graph.membrane_capacitance, membrane_resistance=graph.membrane_resistance,
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
    parser.add_argument("--model", choices=("pc", "rc-spiking"), default="pc", help="Bench-only model experiment; switching models starts a fresh trial.")
    parser.add_argument("--memory-time-constant", type=float, default=1/3, help="RC time constant R*C, in seconds.")
    parser.add_argument("--spike-threshold", type=float, default=.02)
    parser.add_argument("--membrane-capacitance", type=float, default=.02, help="RC capacitance in model units; R = tau/C.")
    parser.add_argument("--resolution", nargs=2, type=int, default=(80, 120), metavar=("ROWS", "COLS"))
    parser.add_argument("--foveal-radius", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8, help="PC settling limit or RC microsteps per physical interval.")
    parser.add_argument("--rate", type=float, default=.1, help="PC line-search step size or RC current gain.")
    parser.add_argument("--dt", type=float, default=1/30, help="Fixed model/world interval per step, in seconds.")
    parser.add_argument("--static-scene", action="store_true", help="Disable Brownian motion for the static-scene control.")
    parser.add_argument("--brownian-scale", type=float, default=4., help="Brownian displacement scale, pixels/sqrt(second).")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = StaticWorldBench(
        resolution=tuple(args.resolution), foveal_radius=args.foveal_radius,
        hidden_units=args.hidden_units, backend=args.backend,
        dt=args.dt, inference_steps=args.steps, inference_rate=args.rate,
        motion_enabled=not args.static_scene, brownian_scale=args.brownian_scale,
        model=args.model, memory_time_constant=args.memory_time_constant, spike_threshold=args.spike_threshold,
        membrane_capacitance=args.membrane_capacitance,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
