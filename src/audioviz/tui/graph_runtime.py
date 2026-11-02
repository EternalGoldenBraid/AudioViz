"""Headless driver for the predictive graph: audio -> readouts -> canvas engine."""

import threading
from dataclasses import replace
from time import monotonic

from audioviz.engine import RippleEngine
from audioviz.sources.audio import AudioSourceConfig
from audioviz.sources.orchestrator import RippleSourceOrchestrator, RippleSourceOrchestratorConfig
from audioviz.sources.synthetic import SyntheticSourceConfig
from audioviz.tui.audio_out import AudioOut
from audioviz.tui.audio_feed import BLOCK_SIZE, SAMPLE_RATE, AudioFeed

DEFAULT_RESOLUTION = 96
STEP_INTERVAL = BLOCK_SIZE / SAMPLE_RATE


class GraphRuntime:
    """Same step order as RippleWaveVisualizer._advance_canvas, without Qt.

    The grid size is the cost knob: step time grows roughly with the pixel count.
    """

    def __init__(self, feed: AudioFeed | None = None, resolution: int = DEFAULT_RESOLUTION) -> None:
        self.feed = feed or AudioFeed()
        self.out = AudioOut()
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step_ms = 0.0
        self._build(resolution, config=RippleSourceOrchestratorConfig(
            audio=AudioSourceConfig(enabled=True),
            synthetic=SyntheticSourceConfig(enabled=False),
        ))

    def _build(self, resolution: int, *, config: RippleSourceOrchestratorConfig, damping: float = 0.995, speed: float = 1.0) -> None:
        grid = (resolution, resolution)
        engine = RippleEngine(
            resolution=grid,
            plane_size_m=(1.0, 1.0),
            speed=speed,
            damping=damping,
            boundary_condition="cyclic",
        )
        orchestrator = RippleSourceOrchestrator(
            config=config,
            processor=self.feed.processor,
            engine=engine,
            resolution=grid,
            n_sources=1,
        )
        orchestrator.set_diagnostics_enabled(True)
        with self.lock:
            self.resolution = resolution
            self.engine = engine
            self.orchestrator = orchestrator
            self.visual_prediction = orchestrator.visual_readout.predict(engine.get_field_numpy())

    def rebuild(self, resolution: int) -> None:
        """Resize the grid, carrying over mixer settings; learned weights are reset."""
        if resolution < 8:
            raise ValueError("resolution must be at least 8")
        with self.lock:
            audio = self.orchestrator.audio_source
            config = replace(
                self.orchestrator.config,
                audio=replace(self.orchestrator.config.audio, observation_gain=audio.observation_gain),
                visual_pathway=self.orchestrator.visual_readout.config,
            )
            self._build(resolution, config=config, damping=self.engine.damping, speed=self.engine.speed)

    @property
    def processor(self):
        return self.feed.processor

    @property
    def audio_readout(self):
        return self.orchestrator.audio_readout

    def step(self) -> None:
        started = monotonic()
        with self.lock:
            self.feed.pump()
            frame = self.orchestrator.resolve()
            self.engine.propagate(frame.drive_grid)
            self.visual_prediction = self.orchestrator.predict_visual()
            self.orchestrator.correct_from_observations()
            self._sound()
        self.step_ms = (monotonic() - started) * 1000.0

    def _sound(self) -> None:
        source = self.orchestrator.audio_source
        bands = source.recent_observation() if self.out.mode == "evidence" else self.orchestrator.audio_prediction
        self.out.submit(bands, source.observation_gain)

    def start(self) -> None:
        """Step on a background thread so a slow grid never stalls the interface."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.out.close()
        self.feed.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.step()
            self._stop.wait(max(STEP_INTERVAL - self.step_ms / 1000.0, 0.0))
