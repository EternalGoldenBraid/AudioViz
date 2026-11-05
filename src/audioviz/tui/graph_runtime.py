"""Headless driver for the predictive graph: audio -> readouts -> canvas engine."""

import threading
from dataclasses import replace
from time import monotonic

from audioviz.audio_processing.cochlea import Cochlea
from audioviz.sources.camera import CameraFrameSourceConfig
from audioviz.engine import RippleEngine
from audioviz.sources.audio import AudioSourceConfig
from audioviz.sources.orchestrator import RippleSourceOrchestrator, RippleSourceOrchestratorConfig
from audioviz.sources.synthetic import SyntheticSourceConfig
from audioviz.tui.audio_out import AudioOut
from audioviz.tui.camera_feed import CameraFeed
from audioviz.tui.audio_feed import BLOCK_SIZE, SAMPLE_RATE, AudioFeed, AudioFrame

DEFAULT_RESOLUTION = 128
STEP_INTERVAL = BLOCK_SIZE / SAMPLE_RATE
MIN_SIDE = 8
AUDIO_MODELS = ("spectral", "cochlea")


def parse_resolution(value: "int | str | tuple[int, int]") -> tuple[int, int]:
    """(rows, cols) from N (square), "RxC" or a pair; every side is at least MIN_SIDE."""
    if isinstance(value, str):
        rows, _, cols = value.lower().partition("x")
        value = (int(rows), int(cols or rows))
    rows, cols = (value, value) if isinstance(value, int) else value
    if min(rows, cols) < MIN_SIDE:
        raise ValueError(f"each side of the grid must be at least {MIN_SIDE}")
    return int(rows), int(cols)


def format_resolution(resolution: tuple[int, int]) -> str:
    return f"{resolution[0]}x{resolution[1]}"


class GraphRuntime:
    """Same step order as RippleWaveVisualizer._advance_canvas, without Qt.

    The grid size is the cost knob: step time grows roughly with the pixel count.
    """

    def __init__(
        self, feed: AudioFeed | None = None, resolution: "int | str | tuple[int, int]" = DEFAULT_RESOLUTION,
        camera: bool = True, audio_model: str = "spectral",
    ) -> None:
        if audio_model not in AUDIO_MODELS:
            raise ValueError(f"audio_model must be one of {AUDIO_MODELS}")
        self.feed = feed or AudioFeed()
        if audio_model == "cochlea":
            self.feed.processor.cochlea = Cochlea(SAMPLE_RATE)
        self.out = AudioOut()
        self.camera = CameraFeed()
        has_camera = camera and self.camera.open()
        self.show_camera = self.feed_canvas = has_camera
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step_ms = 0.0
        self._build(resolution, config=RippleSourceOrchestratorConfig(
            audio=AudioSourceConfig(enabled=True),
            synthetic=SyntheticSourceConfig(enabled=False),
            camera_frame=CameraFrameSourceConfig(enabled=has_camera),
        ))

    def _build(self, resolution: "int | str | tuple[int, int]", *, config: RippleSourceOrchestratorConfig, damping: float = 0.995, speed: float = 1.0) -> None:
        grid = parse_resolution(resolution)
        longest = max(grid)
        engine = RippleEngine(
            resolution=grid,
            plane_size_m=(grid[0] / longest, grid[1] / longest),  # square cells whatever the aspect
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
            camera_capture=self.camera,
        )
        orchestrator.set_diagnostics_enabled(True)
        with self.lock:
            self.resolution = grid
            self.engine = engine
            self.orchestrator = orchestrator
            self.visual_prediction = orchestrator.visual_readout.predict(engine.get_field_numpy())

    def rebuild(self, resolution: "int | str | tuple[int, int]") -> None:
        """Resize the grid, carrying over mixer settings; learned weights are reset."""
        resolution = parse_resolution(resolution)
        with self.lock:
            audio = self.orchestrator.audio_source
            config = replace(
                self.orchestrator.config,
                audio=replace(self.orchestrator.config.audio, observation_gain=audio.observation_gain),
                visual_pathway=self.orchestrator.visual_readout.config,
                camera_frame=replace(
                    self.orchestrator.config.camera_frame,
                    enabled=self.orchestrator.camera_source.enabled,
                    gain=self.orchestrator.camera_source.gain,
                ),
            )
            self._build(resolution, config=config, damping=self.engine.damping, speed=self.engine.speed)

    @property
    def processor(self):
        return self.feed.processor

    @property
    def audio_readout(self):
        return self.orchestrator.audio_readout

    @property
    def audio_model(self) -> str:
        return "cochlea" if self.processor.cochlea is not None else "spectral"

    def set_audio_model(self, model: str) -> None:
        """Changing representation rebuilds graph weights and mutes playback."""
        if model not in AUDIO_MODELS:
            raise ValueError(f"audio_model must be one of {AUDIO_MODELS}")
        with self.lock:
            if model == self.audio_model:
                return
            self.out.set_mode("off")
            self.processor.cochlea = Cochlea(SAMPLE_RATE) if model == "cochlea" else None
            self.rebuild(self.resolution)

    def step(self) -> None:
        started = monotonic()
        with self.lock:
            cochlea = self.processor.cochlea
            blocks = self.feed.pump() if cochlea is None else ()
            frame = self.orchestrator.resolve()
            self.engine.propagate(frame.drive_grid)
            self.visual_prediction = self.orchestrator.predict_visual()
            if cochlea is not None:
                blocks = self.feed.pump()
            self.orchestrator.correct_from_observations()
            source = self.orchestrator.audio_source
            if cochlea is not None and blocks and source.enabled and source.observation_gain > 0:
                cochlea.correct_from_graph(self.orchestrator.audio_prediction / source.observation_gain)
            self._sound(blocks)
        self.step_ms = (monotonic() - started) * 1000.0

    def set_camera(self, on: bool) -> None:
        """Shows or hides the camera tile; the device stays open while the canvas still uses it."""
        self.show_camera = on
        self._sync_camera()

    def set_canvas_feed(self, on: bool) -> None:
        self.feed_canvas = on
        self._sync_camera()

    def _sync_camera(self) -> None:
        if (self.show_camera or self.feed_canvas) and not self.camera.isOpened():
            self.camera.open()
        self.orchestrator.camera_source.set_enabled(self.feed_canvas and self.camera.isOpened())
        if not (self.show_camera or self.feed_canvas):
            self.camera.release()

    def _sound(self, blocks: tuple[AudioFrame, ...]) -> None:
        source = self.orchestrator.audio_source
        if not source.enabled:
            return
        if self.processor.cochlea is not None or self.out.mode == "input":
            for block in blocks:
                if self.out.mode in ("input", "evidence"):
                    self.out.submit_waveform(block.samples)
                elif self.out.mode == "prediction":
                    self.out.submit_waveform(block.cochlea.prediction)
                elif self.out.mode == "error":
                    self.out.submit_waveform(block.cochlea.error)
            return
        bands = source.recent_observation() if self.out.mode == "evidence" else self.orchestrator.audio_prediction
        if self.out.mode == "error":
            observed = source.recent_observation()
            bands = None if observed is None or bands is None else observed - bands
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
        self.camera.release()
        self.feed.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.step()
            self._stop.wait(max(STEP_INTERVAL - self.step_ms / 1000.0, 0.0))
