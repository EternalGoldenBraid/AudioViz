"""Terminal front end: modal, Vim-style control over layered, floating tiles."""

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, RichLog, Static

from audioviz.tui import control
from audioviz.tui.audio_feed import BLOCK_SIZE, SAMPLE_RATE
from audioviz.tui.graph_runtime import GraphRuntime
from audioviz.tui.devices import DevicePicker, list_devices
from audioviz.tui.halfblock import RENDER_MODES, render_cells
from audioviz.tui.mixer import CONTROLS
from audioviz.tui.themes import THEMES
from audioviz.tui.views import VIEWS

MAX_LAYERS = 8
BRAILLE_MIN_CELLS = 1200  # tiles at least this large render at braille density


@dataclass
class Placement:
    """Where a tile sits: cell position, size and opacity; stacking order lives in the app."""

    x: int
    y: int
    width: int
    height: int
    opacity: float = 1.0


class ViewTile(Static, can_focus=True):
    """A focusable pane showing one subgraph view; focus and Placement are shared by keys and gaze."""

    DEFAULT_CSS = """
    ViewTile { position: absolute; border: round $primary-darken-2; padding: 0; background: $surface; }
    ViewTile:focus { border: heavy $accent; }
    """

    def __init__(self, view: str, runtime: GraphRuntime, **kwargs) -> None:
        super().__init__(**kwargs)
        self.runtime = runtime
        self.render_mode: str | None = None  # None: choose by tile size
        self.set_view(view)

    def mode_for(self, cells: int) -> str:
        if self.render_mode:
            return self.render_mode
        return "braille" if cells >= BRAILLE_MIN_CELLS else "half"

    def set_view(self, view: str) -> None:
        self.view = view
        self.border_title = view

    def refresh_view(self) -> None:
        size = self.content_size
        if size.width < 2 or size.height < 1:
            return
        mode = self.mode_for(size.width * size.height)
        sub_rows, sub_cols = RENDER_MODES[mode]
        image = VIEWS[self.view](self.runtime, size.height * sub_rows, size.width * sub_cols)
        if image is None:
            self.update(f"{self.view}: waiting for data")
            return
        self.update(render_cells(image, size.height, size.width, mode))


class MixerTile(Static, can_focus=True):
    """Slider rows for the graph parameters; j/k select a row, h/l adjust it."""

    DEFAULT_CSS = """
    MixerTile { position: absolute; border: round $primary-darken-2; padding: 0 1; background: $surface; }
    MixerTile:focus { border: heavy $accent; }
    """

    def __init__(self, runtime: GraphRuntime, **kwargs) -> None:
        super().__init__(**kwargs)
        self.runtime = runtime
        self.row = 0
        self.border_title = "mixer"

    def select(self, delta: int) -> None:
        self.row = (self.row + delta) % len(CONTROLS)
        self.refresh_view()

    def adjust(self, steps: int) -> None:
        with self.runtime.lock:
            CONTROLS[self.row].nudge(self.runtime, steps)
        self.refresh_view()

    def refresh_view(self) -> None:
        width = self.content_size.width
        if width < 20:
            return
        bar = max(width - 24, 4)
        text = Text()
        for index, control in enumerate(CONTROLS):
            filled = round(control.position(self.runtime) * bar)
            selected = index == self.row and self.has_focus
            text.append(f"{'>' if selected else ' '}{control.name:<15}", style="bold" if selected else "")
            text.append("█" * filled, style="green" if selected else "dark_green")
            text.append("░" * (bar - filled), style="grey37")
            text.append(f" {control.format(self.runtime)}\n")
        self.update(text)


class LogTile(RichLog):
    DEFAULT_CSS = """
    LogTile { position: absolute; border: round $primary-darken-2; background: $surface; }
    LogTile:focus { border: heavy $accent; }
    """


class AudioVizApp(App):
    """NORMAL moves/lifts tiles; ':' opens the command line.

    Placement plus stack order is the only state a key or gaze ray needs to change:
    (x, y) from where the ray hits the display, stacking depth from viewer distance.
    """

    CSS = f"""
    #stage {{ background: $background; layers: {" ".join(f"z{i}" for i in range(MAX_LAYERS))}; }}
    .-immersive #mode {{ display: none; }}
    .-immersive ViewTile {{ border: none !important; }}
    #command {{ dock: bottom; display: none; border: none; height: 1; margin-bottom: 1; padding: 0 1; }}
    #command.-active {{ display: block; }}
    #mode {{ dock: bottom; height: 1; background: $primary-darken-2; padding: 0 1; }}
    """
    SCREENSHOT_DIR = Path("outputs/tui")
    MOVE_STEP = (2, 1)
    OPACITY_STEP = 0.1

    def __init__(self, audio: str = "default") -> None:
        super().__init__()
        self.audio = audio
        self.mode = "NORMAL"
        self.runtime = GraphRuntime()
        self.placements = {
            "canvas": Placement(0, 0, 40, 16),
            "mixer": Placement(0, 34, 40, 9),
            "spectrogram": Placement(40, 0, 40, 16),
            "log": Placement(80, 0, 40, 16),
            "waveform": Placement(0, 16, 40, 9),
            "prediction": Placement(40, 16, 40, 9),
            "error": Placement(80, 16, 40, 9),
            "hidden": Placement(0, 25, 40, 9),
        }
        self.stack = list(self.placements)  # bottom -> top
        self._return_focus = None
        self.zoomed: str | None = None
        self._last_focus = None
        self.immersive = False
        self._tmux_zoomed = False
        self._before_mixer = "canvas"  # where escape returns to
        self.tile_background: str | None = None  # overrides the theme surface

    def compose(self) -> ComposeResult:
        with Container(id="stage"):
            for name in self.placements:
                if name == "log":
                    yield LogTile(id="log", wrap=True)
                elif name == "mixer":
                    yield MixerTile(self.runtime, id="mixer")
                else:
                    yield ViewTile(name, self.runtime, id=name)
        yield Static("NORMAL", id="mode")
        yield Input(placeholder=":", id="command")

    def on_mount(self) -> None:
        for theme in THEMES:
            self.register_theme(theme)
        self.theme = "black"
        self.query_one("#log", LogTile).border_title = "log"
        self.apply_layout()
        self.runtime.start()
        self.set_interval(1 / 15, self.refresh_tiles)
        self.query_one("#canvas").focus()
        self.select_audio(self.audio)
        self.run_worker(self._serve_control(), exclusive=False)

    async def _serve_control(self) -> None:
        server = await control.serve(self)
        async with server:
            await server.serve_forever()

    def describe(self) -> dict:
        """Observable state for the control socket."""
        mixer = self.query_one("#mixer", MixerTile)
        return {
            "focus": self._focused_name(),
            "mode": self.mode,
            "zoomed": self.zoomed,
            "immersive": self.immersive,
            "output_device": self.runtime.out.device,
            "theme": self.theme,
            "audio": self.runtime.feed.source,
            "out": self.runtime.out.mode,
            "grid": self.runtime.resolution,
            "step_ms": round(self.runtime.step_ms, 1),
            "stack": self.stack,
            "placements": {k: vars(v) for k, v in self.placements.items()},
            "mixer": {
                "row": CONTROLS[mixer.row].name,
                **{c.name: c.format(self.runtime) for c in CONTROLS},
            },
        }

    def on_unmount(self) -> None:
        control.SOCKET.unlink(missing_ok=True)
        self.runtime.stop()

    def refresh_tiles(self) -> None:
        for tile in self.query(ViewTile):
            if tile.display:
                tile.refresh_view()
        self.set_mode(self.mode)
        mixer = self.query_one("#mixer", MixerTile)
        if mixer.display:
            mixer.refresh_view()

    def apply_layout(self) -> None:
        stage = self.query_one("#stage").size
        for depth, name in enumerate(self.stack):
            placement = self.placements[name]
            widget = self.query_one(f"#{name}")
            styles = widget.styles
            zoomed = self.zoomed == name
            widget.display = self.zoomed in (None, name)
            styles.offset = (0, 0) if zoomed else (placement.x, placement.y)
            styles.width = stage.width if zoomed else placement.width
            styles.height = stage.height if zoomed else placement.height
            styles.opacity = 1.0 if zoomed else placement.opacity
            styles.layer = f"z{depth}"
        self.set_mode(self.mode)

    def on_resize(self) -> None:
        if self.zoomed:
            self.apply_layout()

    def open_devices(self) -> None:
        try:
            inputs, outputs = list_devices()
        except Exception as error:  # no audio backend should not take down the shell
            self.query_one("#log", LogTile).write(f"devices unavailable: {error}")
            return
        out = self.runtime.out
        self.push_screen(
            DevicePicker(inputs, outputs, self.runtime.feed.source, "default" if out.device is None else str(out.device)),
            self._device_chosen,
        )

    def _device_chosen(self, choice: tuple[str, str] | None) -> None:
        if choice is None:
            return
        kind, value = choice
        log = self.query_one("#log", LogTile)
        if kind == "input":
            self.select_audio(value)
        else:
            try:
                self.runtime.out.use_device(None if value == "default" else int(value))
                log.write(f"output device: {value}")
            except Exception as error:  # a busy device must not take down the shell
                log.write(f"output {value!r} failed: {error}")

    def toggle_immersive(self) -> None:
        """True fullscreen: zoom the tile, drop all chrome, and zoom the tmux pane when inside tmux."""
        self.immersive = not self.immersive
        if self.immersive != bool(self.zoomed):
            self.toggle_zoom()
        self.set_class(self.immersive, "-immersive")
        self.call_after_refresh(self.apply_layout)  # the stage grows once the status bar is gone
        if os.environ.get("TMUX") and self._tmux_zoomed != self.immersive:
            subprocess.run(["tmux", "resize-pane", "-Z"], check=False)
            self._tmux_zoomed = self.immersive

    def toggle_zoom(self) -> None:
        self.zoomed = None if self.zoomed else self._focused_name()
        self.apply_layout()
        self.query_one(f"#{self._focused_name()}").focus()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if isinstance(event.widget, MixerTile) and isinstance(self._last_focus, ViewTile):
            self._before_mixer = self._last_focus.id
        self._last_focus = event.widget

    def _focused_name(self) -> str:
        focused = self.focused
        return focused.id if focused is not None and focused.id in self.placements else self.stack[-1]

    def move_tile(self, dx: int, dy: int) -> None:
        placement = self.placements[self._focused_name()]
        placement.x, placement.y = max(0, placement.x + dx), max(0, placement.y + dy)
        self.apply_layout()

    def lift_tile(self, delta: int) -> None:
        name = self._focused_name()
        index = self.stack.index(name)
        target = min(max(index + delta, 0), len(self.stack) - 1)
        self.stack.insert(target, self.stack.pop(index))
        self.apply_layout()

    def set_tile_background(self, color: str | None) -> None:
        """Colour behind every tile; None follows the theme's surface."""
        from textual.color import Color

        if color is not None:
            Color.parse(color)  # raises on an invalid colour
        self.tile_background = color
        for tile in self.query("ViewTile, MixerTile, LogTile"):
            tile.styles.background = color if color else None

    def cycle_render(self) -> None:
        tile = self.focused
        if not isinstance(tile, ViewTile):
            return
        order = [None, *RENDER_MODES]
        tile.render_mode = order[(order.index(tile.render_mode) + 1) % len(order)]
        self.set_render(tile, tile.render_mode)

    def set_render(self, tile: ViewTile, mode: str | None) -> None:
        tile.render_mode = mode
        tile.border_title = f"{tile.view} [{mode or 'auto'}]"

    def set_opacity(self, value: float) -> None:
        self.placements[self._focused_name()].opacity = min(max(value, 0.1), 1.0)
        self.apply_layout()

    def nudge_opacity(self, delta: float) -> None:
        self.set_opacity(self.placements[self._focused_name()].opacity + delta)

    def neighbour(self, name: str, dx: int, dy: int) -> str | None:
        """Nearest tile in a direction: its centre must lie that way; sideways offset costs double.

        Cells are about twice as tall as wide, so vertical distance is doubled to compare like with like.
        """
        here = self.placements[name]
        cx, cy = here.x + here.width / 2, (here.y + here.height / 2) * 2
        best, best_cost = None, float("inf")
        for other, p in self.placements.items():
            if other == name:
                continue
            ox, oy = p.x + p.width / 2 - cx, (p.y + p.height / 2) * 2 - cy
            along, across = (ox * dx, abs(oy)) if dx else (oy * dy, abs(ox))
            if along <= 0:
                continue
            cost = along + 2 * across
            if cost < best_cost:
                best, best_cost = other, cost
        return best

    def focus_toward(self, dx: int, dy: int) -> None:
        if self.zoomed:
            return
        target = self.neighbour(self._focused_name(), dx, dy)
        if target:
            self.query_one(f"#{target}").focus()

    def _move_focus(self, step: int) -> None:
        names = [name for name in self.placements if self.zoomed in (None, name)]
        current = names.index(self._focused_name())
        self.query_one(f"#{names[(current + step) % len(names)]}").focus()

    def on_key(self, event: events.Key) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        if self.focused is self.query_one("#command", Input):
            if event.key == "escape":
                self._close_command()
                event.stop()
            return
        dx, dy = self.MOVE_STEP
        if isinstance(self.focused, MixerTile) and event.key == "escape":
            self.query_one(f"#{self._before_mixer}").focus()
            event.stop()
            return
        if isinstance(self.focused, MixerTile) and event.key in ("h", "l", "j", "k"):
            mixer = self.focused
            {"h": lambda: mixer.adjust(-1), "l": lambda: mixer.adjust(1),
             "j": lambda: mixer.select(1), "k": lambda: mixer.select(-1)}[event.key]()
            event.stop()
            return
        actions = {
            "tab": lambda: self._move_focus(1),
            "shift+tab": lambda: self._move_focus(-1),
            "w": lambda: self._move_focus(1),
            "z": self.toggle_zoom,
            "F": self.toggle_immersive,
            "a": self.open_devices,
            "r": self.cycle_render,
            "h": lambda: self.focus_toward(-1, 0),
            "k": lambda: self.focus_toward(0, -1),
            "l": lambda: self.focus_toward(1, 0),
            "j": lambda: self.focus_toward(0, 1),
            "H": lambda: self.move_tile(-dx, 0),
            "L": lambda: self.move_tile(dx, 0),
            "K": lambda: self.move_tile(0, -dy),
            "J": lambda: self.move_tile(0, dy),
            "right_square_bracket": lambda: self.lift_tile(1),
            "left_square_bracket": lambda: self.lift_tile(-1),
            "full_stop": lambda: self.nudge_opacity(self.OPACITY_STEP),
            "comma": lambda: self.nudge_opacity(-self.OPACITY_STEP),
            "colon": self._open_command,
            "ctrl+q": self.exit,
        }
        action = actions.get(event.key)
        if action:
            action()
            event.stop()

    def _open_command(self) -> None:
        self._return_focus = self.focused
        command = self.query_one("#command", Input)
        command.add_class("-active")
        command.focus()
        self.set_mode("COMMAND")

    def _close_command(self) -> None:
        command = self.query_one("#command", Input)
        command.value = ""
        command.remove_class("-active")
        self.set_mode("NORMAL")
        (self._return_focus or self.query_one("#canvas")).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.lstrip(":").split()
        self._close_command()
        if line:
            self.run_command(line[0], line[1:])

    def screenshot(self, name: str = "tui") -> Path:
        """Save an SVG (and a PNG when rsvg-convert exists) of the current screen."""
        self.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        svg = Path(self.save_screenshot(f"{name}.svg", str(self.SCREENSHOT_DIR)))
        converter = shutil.which("rsvg-convert")
        if converter is None:
            return svg
        png = svg.with_suffix(".png")
        subprocess.run([converter, "-o", str(png), str(svg)], check=True)
        return png

    def run_command(self, name: str, args: list[str]) -> None:
        log = self.query_one("#log", LogTile)
        try:
            if name in ("help", "h"):
                log.write("damping|speed <v>  opacity <v>  raise|lower")
                log.write("move <dx> <dy>  view <name>  views")
                log.write("audio list|synthetic|default|<index>  gain <v>")
                log.write("theme [name]  bg <color|theme>")
                log.write("out off|prediction|evidence  volume <v>  audio tone <hz>")
                log.write("resolution <n>  render half|quad|braille|auto  zoom")
                log.write("screenshot [name]  q   keys: z zoom, r render, w/tab focus")
                log.write("h/j/k/l focus  H/J/K/L move  [ ] depth  , . opacity")
            elif name == "damping":
                self.runtime.engine.set_damping(float(args[0]))
            elif name == "speed":
                self.runtime.engine.set_speed(float(args[0]))
            elif name == "gain":
                gain = float(args[0])
                if gain < 0:
                    raise ValueError("gain must be non-negative")
                self.runtime.orchestrator.audio_source.observation_gain = gain
            elif name == "theme":
                if not args:
                    log.write("themes: " + " ".join(sorted(self.available_themes)))
                    log.write(f"current: {self.theme}")
                else:
                    if args[0] not in self.available_themes:
                        raise ValueError(f"unknown theme {args[0]!r}")
                    self.theme = args[0]
            elif name in ("bg", "background"):
                self.set_tile_background(None if args[0] == "theme" else args[0])
            elif name == "out":
                self.runtime.out.set_mode(args[0], speakers=len(args) < 2 or args[1] != "silent")
                log.write(f"output: {self.runtime.out.mode}")
            elif name == "volume":
                self.runtime.out.volume = float(args[0])
            elif name == "resolution":
                self.runtime.rebuild(int(args[0]))
                log.write(f"grid {self.runtime.resolution}x{self.runtime.resolution}")
            elif name == "render":
                tile = self.focused
                if not isinstance(tile, ViewTile) or args[0] not in (*RENDER_MODES, "auto"):
                    raise ValueError("focus a view tile; mode is half|quad|braille|auto")
                self.set_render(tile, None if args[0] == "auto" else args[0])
            elif name == "zoom":
                self.toggle_zoom()
            elif name in ("fullscreen", "fs"):
                self.toggle_immersive()
            elif name == "devices":
                self.open_devices()
            elif name == "views":
                log.write(" ".join(VIEWS))
            elif name == "view":
                if args[0] not in VIEWS:
                    raise ValueError(f"unknown view {args[0]!r}")
                focused = self.focused
                if not isinstance(focused, ViewTile):
                    raise ValueError("focus a view tile first")
                focused.set_view(args[0])
            elif name == "audio":
                self._run_audio_command(args, log)
            elif name == "opacity":
                self.set_opacity(float(args[0]))
            elif name == "raise":
                self.lift_tile(1)
            elif name == "lower":
                self.lift_tile(-1)
            elif name == "move":
                self.move_tile(int(args[0]), int(args[1]))
            elif name in ("screenshot", "ss"):
                self.call_after_refresh(lambda: log.write(f"saved {self.screenshot(*args[:1])}"))
            elif name in ("q", "quit"):
                self.exit()
            else:
                log.write(f"unknown command: {name}")
        except Exception as error:  # a bad command must not take down the shell
            log.write(f"error: {error!r}")


    def set_mode(self, mode: str) -> None:
        self.mode = mode
        zoom = f"   zoom: {self.zoomed}" if self.zoomed else ""
        self.query_one("#mode", Static).update(
            f"{mode}   audio: {self.runtime.feed.source}   grid: {self.runtime.resolution}"
            f" ({self.runtime.step_ms:.0f} ms){zoom}"
        )

    def select_audio(self, choice: str) -> None:
        """Switch audio source; a device that fails to open leaves the previous source running."""
        log = self.query_one("#log", LogTile)
        feed = self.runtime.feed
        try:
            if choice == "synthetic":
                feed.use_synthetic()
            else:
                feed.use_device(None if choice == "default" else int(choice))
            log.write(f"audio source: {feed.source}")
        except Exception as error:  # a missing or busy device must not take down the shell
            log.write(f"audio {choice!r} failed: {error}; keeping {feed.source}")
        self.set_mode(self.mode)

    def _run_audio_command(self, args: list[str], log: LogTile) -> None:
        choice = args[0] if args else "list"
        if choice == "list":
            log.write(f"source: {self.runtime.feed.source}")
            for line in self.runtime.feed.list_devices():
                log.write(line)
        elif choice == "tone":
            self.runtime.feed.use_tone(float(args[1]))
            log.write(f"audio source: {self.runtime.feed.source}")
            self.set_mode(self.mode)
        else:
            self.select_audio(choice)


def main() -> None:
    parser = argparse.ArgumentParser(description="AudioViz terminal interface")
    parser.add_argument(
        "--audio", default="default", help="default | synthetic | <input device index>"
    )
    AudioVizApp(audio=parser.parse_args().audio).run()


if __name__ == "__main__":
    main()
