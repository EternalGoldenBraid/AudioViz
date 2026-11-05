"""Terminal front end: modal, Vim-style control over layered, floating tiles."""

import argparse
import json
import os
import shutil
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, RichLog, Static

from audioviz.tui import control
from audioviz.tui.audio_feed import BLOCK_SIZE, SAMPLE_RATE
from audioviz.tui.deck_client import DeckClient
from audioviz.tui.device_checklist import DeviceChecklist
from audioviz.tui.graph_runtime import GraphRuntime, format_resolution
from audioviz.tui.devices import DevicePicker, list_devices
from audioviz.tui.halfblock import RENDER_MODES, render_cells
from audioviz.tui import keys as keys_module
from audioviz.tui.hot_reload import HotReload
from audioviz.tui.keys_overlay import KeysScreen
from audioviz.tui.mixer import CONTROLS
from audioviz.tui.settings import settings_controls
from audioviz.tui.themes import THEMES
from audioviz.tui.tile_list import TileList
from audioviz.tui import views as views_module

MAX_LAYERS = 8
BRAILLE_MIN_CELLS = 1200  # tiles at least this large render at braille density
AUTOSAVE_SECONDS = 5
SEQUENCE_SECONDS = 1.0  # how long the first key of a multi-key binding waits for the next
CACHE_PATH = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "audioviz" / "state.json"
SAVES_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "audioviz" / "saves"


@dataclass
class Placement:
    """Where a tile sits: cell position, size and opacity; stacking order lives in the app."""

    x: int
    y: int
    width: int
    height: int
    opacity: float = 1.0


KNOB_HEIGHT = 3
MIXER_PAGE = 6  # knobs shown at once in the mixer group
HIDDEN_BY_DEFAULT = ("camera_2_prediction", "camera_2_error")  # sit over camera_2 until shown from the tile list


def knob_id(name: str) -> str:
    return "knob-" + name.replace(" ", "-")


def default_layout(width: int, height: int, mixer_ids: list[str], settings_ids: list[str]) -> dict[str, Placement]:
    """Four columns of views over a strip of knob tiles: mixer knobs left, settings knobs right."""
    edges = [round(i * width / 4) for i in range(5)]
    bottom = min(11, height // 2) // KNOB_HEIGHT * KNOB_HEIGHT
    rest = height - bottom
    first, second = round(rest * 0.36), round(rest * 0.32)
    layout = {}
    for row, y, h in ((("canvas", "camera_1", "visual_prediction", "visual_error"), 0, first),
                      (("spectrogram", "waveform", "prediction", "error"), first, second),
                      (("cochlea", "hidden", "camera_2", "log"), first + second, rest - first - second)):
        columns = {2: [0, 2, 4], 3: [0, 1, 2, 4]}.get(len(row), range(len(row) + 1))  # a short row spans whole columns
        for i, name in enumerate(row):
            layout[name] = Placement(edges[columns[i]], y, edges[columns[i + 1]] - edges[columns[i]], h)
    for name in HIDDEN_BY_DEFAULT:
        layout[name] = Placement(**vars(layout["camera_2"]))
    per_column = bottom // KNOB_HEIGHT
    for ids, x0, x1, slots in ((mixer_ids, 0, edges[2], MIXER_PAGE), (settings_ids, edges[2], width, len(settings_ids))):
        columns = -(-slots // per_column)
        cell = (x1 - x0) // columns
        for i, name in enumerate(ids):  # members past the first page reuse the page's slots
            column, row = divmod(i % slots, per_column)
            last = column == columns - 1
            layout[name] = Placement(
                x0 + column * cell, rest + row * KNOB_HEIGHT,
                (x1 - x0 - column * cell) if last else cell, KNOB_HEIGHT,
            )
    return layout


class ViewTile(Static, can_focus=True):
    """A focusable pane showing one subgraph view; focus and Placement are shared by keys and gaze."""

    DEFAULT_CSS = """
    ViewTile { position: absolute; border: round $primary-darken-2; padding: 0; background: $surface; }
    ViewTile:focus { border: heavy $accent; }
    """

    def __init__(self, view: str, runtime: GraphRuntime, deck=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.runtime = runtime
        self.deck = deck  # when attached, the images come from the running window
        self.render_mode: str | None = None  # None: choose by tile size
        self.set_view(view)

    def mode_for(self, cells: int) -> str:
        if self.render_mode:
            return self.render_mode
        return "braille" if cells >= BRAILLE_MIN_CELLS else "half"

    def set_view(self, view: str) -> None:
        self.view = view
        self.border_title = view
        self.border_subtitle = self.view_caption

    @property
    def view_caption(self) -> str:
        return {
            "cochlea": "index / time > | -80..0 dB",
            "waveform": "fixed +/-1 | time >",
        }.get(self.view, "")

    def refresh_view(self) -> None:
        size = self.content_size
        if size.width < 2 or size.height < 1:
            return
        mode = self.mode_for(size.width * size.height)
        sub_rows, sub_cols = RENDER_MODES[mode]
        pixels = (size.height * sub_rows, size.width * sub_cols)
        if self.deck is not None:
            image = self.deck.preview(self.view, *pixels)
        else:
            image = views_module.VIEWS[self.view](self.runtime, *pixels)
        if image is None:
            self.update("Select audio model: cochlea; waiting for audio" if self.view == "cochlea" else f"{self.view}: waiting for data")
            return
        self.update(render_cells(image, size.height, size.width, mode))


class KnobTile(Static, can_focus=True):
    """One Control as a tile: -/+ or arrows turn it; Enter enters it so h/l slide it. h/j/k/l walk between knobs otherwise. It is a leaf node like any view, so it can be moved, grouped and lifted."""

    DEFAULT_CSS = """
    KnobTile { position: absolute; border: round $primary-darken-2; padding: 0 1; background: $surface; }
    KnobTile:focus { border: heavy $accent; }
    KnobTile.-editing { border: double $warning; }
    """

    editing = False

    def __init__(self, context, control, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context = context  # the object the control reads and writes: a runtime or the app
        self.control = control
        self.border_title = control.name

    def adjust(self, steps: int) -> None:
        try:
            with getattr(self.context, "lock", nullcontext()):
                self.control.nudge(self.context, steps)
        except Exception as error:  # a failing setting (busy device, bad value) must not take down the shell
            self.app.query_one("#log", LogTile).write(f"{self.control.name}: {error}")
        self.refresh_view()

    def refresh_view(self) -> None:
        width = self.content_size.width
        if width < 6:
            return
        value = self.control.format(self.context)
        if getattr(self.control, "is_picker", False):  # a device list: no bar, the choice is the content
            self.update(Text.assemble(("◉ ", "bold green" if value != "off" else "grey50"), value.partition(": ")[2] or value))
            return
        bar = max(width - len(value) - 1 - (5 if self.editing else 0), 2)
        filled = round(self.control.position(self.context) * bar)
        text = Text()
        text.append("█" * filled, style="green")
        text.append("░" * (bar - filled), style="grey37")
        text.append(f" {value}")
        if self.editing:
            text.append("  ◀ ▶", style="bold yellow")
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
    .-selected {{ border: double $warning !important; }}
    .-group-focus {{ border: heavy $accent !important; }}
    #command {{ dock: bottom; display: none; border: none; height: 1; margin-bottom: 1; padding: 0 1; }}
    #command.-active {{ display: block; }}
    #mode {{ dock: bottom; height: 1; background: $primary-darken-2; padding: 0 1; }}
    """
    SCREENSHOT_DIR = Path("outputs/tui")
    MOVE_STEP = (2, 1)
    OPACITY_STEP = 0.1

    def __init__(self, audio: str = "default", camera: bool = True, state_dir: Path | None = None,
                 autosave: bool = True, hot_reload: bool = True, attach: bool = False) -> None:
        super().__init__()
        self.audio = audio
        self.mode = "NORMAL"
        self.attached = attach  # the mixer knobs drive a running main.py window; the local graph stays idle
        self.runtime = GraphRuntime(camera=camera and not attach)
        if attach:
            self.deck = DeckClient()
            self.mixer_controls, self.mixer_context = self.deck.controls(), self.deck
        else:
            self.mixer_controls, self.mixer_context = CONTROLS, self.runtime
        name = "state-attached.json" if attach else "state.json"
        self.cache_path = state_dir / name if state_dir else CACHE_PATH.with_name(name)
        self.saves_dir = state_dir / "saves" if state_dir else SAVES_DIR
        self.keys_path = state_dir / "keys.toml" if state_dir else keys_module.CONFIG_PATH
        self.load_keys()
        self.autosave = autosave
        self.settings = settings_controls()
        if attach:  # resolution and sound belong to the local graph, which is idle
            self.settings = tuple(c for c in self.settings if c.name in ("theme", "tile background", "preview px"))
        else:
            self.settings = tuple(c for c in self.settings if c.name != "preview px")
        self.hot_reload = hot_reload
        self._reloader = HotReload()
        self._saved: dict | None = None
        self.knobs = [(knob_id(c.name), self.mixer_context, c) for c in self.mixer_controls]
        self.knobs += [(knob_id(c.name), self, c) for c in self.settings]
        self.mixer_ids = [k for k, ctx, _ in self.knobs if ctx is self.mixer_context]
        self.settings_ids = [k for k, ctx, _ in self.knobs if ctx is self]
        self.layout_size: tuple[int, int] | None = None  # the stage size the placements were laid out for; set on mount
        self.placements = default_layout(120, 36, self.mixer_ids, self.settings_ids)  # resized to the real stage on mount
        self.pages: dict[str, dict[str, int]] = {self.mixer_ids[0]: {"size": MIXER_PAGE, "index": 0}}  # by group's first member
        self.group_focus: list[str] = []  # non-empty while a whole group is the focused node
        self.groups: list[list[str]] = [list(self.mixer_ids), list(self.settings_ids)]  # move, lift and fade together
        self.selection: list[str] = []  # being collected in VISUAL mode
        self.hidden: set[str] = set(HIDDEN_BY_DEFAULT)
        self.hide_history: list[list[str]] = []  # what each hide took away, so `u` can bring it back
        self.pending_keys: list[str] = []  # the first keys of a multi-key binding such as `d d`
        self.stack = list(self.placements)  # bottom -> top
        self._return_focus = None
        self.zoomed: str | None = None
        self.immersive = False
        self._tmux_zoomed = False
        self.tile_background: str | None = None  # overrides the theme surface

    def compose(self) -> ComposeResult:
        with Container(id="stage"):
            for name in self.placements:
                if name == "log":
                    yield LogTile(id="log", wrap=True)
                elif name.startswith("knob-"):
                    _, context, control = next(k for k in self.knobs if k[0] == name)
                    yield KnobTile(context, control, id=name)
                else:
                    yield ViewTile(name, self.runtime, self.deck if self.attached else None, id=name)
        yield Static("NORMAL", id="mode")
        yield Input(placeholder=":", id="command")

    def on_mount(self) -> None:
        for theme in THEMES:
            self.register_theme(theme)
        self.theme = "black"
        self.query_one("#log", LogTile).border_title = "log"
        for name, placement in default_layout(self.size.width, max(self.size.height - 1, 12), self.mixer_ids, self.settings_ids).items():
            self.placements[name] = placement
        self.layout_size = self.stage_size()
        self.load_state(self.cache_path, quiet=True)
        self.refit()
        if self.attached:
            self.set_interval(0.25, self.refresh_deck)
        else:
            self.runtime.start()
        self.set_interval(1 / 15, self.refresh_tiles)
        if self.hot_reload:
            self.set_interval(0.5, self.reload_code)
        if self.autosave:
            self.set_interval(AUTOSAVE_SECONDS, self.autosave_state)
        self.query_one("#canvas").focus()
        if not self.attached:
            self.select_audio(self.audio)
        self.run_worker(self._serve_control(), exclusive=False)

    async def _serve_control(self) -> None:
        server = await control.serve(self)
        if server is None:  # another instance owns the socket; leave it alone
            return
        self.owns_socket = True
        async with server:
            await server.serve_forever()

    def describe(self) -> dict:
        """Observable state for the control socket."""
        return {
            "groups": self.groups,
            "pages": self.pages,
            "camera_error": self.runtime.camera.error,
            "focus": self._focused_name(),
            "group_focus": self.group_focus,
            "mode": self.mode,
            "zoomed": self.zoomed,
            "immersive": self.immersive,
            "output_device": self.runtime.out.device,
            "theme": self.theme,
            "audio": self.runtime.feed.source,
            "out": self.runtime.out.mode,
            "grid": format_resolution(self.runtime.resolution),
            "step_ms": round(self.runtime.step_ms, 1),
            "stack": self.stack,
            "size": list(self.layout_size),
            "placements": {k: vars(v) for k, v in self.placements.items()},
            "mixer": {c.name: c.format(self.mixer_context) for c in self.mixer_controls},
            "settings": {c.name: c.format(self) for c in self.settings},
        }

    def on_unmount(self) -> None:
        if getattr(self, "owns_socket", False):
            control.SOCKET.unlink(missing_ok=True)
        if self.autosave:
            self.autosave_state()
        self.runtime.stop()

    def refresh_deck(self) -> None:
        def poll() -> None:
            try:
                before = self.sensors_enabled()
                self.deck.refresh()
                self.deck.fetch_previews()
                if self.sensors_enabled() != before:  # a sensor was switched on or off: its tile appears or goes
                    self.call_from_thread(self.apply_layout)
            except (OSError, RuntimeError) as error:  # the window closed or stalled; knobs keep their last values
                self.call_from_thread(self.query_one("#log", LogTile).write, f"deck: {error}")

        self.run_worker(poll, thread=True, exclusive=True, group="deck")

    def refresh_tiles(self) -> None:
        self.fit_if_resized()
        for tile in self.query(ViewTile):
            if tile.display:
                tile.refresh_view()
        self.set_mode(self.mode)
        for tile in self.query(KnobTile):
            if tile.display:
                tile.refresh_view()

    def apply_layout(self) -> None:
        stage = self.query_one("#stage").size
        for depth, name in enumerate(self.stack):
            placement = self.placements[name]
            widget = self.query_one(f"#{name}")
            styles = widget.styles
            zoomed = self.zoomed == name
            widget.display = self.zoomed in (None, name) and self.visible(name)
            styles.offset = (0, 0) if zoomed else (placement.x, placement.y)
            styles.width = stage.width if zoomed else placement.width
            styles.height = stage.height if zoomed else placement.height
            styles.opacity = 1.0 if zoomed else placement.opacity
            styles.layer = f"z{depth}"
            widget.border_subtitle = self.page_label(name) or (widget.view_caption if isinstance(widget, ViewTile) else "")
        self.set_mode(self.mode)

    def page_of(self, name: str) -> tuple[list[str], dict[str, int]] | None:
        group = self.members(name)
        page = self.pages.get(group[0])
        return (group, page) if page and len(group) > page["size"] else None

    def sensors_enabled(self) -> dict[str, bool]:
        """Which sensors the graph is using; a tile named after a sensor shows only while it is enabled."""
        if self.attached:
            return {s["id"]: s["enabled"] for s in self.deck.sensors.values()}
        orchestrator = self.runtime.orchestrator
        return {i: self.runtime.show_camera if s is orchestrator.camera_source else s.enabled
                for i, s in orchestrator.sensors.items()}

    def sensor_of(self, name: str) -> str | None:
        """The sensor a tile is named after: `camera_2` and `camera_2_error` both belong to `camera 2`."""
        return next((s for s in self.sensors_enabled() if name == s.replace(" ", "_") or name.startswith(s.replace(" ", "_") + "_")), None)

    def visible(self, name: str) -> bool:
        sensor = self.sensor_of(name)
        if name in self.hidden or (sensor is not None and not self.sensors_enabled()[sensor]):
            return False
        paged = self.page_of(name)
        if paged is None:
            return True
        group, page = paged
        return group.index(name) // page["size"] == page["index"]

    def hide_tiles(self, names: list[str]) -> None:
        """Hides the tiles that are shown, never the last visible one; focus moves to the nearest tile left."""
        names = [n for n in names if n not in self.hidden]
        remaining = [n for n in self.placements if n not in self.hidden and n not in names]
        if not names or not remaining:
            return
        focused = self._focused_name()
        nearest = next((t for t in (self.neighbour(focused, *d) for d in ((0, 1), (0, -1), (1, 0), (-1, 0))) if t in remaining), remaining[0])
        self.hidden |= set(names)
        self.hide_history.append(names)
        self.apply_layout()
        self.query_one(f"#{nearest}").focus()
        self.query_one("#log", LogTile).write(f"hid {', '.join(names)}; u brings it back")

    def hide(self) -> None:
        if not self.zoomed:
            self.hide_tiles(self.group_focus or [self._focused_name()])

    def unhide(self) -> None:
        if not self.hide_history:
            return
        names = self.hide_history.pop()
        self.hidden -= set(names)
        self.apply_layout()
        self.query_one(f"#{names[0]}").focus()

    def open_tiles(self) -> None:
        """One section per group, then the tiles that belong to none."""
        grouped = {n for g in self.groups for n in g}
        labels = {tuple(self.mixer_ids): "mixer", tuple(self.settings_ids): "settings"}
        sections = [(labels.get(tuple(g), f"group of {g[0]}"), g) for g in self.groups]
        loose = [n for n in self.placements if n not in grouped]
        self.push_screen(TileList(sections + [("tiles", loose)], self.hidden), self.apply_tiles)

    def apply_tiles(self, hidden: set[str] | None) -> None:
        if hidden is None or hidden == self.hidden:
            return
        shown = [n for n in self.placements if n not in hidden]
        if not shown:
            self.query_one("#log", LogTile).write("keep at least one tile")
            return
        newly = [n for n in self.placements if n in hidden and n not in self.hidden]
        if newly:
            self.hide_history.append(newly)
        self.hidden = {n for n in hidden if n in self.placements}
        self.apply_layout()
        self.query_one(f"#{next(n for n in shown if self.visible(n))}").focus()

    def page_label(self, name: str) -> str:
        paged = self.page_of(name)
        if paged is None:
            return ""
        group, page = paged
        return f"{page['index'] + 1}/{-(-len(group) // page['size'])}"

    def turn_page(self, delta: int) -> None:
        """Show the group's next page: its members take over the slots of the page being left."""
        name = self._focused_name()
        paged = self.page_of(name)
        if paged is None:
            return
        group, page = paged
        size = page["size"]
        count = -(-len(group) // size)
        new = (page["index"] + delta) % count
        slots = [replace(self.placements[n]) for n in group[page["index"] * size:(page["index"] + 1) * size]]
        arriving = group[new * size:(new + 1) * size]
        for slot, member in zip(slots, arriving):
            self.placements[member] = replace(slot)
        page["index"] = new
        self.apply_layout()
        self.query_one(f"#{arriving[min(group.index(name) % size, len(arriving) - 1)]}").focus()

    def mark_group_focus(self) -> None:
        for name in self.placements:
            self.query_one(f"#{name}").set_class(name in self.group_focus, "-group-focus")

    def rise(self) -> None:
        """Focus the whole group of the focused tile: navigation then steps between top-level nodes."""
        members = self.members(self._focused_name())
        if len(members) > 1 and not self.zoomed:
            self.group_focus = members
            self.mark_group_focus()

    def dive(self) -> None:
        if self.group_focus:
            self.group_focus = []
            self.mark_group_focus()
        elif isinstance(self.focused, KnobTile) and getattr(self.focused.control, "is_picker", False):
            self.open_checklist(self.focused)
        elif isinstance(self.focused, KnobTile):
            self.set_editing(True)

    def open_checklist(self, knob: "KnobTile") -> None:
        control = knob.control
        screen = DeviceChecklist(control.name, control.choices, control.selected, control.capacity)
        # Pushed after the opening Enter has been fully handled, so the modal's own Enter binding never sees it.
        self.call_later(self.push_screen, screen, lambda values: self.apply_checklist(knob, values))

    def apply_checklist(self, knob: "KnobTile", values: list[str] | None) -> None:
        if values is None or values == knob.control.selected:
            return
        log = self.query_one("#log", LogTile)

        def apply() -> None:
            try:
                knob.control.select(values)
                self.call_from_thread(log.write, f"{knob.control.name}: {', '.join(values) or 'off'}")
            except Exception as error:  # a busy or missing device leaves the current one running
                self.call_from_thread(log.write, f"{knob.control.name}: {error}")
            self.call_from_thread(knob.refresh_view)

        self.run_worker(apply, thread=True, group="deck-select")

    def set_editing(self, on: bool) -> None:
        knob = self.focused
        if isinstance(knob, KnobTile):
            knob.editing = on
            knob.set_class(on, "-editing")
            knob.refresh_view()
        self.set_mode("EDIT" if on and isinstance(knob, KnobTile) else "NORMAL")

    def stage_size(self) -> tuple[int, int]:
        return self.size.width, max(self.size.height - 1, 12)

    def refit(self) -> None:
        """Scale placements from the size they were laid out for to the current terminal.

        Knob tiles keep their height and stay anchored to the bottom; every other tile scales
        in both directions over the space above the knob strip.
        """
        (old_w, old_h), (new_w, new_h) = self.layout_size, self.stage_size()
        knobs = [p for n, p in self.placements.items() if n.startswith("knob-")]
        strip = old_h - min((p.y for p in knobs), default=old_h)
        sx, sy = new_w / old_w, max(new_h - strip, 3) / max(old_h - strip, 3)
        for name, p in self.placements.items():
            p.x, p.width = round(p.x * sx), max(round(p.width * sx), 4)
            if name.startswith("knob-"):
                p.y += new_h - old_h
            else:
                p.y, p.height = round(p.y * sy), max(round(p.height * sy), 3)
        self.layout_size = (new_w, new_h)
        self.apply_layout()

    def fit_if_resized(self) -> None:
        if self.layout_size and self.layout_size != self.stage_size():
            self.refit()

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

    def _focused_name(self) -> str:
        focused = self.focused
        return focused.id if focused is not None and focused.id in self.placements else self.stack[-1]

    def members(self, name: str) -> list[str]:
        """The tile's group, or the tile alone."""
        return next((group for group in self.groups if name in group), [name])

    def move_tile(self, dx: int, dy: int) -> None:
        for name in self.members(self._focused_name()):
            placement = self.placements[name]
            placement.x, placement.y = max(0, placement.x + dx), max(0, placement.y + dy)
        self.apply_layout()

    def lift_tile(self, delta: int) -> None:
        """Move the tile's group through the stack as one contiguous block."""
        block = [name for name in self.stack if name in self.members(self._focused_name())]
        first = self.stack.index(block[0])
        rest = [name for name in self.stack if name not in block]
        target = min(max(first + delta, 0), len(rest))
        self.stack = rest[:target] + block + rest[target:]
        self.apply_layout()

    def make_group(self, names: list[str]) -> None:
        self.groups = [[n for n in group if n not in names] for group in self.groups]
        self.groups = [group for group in self.groups if len(group) > 1]
        if len(names) > 1:
            self.groups.append(list(names))
        self.pages = {k: v for k, v in self.pages.items() if any(g[0] == k for g in self.groups)}
        self.apply_layout()

    def start_visual(self) -> None:
        self.selection = [self._focused_name()]
        self.set_mode("VISUAL")
        self.mark_selection()

    def end_visual(self, commit: bool) -> None:
        if commit:
            self.make_group(self.selection)
        self.selection = []
        self.set_mode("NORMAL")
        self.mark_selection()

    def extend_selection(self, dx: int, dy: int) -> None:
        target = self.neighbour(self._focused_name(), dx, dy)
        if target:
            self.query_one(f"#{target}").focus()
            if target not in self.selection:
                self.selection.append(target)
            self.mark_selection()

    def mark_selection(self) -> None:
        for name in self.placements:
            self.query_one(f"#{name}").set_class(name in self.selection, "-selected")

    def set_tile_background(self, color: str | None) -> None:
        """Colour behind every tile; None follows the theme's surface."""
        from textual.color import Color

        if color is not None:
            Color.parse(color)  # raises on an invalid colour
        self.tile_background = color
        for tile in self.query("ViewTile, KnobTile, LogTile"):
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
        tile.border_title = f"{tile.view} [{mode}]" if mode else tile.view

    def set_opacity(self, value: float) -> None:
        for name in self.members(self._focused_name()):
            self.placements[name].opacity = min(max(value, 0.1), 1.0)
        self.apply_layout()

    def nudge_opacity(self, delta: float) -> None:
        self.set_opacity(self.placements[self._focused_name()].opacity + delta)

    def neighbour(self, name: str, dx: int, dy: int, outer: bool = False) -> str | None:
        """Nearest tile in a direction: its centre must lie that way; sideways offset costs double.

        Cells are about twice as tall as wide, so vertical distance is doubled to compare like with like.
        """
        group = self.members(name) if outer else [name]
        rects = [self.placements[n] for n in group]
        left, top = min(p.x for p in rects), min(p.y for p in rects)
        right, bottom = max(p.x + p.width for p in rects), max(p.y + p.height for p in rects)
        cx, cy = (left + right) / 2, (top + bottom)
        best, best_cost = None, float("inf")
        for other, p in self.placements.items():
            if other in group or not self.visible(other):
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
        target = self.neighbour(self._focused_name(), dx, dy, outer=bool(self.group_focus))
        if target is None:
            return
        self.group_focus = self.members(target) if self.group_focus and len(self.members(target)) > 1 else []
        self.mark_group_focus()
        self.query_one(f"#{target}").focus()

    def _move_focus(self, step: int) -> None:
        names = [name for name in self.placements if self.zoomed in (None, name) and self.visible(name)]
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
        mode = "visual" if self.mode == "VISUAL" else "normal"
        knob = self.focused if isinstance(self.focused, KnobTile) else None
        if mode == "normal" and knob is not None and knob.editing:
            mode = "edit"
        elif mode == "normal" and knob is not None and not self.group_focus:
            mode = "knob"
        action = self.lookup(mode, event.key)
        if self.pending_keys:
            sequence = " ".join([*self.pending_keys, event.key])
            self.pending_keys = []
            action = self.lookup(mode, sequence) or action
        if action is None and self.starts_sequence(mode, event.key):
            self.pending_keys = [event.key]
            self.set_timer(SEQUENCE_SECONDS, self.clear_pending)
            event.stop()
            return
        if (action is not None and self.perform(action)) or mode in ("visual", "edit"):
            event.stop()  # selecting and entered knobs swallow every key

    def lookup(self, mode: str, key: str) -> str | None:
        action = self.keymap.get((mode, key))
        if action is None and mode == "knob":
            action = self.keymap.get(("normal", key))
        return action

    def starts_sequence(self, mode: str, key: str) -> bool:
        modes = (mode, "normal") if mode == "knob" else (mode,)
        return any(m in modes and k.startswith(key + " ") for m, k in self.keymap)

    def clear_pending(self) -> None:
        self.pending_keys = []

    def perform(self, action: str) -> bool:
        """False when the action does not apply to the focused tile, so the key stays unhandled."""
        dx, dy = self.MOVE_STEP
        knob = self.focused
        actions = {
            "focus_left": lambda: self.focus_toward(-1, 0), "focus_right": lambda: self.focus_toward(1, 0),
            "focus_up": lambda: self.focus_toward(0, -1), "focus_down": lambda: self.focus_toward(0, 1),
            "next_tile": lambda: self._move_focus(1), "prev_tile": lambda: self._move_focus(-1),
            "rise": self.rise, "dive": self.dive, "exit_edit": lambda: self.set_editing(False),
            "slide_down": lambda: knob.adjust(-1), "slide_up": lambda: knob.adjust(1),
            "move_left": lambda: self.move_tile(-dx, 0), "move_right": lambda: self.move_tile(dx, 0),
            "move_up": lambda: self.move_tile(0, -dy), "move_down": lambda: self.move_tile(0, dy),
            "lift": lambda: self.lift_tile(1), "lower": lambda: self.lift_tile(-1),
            "opacity_up": lambda: self.nudge_opacity(self.OPACITY_STEP),
            "opacity_down": lambda: self.nudge_opacity(-self.OPACITY_STEP),
            "visual": self.start_visual,
            "ungroup": lambda: self.make_group(self.members(self._focused_name())[:1]),
            "page_next": lambda: self.turn_page(1), "page_prev": lambda: self.turn_page(-1),
            "select_left": lambda: self.extend_selection(-1, 0), "select_right": lambda: self.extend_selection(1, 0),
            "select_up": lambda: self.extend_selection(0, -1), "select_down": lambda: self.extend_selection(0, 1),
            "commit_group": lambda: self.end_visual(True), "cancel_group": lambda: self.end_visual(False),
            "zoom": self.toggle_zoom, "immersive": self.toggle_immersive, "render": self.cycle_render,
            "devices": self.open_devices,
            "hide": self.hide, "unhide": self.unhide, "tiles": self.open_tiles,
            "turn_down": lambda: knob.adjust(-1), "turn_up": lambda: knob.adjust(1),
            "command": self._open_command, "help": self.show_keys, "quit": self.exit,
        }
        actions[action]()
        return True

    def load_keys(self) -> list[str]:
        overrides, problems = keys_module.load_overrides(self.keys_path)
        self.bindings = keys_module.bindings(overrides)
        self.keymap = keys_module.keymap(self.bindings)
        self.key_problems = problems
        return problems

    def run_keys_command(self, args: list[str]) -> None:
        log = self.query_one("#log", LogTile)
        if args[:1] == ["write"]:
            if self.keys_path.exists():
                log.write(f"{self.keys_path} exists; edit it, then :keys reload")
            else:
                self.keys_path.parent.mkdir(parents=True, exist_ok=True)
                self.keys_path.write_text(keys_module.default_toml())
                log.write(f"wrote {self.keys_path}")
        elif args[:1] == ["reload"]:
            problems = self.load_keys()
            log.write(f"reloaded {self.keys_path}" + "".join(f"\n{p}" for p in problems))
        else:
            self.show_keys()

    def show_keys(self) -> None:
        self.push_screen(KeysScreen(self.bindings, self.key_problems))

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

    def screenshot(self, name: str = "tui", directory: Path | None = None) -> Path:
        """Save an SVG (and a PNG when rsvg-convert exists) of the current screen."""
        directory = directory or self.SCREENSHOT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        svg = Path(self.save_screenshot(f"{name}.svg", str(directory)))
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
                log.write("theme [name]  bg <color|theme>  group <tiles>  ungroup")
                log.write(":w [name] save state+screenshot  :e [name] load   ? key bindings  :keys [write|reload]")
                log.write("out off|prediction|error|input|evidence  volume <v>  audio tone <hz>")
                log.write("resolution <n|RxC>  render half|quad|braille|auto  zoom")
                log.write("screenshot [name]  q   save  rescan (attached: re-list audio devices)  (attached: window knobs -> config/deck.json, loaded by main.py)")
            elif name == "keys":
                self.run_keys_command(args)
            elif name == "save":
                if not self.attached:
                    raise ValueError("save writes the main.py window's knobs; start with --attach")
                log.write(f"saved {self.deck.save()}")
            elif name == "rescan":
                if not self.attached:
                    raise ValueError("rescan re-lists the window's devices; start attached")
                log.write("rescanning devices (audio drops for a moment)")

                def rescan() -> None:
                    try:
                        self.deck.rescan()
                        self.call_from_thread(log.write, "devices rescanned")
                    except Exception as error:
                        self.call_from_thread(log.write, f"rescan: {error}")

                self.run_worker(rescan, thread=True, group="deck-select")
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
            elif name == "group":
                if any(n not in self.placements for n in args):
                    raise ValueError(f"unknown tile in {args}")
                self.make_group(args)
            elif name == "ungroup":
                self.make_group(self.members(self._focused_name())[:1])
            elif name in ("w", "write"):
                self.write_session(args[0] if args else "state")
            elif name in ("e", "edit"):
                self.load_state(self.saves_dir / f"{args[0] if args else 'state'}.json")
            elif name == "out":
                self.runtime.out.set_mode(args[0], speakers=len(args) < 2 or args[1] != "silent")
                log.write(f"output: {self.runtime.out.mode}")
            elif name == "volume":
                self.runtime.out.volume = float(args[0])
            elif name == "resolution":
                self.runtime.rebuild(args[0])
                log.write(f"grid {format_resolution(self.runtime.resolution)}")
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
                log.write(" ".join(views_module.VIEWS))
            elif name == "view":
                if args[0] not in views_module.VIEWS:
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


    def snapshot(self) -> dict:
        """Everything a restart needs to look and sound the same; sound output stays off for safety."""
        return {
            "placements": {k: vars(v) for k, v in self.placements.items()},
            "stack": self.stack,
            "groups": self.groups,
            "pages": self.pages,
            "theme": self.theme,
            "tile_background": self.tile_background,
            "hidden": sorted(self.hidden),
            "views": {t.id: t.view for t in self.query(ViewTile)},
            "render": {t.id: t.render_mode for t in self.query(ViewTile)},
            "mixer": {c.name: c.get(self.mixer_context) for c in self.mixer_controls},
            "settings": {c.name: c.get(self) for c in self.settings if c.name != "sound out"},
        }

    def restore(self, data: dict) -> None:
        same_tiles = set(data.get("placements", {})) == set(self.placements)  # a changed tile set gets the default layout
        for name, values in data.get("placements", {}).items():
            if same_tiles:
                self.placements[name] = Placement(**values)
        if same_tiles:
            extent = (max(p.x + p.width for p in self.placements.values()),
                      max(p.y + p.height for p in self.placements.values()))
            self.layout_size = tuple(data.get("size", extent))  # layouts saved before sizes were recorded span their extent
        stack = [n for n in data.get("stack", []) if n in self.placements]
        self.stack = stack + [n for n in self.placements if n not in stack]
        hidden = {n for n in data.get("hidden", []) if n in self.placements}
        hidden |= {n for n in HIDDEN_BY_DEFAULT if n not in data.get("placements", {})}  # tiles the save predates
        self.hidden = hidden if len(hidden) < len(self.placements) else set()
        if "groups" in data and same_tiles:
            groups = [[n for n in g if n in self.placements] for g in data["groups"]]
            self.groups = [g for g in groups if len(g) > 1]
            self.pages = {k: v for k, v in data.get("pages", {}).items()
                          if any(g[0] == k and len(g) > v["size"] * v["index"] for g in self.groups)}
        for tile in self.query(ViewTile):
            if data.get("views", {}).get(tile.id) in views_module.VIEWS:
                tile.set_view(data["views"][tile.id])
            self.set_render(tile, data.get("render", {}).get(tile.id))
        for controls, context, saved in ((self.settings, self, data.get("settings", {})),
                                         (self.mixer_controls, self.mixer_context, data.get("mixer", {}))):
            for control in controls:  # settings first: a resolution change rebuilds the graph
                if control.name in saved:
                    control.set(context, saved[control.name])
        if data.get("theme") in self.available_themes:
            self.theme = data["theme"]
        self.set_tile_background(data.get("tile_background"))

    def reload_code(self) -> None:
        log = self.query_one("#log", LogTile)
        try:
            for name in self._reloader.poll():
                log.write(f"reloaded {name}")
        except Exception as error:  # a half-typed edit keeps the previous code running
            log.write(f"reload failed: {error!r}")

    def autosave_state(self) -> None:
        data = self.snapshot()
        if data != self._saved:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(data, indent=1))
            self._saved = data

    def load_state(self, path: Path, quiet: bool = False) -> None:
        try:
            self.restore(json.loads(path.read_text()))
        except FileNotFoundError:
            if not quiet:
                self.query_one("#log", LogTile).write(f"no saved state at {path}")
        except Exception as error:  # a stale or hand-edited file must not stop the app starting
            self.query_one("#log", LogTile).write(f"could not load {path}: {error!r}")
        else:
            self.apply_layout()

    def write_session(self, name: str) -> None:
        """Write the state and a screenshot side by side."""
        self.saves_dir.mkdir(parents=True, exist_ok=True)
        path = self.saves_dir / f"{name}.json"
        path.write_text(json.dumps(self.snapshot(), indent=1))
        self.call_after_refresh(
            lambda: self.query_one("#log", LogTile).write(f"wrote {path} and {self.screenshot(name, self.saves_dir)}")
        )

    def set_mode(self, mode: str) -> None:
        if mode == "EDIT" and not isinstance(self.focused, KnobTile):
            mode = "NORMAL"
        self.mode = mode
        zoom = f"   zoom: {self.zoomed}" if self.zoomed else ""
        if mode == "EDIT":
            mode = f"EDIT {self.focused.control.name}   h/l slide, Enter/Esc leave"
        graph = "attached to main.py" if self.attached else (
            f"audio: {self.runtime.feed.source}   grid: {format_resolution(self.runtime.resolution)}"
            f" ({self.runtime.step_ms:.0f} ms)")
        self.query_one("#mode", Static).update(f"{mode}   {graph}{zoom}")

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
    parser.add_argument("--no-camera", action="store_true", help="do not open the camera")
    parser.add_argument("--attach", action="store_true", help="control a running main.py window instead of a local graph")
    args = parser.parse_args()
    AudioVizApp(audio=args.audio, camera=not args.no_camera, attach=args.attach).run()


if __name__ == "__main__":
    main()
