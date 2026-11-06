"""Device preparation and child-window lifetime owned by the terminal."""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess

import sounddevice as sd
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from audioviz.tui.deck_client import DeckClient
from audioviz.tui.deck_server import SOCKET
from audioviz.tui.device_checklist import DeviceChecklist
from audioviz.utils.audio_devices import load_device_config, make_device_config, validate_device_config


class StartupPanel(Vertical):
    DEFAULT_CSS = """
    StartupPanel { width: 100%; height: 100%; padding: 2 4; background: $background; }
    StartupPanel #startup-title { text-style: bold; color: $accent; height: 3; }
    StartupPanel #startup-status { height: 1fr; overflow-y: auto; }
    StartupPanel OptionList { height: 5; border: round $accent; }
    StartupPanel #startup-hint { height: 2; color: $text-muted; }
    """
    BINDINGS = [
        Binding("j", "down", show=False),
        Binding("k", "up", show=False),
    ]

    class Action(Message):
        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    def __init__(self, *, audio_setup: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.audio_setup = audio_setup

    def compose(self) -> ComposeResult:
        yield Static("AudioViz / startup", id="startup-title")
        yield Static("Preparing the window. No devices are open in this terminal.", id="startup-status", markup=False)
        yield OptionList(
            *([Option("Choose audio devices", id="devices")] if self.audio_setup else []),
            Option("Retry startup", id="retry"),
            Option("Quit", id="quit"),
        )
        hint = "j/k move · Enter select · " + ("a devices · " if self.audio_setup else "") + "r retry · q quit"
        yield Static(hint, id="startup-hint")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def status(self, message: str) -> None:
        self.query_one("#startup-status", Static).update(message)

    def action_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.post_message(self.Action(event.option.id))


class WindowSession:
    """Only the process started here is terminated when the terminal exits."""

    def __init__(self, command: list[str], config_file: Path | None, *,
                 socket_path: Path = SOCKET, log_path: Path = Path("outputs/main.log"),
                 timeout: float = 60.0) -> None:
        self.command, self.config_file = command, config_file
        self.socket_path, self.log_path, self.timeout = socket_path, log_path, timeout
        self.process: subprocess.Popen | None = None
        self.deck: DeckClient | None = None
        self._log = None

    def load_config(self) -> dict:
        return load_device_config(self.config_file)

    @staticmethod
    def validate_config(config: dict) -> None:
        validate_device_config(config)

    async def choose_devices(self, app) -> dict | None:
        devices = await asyncio.to_thread(sd.query_devices)
        selected = []
        for kind in ("input", "output"):
            choices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d[f"max_{kind}_channels"] > 0]
            values = await app.push_screen_wait(DeviceChecklist(
                f"Startup / audio {kind}", choices, [], capacity=1, live=False,
            ))
            if values is None:
                return None
            selected.append(int(values[0].split(":", 1)[0]) if values else -1)
        return make_device_config(devices, *selected)

    def save_config(self, config: dict) -> None:
        self.validate_config(config)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(config, indent=4) + "\n")

    async def connect(self) -> DeckClient:
        if self.socket_path.exists():
            with socket.socket(socket.AF_UNIX) as probe:
                probe.settimeout(1)
                try:
                    probe.connect(str(self.socket_path))
                except (ConnectionRefusedError, FileNotFoundError):
                    self.socket_path.unlink(missing_ok=True)
                else:
                    raise RuntimeError(
                        "A window already owns the deck socket.\n"
                        "Run python -m audioviz.tui.app --attach to control it."
                    )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        try:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.DEVNULL, stdout=self._log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "AUDIOVIZ_DECK_SOCKET": str(self.socket_path), "AUDIOVIZ_TUI": "1"},
            )
            async with asyncio.timeout(self.timeout):
                while not self.socket_path.exists():
                    if self.process.poll() is not None:
                        raise RuntimeError(f"Window exited during startup (code {self.process.returncode}).")
                    await asyncio.sleep(0.05)
                self.deck = await asyncio.to_thread(DeckClient, self.socket_path)
            return self.deck
        except asyncio.CancelledError:
            self.stop()
            raise
        except (OSError, RuntimeError, ValueError, TimeoutError) as error:
            self.stop()
            details = self.log_path.read_text(errors="replace")[-6000:]
            reason = f"No deck connection within {self.timeout:g} seconds." if isinstance(error, TimeoutError) else str(error)
            raise RuntimeError(f"Window startup failed: {reason}\n{details}") from error

    def stop(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
            self.socket_path.unlink(missing_ok=True)
        if self.deck is not None:
            self.deck.close()
            self.deck = None
        if self._log is not None:
            self._log.close()
            self._log = None
