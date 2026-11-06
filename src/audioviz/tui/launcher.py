"""Open the terminal first; it prepares devices and owns the child Qt window."""

import signal
from pathlib import Path

from audioviz.tui.startup import WindowSession


def _exit_cleanly(signum, frame) -> None:
    raise SystemExit(128 + signum)


def run_with_tui(window_command: list[str], *, config_file: Path | None = None) -> int:
    from audioviz.tui.app import AudioVizApp

    for signum in (signal.SIGHUP, signal.SIGTERM):  # a closed terminal must still stop the window
        signal.signal(signum, _exit_cleanly)
    session = WindowSession(window_command, config_file)
    try:
        AudioVizApp(startup=session).run()
    finally:
        session.stop()
    return 0
