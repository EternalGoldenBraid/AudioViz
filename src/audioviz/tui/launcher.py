"""Run the Qt window as a child process and the terminal controller in this one."""

import signal
import subprocess
import sys
import time
from pathlib import Path

from audioviz.tui.deck_server import SOCKET

LOG = Path("outputs/main.log")
STARTUP_SECONDS = 60


def _exit_cleanly(signum, frame) -> None:
    raise SystemExit(128 + signum)


def run_with_tui(window_command: list[str]) -> int:
    from audioviz.tui.app import AudioVizApp

    for signum in (signal.SIGHUP, signal.SIGTERM):  # a closed terminal must still stop the window
        signal.signal(signum, _exit_cleanly)
    SOCKET.unlink(missing_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w") as log:
        window = subprocess.Popen(
            window_command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + STARTUP_SECONDS
            while not SOCKET.exists():
                if window.poll() is not None or time.monotonic() > deadline:
                    print(f"the window did not start; see {LOG}", file=sys.stderr)
                    return 1
                time.sleep(0.1)
            AudioVizApp(attach=True).run()
        finally:
            window.terminate()
            try:
                window.wait(timeout=5)
            except subprocess.TimeoutExpired:
                window.kill()
    return 0
