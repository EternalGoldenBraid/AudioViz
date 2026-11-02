"""Scripted live checks: drive the running TUI over its control socket and assert on `state`.

    python -m audioviz.tui.checks [checks/tui.json] [--keep] [--attach]

Steps: {"keys": [...]} | {"run": "cmd"} | {"focus": "tile-id"} | {"expect": {"dotted.path": value}}.
Each step is appended to a JSONL trace (state before, action, state after, verdict), which is
the dataset for fine-tuning the small agent later. Exit code is non-zero on any failure.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from audioviz.tui import control

LAUNCH = "uv run --group tui --with-editable . python -m audioviz.tui.app; read"
TRACES = Path("outputs/traces")
FOCUS_TRIES = 40


async def call(request: dict) -> dict:
    reader, writer = await asyncio.open_unix_connection(str(control.SOCKET))
    writer.write(json.dumps(request).encode() + b"\n")
    await writer.drain()
    reply = json.loads(await reader.readline())
    writer.close()
    return reply


def lookup(state: dict, path: str):
    for part in path.split("."):
        state = state[part]
    return state


def diff(before: dict, after: dict) -> dict:
    return {k: [before.get(k), after[k]] for k in after if before.get(k) != after[k]}


async def step(action: dict, state: dict) -> tuple[bool, dict, str]:
    """Run one step; returns (ok, new state, evidence)."""
    if "keys" in action:
        new = await call({"op": "keys", "keys": action["keys"]})
        return True, new, ""
    if "run" in action:
        new = await call({"op": "run", "line": action["run"]})
        return True, new, ""
    if "focus" in action:
        for _ in range(FOCUS_TRIES):
            if state["focus"] == action["focus"]:
                return True, state, ""
            state = await call({"op": "keys", "keys": ["w"]})
        return state["focus"] == action["focus"], state, f"focus is {state['focus']!r}"
    if "expect" in action:
        state = await call({"op": "state"})
        bad = {p: [want, lookup(state, p)] for p, want in action["expect"].items() if lookup(state, p) != want}
        return not bad, state, json.dumps({"expected_vs_observed": bad}) if bad else ""
    raise ValueError(f"unknown step {action!r}")


async def run_check(check: dict, trace) -> bool:
    state = await call({"op": "state"})
    ok = True
    for action in check["steps"]:
        before = state
        passed, state, evidence = await step(action, state)
        trace.write(json.dumps({"check": check["name"], "action": action, "ok": passed,
                                "evidence": evidence, "diff": diff(before, state)}) + "\n")
        if not passed:
            ok = False
            print(f"FAIL {check['name']}: {action} {evidence}")
            break
    if ok:
        print(f"PASS {check['name']}")
    return ok


def launch() -> str:
    """Start a private app instance on its own socket so a running app is never disturbed."""
    control.SOCKET = Path(f"/tmp/audioviz-checks-{os.getpid()}.sock")
    pane = subprocess.check_output(
        ["tmux", "split-window", "-d", "-h", "-c", os.getcwd(), "-e", f"AUDIOVIZ_SOCKET={control.SOCKET}",
         "-P", "-F", "#{pane_id}", LAUNCH],
        text=True).strip()
    deadline = time.time() + 90
    while time.time() < deadline:
        if control.SOCKET.exists():
            try:
                asyncio.run(call({"op": "state"}))
                time.sleep(12)  # first frames
                return pane
            except OSError:
                pass
        time.sleep(1)
    subprocess.run(["tmux", "kill-pane", "-t", pane])
    sys.exit("app did not come up")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checks", nargs="?", default="checks/tui.json")
    parser.add_argument("--attach", action="store_true", help="use an already running app")
    parser.add_argument("--keep", action="store_true", help="leave the launched app running")
    args = parser.parse_args()
    checks = json.loads(Path(args.checks).read_text())
    pane = None if args.attach else launch()
    TRACES.mkdir(parents=True, exist_ok=True)
    try:
        with open(TRACES / f"{int(time.time())}.jsonl", "w") as trace:
            results = [asyncio.run(run_check(c, trace)) for c in checks]
    finally:
        if pane and not args.keep:
            subprocess.run(["tmux", "kill-pane", "-t", pane])
            control.SOCKET.unlink(missing_ok=True)
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
