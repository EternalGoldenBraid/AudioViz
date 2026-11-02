"""Live test agent: a small Ollama model drives the TUI through the checks runner's tools.

    python -m audioviz.tui.agent "turn camera view off and confirm canvas feed stays on" [--model qwen3.5:4b]

The model gets tools over the control socket and ends with `report`. The whole conversation is
traced to outputs/traces/agent-*.jsonl as fine-tuning data. Exit code is non-zero on a failed report.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request

from audioviz.tui import checks, control

OLLAMA = "http://localhost:11434/api/chat"
MAX_TURNS = 20
SYSTEM = (
    "You check behaviour of a running terminal app by calling tools. Do exactly what the task says, "
    "read `state` to observe values, and never guess. Knobs are walked with focus(tile id) and turned "
    "with keys minus/plus. Finish by calling report with passed (bool) and evidence (observed values)."
)
TOOLS = [
    {"type": "function", "function": {
        "name": "keys", "description": "Press Textual key names, e.g. minus, plus, h, escape, n.",
        "parameters": {"type": "object", "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
                       "required": ["keys"]}}},
    {"type": "function", "function": {
        "name": "focus", "description": "Move focus to a tile id such as knob-camera-view.",
        "parameters": {"type": "object", "properties": {"tile": {"type": "string"}}, "required": ["tile"]}}},
    {"type": "function", "function": {
        "name": "state", "description": "Read values. Dotted paths such as 'mixer.camera view' or 'focus'.",
        "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                       "required": ["paths"]}}},
    {"type": "function", "function": {
        "name": "report", "description": "Finish with the verdict.",
        "parameters": {"type": "object", "properties": {"passed": {"type": "boolean"}, "evidence": {"type": "string"}},
                       "required": ["passed", "evidence"]}}},
]


def chat(model: str, messages: list[dict]) -> dict:
    body = json.dumps({"model": model, "messages": messages, "tools": TOOLS, "stream": False,
                       "think": False, "options": {"temperature": 0}}).encode()
    request = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as reply:
        return json.load(reply)["message"]


async def act(name: str, args: dict, state: dict) -> tuple[str, dict]:
    if name == "keys":
        state = await checks.call({"op": "keys", "keys": args["keys"]})
        return "ok", state
    if name == "focus":
        ok, state, evidence = await checks.step({"focus": args["tile"]}, state)
        return "ok" if ok else f"failed: {evidence}", state
    if name == "state":
        state = await checks.call({"op": "state"})
        return json.dumps({p: checks.lookup(state, p) for p in args["paths"]}), state
    return f"unknown tool {name}", state


async def loop(model: str, task: str, trace) -> bool:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    state = await checks.call({"op": "state"})
    for _ in range(MAX_TURNS):
        message = await asyncio.to_thread(chat, model, messages)
        messages.append(message)
        trace.write(json.dumps(message) + "\n")
        for call in message.get("tool_calls", []):
            name, args = call["function"]["name"], call["function"]["arguments"]
            if name == "report":
                print(f"{'PASS' if args['passed'] else 'FAIL'}: {args['evidence']}")
                return bool(args["passed"])
            try:
                result, state = await act(name, args, state)
            except Exception as error:  # feed the mistake back so the model can correct itself
                result = f"error: {error!r}"
            messages.append({"role": "tool", "tool_name": name, "content": result})
            trace.write(json.dumps(messages[-1]) + "\n")
        if not message.get("tool_calls"):
            messages.append({"role": "user", "content": "Call a tool; finish with report."})
    print("FAIL: no report within the turn limit")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    pane = None if args.attach else checks.launch()
    checks.TRACES.mkdir(parents=True, exist_ok=True)
    try:
        with open(checks.TRACES / f"agent-{int(time.time())}.jsonl", "w") as trace:
            ok = asyncio.run(loop(args.model, args.task, trace))
    finally:
        if pane and not args.keep:
            subprocess.run(["tmux", "kill-pane", "-t", pane])
            control.SOCKET.unlink(missing_ok=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
