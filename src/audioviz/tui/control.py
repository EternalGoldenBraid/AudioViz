"""Control socket: JSON lines over a Unix socket, so scripts and agents can drive the live app.

Requests: {"op": "keys", "keys": ["w", "j"]} | {"op": "run", "line": "theme amber"}
          | {"op": "state"} | {"op": "screenshot", "name": "x"}
The same channel is where a gaze daemon can later publish focus and lift events.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

SOCKET = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "audioviz.sock"


async def serve(app) -> asyncio.AbstractServer:
    SOCKET.unlink(missing_ok=True)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        async for line in reader:
            try:
                reply = await dispatch(app, json.loads(line))
            except Exception as error:  # a bad request must not drop the connection
                reply = {"error": repr(error)}
            writer.write(json.dumps(reply).encode() + b"\n")
            await writer.drain()
        writer.close()

    return await asyncio.start_unix_server(handle, path=str(SOCKET))


async def dispatch(app, request: dict) -> dict:
    op = request["op"]
    if op == "keys":
        await app._press_keys(request["keys"])
        await asyncio.sleep(0.15)
    elif op == "run":
        name, *args = request["line"].lstrip(":").split()
        app.run_command(name, args)
        await asyncio.sleep(0.15)
    elif op == "screenshot":
        path = app.screenshot(request.get("name", "ctl"))
        return {"path": str(path)}
    elif op != "state":
        raise ValueError(f"unknown op {op!r}")
    return app.describe()


def main() -> None:
    """CLI: ctl.py keys w j | run 'theme amber' | state | screenshot [name]"""
    op, *rest = sys.argv[1:] or ["state"]
    request = {"op": op}
    if op == "keys":
        request["keys"] = rest
    elif op == "run":
        request["line"] = " ".join(rest)
    elif op == "screenshot" and rest:
        request["name"] = rest[0]

    async def call() -> dict:
        reader, writer = await asyncio.open_unix_connection(str(SOCKET))
        writer.write(json.dumps(request).encode() + b"\n")
        await writer.drain()
        reply = json.loads(await reader.readline())
        writer.close()
        return reply

    print(json.dumps(asyncio.run(call()), indent=1))


if __name__ == "__main__":
    main()
