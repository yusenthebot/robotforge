"""Headless smoke test: the viewer server and an MCP stdio session, run the way a user's machine runs them.

    uv run python scripts/smoke.py

It starts `robotforge demo --no-browser` as a separate process and checks its HTTP routes and WebSocket
stream, then drives `robotforge mcp` over stdio with the official MCP client. No display or GPU needed:
when offscreen rendering is missing the check says so instead of failing, unless ROBOTFORGE_REQUIRE_GL is set.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from websockets.sync.client import connect

from robotforge.cli import DEFAULT_PORT, find_free_port

HOST = "127.0.0.1"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
STARTUP_SECONDS = 90  # a cold Windows runner needs a while to import mujoco and numpy
STREAM_SECONDS = 1.5
OVERALL_SECONDS = 240

SCRIPT = """\
from robotforge.script import connect

sim = connect()
start = sim.position("rover")
sim.move("rover", speed=0.3)
sim.step(2.0)
sim.stop("rover")
end = sim.position("rover")
print(f"rover drove {end[0] - start[0]:.2f} m \\u2713")
"""


class SmokeFailure(AssertionError):
    pass


def check(condition: object, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"ok    {message}", flush=True)


def note(message: str) -> None:
    print(f"info  {message}", flush=True)


def http_get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@contextmanager
def background(args: list[str], log: Any) -> Iterator[subprocess.Popen]:
    """Run a robotforge command in the background; always stop it, on Windows too."""
    proc = subprocess.Popen([sys.executable, "-m", "robotforge", *args], stdout=log, stderr=subprocess.STDOUT)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def wait_for_viewer(base: str, proc: subprocess.Popen) -> dict[str, Any]:
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SmokeFailure(f"robotforge demo exited early with code {proc.returncode}")
        try:
            status, body = http_get(f"{base}/scene.json", timeout=2)
        except OSError:
            time.sleep(0.25)
            continue
        if status == 200:
            return json.loads(body)
    raise SmokeFailure(f"viewer did not answer on {base} within {STARTUP_SECONDS} s")


def check_viewer(log: Any) -> None:
    port = find_free_port(DEFAULT_PORT)
    base = f"http://{HOST}:{port}"
    with background(["demo", "--no-browser", "--port", str(port)], log) as proc:
        scene = wait_for_viewer(base, proc)
        check(scene.get("nbody", 0) > 1 and scene.get("geoms"), f"GET /scene.json describes {scene['nbody']} bodies")
        status, body = http_get(f"{base}/")
        check(status == 200 and b"robotforge" in body, "GET / serves the viewer page")
        status, body = http_get(f"{base}/static/app.js")
        check(status == 200 and len(body) > 1000, "GET /static/app.js serves the client")

        with connect(f"ws://{HOST}:{port}/ws", open_timeout=10) as ws:
            first = json.loads(ws.recv(timeout=10))
            check(first["type"] == "scene" and first["nbody"] == scene["nbody"], "WebSocket opens with the scene")
            frames, times, started = 0, [], time.monotonic()
            while time.monotonic() - started < STREAM_SECONDS:
                message = ws.recv(timeout=5)
                if isinstance(message, bytes):
                    if len(message) != 4 * (1 + 7 * scene["nbody"]):
                        raise SmokeFailure(f"pose frame has {len(message)} bytes for {scene['nbody']} bodies")
                    frames += 1
                    times.append(struct.unpack_from("<f", message)[0])
            fps = frames / (time.monotonic() - started)
            check(frames >= 5, f"WebSocket streams pose frames ({fps:.0f} fps)")
            check(times[-1] > times[0], f"simulation clock runs while watched ({times[0]:.2f} s -> {times[-1]:.2f} s)")
    check(proc.returncode is not None, "viewer process stops on request")


def text_of(result: Any) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", "") == "text")


async def call(client: Client, tool: str, **args: Any) -> Any:
    result = await client.call_tool(tool, args)
    if result.is_error:
        raise SmokeFailure(f"{tool} failed: {text_of(result)}")
    return result


async def call_json(client: Client, tool: str, **args: Any) -> dict[str, Any]:
    return json.loads(text_of(await call(client, tool, **args)))


async def check_mcp(home: Path) -> None:
    port = find_free_port(DEFAULT_PORT)
    # The client starts the server with a minimal environment; pass ours so GL settings (MUJOCO_GL...) arrive.
    env = {**os.environ, "ROBOTFORGE_NO_BROWSER": "1", "ROBOTFORGE_HOME": str(home)}
    # Launch it the way an MCP client config does: through the installed `robotforge` command.
    command = shutil.which("robotforge", path=str(Path(sys.executable).parent))
    check(command, "the robotforge command is installed next to this Python")
    params = StdioServerParameters(command=command, args=["mcp", "--port", str(port)], env=env)
    async with Client(params, read_timeout_seconds=STARTUP_SECONDS) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        wanted = {
            "describe_model",
            "simulate",
            "observe",
            "new_world",
            "add_robot",
            "move",
            "run_script",
            "open_viewer",
        }
        check(wanted <= tools, f"MCP lists {len(tools)} tools")

        described = await call_json(client, "describe_model")
        check([a["name"] for a in described["actuators"]] == ["yaw", "shoulder", "elbow"], "describe_model: demo arm")

        simulated = await call_json(client, "simulate", seconds=1.0, controls={"shoulder": 0.8})
        check(abs(simulated["time"] - 1.0) < 0.01, f"simulate: 1 s of physics, t = {simulated['time']}")

        observed = await call(client, "observe", view="front", width=160, height=120)
        state = json.loads(text_of(observed))
        images = [c for c in observed.content if getattr(c, "type", "") == "image"]
        check(abs(state["joints"]["shoulder"] - 0.8) < 0.1, "observe: state follows the controls")
        if images:
            check(images[0].mime_type == "image/png", "observe: renders a PNG")
        elif os.environ.get("ROBOTFORGE_REQUIRE_GL"):
            raise SmokeFailure(f"ROBOTFORGE_REQUIRE_GL is set but observe returned no image: {state.get('note')}")
        else:
            check("No image" in state.get("note", ""), "observe: no OpenGL here, state returned with a note")
            note(state["note"])

        viewer = await call_json(client, "open_viewer", launch_browser=False)
        status, _ = await asyncio.to_thread(http_get, f"{viewer['viewer_url']}scene.json")
        check(status == 200, "open_viewer: the viewer answers inside the MCP server process")

        await call_json(client, "new_world", name="smoke")
        await call_json(client, "add_robot", name="rover", template="rover")
        moved = await call_json(client, "move", robot="rover", speed=0.3, seconds=2.0)
        check(moved["distance_m"] > 0.2, f"design + move: the rover drove {moved['distance_m']} m")

        await call(client, "write_script", name="drive", code=SCRIPT)
        ran = await call_json(client, "run_script", name="drive")
        check(ran["ok"] and "rover drove" in ran["output"], f"run_script: {ran['output'].strip()!r}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to a legacy code page
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="robotforge-smoke-") as tmp, tempfile.TemporaryFile() as log:
        try:
            check_viewer(log)
            asyncio.run(asyncio.wait_for(check_mcp(Path(tmp)), timeout=OVERALL_SECONDS))
        except (SmokeFailure, OSError, TimeoutError) as exc:
            log.seek(0)
            print(f"\nFAIL  {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"--- viewer process output ---\n{log.read().decode('utf-8', 'replace')}", file=sys.stderr)
            return 1
    print(f"\nsmoke test passed in {time.monotonic() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
