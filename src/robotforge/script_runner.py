"""Run a robot script (see robotforge.script) in a child process, in lockstep with the live simulation.

The script talks to us over a localhost socket guarded by a one-time token. While
it runs it owns the simulation clock: time advances only when it asks, paced in
real time if someone is watching the viewer so they see what the script does.

This is not a security sandbox. Scripts are ordinary Python on the user's machine
with a time limit, run in the world folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.manager import BehaviorError
from robotforge.playground import Playground, PlaygroundError
from robotforge.script import ENV_PORT, ENV_TOKEN

OUTPUT_LIMIT = 6000  # characters of script output returned to the agent
FRAME = 1 / 60


class ScriptRunner:
    def __init__(self, playground: Playground, watched: Callable[[], bool]):
        self.pg = playground
        self.watched = watched
        self.sim_budget = 0.0
        self._sim_start = 0.0

    async def run(self, path: Path, max_sim_seconds: float = 60.0, max_wall_seconds: float = 180.0) -> dict[str, Any]:
        token = secrets.token_hex(16)
        session = self.pg.session
        self.sim_budget = max_sim_seconds
        self._sim_start = session.data.time
        connected = asyncio.Event()

        async def on_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                hello = json.loads(await reader.readline() or b"{}")
                if hello.get("op") != "hello" or hello.get("token") != token:
                    return
                connected.set()
                writer.write(b'{"result": "ok"}\n')
                await writer.drain()
                while line := await reader.readline():
                    reply = await self._dispatch(json.loads(line))
                    writer.write((json.dumps(reply) + "\n").encode())
                    await writer.drain()
            except (ConnectionError, json.JSONDecodeError):
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(on_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        # PYTHONUTF8: on Windows a piped child would print in the ANSI code page and crash on non-Latin text.
        env = {**os.environ, ENV_PORT: str(port), ENV_TOKEN: token, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
        session.clock_owner = "script"
        started = time.monotonic()
        output = bytearray()
        timed_out = False
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(path),
                cwd=str(path.parent),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            async def pump() -> None:
                assert proc.stdout is not None
                while chunk := await proc.stdout.read(4096):
                    output.extend(chunk)
                    if len(output) > 4 * OUTPUT_LIMIT:
                        del output[: len(output) - 2 * OUTPUT_LIMIT]

            pumping = asyncio.create_task(pump())
            try:
                await asyncio.wait_for(proc.wait(), timeout=max_wall_seconds)
            except TimeoutError:
                timed_out = True
                proc.kill()
                await proc.wait()
            await pumping
        finally:
            session.clock_owner = None
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()

        text = output.decode("utf-8", errors="replace")
        if len(text) > OUTPUT_LIMIT:
            text = "...\n" + text[-OUTPUT_LIMIT:]
        result: dict[str, Any] = {
            "script": path.name,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0 and not timed_out,
            "output": text,
            "sim_seconds": round(float(session.data.time - self._sim_start), 2),
            "wall_seconds": round(time.monotonic() - started, 2),
            "shown_live": self.watched(),
        }
        if timed_out:
            result["error"] = f"stopped after {max_wall_seconds:.0f} s of wall time"
        elif not connected.is_set() and proc.returncode == 0:
            result["note"] = "the script never called robotforge.script.connect(), so it did not touch the world"
        return result

    # ---- requests from the script ----------------------------------------------------------

    async def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        op = req.get("op")
        try:
            handler = getattr(self, f"_op_{op}", None)
            if handler is None:
                raise PlaygroundError(f"unknown request {op!r}")
            result = handler(req)
            if asyncio.iscoroutine(result):
                result = await result
            return {"result": result}
        except (PlaygroundError, BehaviorError, ValueError, KeyError, IndexError, TypeError) as exc:
            return {"error": str(exc)}

    def _op_time(self, _req) -> float:
        return round(float(self.pg.session.data.time), 4)

    async def _op_step(self, req) -> float:
        seconds = float(req.get("seconds", 0.02))
        if not 0 < seconds <= 60:
            raise PlaygroundError("step(seconds) needs 0 < seconds <= 60")
        await self._advance(lambda: False, seconds)
        return self._op_time(req)

    def _op_position(self, req) -> list[float]:
        name = req["name"]
        m, d = self.pg.session.model, self.pg.session.data
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b >= 0:
            return [round(float(v), 4) for v in d.xpos[b]]
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g >= 0:
            return [round(float(v), 4) for v in d.geom_xpos[g]]
        return self.pg.behaviors.pose(name)["position"]

    def _op_heading(self, req) -> float:
        return self.pg.behaviors.pose(req["name"])["heading_deg"]

    def _op_joints(self, req) -> dict[str, float]:
        prefix = f"{req['robot']}/"
        joints = self.pg.state()["joints"]
        out = {k[len(prefix) :]: v for k, v in joints.items() if k.startswith(prefix)}
        if not out:
            raise PlaygroundError(f"no joints found for robot {req['robot']!r}")
        return out

    def _op_sensors(self, req) -> dict[str, Any]:
        prefix = f"{req['robot']}/"
        sensors = self.pg.state()["sensors"]
        return {k[len(prefix) :]: v for k, v in sensors.items() if k.startswith(prefix)}

    def _op_set_controls(self, req) -> None:
        self.pg.set_controls({str(k): float(v) for k, v in req["controls"].items()})

    def _op_move(self, req) -> dict:
        return self.pg.behaviors.move(req["robot"], float(req.get("speed", 0.2)), float(req.get("turn", 0.0)))

    def _op_stop(self, req) -> dict:
        return self.pg.behaviors.stop(req["robot"])

    async def _op_pose(self, req) -> bool:
        ctrl = self.pg.behaviors.strike_pose(req["robot"], str(req["name"]))
        await self._advance(lambda: ctrl.done, 20.0)
        return bool(self.pg.behaviors.pose(req["robot"])["upright"])

    async def _op_move_arm(self, req) -> list[float]:
        arm = self.pg.behaviors.move_arm(req["robot"], req["position"], req.get("gripper"))
        await self._advance(lambda: arm.done, 20.0)
        return [round(float(v), 4) for v in self.pg.session.data.site_xpos[arm.site]]

    async def _op_gripper(self, req) -> None:
        arm = self.pg.behaviors.set_gripper(req["robot"], req["state"])
        await self._advance(lambda: arm.done, 5.0)

    async def _op_pick_and_place(self, req) -> bool:
        target = req["target"]
        arm, info = self.pg.behaviors.pick_and_place(req["robot"], req["object"], target)
        await self._advance(lambda: arm.done, 40.0)
        await self._advance(lambda: False, 0.5)
        return bool(self.pg.behaviors.placed(req["object"], info["region"])["success"])

    async def _advance(self, done: Callable[[], bool], max_seconds: float) -> None:
        """Advance physics, paced to real time when watched, within the script's time budget."""
        session = self.pg.session
        timestep = session.model.opt.timestep
        used = session.data.time - self._sim_start
        # Less than one physics step left counts as used up; otherwise step() would advance
        # nothing and a `while True: step()` script would spin until the wall-clock limit.
        if self.sim_budget - used < timestep:
            raise PlaygroundError(f"time limit of {self.sim_budget:.0f} simulated seconds reached")
        max_seconds = min(max_seconds, self.sim_budget - used)
        start = session.data.time
        while not done() and session.data.time - start < max_seconds - timestep / 2:
            if self.watched():
                tick = time.perf_counter()
                remaining = max_seconds - (session.data.time - start)
                session.step(max(1, round(min(FRAME * session.speed, remaining) / timestep)))
                await asyncio.sleep(max(0.0, FRAME - (time.perf_counter() - tick)))
            else:
                remaining = max_seconds - (session.data.time - start)
                session.step(max(1, round(min(0.02, remaining) / timestep)))
        if not np.isfinite(session.data.qpos).all():
            raise PlaygroundError("the simulation became unstable; reset the world")
