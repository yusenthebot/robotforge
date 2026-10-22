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
import re
import secrets
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco

import robotforge
from robotforge.behaviors.manager import BehaviorError
from robotforge.demonstrations import RATE_HZ
from robotforge.playground import Playground, PlaygroundError
from robotforge.script import ENV_PORT, ENV_TOKEN
from robotforge.sim.execution import EndReason, ExecutionError, ExecutionRun, RunLimits

OUTPUT_LIMIT = 6000  # characters of script output returned to the agent
CLEANUP_SECONDS = 1.0
# Windows has no killpg. Gate user code until its process belongs to the kill-on-close job.
WINDOWS_BOOTSTRAP = """import os, runpy, sys
ready = sys.stdin.buffer.read(1)
sys.stdin.close()
sys.stdin = open(os.devnull, encoding='utf-8')
sys.argv = sys.argv[1:]
if ready == b'1':
    runpy.run_path(sys.argv[0], run_name='__main__')
"""
# One frame of a Python traceback: its "File ..." line and the source lines under it.
TRACEBACK_FRAME = re.compile(r'  File "(?P<file>[^"\n]+)", line \d+, in [^\n]*\n(?:    [^\n]*\n)*')


def _windows_job(pid: int) -> Callable[[], None]:
    """Own the script's process tree, including descendants that outlive their parent.

    https://learn.microsoft.com/windows/win32/procthread/job-objects
    The child is waiting on its stdin gate until assignment succeeds.
    """
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_int64),
            ("job_time", ctypes.c_int64),
            ("flags", wintypes.DWORD),
            ("min_working_set", ctypes.c_size_t),
            ("max_working_set", ctypes.c_size_t),
            ("active_processes", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits),
            ("io_counters", ctypes.c_uint64 * 6),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateJobObjectW
    create.argtypes, create.restype = [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE
    configure = kernel.SetInformationJobObject
    configure.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    configure.restype = wintypes.BOOL
    open_process = kernel.OpenProcess
    open_process.argtypes, open_process.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
    assign = kernel.AssignProcessToJobObject
    assign.argtypes, assign.restype = [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
    job = create(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not configure(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        process = open_process(0x0100 | 0x0001, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not assign(job, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            close(process)
    except BaseException:
        close(job)
        raise

    def close_job() -> None:
        if not close(job):
            raise ctypes.WinError(ctypes.get_last_error())

    return close_job


async def _wait_exit(proc: asyncio.subprocess.Process) -> None:
    # Process.wait() can wait for EOF on stdout after the process has already been reaped.
    # A descendant inheriting that pipe must not extend either the wall budget or cleanup.
    while proc.returncode is None:
        await asyncio.sleep(0.01)


def tidy_output(text: str, script_dir: Path) -> str:
    """A script's output as an agent should read it: the traceback keeps the script's own lines and the error,
    without frames inside robotforge's script client (a wrong robot name showed three of them) or the long
    path to the world folder."""
    package = Path(robotforge.__file__).parent

    def drop_internal(frame: re.Match[str]) -> str:
        return "" if Path(frame["file"]).is_relative_to(package) else frame[0]

    return TRACEBACK_FRAME.sub(drop_internal, text).replace(f"{script_dir}{os.sep}", "")


class ScriptRunner:
    def __init__(self, playground: Playground, watched: Callable[[], bool]):
        self.pg = playground
        self.watched = watched
        self._execution: ExecutionRun | None = None
        self._command_robots: set[str] = set()
        self._controlled: dict[str, Any] = {}
        self._replays: list[Any] = []

    async def run(
        self,
        path: Path,
        max_sim_seconds: float = 60.0,
        max_wall_seconds: float = 180.0,
        args: list[str] | None = None,
    ) -> dict[str, Any]:
        limits = RunLimits(sim_seconds=max_sim_seconds, wall_seconds=max_wall_seconds)
        try:
            with self.pg.session.execution.run("script", limits, self.watched) as execution:
                self._execution = execution
                self._command_robots = set()
                self._controlled = {}
                self._replays = []
                try:
                    return await self._run_script(path, args, execution)
                finally:
                    self._execution = None
        except ExecutionError as exc:
            raise PlaygroundError(str(exc)) from exc

    async def _run_script(self, path: Path, args: list[str] | None, execution: ExecutionRun) -> dict[str, Any]:
        token = secrets.token_hex(16)
        connected = asyncio.Event()
        clients: set[asyncio.Task] = set()
        writers: set[asyncio.StreamWriter] = set()
        server = None
        proc = None
        spawning = None
        pumping = None
        close_job = None
        started = execution.started
        output = bytearray()
        timed_out = False
        interrupted: str | None = None

        async def on_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            clients.add(task)
            writers.add(writer)
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
                writers.discard(writer)
                clients.discard(task)

        async def pump() -> None:
            assert proc is not None and proc.stdout is not None
            while chunk := await proc.stdout.read(4096):
                output.extend(chunk)
                if len(output) > 4 * OUTPUT_LIMIT:
                    del output[: len(output) - 2 * OUTPUT_LIMIT]

        try:
            self.pg.flow.clear()  # the last script's tree is not this one's
            server = await asyncio.start_server(on_client, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            # PYTHONUTF8: a piped Windows child otherwise prints in the ANSI code page.
            env = {**os.environ, ENV_PORT: str(port), ENV_TOKEN: token, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
            spawning = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    *(("-c", WINDOWS_BOOTSTRAP) if os.name == "nt" else ()),
                    str(path),
                    *(args or []),
                    cwd=str(path.parent),
                    env=env,
                    stdin=asyncio.subprocess.PIPE if os.name == "nt" else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    **({"start_new_session": True} if os.name != "nt" else {}),
                )
            )
            # If cancelled during spawn, cleanup still receives the process and reaps it.
            proc = await asyncio.shield(spawning)
            if os.name == "nt":
                close_job = _windows_job(proc.pid)
                proc.stdin.write(b"1")
                proc.stdin.close()

            pumping = asyncio.create_task(pump())
            try:
                await execution.wait(lambda: proc.returncode is not None)
            except ExecutionError as exc:
                execution.reason = exc.reason
                if exc.reason == EndReason.WALL_LIMIT:
                    timed_out = True
                else:
                    interrupted = str(exc)
        finally:

            async def cleanup() -> None:
                nonlocal proc, pumping
                try:
                    if server is not None:
                        server.close()
                    for writer in tuple(writers):
                        writer.close()
                    pending = tuple(clients)
                    for client in pending:
                        client.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    if server is not None:
                        await server.wait_closed()
                    if proc is None and spawning is not None:
                        with contextlib.suppress(Exception):
                            proc = await spawning
                    if proc is not None:
                        if pumping is None:
                            # Cancellation can arrive after spawn but before its output pump starts.
                            pumping = asyncio.create_task(pump())
                        if close_job is not None:
                            close_job()
                        elif os.name != "nt":
                            # Kill the group even if the script leader exited before its descendants.
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(proc.pid, signal.SIGKILL)
                        if proc.returncode is None:
                            with contextlib.suppress(ProcessLookupError):
                                proc.kill()
                        await asyncio.wait_for(_wait_exit(proc), CLEANUP_SECONDS)
                    if pumping is not None:
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(pumping, CLEANUP_SECONDS)
                finally:
                    if pumping is not None and not pumping.done():
                        pumping.cancel()
                        await asyncio.gather(pumping, return_exceptions=True)
                    if proc is not None:
                        # Detached descendants can retain stdout. StreamReader has no public close;
                        # close the subprocess pipe explicitly after the bounded drain, even on error.
                        pipe = proc._transport.get_pipe_transport(1)
                        if pipe is not None:
                            pipe.close()
                    # Threads are cooperative: retain the lease until a running planner finishes, but reap
                    # the child first. A queued planner checks validity before it can mutate the session.
                    await execution.drain_workers(cancel_pending=True)
                    if proc is None or proc.returncode != 0 or timed_out or interrupted is not None:
                        with self.pg.session.lock:
                            if execution.current:
                                for replay in self._replays:
                                    if self.pg.demonstrations.replays.get(replay.robot) is replay and not replay.done:
                                        replay.cancel()
                                for robot, controller in self._controlled.items():
                                    if self.pg.session.controllers.get(robot) is controller:
                                        self.pg.behaviors.stop(robot)

            cleaning = asyncio.create_task(cleanup())
            try:
                await asyncio.shield(cleaning)
            except asyncio.CancelledError:
                await cleaning
                raise

        text = tidy_output(output.decode("utf-8", errors="replace"), path.parent)
        if len(text) > OUTPUT_LIMIT:
            text = "...\n" + text[-OUTPUT_LIMIT:]
        result: dict[str, Any] = {
            "script": path.name,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0 and not timed_out and interrupted is None,
            "output": text,
            "sim_seconds": round(execution.sim_seconds, 2),
            "wall_seconds": round(time.monotonic() - started, 2),
            "shown_live": execution.shown_live,
        }
        if timed_out:
            result["error"] = f"stopped after {execution.limits.wall_seconds:.0f} s of wall time"
        elif interrupted is not None:
            result["error"] = interrupted
        elif not connected.is_set() and proc.returncode == 0:
            result["note"] = "the script never called robotforge.script.connect(), so it did not touch the world"
        if not result["ok"] and execution.reason == EndReason.COMPLETED:
            execution.reason = EndReason.FAILED
        return result

    # ---- requests from the script ----------------------------------------------------------

    async def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        op = req.get("op")
        try:
            if self._execution is not None:
                self._execution.check()
            handler = getattr(self, f"_op_{op}", None)
            if handler is None:
                raise PlaygroundError(f"unknown request {op!r}")
            result = handler(req)
            if asyncio.iscoroutine(result):
                result = await result
            if op in {"move", "run_policy", "start_controller"}:
                self._remember_command(self.pg.behaviors.resolve(req["robot"]))
            self._remember_controllers()
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
        with self.pg.session.lock:
            m, d = self.pg.session.model, self.pg.session.data
            out = {}
            for j in range(m.njnt):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if name.startswith(prefix) and int(m.jnt_type[j]) in (
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                ):
                    out[name[len(prefix) :]] = round(float(d.qpos[m.jnt_qposadr[j]]), 4)
        if not out:
            raise PlaygroundError(f"no joints found for robot {req['robot']!r}")
        return out

    def _op_sensors(self, req) -> dict[str, Any]:
        prefix = f"{req['robot']}/"
        sensors = self.pg.sensor_readings()  # all of them: observe's list stops at 40, and a lidar's rays went missing
        return {k[len(prefix) :]: v for k, v in sensors.items() if k.startswith(prefix)}

    def _op_sensor(self, req) -> dict[str, Any]:
        """One sensor's reading by name, the way read_sensors gives it (no pictures: a script reads numbers)."""
        reading, _ = self.pg.read_sensors(str(req["robot"]), [str(req["name"])], full=bool(req.get("full")))
        return next(iter(reading["sensors"].values()))

    def _op_sensor_names(self, req) -> list[str]:
        return self.pg.sensors.names(self.pg.behaviors.resolve(req["robot"]))

    def _op_robots(self, _req) -> list[str]:
        return self.pg.behaviors.robot_names()

    def _op_describe(self, req) -> dict[str, Any]:
        """What a robot is, for the script's Robot: its kind, the arms it carries, how fast it can go."""
        name = self.pg.behaviors.resolve(req["robot"])
        for info in self.pg.behaviors.robots():
            if info["name"] == name:
                return info
        raise PlaygroundError(f"no robot named {req['robot']!r}")

    def _op_start_controller(self, req) -> dict:
        """Hand a robot to a plugin's controller (robotforge.extend.controller)."""
        from robotforge.extend.routines import start_controller

        robot = self.pg.behaviors.resolve(req["robot"])

        def read(name: str) -> Any:
            reading, _ = self.pg.read_sensors(robot, [name])
            return next(iter(reading["sensors"].values()))

        return start_controller(self.pg.session, robot, str(req["name"]), req.get("params"), read)

    def _op_show_flow(self, req) -> None:
        """A script's behavior tree for the viewer's tree panel (robotforge.flow)."""
        self.pg.flow.update(
            str(req.get("name", "tree")), req.get("nodes"), str(req.get("status", "")), req.get("notes")
        )

    # ---- demonstrations (demonstrations.py) ----

    def _op_demo_start(self, req) -> dict:
        return self.pg.demonstrations.start(
            str(req["robot"]),
            req.get("name"),
            float(req.get("rate_hz", RATE_HZ)),
            req.get("sensors"),
            bool(req.get("overwrite")),
        )

    def _op_demo_stop(self, req) -> dict:
        return self.pg.demonstrations.stop(req.get("name"))

    async def _op_demo_replay(self, req) -> dict:
        replay = self.pg.demonstrations.replay(str(req["name"]), req.get("robot"), str(req.get("mode", "commands")))
        self._replays.append(replay)
        self._remember_command(replay.robot)
        if req.get("wait", True):
            try:
                await self._advance(lambda: replay.done, replay.demo.seconds + 3.0)
            finally:
                if not replay.done:
                    replay.cancel()  # out of time, or the script is over: it does not play on by itself
        return replay.status()

    def _op_demo_export(self, req) -> dict:
        fmt = req.get("format")
        return self.pg.demonstrations.export(
            str(req["name"]), str(fmt) if fmt else None, req.get("to"), bool(req.get("overwrite"))
        )

    def _op_demo_list(self, _req) -> dict:
        return self.pg.demonstrations.listing()

    def _op_set_controls(self, req) -> None:
        self.pg.set_controls({str(k): float(v) for k, v in req["controls"].items()})

    def _op_move(self, req) -> dict:
        return self.pg.behaviors.move(req["robot"], float(req.get("speed", 0.2)), float(req.get("turn", 0.0)))

    def _op_stop(self, req) -> dict:
        return self.pg.behaviors.stop(req["robot"])

    async def _op_pose(self, req) -> bool:
        ctrl = self.pg.behaviors.strike_pose(req["robot"], str(req["name"]))
        robot = self.pg.behaviors.resolve(req["robot"])
        self._remember_command(robot)
        await self._advance_controller(robot, ctrl, 20.0)
        return bool(self.pg.behaviors.pose(req["robot"])["upright"])

    async def _carrier_stands(self, robot: str) -> None:
        """An arm riding on a moving robot works from it standing still: stop the robot and wait until it stands."""
        stop = self.pg.behaviors.stop_carrier(robot)
        if stop is not None:
            self._remember_command(stop.robot)
            await self._advance(stop.still, stop.seconds)

    async def _op_move_arm(self, req) -> list[float]:
        await self._carrier_stands(req["robot"])
        arm = self.pg.behaviors.move_arm(req["robot"], req["position"], req.get("gripper"))
        self._remember_command(arm.robot)
        await self._advance(lambda: arm.done, 20.0)
        return [round(float(v), 4) for v in self.pg.session.data.site_xpos[arm.site]]

    async def _op_gripper(self, req) -> None:
        arm = self.pg.behaviors.set_gripper(req["robot"], req["state"])
        self._remember_command(arm.robot)
        await self._advance_controller(arm.robot, arm, 5.0)

    async def _advance_controller(self, robot: str, controller: Any, seconds: float) -> None:
        def done() -> bool:
            if self.pg.session.controllers.get(robot) is not controller:
                raise ExecutionError(EndReason.INVALIDATED, "the robot's controller changed during the script")
            return bool(controller.done)

        await self._advance(done, seconds)

    async def _op_pick_and_place(self, req) -> bool:
        target = req["target"]
        await self._carrier_stands(req["robot"])
        # Planned off the event loop: working out a trip and a grasp takes seconds, and the viewer's stream (and the
        # tree panel a script shows) stood still for that long
        assert self._execution is not None

        def plan():
            motion, info = self.pg.behaviors.pick_and_place(req["robot"], req["object"], target)
            self._remember_command(motion.robot)
            return motion, info

        motion, info = await self._execution.call_thread(plan)
        await self._advance(lambda: motion.done, getattr(motion, "time_limit", 40.0))  # an errand goes over first
        await self._advance(lambda: False, 0.5)
        return bool(self.pg.behaviors.placed(req["object"], info["region"])["success"])

    def _op_experiments(self, req) -> dict[str, Any]:
        from robotforge.experiments.live import history

        return history(self.pg.session, self.pg.store.dir if self.pg.store is not None else None, req.get("limit", 20))

    def _op_experiment(self, req) -> dict[str, Any]:
        from robotforge.experiments import ExperimentStore
        from robotforge.experiments.live import experiment_home

        home = experiment_home(self.pg.store.dir if self.pg.store is not None else None)
        return ExperimentStore(home).open(req["experiment_id"]).record.model_dump(mode="json")

    def _op_task_catalog(self, _req) -> dict[str, Any]:
        from robotforge.control.runtime import contract_catalog

        return contract_catalog(self.pg.current_world())

    async def _op_run_controller(self, req) -> dict[str, Any]:
        from robotforge.control.runtime import start_control

        if self._execution is None:
            raise PlaygroundError("the script has no active execution")

        def attach() -> dict[str, Any]:
            result = start_control(
                self.pg,
                req["robot"],
                req["task"],
                req["method"],
                req.get("task_params"),
                req.get("method_params"),
            )
            # A cancelled dispatch may never resume; cleanup still owns a controller attached by its worker.
            self._remember_command(result["robot"])
            self._remember_controllers()
            return result

        return await self._execution.call_thread(attach)

    def _op_run_policy(self, req) -> dict:
        from robotforge.control.rl.runtime import attach_policy

        world_dir = self.pg.store.dir if self.pg.store is not None else None
        return attach_policy(
            self.pg.session,
            req["robot"],
            req["name"],
            world_dir,
            req.get("task_params"),
            world=self.pg.current_world(),
        )

    async def _advance(self, done: Callable[[], bool], max_seconds: float) -> None:
        """Use the same clock driver as move, keeping the script's public limit errors."""
        execution = self._execution
        if execution is None:
            raise PlaygroundError("the script has no active execution")
        self._remember_controllers()
        before = execution.sim_seconds
        result = await execution.advance(done, max_seconds)
        if result.reason == EndReason.WALL_LIMIT:
            raise PlaygroundError("script wall time limit reached")
        # A last, clipped step request still succeeds as before; the next request reports exhaustion.
        if result.reason in (EndReason.SIM_LIMIT, EndReason.STEP_LIMIT) and execution.sim_seconds == before:
            raise PlaygroundError(f"time limit of {execution.limits.sim_seconds:.0f} simulated seconds reached")

    def _remember_controllers(self) -> None:
        with self.pg.session.lock:
            for name in self._command_robots:
                controller = self.pg.session.controllers.get(name)
                if controller is not None and name not in self._controlled:
                    self._controlled[name] = controller

    def _remember_command(self, robot: str) -> None:
        self._command_robots.add(robot)
        if self._execution is not None:
            self._execution.resources.add(f"robot:{robot.split('/', 1)[0]}")
        # Only an explicit successful command may adopt a replacement controller. Polling must not claim it.
        controller = self.pg.session.controllers.get(robot)
        if controller is not None:
            self._controlled[robot] = controller
        self._remember_controllers()
