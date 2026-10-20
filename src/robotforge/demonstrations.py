"""Recording, replaying and exporting demonstrations of the open world's robots (see sim/demonstration.py).

A `Recording` watches the physics step and the commands the robot is given; a `Replay` gives them again, or plays the
actuator targets back. Both run beside the simulation, whoever steps it (the viewer's clock, a tool, a script).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import mujoco
import numpy as np

from robotforge import wording
from robotforge.behaviors.manager import MOBILE, BehaviorError
from robotforge.design.store import worlds_home
from robotforge.sim.demonstration import (
    FOLDER,
    MAX_FRAMES,
    MAX_RATE_HZ,
    RATE_HZ,
    SKIPPED_SENSORS,
    SUFFIXES,
    Demonstration,
    DemonstrationError,
    clean_name,
    flatten,
)

TAIL = 0.0  # s a replay runs past the demonstration's last frame
PROBLEMS_SHOWN = 8  # refused commands a replay lists


def _a(kind: str) -> str:
    return f"an {kind}" if kind[:1] in "aeiou" else f"a {kind}"


def _yaw(rot: np.ndarray) -> float:
    return math.atan2(rot[1, 0], rot[0, 0])


def _to_robot(root_pos: np.ndarray, yaw: float, world: np.ndarray, vector: bool = False) -> list[float]:
    """A world point (or, with `vector`, a direction) in the frame of a robot at `root_pos` facing `yaw`."""
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(world, float) - (0 if vector else root_pos)
    return [round(float(c * d[0] + s * d[1]), 5), round(float(-s * d[0] + c * d[1]), 5), round(float(d[2]), 5)]


def _to_world(root_pos: np.ndarray, yaw: float, local: list[float], vector: bool = False) -> list[float]:
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = (float(v) for v in local)
    out = np.array([c * x - s * y, s * x + c * y, z])
    return [float(v) for v in (out if vector else out + root_pos)]


class Recording:
    """One robot being recorded: its frames at `rate_hz` and the commands it is given."""

    def __init__(self, pg: Any, robot: str, name: str, rate_hz: float, sensors: list[str] | None) -> None:
        self.pg = pg
        self.name = name
        self.rate_hz = rate_hz
        self.requested_sensors = sensors
        self.owner = pg.behaviors.resolve(robot).split("/", 1)[0]
        self.stopped_by: str | None = None
        self.active = False
        self._model: mujoco.MjModel | None = None
        self._version = 0
        self._t0 = 0.0
        self._next = 0.0
        self._joint_ids: list[int] = []
        self._joint_adr: list[int] = []
        self._actuators: list[int] = []
        self._root = 0
        self._sensors: list[str] = []
        self._raw = False  # the robot's own MuJoCo sensors no definition reads (a library robot's IMU) are kept too
        self._sensor_keys: list[str] = []
        self._times: list[float] = []
        self._joints: list[np.ndarray] = []
        self._ctrl: list[np.ndarray] = []
        self._pos: list[np.ndarray] = []
        self._quat: list[np.ndarray] = []
        self._readings: list[dict[str, float]] = []
        self._commands: list[dict[str, Any]] = []
        self._kind = ""
        self._info: dict[str, Any] = {}

    # ---- starting and stopping ----

    def start(self) -> Recording:
        pg, s = self.pg, self.pg.session
        with s.lock:
            m = s.model
            self._model, self._version = m, s.version
            prefix = f"{self.owner}/"
            for j in range(m.njnt):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if name.startswith(prefix) and int(m.jnt_type[j]) in (
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                ):
                    self._joint_ids.append(j)
                    self._joint_adr.append(int(m.jnt_qposadr[j]))
            self._actuators = [
                a
                for a in range(m.nu)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(prefix)
            ]
            self._root = pg.behaviors._root_body(self.owner)
            self._kind = pg.behaviors.kind(self.owner).value
            self._sensors = self._choose_sensors()
            self._raw = self.requested_sensors is None and pg.sensors._has_other(self.owner)
            self._info = self._robot_info()
            self._t0 = self._next = float(s.data.time)
            self.active = True
            self._grab()
            s.step_hooks.append(self._grab)
            pg.behaviors.command_hooks.append(self._command)
        return self

    def _robot_info(self) -> dict[str, Any]:
        info = next((r for r in self.pg.behaviors.robots() if r["name"] == self.owner), {})
        return {k: info[k] for k in ("model", "max_speed", "arms") if k in info}

    def _choose_sensors(self) -> list[str]:
        """The sensors read at every frame: the ones asked for, else every sensor on the robot that is cheap to read (a
        camera or a lidar is a picture or a thousand rays: named on purpose)."""
        hub = self.pg.sensors
        have = hub.suite.of(self.owner)
        if self.requested_sensors is not None:
            return [s.local for s in hub.chosen(self.owner, self.requested_sensors)] if self.requested_sensors else []
        return [s.local for s in have if not s.renders and s.type not in SKIPPED_SENSORS]

    def stop(self) -> Demonstration:
        with self.pg.session.lock:
            self._end()
        return self.demonstration()

    def _end(self) -> None:
        if not self.active:
            return
        self.active = False
        hooks = self.pg.session.step_hooks
        if self._grab in hooks:
            hooks.remove(self._grab)
        if self._command in self.pg.behaviors.command_hooks:
            self.pg.behaviors.command_hooks.remove(self._command)

    # ---- every step ----

    def _grab(self) -> None:
        s = self.pg.session
        if s.version != self._version or s.model is not self._model:
            self.stopped_by = "the world changed (an edit or a reset), so the recording ends there"
            self._end()
            return
        t = float(s.data.time)
        if self._times and t < self._t0 + self._times[-1]:
            self.stopped_by = "the simulation was reset, so the recording ends there"
            self._end()
            return
        if t + 1e-9 < self._next:
            return
        if len(self._times) >= MAX_FRAMES:
            self.stopped_by = f"the recording reached its {MAX_FRAMES} frames"
            self._end()
            return
        d = s.data
        self._times.append(t - self._t0)
        self._joints.append(d.qpos[self._joint_adr].copy())
        self._ctrl.append(d.ctrl[self._actuators].copy())
        self._pos.append(d.xpos[self._root].copy())
        self._quat.append(d.xquat[self._root].copy())
        if self._sensors or self._raw:
            self._readings.append(self._read())
        self._next = t + 1.0 / self.rate_hz

    def _read(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._sensors:
            reading, _ = self.pg.sensors.read(self.owner, self._sensors)
            for name, value in reading["sensors"].items():
                if value is not None:
                    flatten(name, value, out)
        if self._raw:
            for name, value in self.pg.sensors.raw(self.owner).items():
                flatten(name, value, out)
        return out

    def _command(self, robot: str, event: dict[str, Any]) -> None:
        """A command the robot was given (the behavior manager calls this inside the command)."""
        if not self.active or robot.split("/", 1)[0] != self.owner:
            return
        d = self.pg.session.data
        root = d.xpos[self._root]
        yaw = _yaw(d.xmat[self._root].reshape(3, 3))
        event = dict(event)
        for key in ("position",):  # targets in the robot's own frame
            if key in event:
                event[key] = _to_robot(root, yaw, np.asarray(event[key], float))
        if "velocity" in event:
            event["velocity"] = _to_robot(root, yaw, np.asarray(event["velocity"], float), vector=True)
        if "/" in robot:
            event["arm"] = robot.split("/", 1)[1]
        self._commands.append({"t": round(float(d.time) - self._t0, 4), **event})

    # ---- the result ----

    def demonstration(self) -> Demonstration:
        if len(self._times) < 2:
            raise DemonstrationError(
                f"{self.name}: nothing was recorded (no time passed): let the robot act, then stop the recording"
            )
        keys = sorted(
            {k for r in self._readings for k in r},
            key=lambda k: next(i for i, r in enumerate(self._readings) if k in r),
        )
        sensors = np.array([[r.get(k, math.nan) for k in keys] for r in self._readings], dtype=float)
        prefix = f"{self.owner}/"
        m = self._model
        assert m is not None
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in self._joint_ids]
        actuators = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "" for a in self._actuators]
        world = self.pg.current_world()
        return Demonstration(
            name=self.name,
            robot=self.owner,
            kind=self._kind,
            rate_hz=self.rate_hz,
            created=dt.datetime.now().isoformat(timespec="seconds"),
            world=world.name if world is not None else "",
            joint_names=[n.removeprefix(prefix) for n in names],
            actuator_names=[n.removeprefix(prefix) for n in actuators],
            sensor_names=keys,
            time=np.array(self._times),
            joints=np.array(self._joints).reshape(len(self._times), len(names)),
            ctrl=np.array(self._ctrl).reshape(len(self._times), len(actuators)),
            base_pos=np.array(self._pos),
            base_quat=np.array(self._quat),
            sensors=sensors.reshape(len(self._times), len(keys)) if keys else np.zeros((len(self._times), 0)),
            commands=list(self._commands),
            info=self._info,
        )

    def status(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "robot": self.owner,
            "recording": self.active,
            "seconds": round(self._times[-1], 2) if self._times else 0.0,
            "frames": len(self._times),
            "commands": len(self._commands),
        }
        if self.stopped_by:
            out["stopped_by"] = self.stopped_by
        return out


class Replay:
    """A demonstration played on a robot: its commands again (mode "commands") or its actuator targets ("joints")."""

    def __init__(self, pg: Any, demo: Demonstration, robot: str, mode: str) -> None:
        self.pg = pg
        self.demo = demo
        self.mode = mode
        self.robot = pg.behaviors.resolve(robot).split("/", 1)[0]
        self.done = False
        self.stopped_by: str | None = None
        self.problems: list[str] = []
        self.issued = 0
        self.speed_scale = 1.0
        self._t0 = 0.0
        self._due = 0
        self._model: mujoco.MjModel | None = None
        self._version = 0
        self._ctrl_adr: list[int] = []
        self._notes: list[str] = []

    # ---- checks before it starts ----

    def check(self) -> None:
        b, demo = self.pg.behaviors, self.demo
        kind = b.kind(self.robot).value
        if kind != demo.kind:
            raise DemonstrationError(
                f"{demo.name} is {_a(demo.kind)}'s demonstration ({demo.robot}); {self.robot} is {_a(kind)}: "
                f"replay it on {_a(demo.kind)}"
            )
        arm_ops = {c["op"] for c in demo.commands} & {"move_arm", "gripper", "jog", "pick_and_place"}
        if arm_ops and not b.arms(self.robot):
            raise DemonstrationError(f"{demo.name} works an arm ({', '.join(sorted(arm_ops))}); {self.robot} has none")
        if self.mode == "joints":
            m = self.pg.session.model
            wanted = [f"{self.robot}/{n}" for n in demo.actuator_names]
            self._ctrl_adr = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in wanted]
            missing = [n for n, a in zip(demo.actuator_names, self._ctrl_adr, strict=True) if a < 0]
            if missing:
                raise DemonstrationError(
                    f'mode "joints" plays the actuator targets back, and {self.robot} has no actuator '
                    f'{", ".join(missing[:4])}{" ..." if len(missing) > 4 else ""}; mode "commands" works on any robot '
                    "of the kind"
                )

    # ---- running ----

    def start(self) -> Replay:
        self.check()
        s = self.pg.session
        with s.lock:
            self._model, self._version = s.model, s.version
            self._t0 = float(s.data.time)
            theirs = next((r for r in self.pg.behaviors.robots() if r["name"] == self.robot), {})
            ours = float(self.demo.info.get("max_speed", 0) or 0)
            if ours and theirs.get("max_speed") and theirs["max_speed"] < ours:
                self.speed_scale = round(float(theirs["max_speed"]) / ours, 3)
                self._notes.append(
                    f"{self.robot} is slower than {self.demo.robot} ({theirs['max_speed']} against {ours} m/s): "
                    f"speeds are scaled to {self.speed_scale:.2f}"
                )
            if self.mode == "joints":
                self._to_start()
            s.step_hooks.append(self._step)
        return self

    def _to_start(self) -> None:
        """The robot in the demonstration's first pose, still, with its own controllers taken off."""
        s, demo = self.pg.session, self.demo
        m, d = s.model, s.data
        for name in [n for n in s.controllers if n.split("/", 1)[0] == self.robot]:
            s.set_controller(name, None)
        for joint, value in zip(demo.joint_names, demo.joints[0], strict=True):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.robot}/{joint}")
            if j >= 0:
                d.qpos[m.jnt_qposadr[j]] = value
                d.qvel[m.jnt_dofadr[j]] = 0.0
        for a, value in zip(self._ctrl_adr, demo.ctrl[0], strict=True):
            d.ctrl[a] = value
        mujoco.mj_forward(m, d)

    def _step(self) -> None:
        s = self.pg.session
        if s.version != self._version or s.model is not self._model:
            self.stopped_by = "the world changed (an edit or a reset), so the replay ends there"
            self._finish()
            return
        elapsed = float(s.data.time) - self._t0
        if elapsed < 0:
            self.stopped_by = "the simulation was reset, so the replay ends there"
            self._finish()
            return
        if self.mode == "joints":
            self._play_targets(elapsed)
        else:
            commands = self.demo.commands
            while self._due < len(commands) and commands[self._due]["t"] <= elapsed + 1e-9:
                self._issue(commands[self._due])
                self._due += 1
        if elapsed >= self.demo.seconds + TAIL - 1e-9:
            self._finish()

    def _play_targets(self, elapsed: float) -> None:
        demo, d = self.demo, self.pg.session.data
        k = int(np.searchsorted(demo.time, elapsed, side="right")) - 1
        k = min(max(k, 0), demo.frames - 1)
        if k + 1 < demo.frames and demo.time[k + 1] > demo.time[k]:
            u = (elapsed - demo.time[k]) / (demo.time[k + 1] - demo.time[k])
            ctrl = demo.ctrl[k] + (demo.ctrl[k + 1] - demo.ctrl[k]) * u
        else:
            ctrl = demo.ctrl[k]
        d.ctrl[self._ctrl_adr] = ctrl

    def _issue(self, c: dict[str, Any]) -> None:
        b = self.pg.behaviors
        s, root = self.pg.session, b._root_body(self.robot)
        pos, yaw = s.data.xpos[root].copy(), _yaw(s.data.xmat[root].reshape(3, 3))
        owner = self.robot
        arm = f"{owner}/{c['arm']}" if c.get("arm") else owner
        if c.get("arm") and arm not in b.arms(owner):  # another robot's arm has another name: the one it has
            arms = b.arms(owner)
            arm = arms[0] if len(arms) == 1 else arm
        scale = self.speed_scale
        try:
            op = c["op"]
            if op == "move":
                b.move(owner, c["speed"] * scale, c["turn"], side=c.get("side", 0.0) * scale, climb=c.get("climb", 0.0))
            elif op == "stop":
                b.stop(owner)
            elif op == "heave":
                b.heave(owner, c["rate"])
            elif op == "move_arm":
                b.move_arm(arm, _to_world(pos, yaw, c["position"]), c.get("gripper"))
            elif op == "jog":
                b.jog_arm(arm, _to_world(pos, yaw, c["velocity"], vector=True), c.get("gripper"))
            elif op == "gripper":
                b.set_gripper(arm, c["state"])
            elif op == "pose":
                b.strike_pose(owner, c["name"])
            elif op == "go":
                b.navigate(owner, c["to"], None, None)
            elif op == "pick_and_place":
                b.pick_and_place(arm, c["object"], c["target"])
            else:
                raise DemonstrationError(f"a command {op!r} that this robotforge does not know")
            self.issued += 1
        except (BehaviorError, DemonstrationError, ValueError, KeyError) as exc:
            if len(self.problems) < PROBLEMS_SHOWN:
                self.problems.append(f"{c.get('t', 0):.2f} s {c.get('op')}: {exc}")

    def _finish(self) -> None:
        if self.done:
            return
        self.done = True
        hooks = self.pg.session.step_hooks
        if self._step in hooks:
            hooks.remove(self._step)
        behaviors = self.pg.behaviors
        if self.mode == "commands" and behaviors.kind(self.robot) in MOBILE:  # an arm holds where it is
            with contextlib.suppress(BehaviorError):
                behaviors.stop(self.robot)

    def cancel(self) -> None:
        with self.pg.session.lock:
            self.stopped_by = self.stopped_by or "stopped"
            self._finish()

    def status(self) -> dict[str, Any]:
        s = self.pg.session
        elapsed = (
            min(max(float(s.data.time) - self._t0, 0.0), self.demo.seconds) if not self.done else self.demo.seconds
        )
        out: dict[str, Any] = {
            "demonstration": self.demo.name,
            "robot": self.robot,
            "mode": self.mode,
            "done": self.done,
            "seconds": round(elapsed, 2),
            "of_seconds": round(self.demo.seconds, 2),
            "commands_issued": self.issued,
        }
        if self.speed_scale != 1.0:
            out["speed_scale"] = self.speed_scale
        notes = [*self._notes]
        if self.stopped_by:
            notes.append(self.stopped_by)
        if notes:
            out["note"] = "; ".join(notes)
        if self.problems:
            out["refused"] = self.problems
        return out


class Demonstrations:
    """The demonstrations of the open world: the recordings under way, the replays, and the saved files."""

    def __init__(self, pg: Any) -> None:
        self.pg = pg
        self.recordings: dict[str, Recording] = {}
        self.replays: dict[str, Replay] = {}  # robot -> its latest replay
        self.kept: dict[str, Demonstration] = {}

    # ---- where they live ----

    def folder(self) -> Path:
        store = self.pg.store
        return (store.dir if store is not None else worlds_home().parent) / FOLDER

    def _path(self, name: str, suffix: str = ".json") -> Path:
        return self.folder() / f"{name}{suffix}"

    def target(self, to: str, name: str, fmt: str) -> Path:
        """Where an export goes: always in the demonstrations folder (a name or a relative path with no ..)."""
        given = str(to).strip()
        ways = (PurePosixPath(given), PureWindowsPath(given))
        if given.startswith("~") or any(w.anchor or ".." in w.parts for w in ways):
            raise DemonstrationError(
                f'to {wording.short(given)} is a file name, e.g. "{name}.{fmt}": exports go in the world\'s '
                f"{FOLDER} folder, and the result's path says where"
            )
        path = self.folder() / given
        if path.suffix.lower() not in SUFFIXES:
            path = path.with_suffix(f".{fmt}")
        if not path.resolve().is_relative_to(self.folder().resolve()):
            raise DemonstrationError(f"to {wording.short(given)} leads out of the {FOLDER} folder; give a file name")
        return path

    # ---- recording ----

    def start(
        self,
        robot: str,
        name: str | None = None,
        rate_hz: float = RATE_HZ,
        sensors: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not 1 <= rate_hz <= MAX_RATE_HZ:
            raise DemonstrationError(
                wording.seconds_range("rate_hz", rate_hz, MAX_RATE_HZ).replace("seconds", "frames a second")
            )
        owner = self.pg.behaviors.resolve(robot).split("/", 1)[0]
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        called = clean_name(name) if name else clean_name(f"{owner}_{stamp}")
        if called in self.recordings:
            raise DemonstrationError(f"{called} is already being recorded: stop it first")
        if not overwrite and (called in self.kept or self._path(called).exists()):
            raise DemonstrationError(f"a demonstration called {called} exists: choose another name, or overwrite=true")
        busy = [r.name for r in self.recordings.values() if r.owner == owner]
        if busy:
            raise DemonstrationError(f"{owner} is already being recorded as {busy[0]}: stop that one first")
        rec = Recording(self.pg, owner, called, float(rate_hz), sensors).start()
        self.recordings[called] = rec
        return {"recording": called, "robot": owner, "kind": rec._kind, "rate_hz": rec.rate_hz,
                "sensors": rec._sensors, "note": "act (keys, a gamepad, move, a script), then stop it"}  # fmt: skip

    def stop(self, name: str | None = None) -> dict[str, Any]:
        rec = self._recording(name)
        demo = rec.stop()
        self.recordings.pop(rec.name, None)
        self.kept[demo.name] = demo
        out = demo.summary()
        try:
            path = demo.save(self._path(demo.name))
            out["path"] = str(path)
        except OSError as exc:
            out["note"] = f"kept in memory only: it could not be written ({exc})"
        if rec.stopped_by:
            out["note"] = rec.stopped_by
        return out

    def _recording(self, name: str | None) -> Recording:
        if name is None:
            if not self.recordings:
                raise DemonstrationError("nothing is being recorded; start a recording first")
            if len(self.recordings) > 1:
                raise DemonstrationError(f"several recordings are going: name one of {', '.join(self.recordings)}")
            return next(iter(self.recordings.values()))
        called = clean_name(name)
        if called not in self.recordings:
            running = ", ".join(self.recordings) or "none"
            raise DemonstrationError(f"{called} is not being recorded (recording now: {running})")
        return self.recordings[called]

    # ---- keeping, replaying and exporting ----

    def get(self, name: str) -> Demonstration:
        called = clean_name(name)
        if called in self.kept:
            return self.kept[called]
        for suffix in SUFFIXES:
            path = self._path(called, suffix)
            if path.is_file():
                self.kept[called] = Demonstration.load(path)
                return self.kept[called]
        have = ", ".join(self.names()) or "none yet"
        raise DemonstrationError(f"no demonstration called {called} (saved: {have}); record one first")

    def names(self) -> list[str]:
        found = set(self.kept)
        folder = self.folder()
        if folder.is_dir():
            found |= {p.stem for p in folder.iterdir() if p.suffix.lower() in SUFFIXES}
        return sorted(found)

    def replay(self, name: str, robot: str | None = None, mode: str = "commands") -> Replay:
        if mode not in ("commands", "joints"):
            raise DemonstrationError('mode is "commands" (the recorded commands again, any robot of the same kind) '
                                     'or "joints" (the recorded actuator targets, the same robot)')  # fmt: skip
        demo = self.get(name)
        target = robot or demo.robot
        replay = Replay(self.pg, demo, target, mode).start()
        self.replays[replay.robot] = replay
        return replay

    def export(self, name: str, fmt: str = "json", to: str | None = None) -> dict[str, Any]:
        fmt = fmt.lower().lstrip(".")
        if fmt not in ("json", "npz"):
            raise DemonstrationError(
                f"format is json or npz, not {fmt!r}; LeRobot's parquet needs a library robotforge does not install"
            )
        demo = self.get(name)
        path = self.target(to, demo.name, fmt) if to else self._path(demo.name, f".{fmt}")
        demo.save(path)
        return {**demo.summary(), "path": str(path), "format": fmt, "size_kb": round(path.stat().st_size / 1e3, 1)}

    def listing(self) -> dict[str, Any]:
        out: dict[str, Any] = {"demonstrations": [], "recording": [r.status() for r in self.recordings.values()]}
        for called in self.names():
            try:
                out["demonstrations"].append(self.get(called).summary())
            except DemonstrationError as exc:
                out["demonstrations"].append({"name": called, "error": str(exc)})
        out["replaying"] = [r.status() for r in self.replays.values() if not r.done]
        out["folder"] = str(self.folder())
        return out

    def viewer_state(self) -> dict[str, Any]:
        """What the viewer's state message says: what is being recorded and replayed (for its REC chip)."""
        out: dict[str, Any] = {}
        if self.recordings:
            out["recording"] = [
                {"name": r.name, "robot": r.owner, "seconds": round(r._times[-1], 1) if r._times else 0.0}
                for r in self.recordings.values()
            ]
        playing = [r for r in self.replays.values() if not r.done]
        if playing:
            out["replaying"] = [{"name": r.demo.name, "robot": r.robot} for r in playing]
        return out
