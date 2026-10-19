"""Plugin controllers and behaviors.

A controller runs inside the simulation, every 1/rate_hz seconds of simulated time, until it is stopped or replaced
(stop(robot), a move, another controller): feedback that reads the robot and sets its actuators. A behavior is a
routine written against the same robot interface scripts use (robot.move, robot.arm, robot.sensors, ...), so one
behavior runs on a template dog and on a Go2; it runs like a script (its own process, in lockstep, with a time limit).
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.extend.registry import REGISTRY, Item, PluginError

log = logging.getLogger(__name__)
MAX_RATE_HZ = 500.0


def controller(
    name: str,
    *,
    summary: str = "",
    params: dict[str, Any] | None = None,
    rate_hz: float = 50.0,
) -> Callable[[Any], Any]:
    """Register a controller, started with plugins(action="run", name=..., robot=...) or robot.start(name).

    A function `update(ctx)` is called every 1/rate_hz seconds of simulated time (ctx: StepContext), or a class
    with `__init__(self, ctx)` and `update(self, ctx)` when it keeps state. params are its options with their
    defaults (`ctx.params`); one not listed is refused when it starts."""

    def register(obj: Any) -> Any:
        _check_callable(obj, "controller", name, "update")
        if not 0 < rate_hz <= MAX_RATE_HZ:
            raise PluginError(f"controller {name!r}: rate_hz is above 0 and at most {MAX_RATE_HZ:g}, got {rate_hz}")
        defaults = dict(params or {})
        REGISTRY.add(
            Item(
                "controller",
                name,
                summary or f"a {name} controller (from a plugin)",
                obj,
                meta={
                    "params": defaults,
                    "rate_hz": float(rate_hz),
                    "describe": {"options": defaults, "rate_hz": float(rate_hz)},
                },
            )
        )
        return obj

    return register


def behavior(name: str, *, summary: str = "", params: dict[str, Any] | None = None) -> Callable[[Any], Any]:
    """Register a behavior: a function `run(robot, **params)` written with the script robot interface
    (robotforge.script's Robot: robot.move, robot.arm, robot.gripper, robot.sensors, robot.run, ...). It runs as a
    script does, started with plugins(action="run", name=..., robot=...) or called from your own script as
    `from robotforge.extend import call; call("name", robot)`. What it returns is the result."""

    def register(fn: Any) -> Any:
        if not callable(fn) or inspect.isclass(fn):
            raise PluginError(f"behavior {name!r}: decorate a function run(robot, **params)")
        defaults = dict(params or {})
        REGISTRY.add(
            Item(
                "behavior",
                name,
                summary or f"a {name} behavior (from a plugin)",
                fn,
                meta={"params": defaults, "describe": {"options": defaults}},
            )
        )
        return fn

    return register


def _check_callable(obj: Any, kind: str, name: str, method: str) -> None:
    if inspect.isclass(obj):
        if not callable(getattr(obj, method, None)):
            raise PluginError(f"{kind} {name!r}: the class needs a {method}(self, ctx) method")
    elif not callable(obj):
        raise PluginError(f"{kind} {name!r}: decorate a function {method}(ctx) or a class with {method}(self, ctx)")


def check_params(kind: str, name: str, item: Item, given: dict[str, Any] | None) -> dict[str, Any]:
    """The options with their defaults; one it does not have is refused, naming the ones it has."""
    known = item.meta["params"]
    extra = sorted(set(given or {}) - set(known))
    if extra:
        have = ", ".join(known) if known else "none"
        raise PluginError(f"{kind} {name!r} takes no option {', '.join(map(repr, extra))}; its options are {have}")
    return {**known, **(given or {})}


# ---- controllers at run time -------------------------------------------------------------------------------------


class StepContext:
    """What a plugin controller sees and sets, once per update. Joint and actuator names are the robot's own, without
    its "name/" prefix ("front_left_hip"); the world's full names work too. Angles are radians, slides meters."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: str,
        params: dict[str, Any],
        read_sensor: Callable[[str], Any] | None = None,
    ) -> None:
        self.model, self.data, self.robot, self.params = model, data, robot, params
        self.rng = np.random.default_rng(0)
        self.time = float(data.time)
        self.dt = 0.0
        self._read_sensor = read_sensor
        prefix = f"{robot}/"
        self._joints = {
            _short(name, prefix): j
            for j in range(model.njnt)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(prefix)
            and int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        }
        self._actuators = {
            _short(name, prefix): a
            for a in range(model.nu)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(prefix)
        }
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot)
        if root < 0:  # a template's base body is named after its first part: the first body under the robot's name
            root = next(
                (
                    b
                    for b in range(1, model.nbody)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)
                ),
                -1,
            )
        self._root = root

    # reading
    @property
    def joint_names(self) -> list[str]:
        return list(self._joints)

    @property
    def actuator_names(self) -> list[str]:
        return list(self._actuators)

    def joint(self, name: str) -> float:
        """A joint's angle (rad) or slide (m)."""
        return float(self.data.qpos[self.model.jnt_qposadr[self._joint_id(name)]])

    def velocity(self, name: str) -> float:
        """A joint's speed (rad/s or m/s)."""
        return float(self.data.qvel[self.model.jnt_dofadr[self._joint_id(name)]])

    def joints(self) -> dict[str, float]:
        return {name: self.joint(name) for name in self._joints}

    @property
    def position(self) -> np.ndarray:
        """The robot's base, [x, y, z] m."""
        return np.array(self.data.xpos[self._root]) if self._root >= 0 else np.zeros(3)

    @property
    def heading_deg(self) -> float:
        """Which way the base faces, degrees counter-clockwise from +x."""
        if self._root < 0:
            return 0.0
        rot = np.array(self.data.xmat[self._root]).reshape(3, 3)
        return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

    def position_of(self, name: str) -> np.ndarray:
        """Where a body of the world is ("cube", "dog/trunk"), m."""
        found = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if found < 0:
            raise ValueError(f"the world has no body named {name!r}")
        return np.array(self.data.xpos[found])

    def sensor(self, name: str) -> Any:
        """A sensor of this robot read by name, as read_sensors gives it."""
        if self._read_sensor is None:
            raise ValueError("this controller runs without a sensor hub")
        return self._read_sensor(name)

    # acting
    def set(self, name: str, value: float) -> None:
        """Set an actuator's target (a position for a servo, a speed, a force: as the actuator is), within its range."""
        a = self._actuator_id(name)
        if self.model.actuator_ctrllimited[a]:
            low, high = self.model.actuator_ctrlrange[a]
            value = float(np.clip(value, low, high))
        self.data.ctrl[a] = value

    def set_all(self, targets: dict[str, float]) -> None:
        for name, value in targets.items():
            self.set(name, value)

    def _joint_id(self, name: str) -> int:
        try:
            return self._joints[name.removeprefix(f"{self.robot}/")]
        except KeyError:
            raise ValueError(f"{self.robot} has no joint {name!r}; it has {', '.join(self._joints)}") from None

    def _actuator_id(self, name: str) -> int:
        try:
            return self._actuators[name.removeprefix(f"{self.robot}/")]
        except KeyError:
            raise ValueError(f"{self.robot} has no actuator {name!r}; it has {', '.join(self._actuators)}") from None


def _short(name: str, prefix: str) -> str:
    return name[len(prefix) :]


class PluginController:
    """A plugin's controller attached to a session (the Controller protocol of sim.session)."""

    def __init__(
        self,
        name: str,
        item: Item,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: str,
        params: dict[str, Any],
        read_sensor: Callable[[str], Any] | None = None,
    ) -> None:
        self.name = name
        self.rate_hz = float(item.meta["rate_hz"])
        self.context = StepContext(model, data, robot, params, read_sensor)
        self.failure: str | None = None
        self.updates = 0
        self._last = float(data.time)
        self._update: Callable[[StepContext], Any]
        if inspect.isclass(item.obj):
            instance = item.obj(self.context)
            self._update = instance.update
        else:
            self._update = item.obj

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.failure is not None:
            return
        ctx = self.context
        ctx.model, ctx.data = model, data
        ctx.time = float(data.time)
        ctx.dt = max(ctx.time - self._last, 0.0)
        self._last = ctx.time
        try:
            self._update(ctx)
        except Exception as exc:  # the user's code: stop this controller, say why, keep the simulation going
            self.failure = f"{type(exc).__name__}: {exc}"
            log.warning("plugin controller %s stopped: %s", self.name, self.failure)
        self.updates += 1

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"controller": self.name, "plugin": True, "updates": self.updates}
        if self.failure:
            out["stopped_with_error"] = self.failure
        return out


def start_controller(
    session: Any,
    robot: str,
    name: str,
    params: dict[str, Any] | None,
    read_sensor: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Attach the plugin controller `name` to `robot` in the live session (it replaces what drove the robot)."""
    item = REGISTRY.get("controller", name)
    if item is None:
        known = ", ".join(REGISTRY.of("controller")) or "none loaded"
        raise PluginError(f"no controller named {name!r}; the plugin controllers are: {known} (plugins() lists them)")
    options = check_params("controller", name, item, params)
    with session.lock:
        try:
            controller_ = PluginController(name, item, session.model, session.data, robot, options, read_sensor)
        except Exception as exc:  # the class's __init__ is the user's code
            raise PluginError(f"controller {name!r} could not start: {type(exc).__name__}: {exc}") from exc
        session.set_controller(robot, controller_)
    return {"robot": robot, "controller": name, "options": options, "rate_hz": controller_.rate_hz}


# ---- behaviors: the script that runs one ---------------------------------------------------------------------------

BEHAVIOR_SCRIPT = """\
import json
import sys
from pathlib import Path

from robotforge.extend.registry import REGISTRY
from robotforge.script import connect

# The same plugins as the server has: the user's, and the world's own when they are trusted, so that a behavior
# can call another one that is defined in another file with call().
name, robot, params, world = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), Path(sys.argv[4])
REGISTRY.load_world(world)
item = REGISTRY.get("behavior", name)
if item is None:
    raise SystemExit(f"the behavior {name!r} did not load: {json.dumps(REGISTRY.problems())}")
sim = connect()
result = item.obj(sim.robot(robot), **params)
print("RESULT " + json.dumps(result, default=str))
"""

WALL_FACTOR, MIN_WALL_SECONDS, MAX_WALL_SECONDS = 3.0, 20.0, 600.0


def wall_limit(sim_seconds: float) -> float:
    """How long a behavior may run on the clock: `seconds` limits simulated time, which a loop that never steps the
    simulation does not use up."""
    return min(MAX_WALL_SECONDS, max(MIN_WALL_SECONDS, WALL_FACTOR * sim_seconds))


def behavior_command(
    name: str, robot: str, params: dict[str, Any] | None, world_dir: Path
) -> tuple[Path, list[str], dict[str, Any]]:
    """The plugin file that defines behavior `name`, the arguments the runner script takes, and the options."""
    item = REGISTRY.get("behavior", name)
    if item is None:
        known = ", ".join(REGISTRY.of("behavior")) or "none loaded"
        raise PluginError(f"no behavior named {name!r}; the plugin behaviors are: {known} (plugins() lists them)")
    options = check_params("behavior", name, item, params)
    path = item.path
    if path is None:
        raise PluginError(f"behavior {name!r} was not defined in a plugin file, so it cannot be run as a script")
    return path, [name, robot, json.dumps(options), str(world_dir)], options


def parse_result(output: str) -> tuple[Any, str]:
    """The RESULT line the runner script prints, and the rest of the output."""
    keep, result = [], None
    for line in output.splitlines():
        if line.startswith("RESULT "):
            try:
                result = json.loads(line[len("RESULT ") :])
            except ValueError:
                keep.append(line)
        else:
            keep.append(line)
    return result, "\n".join(keep)


def call(name: str, robot: Any, **params: Any) -> Any:
    """From a script: run the behavior `name` on `robot` (a robot from sim.robot(...)) and return what it returns.
    The script's process reads the plugin folders itself (the user's, and the world's own when they are trusted)."""
    item = REGISTRY.get("behavior", name)
    if item is None:
        REGISTRY.load_user()
        if not REGISTRY.worlds:  # a script of the world runs in <world>/behaviors; a behavior's runner loaded it
            REGISTRY.load_world(Path.cwd().parent)
        item = REGISTRY.get("behavior", name)
    if item is None:
        known = ", ".join(REGISTRY.of("behavior")) or "none loaded"
        raise PluginError(f"no behavior named {name!r}; the plugin behaviors are: {known}")
    return item.obj(robot, **check_params("behavior", name, item, params))


__all__: Sequence[str] = ("behavior", "call", "controller")
