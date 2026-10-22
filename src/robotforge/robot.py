"""One interface to any robot in a script: robot.sensors, robot.move, robot.arm, robot.gripper, a per-step callback.

    from robotforge.script import connect

    sim = connect()
    robot = sim.robot("dog")                 # a template dog, a Go2, a Spot...: the same calls
    robot.on_step(lambda r: print(r.sensors["imu"]["orientation_deg"]), every=0.5)
    robot.move(speed=0.3)                    # walks (or drives) until robot.stop()
    robot.run(4.0)                           # let 4 simulated seconds pass; the callback fires as they do
    robot.stop()

    arm = sim.robot("franka")                # a template arm, a Panda, a UR5e, a Go2 carrying a PiPER...
    arm.arm.move_to([0.4, 0.0, 0.25])
    arm.gripper.close()

What a robot cannot do says so (a dog has no arm, an arm cannot walk) with what it can do instead, so a script written
for one robot fails clearly on another instead of doing nothing. This module needs only what scripts have (a Sim).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MOBILE_KINDS = ("quadruped", "hexapod", "humanoid", "rover", "balancer", "snake", "boat", "rov", "aircraft")


class Sensors:
    """A robot's sensors by name: `robot.sensors["lidar"]`, `robot.sensors.imu`, `robot.sensors.names()`."""

    def __init__(self, robot: Robot) -> None:
        self._robot = robot

    def read(self) -> dict[str, Any]:
        """Every sensor's reading: the ones added with add_sensor, the template's own, MuJoCo's raw ones."""
        return self._robot.sim.sensors(self._robot.name)

    def names(self) -> list[str]:
        """The names of the sensors added with add_sensor and the template's own."""
        return self._robot.sim.sensor_names(self._robot.name)

    def __getitem__(self, name: str) -> Any:
        return self._robot.sim.sensor(self._robot.name, name)

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except Exception:  # a script asking "is there a lidar?" wants the default, whatever the reason it is missing
            return default

    def __contains__(self, name: str) -> bool:
        return name in self.names()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


class Gripper:
    """The hand of an arm: fingers, a suction cup or a magnet."""

    def __init__(self, robot: Robot, name: str | None = None) -> None:
        self._robot = robot
        self.name = name or robot.name

    def open(self) -> None:
        """Open the fingers (suction or magnet off); returns when it is done."""
        self._robot.sim.gripper(self.name, "open")

    def close(self) -> None:
        """Close the fingers (suction or magnet on); returns when it is done."""
        self._robot.sim.gripper(self.name, "close")


class Arm:
    """An arm: the robot itself, or the one it carries (a PiPER on a Go2's back is `go2/arm`)."""

    def __init__(self, robot: Robot, name: str) -> None:
        self._robot = robot
        self.name = name

    def move_to(self, position: list[float], gripper: str | None = None) -> list[float]:
        """Move the tool to [x, y, z] m in the world, pointing down; returns when it is there (the tool position).
        gripper="open" or "close" also sets the hand."""
        return self._robot.sim.move_arm(self.name, list(position), gripper=gripper)

    @property
    def gripper(self) -> Gripper:
        """This arm's hand, including when the robot carries several arms."""
        caps = self._robot.capabilities
        if caps is not None and self.name not in caps["gripper"]["endpoints"]:
            raise self._robot._cannot("hold things", "arm")
        return Gripper(self._robot, self.name)

    def pick_and_place(self, obj: str, target: str | list[float]) -> bool:
        """Pick the object up and put it on or in `target` (an object's name or [x, y]); True if it landed there.
        An arm on a robot that walks or drives fetches: the robot goes over first."""
        return self._robot.sim.pick_and_place(self.name, obj, target)


class DemoRecording:
    """A demonstration being recorded (see Robot.record): `stop()` ends it, and so does leaving a `with` block."""

    def __init__(self, robot: Robot, started: dict[str, Any]) -> None:
        self._robot = robot
        self.name: str = started["recording"]
        self.started = started
        self.result: dict[str, Any] | None = None

    def stop(self) -> dict[str, Any]:
        """End the recording; returns what it holds (seconds, frames, commands) and where it was saved."""
        if self.result is None:
            self.result = self._robot.sim.stop_recording(self.name)
        return self.result

    def __enter__(self) -> DemoRecording:
        return self

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        if self.result is not None:
            return
        try:
            self.stop()
        except RuntimeError:
            if exc_type is None:  # a body that failed keeps its own error; an empty recording is the only one to report
                raise


class Robot:
    """A robot of the world as a script sees it. Make one with `sim.robot(name)`."""

    def __init__(self, sim: Any, name: str, info: dict[str, Any]) -> None:
        self.sim = sim
        self.name = name
        self.info = info
        self.kind: str = info.get("kind", "other")
        self.sensors = Sensors(self)
        caps = self.capabilities
        arms = caps["move_arm"]["endpoints"] if caps is not None else info.get("arms", [])
        self.arms: list[Arm] = [Arm(self, a) for a in arms]
        if caps is None and not self.arms and self.kind == "arm":
            self.arms = [Arm(self, name)]
        self._gripper = Gripper(self)

    def __repr__(self) -> str:
        return f"<Robot {self.name!r}: {self.kind}>"

    # ---- where it is ----------------------------------------------------------------------------------------------

    @property
    def time(self) -> float:
        return self.sim.time

    @property
    def position(self) -> list[float]:
        """[x, y, z] m of the robot's base."""
        return self.sim.position(self.name)

    @property
    def heading(self) -> float:
        """Degrees counter-clockwise from +x."""
        return self.sim.heading(self.name)

    @property
    def joints(self) -> dict[str, float]:
        """Joint angles (rad) or slides (m) by name, without the robot's prefix."""
        return self.sim.joints(self.name)

    def set_controls(self, controls: dict[str, float]) -> None:
        """Raw actuator targets by the robot's own names ("front_left_hip"): the same on any robot that has them."""
        prefix = f"{self.name}/"
        self.sim.set_controls({k if k.startswith(prefix) else prefix + k: v for k, v in controls.items()})

    # ---- moving ---------------------------------------------------------------------------------------------------

    @property
    def can_move(self) -> bool:
        if self.capabilities is not None:
            return self.capabilities["move"]["supported"]
        return self.kind in MOBILE_KINDS or "max_speed" in self.info

    @property
    def capabilities(self) -> dict[str, Any] | None:
        """The server's capability snapshot; None when connected to an older server."""
        return self.info.get("capabilities")

    def refresh(self) -> Robot:
        """Read a new snapshot, for example after keeping or removing a learned walk."""
        fresh = self.sim.robot(self.name)
        self.__init__(self.sim, fresh.name, fresh.info)
        return self

    def move(self, speed: float = 0.2, turn: float = 0.0, seconds: float | None = None) -> None:
        """Walk, drive or swim at `speed` m/s turning `turn` rad/s (left is positive); keeps going until stop().
        With `seconds`, runs that long and stops."""
        if not self.can_move:
            raise self._cannot("walk or drive", "arm" if self.arms else None)
        self.sim.move(self.name, speed, turn)
        if seconds is not None:
            self.run(seconds)
            self.stop()

    def stop(self) -> None:
        self.sim.stop(self.name)

    def pose(self, name: str) -> bool:
        """A humanoid's pose ("wave", "squat", "arms_up", "bow", "look_around", "stand"); True if still upright."""
        return self.sim.pose(self.name, name)

    # ---- the arm and its hand --------------------------------------------------------------------------------------

    @property
    def arm(self) -> Arm:
        """The arm (the first, for a robot with several: see .arms)."""
        if not self.arms:
            raise self._cannot("work an arm", "move" if self.can_move else None)
        return self.arms[0]

    @property
    def gripper(self) -> Gripper:
        if self.capabilities is not None:
            endpoints = self.capabilities["gripper"]["endpoints"]
            if len(endpoints) == 1:
                return Gripper(self, endpoints[0])
            if len(endpoints) > 1:
                from robotforge.script import ScriptError

                raise ScriptError(f"{self.name} has several grippers; choose robot.arms[i].gripper")
            raise self._cannot("hold things", "move" if self.can_move else None)
        if not self.arms:
            raise self._cannot("hold things", "move" if self.can_move else None)
        return self._gripper

    @property
    def has_arm(self) -> bool:
        return bool(self.arms)

    # ---- demonstrations --------------------------------------------------------------------------------------------

    def record(
        self, name: str | None = None, rate_hz: float = 20.0, sensors: list[str] | None = None, overwrite: bool = False
    ) -> DemoRecording:
        """Start keeping what this robot does (joints, actuator targets, base pose, sensors and every command it is
        given, from a script, a key or a gamepad): `with robot.record("pick") as demo: ...`, or demo = robot.record(),
        then demo.stop(). The result is saved in the world's demonstrations folder."""
        return DemoRecording(self, self.sim.start_recording(self.name, name, rate_hz, sensors, overwrite))

    def replay(self, name: str, mode: str = "commands", wait: bool = True) -> dict[str, Any]:
        """Play a demonstration on this robot, which may be another one than recorded it, if it is of the same kind:
        "commands" gives the recorded commands again, "joints" the recorded actuator targets (same actuators). With
        wait=True it returns when the replay is over."""
        return self.sim.replay(name, self.name, mode, wait)

    # ---- plugin controllers ----------------------------------------------------------------------------------------

    def start(self, controller: str, **options: Any) -> dict[str, Any]:
        """Hand the robot to a plugin controller (plugins() lists them); it keeps driving while time passes, until
        stop() or another command."""
        return self.sim.start_controller(self.name, controller, options)

    # ---- time and the per-step callback ----------------------------------------------------------------------------

    def on_step(self, callback: Callable[[Robot], Any], every: float = 0.02) -> Callable[[Robot], Any]:
        """Call `callback(robot)` every `every` simulated seconds while time passes through run() or sim.step().
        Returns the callback, so it works as a decorator. Blocking calls (arm.move_to, pick_and_place) run to their
        end without calling it."""
        self.sim.on_step(lambda: callback(self), every)
        return callback

    def run(self, seconds: float) -> float:
        """Let `seconds` of simulated time pass, calling the per-step callbacks as it does. Returns the new time."""
        return self.sim.step(seconds)

    def _cannot(self, what: str, instead: str | None) -> RuntimeError:
        from robotforge.script import ScriptError

        hint = {
            "arm": "it has an arm: use robot.arm and robot.gripper",
            "move": "it can walk or drive: use robot.move",
            None: "check robot.kind and robot.arms first",
        }[instead]
        return ScriptError(f"{self.name} is a {self.kind} and cannot {what}; {hint}")
