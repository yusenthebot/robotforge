"""The API that robot scripts use. Scripts run in their own process, in lockstep with the simulation.

    from robotforge.script import connect

    sim = connect()
    print("time is", sim.time)
    sim.move("dog", speed=0.2)          # start walking (returns at once)
    sim.step(3.0)                       # let 3 seconds pass
    sim.stop("dog")
    ok = sim.pick_and_place("arm", "cube", "tray")   # runs the whole motion, True if it landed
    sim.pose("kid", "wave")             # a humanoid waves hello, then stands

Time only moves when the script calls step() or a blocking action, so a script
can read the world, think, and act without the world running away from it.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from typing import Any

ENV_PORT = "ROBOTFORGE_SCRIPT_PORT"
ENV_TOKEN = "ROBOTFORGE_SCRIPT_TOKEN"


class ScriptError(RuntimeError):
    """The playground refused a request; the message says why."""


class ScriptTimeLimit(ScriptError):
    """The run's simulated-time budget is used up, or the run was stopped: nothing a script does can go on. Behavior
    trees let this one through instead of counting it as a failed node."""


class Sim:
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._file = sock.makefile("rw", encoding="utf-8", newline="\n")
        self._callbacks: list[list[Any]] = []  # [function, every, due]: see on_step

    def _call(self, op: str, **args: Any) -> Any:
        self._file.write(json.dumps({"op": op, **args}) + "\n")
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise ScriptTimeLimit("the playground closed the connection (time limit reached or run stopped)")
        reply = json.loads(line)
        if "error" in reply:
            raise (ScriptTimeLimit if reply["error"].startswith("time limit of") else ScriptError)(reply["error"])
        return reply.get("result")

    # ---- time ----
    @property
    def time(self) -> float:
        """Simulated seconds since the world started."""
        return self._call("time")

    def step(self, seconds: float = 0.02) -> float:
        """Let `seconds` of simulated time pass (physics and running behaviors). Returns the new time. Callbacks from
        on_step are called as it passes."""
        if not self._callbacks:
            return self._call("step", seconds=seconds)
        now = self.time
        end = now + seconds
        while end - now > 1e-9:
            nearest = min(due for _, _, due in self._callbacks)
            now = self._call("step", seconds=max(min(nearest, end) - now, 1e-3))
            for entry in self._callbacks:
                if now + 1e-9 >= entry[2]:
                    entry[2] = now + entry[1]
                    entry[0]()
        return now

    def on_step(self, callback: Callable[[], Any], every: float = 0.02) -> None:
        """Call `callback()` every `every` simulated seconds while step() lets time pass (robot.on_step wraps it)."""
        if every < 0.002:
            raise ScriptError("on_step(every=...) is at least 0.002 s, one physics step")
        self._callbacks.append([callback, float(every), self.time + float(every)])

    # ---- robots as one interface ----
    def sensor_names(self, robot: str) -> list[str]:
        """The names of the sensors a robot has (add_sensor's, the builder's and the templates')."""
        return self._call("sensor_names", robot=robot)

    def robots(self) -> list[str]:
        """The names of the world's robots."""
        return self._call("robots")

    def robot(self, name: str):
        """A robot as one interface whatever it is (see robotforge.robot): robot.sensors, robot.move(), robot.arm,
        robot.gripper, robot.on_step(). The same script runs on a template dog and on a Go2."""
        from robotforge.robot import Robot

        return Robot(self, name, self._call("describe", robot=name))

    def show_flow(
        self,
        name: str = "tree",
        nodes: list[dict[str, Any]] | None = None,
        status: str = "",
        notes: dict[str, str] | None = None,
    ) -> None:
        """Show a behavior tree or state machine in the viewer (robotforge.flow does this for Tree.run): `nodes` are
        {"id", "parent", "kind", "name"} (leave out to keep the ones shown), `status` one letter per node: . not run,
        r running, s succeeded, f failed; `notes` short texts by node id. nodes=[] takes the panel away."""
        self._call("show_flow", name=name, nodes=nodes, status=status, notes=notes or {})

    # ---- demonstrations: record what a robot does, replay it, export it (robot.record and replay wrap these) ----
    def start_recording(
        self,
        robot: str,
        name: str | None = None,
        rate_hz: float = 20.0,
        sensors: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Keep what `robot` does from now on: joint positions, actuator targets, base pose, sensor readings (the cheap
        ones; sensors=[...] names them) and the commands it is given, at rate_hz frames a simulated second."""
        return self._call("demo_start", robot=robot, name=name, rate_hz=rate_hz, sensors=sensors, overwrite=overwrite)

    def stop_recording(self, name: str | None = None) -> dict[str, Any]:
        """End the recording (the only one going, or `name`): it is saved in the world's demonstrations folder."""
        return self._call("demo_stop", name=name)

    def replay(self, name: str, robot: str | None = None, mode: str = "commands", wait: bool = True) -> dict[str, Any]:
        """Play a demonstration on `robot` (default: the one that recorded it, or another of its kind). mode "commands"
        gives the recorded commands again at their times, "joints" plays the actuator targets back. wait=True returns
        when it is over (time passes meanwhile), else it plays as time passes through step()."""
        return self._call("demo_replay", name=name, robot=robot, mode=mode, wait=wait)

    def export_demo(self, name: str, format: str = "json", to: str | None = None) -> dict[str, Any]:
        """Write a demonstration as "json" or "npz" in the world's demonstrations folder; the result has the path."""
        return self._call("demo_export", name=name, format=format, to=to)

    def demos(self) -> dict[str, Any]:
        """The demonstrations kept and what is being recorded or replayed."""
        return self._call("demo_list")

    def start_controller(self, robot: str, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Hand `robot` to a plugin's controller (see the plugins tool); it runs until stop(robot)."""
        return self._call("start_controller", robot=robot, name=name, params=params or {})

    # ---- reading the world ----
    def position(self, name: str) -> list[float]:
        """[x, y, z] in meters of an object, a robot (its base) or any body like "dog/trunk"."""
        return self._call("position", name=name)

    def heading(self, robot: str) -> float:
        """Which way a robot faces, in degrees counter-clockwise from +x."""
        return self._call("heading", name=robot)

    def joints(self, robot: str) -> dict[str, float]:
        """Joint angles (radians) or slide positions (meters) of a robot."""
        return self._call("joints", robot=robot)

    def sensors(self, robot: str) -> dict[str, Any]:
        """Sensor readings of a robot (IMU, touch, tool position...); rangefinders and lidar rays read past the robot's
        own parts (-1, or their full range: nothing within it)."""
        return self._call("sensors", robot=robot)

    def sensor(self, robot: str, name: str, full: bool = False) -> dict[str, Any]:
        """One sensor's reading by name (sensors added with add_sensor, the builder's and the templates'), the
        same as read_sensors gives, e.g. sim.sensor("go2", "lidar")["ahead"]["distance_m"]. full=True adds a
        lidar's every range ("ranges_m", ring by ring) and a 3D lidar's points."""
        return self._call("sensor", robot=robot, name=name, full=full)

    # ---- acting ----
    def set_controls(self, controls: dict[str, float]) -> None:
        """Set actuator targets, e.g. {"arm/shoulder": 0.5}. Names come from describe_model."""
        self._call("set_controls", controls=controls)

    def move(self, robot: str, speed: float = 0.2, turn: float = 0.0) -> None:
        """Start walking (quadruped or humanoid) or driving (rover); keeps going until stop(). m/s and rad/s."""
        self._call("move", robot=robot, speed=speed, turn=turn)

    def pose(self, robot: str, name: str) -> bool:
        """A humanoid strikes a pose: "wave", "squat", "arms_up", "bow", "look_around" or "stand".
        Waits until it is done; True if the robot is still upright."""
        return self._call("pose", robot=robot, name=name)

    def stop(self, robot: str) -> None:
        self._call("stop", robot=robot)

    def move_arm(self, robot: str, position: list[float], gripper: str | None = None) -> list[float]:
        """Move an arm's gripper to [x, y, z] pointing down; waits until it arrives. Returns tool position."""
        return self._call("move_arm", robot=robot, position=position, gripper=gripper)

    def gripper(self, robot: str, state: str) -> None:
        """ "open" or "close"; waits until done."""
        self._call("gripper", robot=robot, state=state)

    def pick_and_place(self, robot: str, obj: str, target: str | list[float]) -> bool:
        """Pick `obj` up and put it on/in `target` (object name or [x, y]); waits. True if it landed there."""
        return self._call("pick_and_place", robot=robot, object=obj, target=target)

    def run_policy(self, robot: str, name: str, task_params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Hand `robot` to a trained policy (see list_policies); it keeps driving while time passes, until
        stop(robot). task_params changes the goal, e.g. {"target": "cube"} for a reach policy."""
        return self._call("run_policy", robot=robot, name=name, task_params=task_params)


def connect() -> Sim:
    """Connect to the playground that launched this script."""
    port = os.environ.get(ENV_PORT)
    token = os.environ.get(ENV_TOKEN, "")
    if not port:
        raise ScriptError("run this script from robotforge (run_script), not directly")
    sock = socket.create_connection(("127.0.0.1", int(port)), timeout=None)
    sim = Sim(sock)
    sim._call("hello", token=token)
    return sim
