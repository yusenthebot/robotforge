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
from typing import Any

ENV_PORT = "ROBOTFORGE_SCRIPT_PORT"
ENV_TOKEN = "ROBOTFORGE_SCRIPT_TOKEN"


class ScriptError(RuntimeError):
    """The playground refused a request; the message says why."""


class Sim:
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._file = sock.makefile("rw", encoding="utf-8", newline="\n")

    def _call(self, op: str, **args: Any) -> Any:
        self._file.write(json.dumps({"op": op, **args}) + "\n")
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise ScriptError("the playground closed the connection (time limit reached or run stopped)")
        reply = json.loads(line)
        if "error" in reply:
            raise ScriptError(reply["error"])
        return reply.get("result")

    # ---- time ----
    @property
    def time(self) -> float:
        """Simulated seconds since the world started."""
        return self._call("time")

    def step(self, seconds: float = 0.02) -> float:
        """Let `seconds` of simulated time pass (physics and running behaviors). Returns the new time."""
        return self._call("step", seconds=seconds)

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
        """Sensor readings of a robot (IMU, touch, tool position...)."""
        return self._call("sensors", robot=robot)

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
