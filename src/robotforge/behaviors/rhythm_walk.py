"""A walk a robot learned for itself, kept with its world and played on command.

The RL II lesson teaches any freely moving robot a rhythm per joint (control/cpg.py). Keeping one writes
it to walks/<robot>.json in the world's folder: the rhythm's numbers, the joint each belongs to (without
the robot's name, so the file does not care what the robot is called in the lesson) and how fast it went.
From then on `move` and the keyboard drive that robot with it, even when no gait code knows its body.

A rhythm is learned going forward only, at its own pace: any forward command plays it from the start,
anything else holds the home pose. It cannot turn or back up, and the status says so.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from robotforge.control.cpg import CONTROL_HZ, FELL, SENSES, Body, CPGError, Rhythm, trunk_senses
from robotforge.design.world import NAME_RE

WALKS_DIR = "walks"
FORMAT = 1


class SavedWalk(BaseModel):
    """What a learned walk keeps: the rhythm and what it did in training."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    format: int = FORMAT
    robot: str
    joints: list[str] = Field(min_length=2)  # the rhythm's joint order, names without "<robot>/"
    freq: float = Field(gt=0, le=10)  # Hz
    reflexes: bool = False
    theta: list[float]
    speed: float = Field(ge=0)  # m/s it covered in training
    distance: float  # m in `seconds`
    seconds: float = Field(gt=0)
    survived: int | None = None  # shoves it stayed up for, of `shoves`
    shoves: int | None = None

    @model_validator(mode="after")
    def _shape(self) -> SavedWalk:
        if not NAME_RE.match(self.robot):
            raise ValueError(f"robot name {self.robot!r} is not a robotforge name")
        if self.format != FORMAT:
            raise ValueError(f"walk file format {self.format}; this robotforge reads format {FORMAT}")
        want = (3 + (SENSES if self.reflexes else 0)) * len(self.joints)
        if len(self.theta) != want:
            raise ValueError(f"{len(self.theta)} rhythm numbers for {len(self.joints)} joints; expected {want}")
        if not all(math.isfinite(v) for v in self.theta):
            raise ValueError("the rhythm has numbers that are not finite")
        return self

    def describe(self) -> dict[str, Any]:
        """What `robots` reports for a robot that has one."""
        out: dict[str, Any] = {"speed": round(self.speed, 3), "forward_only": True, "walker": "rhythm"}
        if self.survived is not None:
            out["stayed_up"] = f"{self.survived} of {self.shoves} shoves"
        return out


def walk_file(world_dir: Path, robot: str) -> Path:
    if not NAME_RE.match(robot):  # names are the file name: keep them to safe characters
        raise ValueError(f"robot name {robot!r} is not a robotforge name")
    return world_dir / WALKS_DIR / f"{robot}.json"


def save_walk(world_dir: Path, walk: SavedWalk) -> Path:
    """Write the walk next to world.yaml; a half-written file never replaces a good one."""
    path = walk_file(world_dir, walk.robot)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(walk.model_dump(), indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_walk(world_dir: Path, robot: str) -> SavedWalk | None:
    """The walk kept for `robot`, None if there is none. A damaged file raises with what is wrong."""
    path = walk_file(world_dir, robot)
    if not path.is_file():
        return None
    try:
        return SavedWalk.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError(f"{path} cannot be read as a learned walk ({exc}); delete it and keep the walk again") from exc


def joint_names(model: mujoco.MjModel, body: Body) -> list[str]:
    """The joints a rhythm drives, in its order, without the robot's name in front."""
    prefix = f"{body.robot}/"
    names = [model.joint(int(model.actuator_trnid[a, 0])).name for a in body.actuators]
    return [n.removeprefix(prefix) for n in names]


class RhythmWalker:
    """Drives a robot with a kept walk: forward plays the rhythm from its start, stop holds home."""

    rate_hz = CONTROL_HZ

    def __init__(self, model: mujoco.MjModel, robot: str, walk: SavedWalk, home_qpos: np.ndarray) -> None:
        try:
            body = Body.find(model, robot)
        except CPGError as exc:
            raise ValueError(str(exc)) from exc
        names = joint_names(model, body)
        if sorted(names) != sorted(walk.joints):
            raise ValueError(
                f"{robot}'s walk was learned for the joints {', '.join(walk.joints)}, but {robot} now has "
                f"{', '.join(names)}; learn the walk again in the RL II lesson"
            )
        order = np.array([names.index(j) for j in walk.joints])
        self.robot = robot
        self.body = body
        self.walk = walk
        self.actuators = body.actuators[order]
        self.lower, self.upper = body.lower[order], body.upper[order]
        self.home = np.asarray(home_qpos)[body.qadr[order]].copy()
        self.rhythm = Rhythm(np.array(walk.theta), walk.freq, walk.reflexes)
        self.max_speed = walk.speed
        self.walking = False
        self.asked: tuple[float, float] = (0.0, 0.0)
        self.t0: float | None = None
        self.tilt_cos = 1.0

    def command(self, speed: float, turn: float) -> None:
        go = speed > 0
        if go and not self.walking:
            self.t0 = None  # the rhythm starts from its beginning at the next tick, as in training
        self.walking = go
        self.asked = (speed, turn)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.tilt_cos = float(data.xmat[self.body.trunk][8])
        if not self.walking:
            data.ctrl[self.actuators] = np.clip(self.home, self.lower, self.upper)
            return
        if self.t0 is None:
            self.t0 = float(data.time)
        senses = trunk_senses(data, self.body) if self.rhythm.reflexes else None
        target = self.rhythm.targets(self.home, float(data.time) - self.t0, senses=senses)
        data.ctrl[self.actuators] = np.clip(target, self.lower, self.upper)

    @property
    def fallen(self) -> bool:
        return self.tilt_cos < FELL

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "controller": "learned rhythm",
            "robot": self.robot,
            "walking": self.walking,
            "speed": round(self.max_speed, 3) if self.walking else 0.0,
        }
        speed, turn = self.asked
        if turn != 0.0 or speed < 0.0 or (speed > 0.0 and abs(speed - self.max_speed) > 0.02):
            out["note"] = (
                f"this walk was learned going forward only, at its own pace (about {self.max_speed:.2f} m/s): "
                "it cannot turn, back up or change speed"
            )
        return out
