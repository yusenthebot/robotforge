"""A walk a robot learned for itself, kept with its world and played on command.

The RL II lesson teaches any freely moving robot a rhythm per joint (control/cpg.py). Keeping one writes
it to walks/<robot>.json in the world's folder: the rhythm's numbers, the joint each belongs to (without
the robot's name, so the file does not care what the robot is called in the lesson) and how fast it went.
From then on `move` and the keyboard drive that robot with it, even when no gait code knows its body.

A rhythm is learned going forward, at its own pace: any forward command plays it from the start. After
keeping it the robot also learns to turn on the spot, one rhythm each way (`learn_turns`), saved in the same
file once learned; a turn command plays those. It cannot back up or change speed, and the status says so.
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

from robotforge.control.cpg import CONTROL_HZ, FELL, SENSES, Body, CPGError, Reward, Rhythm, Trainer, trunk_senses
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
    # Turning on the spot, learned after the walk (3 numbers per joint, no reflexes, the walk's frequency).
    turn_left: list[float] | None = None
    turn_right: list[float] | None = None
    turn_rate: float | None = Field(None, ge=0)  # rad/s, the slower of the two turns

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
        for side in (self.turn_left, self.turn_right):
            if side is not None and (len(side) != 3 * len(self.joints) or not all(math.isfinite(v) for v in side)):
                raise ValueError(f"a turn rhythm needs {3 * len(self.joints)} finite numbers")
        return self

    @property
    def turns(self) -> bool:
        return self.turn_left is not None and self.turn_right is not None

    def describe(self) -> dict[str, Any]:
        """What `robots` reports for a robot that has one."""
        out: dict[str, Any] = {"speed": round(self.speed, 3), "turns": self.turns, "walker": "rhythm"}
        if self.turns and self.turn_rate is not None:
            out["turn_deg_s"] = round(math.degrees(self.turn_rate))
        if self.survived is not None:
            out["stayed_up"] = f"{self.survived} of {self.shoves} shoves"
        return out


TURN_ROUNDS = 30  # search rounds per start: 6 s on a laptop for the starfish
TURN_STARTS = 2  # random starts per side; the better one is kept
# A rhythm starts once the robot stands still, as in training: trunk and joint speeds under these, or after
# SETTLE_MAX seconds of holding the home pose at most.
STILL_SPEED = 0.02  # m/s
STILL_SPIN = 0.1  # rad/s
STILL_JOINT = 0.1  # rad/s
SETTLE_MAX = 1.0  # s


def learn_turns(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """Teach the robot (alone in `model`, never the live world) to turn on the spot each way, by the same random
    search as the walk with a reward for the angle turned, and return the walk with both turns added."""
    body = Body.find(model, robot)
    names = joint_names(model, body)
    if sorted(names) != sorted(walk.joints):
        raise ValueError(f"{robot}'s joints are not the ones its walk was learned for")
    learned: dict[str, list[float]] = {}
    rates = []
    order = [names.index(j) for j in walk.joints]  # the trainer's order is the model's; the file keeps the walk's
    n = body.joints
    for side, sign in (("turn_left", 1.0), ("turn_right", -1.0)):
        trainer = Trainer(model, robot, Reward(turn=sign), walk.seconds, walk.freq)
        # Random search lands somewhere different from every start (one tripod turn came out at 16 deg/s, the
        # next at 87): two starts each way, the better one kept.
        best: tuple[float, Rhythm, dict[str, float]] | None = None
        for start in range(TURN_STARTS):
            rhythm = Rhythm.still(body.joints, walk.freq)
            trainer.train(rhythm, iterations=TURN_ROUNDS, seed=seed + 10 * start + (1 if sign > 0 else 2))
            result = trainer.rollout(rhythm)
            if best is None or result["reward"] > best[0]:
                best = (result["reward"], rhythm, result)
        assert best is not None
        _, rhythm, result = best
        theta = np.concatenate([rhythm.theta[k * n : (k + 1) * n][order] for k in range(3)])
        learned[side] = [float(v) for v in theta]
        rates.append(max(0.0, sign * result["turned"]) / walk.seconds)
    return walk.model_copy(update={**learned, "turn_rate": min(rates)})


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
    """Drives a robot with a kept walk: forward plays the walk, a turn plays a turn (once learned), each from
    its start as in training, standing still at home; stop holds the home pose."""

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
        self.rhythms = {"forward": Rhythm(np.array(walk.theta), walk.freq, walk.reflexes)}
        if walk.turns:
            self.rhythms["left"] = Rhythm(np.array(walk.turn_left), walk.freq)
            self.rhythms["right"] = Rhythm(np.array(walk.turn_right), walk.freq)
        self.max_speed = walk.speed
        self.mode: str | None = None  # forward, left, right, or None: standing
        self.asked: tuple[float, float] = (0.0, 0.0)
        self.t0: float | None = None  # when the playing rhythm started; None while standing or settling
        self.settle_from: float | None = None
        self.tilt_cos = 1.0

    @property
    def walking(self) -> bool:
        return self.mode is not None

    def command(self, speed: float, turn: float) -> None:
        """A turn wins over going forward (it turns on the spot); without learned turns it is ignored."""
        if turn > 0 and "left" in self.rhythms:
            mode = "left"
        elif turn < 0 and "right" in self.rhythms:
            mode = "right"
        elif speed > 0:
            mode = "forward"
        else:
            mode = None
        if mode != self.mode:
            self.t0 = self.settle_from = None  # the new rhythm starts from its beginning, as in training
        self.mode = mode
        self.asked = (speed, turn)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.tilt_cos = float(data.xmat[self.body.trunk][8])
        if self.mode is not None and self.t0 is None and self._ready(data):
            self.t0 = float(data.time)
        if self.t0 is None:  # standing, or settling before a rhythm starts
            data.ctrl[self.actuators] = np.clip(self.home, self.lower, self.upper)
            return
        assert self.mode is not None
        rhythm = self.rhythms[self.mode]
        senses = trunk_senses(data, self.body) if rhythm.reflexes else None
        target = rhythm.targets(self.home, float(data.time) - self.t0, senses=senses)
        data.ctrl[self.actuators] = np.clip(target, self.lower, self.upper)

    def _ready(self, data: mujoco.MjData) -> bool:
        """Whether the asked rhythm may start. Each was learned from standing still at home. Started mid-stride
        (walking straight into a turn, one turn into the other, a turn into walking), the imported A1 fell in 4 of
        10 switches over two learned walks; holding its home pose until it was still first (at once if it already
        is), in none."""
        now = float(data.time)
        if self.settle_from is None:
            self.settle_from = now
        b = self.body
        still = (
            float(np.linalg.norm(data.qvel[b.qvel[:3]])) < STILL_SPEED
            and float(np.linalg.norm(data.qvel[b.qvel[3:6]])) < STILL_SPIN
            and float(np.abs(data.qvel[b.dadr]).max()) < STILL_JOINT
        )
        return still or now - self.settle_from >= SETTLE_MAX

    @property
    def fallen(self) -> bool:
        return self.tilt_cos < FELL

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "controller": "learned rhythm",
            "robot": self.robot,
            "walking": self.mode == "forward",
            "turning": self.mode if self.mode in ("left", "right") else None,
            "speed": round(self.max_speed, 3) if self.mode == "forward" else 0.0,
        }
        speed, turn = self.asked
        notes = []
        if turn != 0.0 and "left" not in self.rhythms:
            notes.append("it has not learned to turn yet (keep its walk again in the RL II lesson to learn turns)")
        elif turn != 0.0 and speed > 0.0:
            notes.append("it turns on the spot, then walks when the turn command stops")
        if speed < 0.0:
            notes.append("it cannot back up")
        if self.mode == "forward" and abs(speed - self.max_speed) > 0.02:
            notes.append(f"it walks at its own pace, about {self.max_speed:.2f} m/s")
        if notes:
            out["note"] = "; ".join(notes)
        return out
