"""A walk a robot learned for itself, kept with its world and played on command.

The RL II lesson teaches any freely moving robot a rhythm per joint (control/cpg.py). Keeping one writes
it to walks/<robot>.json in the world's folder: the rhythm's numbers, the joint each belongs to (without
the robot's name, so the file does not care what the robot is called in the lesson) and how fast it went.
From then on `move` and the keyboard drive that robot with it, even when no gait code knows its body.

A rhythm is learned going forward, at its own pace, from standing still. After keeping it the robot also
learns to turn on the spot, one rhythm each way (`learn_turns`), from standing still and right out of its walk,
saved in the same file once learned. Commands switch between them in a fraction of a second: standing still, a
rhythm starts as it was learned; a turn out of the walk starts at once, faded in from the stride, as it learned
to; any other switch while moving first steadies the robot on its home pose until it is calm, then fades the
new rhythm in. It cannot back up or change speed, and the status says so.
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

from robotforge.control.cpg import (
    CONTROL_HZ,
    FADE,
    FELL,
    SENSES,
    Body,
    CPGError,
    Reward,
    Rhythm,
    Start,
    Trainer,
    fade_in,
    trunk_senses,
)
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
    turns_from_walking: bool = False  # the turns also learned to start right out of the walk (older files: no)

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
TURN_STARTS = 2  # random starts per side; the one that turns best from its worst beginning is kept
# Moments of the walk (s after it starts from standing) a turn also learns to begin at: its first steps, then its
# stride at different points of a step (0.3 s apart at the usual 2 Hz). Half of each round's tries begin at one.
WALK_STARTS = (0.5, 0.8, 1.1, 1.4, 1.7, 2.0)
# Standing still (trunk and joint speeds under these), a rhythm starts exactly as it was learned.
STILL_SPEED = 0.02  # m/s
STILL_SPIN = 0.1  # rad/s
STILL_JOINT = 0.1  # rad/s
# Before a switch it never learned (out of a turn, or into one with turns learned only from standing) the robot
# steadies on its home pose until its trunk turns slower than CALM_SPIN and leans less than CALM_TILT, at most
# CALM_MAX; standing on home until perfectly still took the whole of a 1 s command.
CALM_SPIN = 2.0  # rad/s, the trunk's angular speed
CALM_TILT = math.radians(25.0)  # a walking A1 rocks up to 20 degrees
CALM_MAX = 0.6  # s


def learn_turns(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """Teach the robot (alone in `model`, never the live world) to turn on the spot each way, by the same random
    search as the walk with a reward for the angle turned, from standing still and from moments of its walk, and
    return the walk with both turns added."""
    body = Body.find(model, robot)
    names = joint_names(model, body)
    if sorted(names) != sorted(walk.joints):
        raise ValueError(f"{robot}'s joints are not the ones its walk was learned for")
    learned: dict[str, list[float]] = {}
    rates = []
    order = [names.index(j) for j in walk.joints]  # the trainer's order is the model's; the file keeps the walk's
    n = body.joints
    in_model_order = [walk.joints.index(j) for j in names]
    forward = Rhythm(_reordered(walk.theta, in_model_order, walk.reflexes), walk.freq, walk.reflexes)
    # Learned only from standing still, a turn asked for mid-stride either waited for the robot to stand still (a
    # 1 s turn right after walking turned the imported A1 16 degrees of the 88 it promised) or started from a
    # pose and speed it never met (3 A1 walks tipped past 45 degrees in 64 of 561 random switches started at once).
    # So half the tries begin out of the walk.
    walking = Trainer(model, robot, Reward(), walk.seconds, walk.freq).moments(forward, WALK_STARTS)
    starts: list[Start | None] = [None] * len(walking) + list(walking)
    for side, sign in (("turn_left", 1.0), ("turn_right", -1.0)):
        trainer = Trainer(model, robot, Reward(turn=sign), walk.seconds, walk.freq)
        # Random search lands somewhere different from every start (one tripod turn came out at 16 deg/s, the
        # next at 87): two starts each way, keeping the one whose worst beginning (standing or walking) is best.
        best: tuple[float, Rhythm, dict[str, float]] | None = None
        for start in range(TURN_STARTS):
            rhythm = Rhythm.still(body.joints, walk.freq)
            trainer.train(
                rhythm, iterations=TURN_ROUNDS, seed=seed + 10 * start + (1 if sign > 0 else 2), starts=starts
            )
            standing = trainer.rollout(rhythm)
            worst = min([standing["reward"]] + [trainer.rollout(rhythm, start=s)["reward"] for s in walking])
            if best is None or worst > best[0]:
                best = (worst, rhythm, standing)
        assert best is not None
        _, rhythm, standing = best
        theta = np.concatenate([rhythm.theta[k * n : (k + 1) * n][order] for k in range(3)])
        learned[side] = [float(v) for v in theta]
        rates.append(max(0.0, sign * standing["turned"]) / walk.seconds)
    return walk.model_copy(update={**learned, "turn_rate": min(rates), "turns_from_walking": True})


def _reordered(theta: list[float], order: list[int], reflexes: bool) -> np.ndarray:
    """A rhythm's numbers with its joints put in `order`: offsets, amplitudes and phases, then any reflex gains."""
    n = len(order)
    parts = [np.asarray(theta[k * n : (k + 1) * n])[order] for k in range(3)]
    if reflexes:
        parts.append(np.asarray(theta[3 * n :]).reshape(n, SENSES)[order].ravel())
    return np.concatenate(parts)


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
        self.playing: str | None = None  # the mode the joints follow now (its rhythm, or home while steadying)
        self.t0: float | None = None  # when the playing rhythm started; None while standing or steadying
        self.steady_from: float | None = None  # steadying on home since then, before the asked rhythm starts
        self.fade_from: np.ndarray | None = None  # the joint targets of the last switch, fading out
        self.fade_t0 = 0.0
        self.walked_until = -math.inf  # sim time the walk last played
        self.clock = -math.inf  # sim time of the last update
        self.tilt_cos = 1.0

    @property
    def walking(self) -> bool:
        return self.mode is not None

    def command(self, speed: float, turn: float) -> None:
        """A turn wins over going forward (it turns on the spot); without learned turns it is ignored. The switch
        happens on the next update, so a stop followed at once by another command (as `move` gives) is skipped."""
        if turn > 0 and "left" in self.rhythms:
            mode = "left"
        elif turn < 0 and "right" in self.rhythms:
            mode = "right"
        elif speed > 0:
            mode = "forward"
        else:
            mode = None
        self.mode = mode
        self.asked = (speed, turn)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        now = float(data.time)
        self.tilt_cos = float(data.xmat[self.body.trunk][8])
        if now < self.clock:  # the clock was turned back (a reset): times kept from before lie in the future
            self.playing, self.t0, self.steady_from, self.fade_from = None, None, None, None
            self.walked_until = -math.inf
        self.clock = now
        if self.mode != self.playing:
            self._switch(data, now)
        if self.steady_from is not None and (self._calm(data) or now - self.steady_from >= CALM_MAX):
            self.steady_from = None
            self._start(data, now, fade=True)
        if self.t0 is None:  # standing, or steadying before a rhythm starts
            target = self.home
        else:
            assert self.playing is not None
            rhythm = self.rhythms[self.playing]
            senses = trunk_senses(data, self.body) if rhythm.reflexes else None
            target = rhythm.targets(self.home, now - self.t0, senses=senses)
            if self.playing == "forward":
                self.walked_until = now
        if self.fade_from is not None:
            target = fade_in(self.fade_from, target, now - self.fade_t0)
            if now - self.fade_t0 >= FADE:
                self.fade_from = None
        data.ctrl[self.actuators] = np.clip(target, self.lower, self.upper)

    def _switch(self, data: mujoco.MjData, now: float) -> None:
        """The asked mode takes over. Each rhythm was learned from standing still, and the turns (in files that say
        so) also right out of the walk; those start at once. Any other switch while moving steadies on home first.
        When every switch waited until the robot stood still, a 1 s turn right after walking turned the imported A1
        16 degrees of the 88 it promised. Over 6 learned A1 walks (scripts/rhythm_bench.py switch --stress 4) that
        turn now reaches 67 to 100 percent of the promise (12 to 107 before), and 1440 random switches made it
        fall 13 times (94 before); the template dog fell 11 times (none before) and the starfish never."""
        out_of_walk = now - self.walked_until <= FADE + 1.0 / CONTROL_HZ
        self.playing, self.steady_from = self.mode, None
        if self.mode is None:  # stop: fade to the home pose
            self.t0 = None
            self._fade(data, now)
        elif self._still(data):
            self._start(data, now, fade=False)
        elif self.mode != "forward" and out_of_walk and self.walk.turns_from_walking:
            self._start(data, now, fade=True)
        else:
            self.t0, self.steady_from = None, now
            self._fade(data, now)

    def _start(self, data: mujoco.MjData, now: float, fade: bool) -> None:
        self.t0 = now
        if fade:
            self._fade(data, now)
        else:
            self.fade_from = None

    def _fade(self, data: mujoco.MjData, now: float) -> None:
        self.fade_from, self.fade_t0 = data.ctrl[self.actuators].copy(), now

    def _still(self, data: mujoco.MjData) -> bool:
        b = self.body
        return (
            float(np.linalg.norm(data.qvel[b.qvel[:3]])) < STILL_SPEED
            and float(np.linalg.norm(data.qvel[b.qvel[3:6]])) < STILL_SPIN
            and float(np.abs(data.qvel[b.dadr]).max()) < STILL_JOINT
        )

    def _calm(self, data: mujoco.MjData) -> bool:
        spin = float(np.linalg.norm(data.qvel[self.body.qvel[3:6]]))
        return spin < CALM_SPIN and self.tilt_cos > math.cos(CALM_TILT)

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
