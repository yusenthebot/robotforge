"""A walk a robot learned for itself, kept with its world and played on command.

The RL II lesson teaches any freely moving robot a rhythm per joint (control/cpg.py). Keeping one writes
it to walks/<robot>.json in the world's folder: the rhythm's numbers, the joint each belongs to (without
the robot's name, so the file does not care what the robot is called in the lesson), the joint angles it swings
around and how fast it went. From then on `move` and the keyboard drive that robot with it, even when no gait code
knows its body.

A rhythm is learned going forward, at its own pace, from standing still. After keeping it the robot learns more on
its own, each saved in the same file once learned: a walk that curves learns to hold its heading
(`learn_steering`), and it learns to turn on the spot, one rhythm each way from standing still, then each tuned
again to begin straight out of its walk (`learn_turns`). Standing still, a rhythm starts exactly as it was learned;
a turn asked for mid-stride starts at once, faded in from the stride, as its walking version learned to; any other
switch while moving first steadies the robot on its home pose until it is calm, then fades the new rhythm in. It
cannot back up or change speed, and the status says so.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from robotforge.control.cpg import (
    CONTROL_HZ,
    FADE,
    FELL,
    SENSES,
    STEER_SPAN,
    Body,
    CPGError,
    Reward,
    Rhythm,
    Start,
    Steering,
    Trainer,
    fade_in,
    trunk_senses,
    trunk_yaw,
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
    # Added later; files without them load and play as before.
    home: dict[str, float] | None = None  # joint -> the angle (rad) the rhythms swing around, as when learned
    steer: list[float] | None = None  # how the walk holds its heading (cpg.Steering): 3 numbers per joint
    turn_left_walking: list[float] | None = None  # the turns tuned again to begin straight out of the walk
    turn_right_walking: list[float] | None = None
    # How it switches rhythms while moving: "fade" (kept since steering and walking turns); None, as files kept
    # before: stop holds home at once, and another rhythm waits on home until the robot stands still.
    switching: Literal["fade"] | None = None

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
        for side in (self.turn_left, self.turn_right, self.turn_left_walking, self.turn_right_walking):
            if side is not None and not _finite(side, 3 * len(self.joints)):
                raise ValueError(f"a turn rhythm needs {3 * len(self.joints)} finite numbers")
        if self.steer is not None and not _finite(self.steer, 3 * len(self.joints)):
            raise ValueError(f"steering needs {3 * len(self.joints)} finite numbers")
        if self.home is not None and (sorted(self.home) != sorted(self.joints) or not _finite(self.home.values())):
            raise ValueError("the home pose needs a finite angle for each of the rhythm's joints and no others")
        return self

    @property
    def turns(self) -> bool:
        return self.turn_left is not None and self.turn_right is not None

    @property
    def turns_from_walking(self) -> bool:
        return self.turn_left_walking is not None and self.turn_right_walking is not None

    def describe(self) -> dict[str, Any]:
        """What `robots` reports for a robot that has one."""
        out: dict[str, Any] = {"speed": round(self.speed, 3), "turns": self.turns, "walker": "rhythm"}
        if self.steer is not None:
            out["holds_heading"] = True
        if self.turns and self.turn_rate is not None:
            out["turn_deg_s"] = round(math.degrees(self.turn_rate))
        if self.survived is not None:
            out["stayed_up"] = f"{self.survived} of {self.shoves} shoves"
        return out


def _finite(values: Any, count: int | None = None) -> bool:
    values = list(values)
    return (count is None or len(values) == count) and all(math.isfinite(v) for v in values)


TURN_ROUNDS = 30  # search rounds per start: 6 s on a laptop for the starfish
TURN_STARTS = 2  # random starts per side; the better one is kept
# Moments of the walk (s after it starts from standing) the walking version of a turn learns to begin at: its first
# steps, then its stride at different points of a step (0.3 s apart at the usual 2 Hz).
WALK_STARTS = (0.5, 0.8, 1.1, 1.4, 1.7, 2.0)
WALKING_ROUNDS = 20  # search rounds per side tuning a turn learned from standing to begin out of the walk
# A turn's walking version is checked from more moments of the walk, a tenth of a second apart, each for TURN_CHECK
# seconds: of the tuned turn and the one from standing (faded in from the stride too), the one that falls from none
# of them and turns more begins straight out of the walk. If both fall from some moment, the robot steadies first.
# Always starting the tuned turn, 6 template dog walks fell 16 times and 6 A1 walks 11 in 810 tries of each switch
# on its own (scripts/rhythm_bench.py switch --trials 15); choosing this way, 12 and 8.
WALK_CHECKS = tuple(round(0.5 + 0.1 * k, 1) for k in range(21))  # s after the walk starts
TURN_CHECK = 2.0  # s
# A kept walk whose heading turns more than STEER_FROM in STEER_CHECK seconds (twice a training's, since a slow curve
# shows late) learns to hold it, STEER_ROUNDS rounds aiming STEER_AIMS (rad, 17 degrees to either side, so it learns
# to correct both ways), losing HOLD_WEIGHT of its progress per rad it strays on average.
STEER_FROM = math.radians(15.0)
STEER_CHECK = 8.0  # s
STEER_ROUNDS = 30
STEER_AIMS = (0.3, -0.3)
HOLD_WEIGHT = 1.0
# Playing it, the aim follows a robot spun or pushed further off than it learned to come back from (rad), and the
# steering comes in over the walk's first STEER_RAMP seconds. At full strength from the start it threw an A1 still
# swinging out of a turn from side to side until it fell: of 810 tries of each switch on its own (6 A1 walks,
# scripts/rhythm_bench.py switch --trials 15), 8 fell at once, 7 eased in over 1 s and 4 over 2 s.
STEER_LEASH = max(abs(a) for a in STEER_AIMS)
STEER_RAMP = 2.0  # s
# Standing still (trunk and joint speeds under these), a rhythm starts exactly as it was learned.
STILL_SPEED = 0.02  # m/s
STILL_SPIN = 0.1  # rad/s
STILL_JOINT = 0.1  # rad/s
# Before a switch it never learned (out of a turn, or into a turn without a walking version) the robot steadies:
# its joints fade to the home pose and hold it until its trunk turns slower than CALM_SPIN and leans less than
# CALM_TILT, at most CALM_MAX, and stop fades to home the same way. Holding home at once and until the robot stood
# still (SETTLE_MAX at most), as walks kept before still do, took the whole of a 1 s command, and in 810 tries of
# each switch on its own 6 A1 walks fell 119 times; steadying until calm and fading, 4 (6 template dog walks: none
# and 8).
CALM_SPIN = 2.0  # rad/s, the trunk's angular speed
CALM_TILT = math.radians(25.0)  # a walking A1 rocks up to 20 degrees
CALM_MAX = 0.6  # s
SETTLE_MAX = 1.0  # s


class _Joints:
    """A robot's driven joints in the model's order (the trainer's) and in its walk file's."""

    def __init__(self, model: mujoco.MjModel, robot: str, walk: SavedWalk) -> None:
        self.body = Body.find(model, robot)
        names = joint_names(model, self.body)
        if sorted(names) != sorted(walk.joints):
            raise ValueError(f"{robot}'s joints are not the ones its walk was learned for")
        self.to_file = [names.index(j) for j in walk.joints]
        self.to_model = [walk.joints.index(j) for j in names]
        self.walk = walk

    def rhythm(self, theta: list[float], reflexes: bool = False) -> np.ndarray:
        return _reordered(theta, self.to_model, reflexes)

    def to_saved(self, theta: np.ndarray) -> list[float]:
        n = self.body.joints
        return [float(v) for v in np.concatenate([theta[k * n : (k + 1) * n][self.to_file] for k in range(3)])]

    def home(self) -> np.ndarray | None:
        """The angles the walk swings around in the model's order, if its file says."""
        home = self.walk.home
        return None if home is None else np.array([home[self.walk.joints[i]] for i in self.to_model])

    def forward(self) -> Rhythm | Steering:
        walk = self.walk
        rhythm = Rhythm(self.rhythm(walk.theta, walk.reflexes), walk.freq, walk.reflexes)
        return rhythm if walk.steer is None else Steering(rhythm, self.rhythm(walk.steer))


def _check(model: mujoco.MjModel, robot: str, walk: SavedWalk) -> tuple[Trainer, Rhythm, dict[str, float]]:
    """The walk played as it was learned for STEER_CHECK seconds from standing, the robot alone in `model`."""
    joints = _Joints(model, robot, walk)
    forward = Rhythm(joints.rhythm(walk.theta, walk.reflexes), walk.freq, walk.reflexes)
    check = Trainer(model, robot, Reward(), STEER_CHECK, walk.freq, home=joints.home())
    return check, forward, check.rollout(forward)


def heading_drift(model: mujoco.MjModel, robot: str, walk: SavedWalk) -> float | None:
    """How far the walk's heading turns (rad, positive to the left) in STEER_CHECK seconds from standing, played as
    it was learned (the robot alone in `model`); None if it falls."""
    loose = _check(model, robot, walk)[2]
    return None if loose["fell"] else loose["turned"]


def needs_steering(drift: float | None) -> bool:
    """Whether a walk whose heading turns `drift` (heading_drift) learns to hold it."""
    return drift is not None and abs(drift) >= STEER_FROM


def learn_steering(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """Teach a walk that curves (alone in `model`, never the live world) to hold its heading, by the same random
    search with the walk itself fixed (cpg.Steering), and return it with the steering added. A walk that goes
    straight, falls, or is not helped (it scores no better over STEER_CHECK seconds) comes back as it was. The
    reward's heading term straightened most walks, but not the imported A1's: open loop, its walks turned 46 degrees
    in 4 s on average over 20 trainings, with or without the term."""
    check, forward, loose = _check(model, robot, walk)
    if not needs_steering(None if loose["fell"] else loose["turned"]):
        return walk
    steering = Steering.still(forward, check.body.joints)
    reward = Reward(hold=HOLD_WEIGHT, aims=STEER_AIMS)
    Trainer(model, robot, reward, walk.seconds, walk.freq, home=check.home).train(
        steering, iterations=STEER_ROUNDS, seed=seed + 7
    )
    held = check.rollout(steering)
    if held["fell"] or held["reward"] <= loose["reward"]:
        return walk
    return walk.model_copy(update={"steer": _Joints(model, robot, walk).to_saved(steering.theta)})


def learn_turns(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """Teach the robot (alone in `model`, never the live world) to turn on the spot each way, by the same random
    search as the walk with a reward for the angle turned, from standing still; then tune each again to begin
    straight out of the walk. Returns the walk with the turns added."""
    joints = _Joints(model, robot, walk)
    home = joints.home()
    learned: dict[str, list[float]] = {}
    rates = []
    standing: dict[str, Rhythm] = {}
    for side, sign in (("turn_left", 1.0), ("turn_right", -1.0)):
        trainer = Trainer(model, robot, Reward(turn=sign), walk.seconds, walk.freq, home=home)
        # Random search lands somewhere different from every start (one tripod turn came out at 16 deg/s, the
        # next at 87): two starts each way, the better one kept.
        best: tuple[float, Rhythm, dict[str, float]] | None = None
        for start in range(TURN_STARTS):
            rhythm = Rhythm.still(joints.body.joints, walk.freq)
            trainer.train(rhythm, iterations=TURN_ROUNDS, seed=seed + 10 * start + (1 if sign > 0 else 2))
            result = trainer.rollout(rhythm)
            if best is None or result["reward"] > best[0]:
                best = (result["reward"], rhythm, result)
        assert best is not None
        _, rhythm, result = best
        standing[side] = rhythm
        learned[side] = joints.to_saved(rhythm.theta)
        rates.append(max(0.0, sign * result["turned"]) / walk.seconds)
    # Learned only from standing still, a turn asked for mid-stride either waited for the robot to stand still (a
    # 1 s turn right after walking turned the imported A1 16 degrees of the 88 it promised) or started from a pose
    # and speed it never met. So each is tuned again on moments of the walk and kept apart: tuning the turns
    # themselves on both left them slower to start from standing (5 of 6 walks turned less in a 1 s command after
    # the robot had stood still).
    stride = Trainer(model, robot, Reward(), walk.seconds, walk.freq, home=home)
    walking = stride.moments(joints.forward(), WALK_STARTS)
    checks = stride.moments(joints.forward(), WALK_CHECKS)
    for side, sign in (("turn_left", 1.0), ("turn_right", -1.0)):
        if not walking or not checks:  # it falls before its first moments: nothing to begin from
            break
        trainer = Trainer(model, robot, Reward(turn=sign), walk.seconds, walk.freq, home=home)
        tuned = Rhythm(standing[side].theta.copy(), walk.freq)
        trainer.train(tuned, iterations=WALKING_ROUNDS, seed=seed + 100 + (1 if sign > 0 else 2), starts=walking)
        check = Trainer(model, robot, Reward(turn=sign), TURN_CHECK, walk.freq, home=home)
        safest = _safest((tuned, standing[side]), check, checks)
        if safest is not None:
            learned[f"{side}_walking"] = joints.to_saved(safest.theta)
    return walk.model_copy(update={**learned, "turn_rate": min(rates)})


def _safest(turns: Sequence[Rhythm], check: Trainer, moments: Sequence[Start]) -> Rhythm | None:
    """Of `turns`, the one that falls from the fewest `moments` (then turns the most), None if every one falls from
    some."""
    scored = []
    for turn in turns:
        tries = [check.rollout(turn, start=m) for m in moments]
        scored.append((sum(t["fell"] for t in tries), -sum(t["reward"] for t in tries), turn))
    falls, _, best = min(scored, key=lambda s: s[:2])
    return best if falls == 0 else None


def _wrapped(angle: float) -> float:
    """`angle` (rad) brought within half a turn either way."""
    return math.atan2(math.sin(angle), math.cos(angle))


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
    """Drives a robot with a kept walk: forward plays the walk (holding the heading it started with, if it learned
    to), a turn plays a turn (once learned), each from its start as in training, standing still at home; stop goes
    back to the home pose."""

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
        # The angles the rhythms were learned around: the file's, else the world's home pose.
        if walk.home is not None:
            self.home = np.array([walk.home[j] for j in walk.joints])
        else:
            self.home = np.asarray(home_qpos)[body.qadr[order]].copy()
        # A walk kept before files said how it switches does as it always did: stop holds home at once, and a rhythm
        # asked for while moving waits on home until the robot stands still. Switched the new way, the walk the
        # imported A1 kept first ended on its back in 7 of 8 move sequences with 0 to 5 s between the calls (in 1 as
        # before): a faded stop leaves its feet elsewhere, and its fast turn from there tips it over.
        self.kept_before = walk.switching is None
        forward = Rhythm(np.array(walk.theta), walk.freq, walk.reflexes)
        self.rhythms: dict[str, Rhythm | Steering] = {
            "forward": forward if walk.steer is None else Steering(forward, np.array(walk.steer))
        }
        if walk.turns:
            self.rhythms["left"] = Rhythm(np.array(walk.turn_left), walk.freq)
            self.rhythms["right"] = Rhythm(np.array(walk.turn_right), walk.freq)
            for side, theta in (("left", walk.turn_left_walking), ("right", walk.turn_right_walking)):
                if theta is not None:
                    self.rhythms[f"{side} walking"] = Rhythm(np.array(theta), walk.freq)
        self.max_speed = walk.speed
        self.mode: str | None = None  # forward, left, right, or None: standing
        self.asked: tuple[float, float] = (0.0, 0.0)
        self.playing: str | None = None  # the mode the joints follow now (its rhythm, or home while steadying)
        self.rhythm: str | None = None  # the rhythm playing (a key of `rhythms`); None while standing or steadying
        self.t0: float | None = None  # when it started
        self.aim = 0.0  # rad: the heading the walk holds, where the robot faced as it started walking
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
            self.playing, self.rhythm, self.t0, self.steady_from, self.fade_from = None, None, None, None, None
            self.walked_until = -math.inf
        self.clock = now
        if self.mode != self.playing:
            self._switch(data, now)
        if self.steady_from is not None and self._steadied(data, now - self.steady_from):
            self.steady_from = None
            assert self.mode is not None
            self._start(data, now, self.mode, fade=not self.kept_before)
        if self.rhythm is None or self.t0 is None:  # standing, or steadying before a rhythm starts
            target = self.home
        else:
            rhythm = self.rhythms[self.rhythm]
            senses = trunk_senses(data, self.body) if rhythm.reflexes else None
            error = 0.0
            if self.rhythm == "forward":
                error = self._heading_error(data, now - self.t0)
                self.walked_until = now
            target = rhythm.targets(self.home, now - self.t0, senses=senses, error=error)
        if self.fade_from is not None:
            target = fade_in(self.fade_from, target, now - self.fade_t0)
            if now - self.fade_t0 >= FADE:
                self.fade_from = None
        data.ctrl[self.actuators] = np.clip(target, self.lower, self.upper)

    def _heading_error(self, data: mujoco.MjData, walked: float) -> float:
        """How far (rad) the walk should steer back, `walked` seconds after it started: the aim follows a robot
        further off than STEER_LEASH, and the correction comes in over STEER_RAMP."""
        yaw = trunk_yaw(data, self.body)
        error = _wrapped(self.aim - yaw)
        if abs(error) > STEER_LEASH:
            error = math.copysign(STEER_LEASH, error)
            self.aim = yaw + error
        return max(-STEER_SPAN, min(STEER_SPAN, error)) * min(1.0, walked / STEER_RAMP)

    def _switch(self, data: mujoco.MjData, now: float) -> None:
        """The asked mode takes over. Each rhythm was learned from standing still, and the turns' walking versions
        (in files that have them) right out of the walk; those start at once. Any other switch while moving steadies
        on home first. When every switch waited until the robot stood still, a 1 s turn right after walking turned
        the imported A1 16 degrees of the 88 it promised."""
        out_of_walk = now - self.walked_until <= FADE + 1.0 / CONTROL_HZ
        self.playing, self.steady_from = self.mode, None
        if self.mode is None:  # stop
            self.rhythm, self.t0 = None, None
            self._settle(data, now)
        elif self._still(data):
            self._start(data, now, self.mode, fade=False)
        elif self.mode != "forward" and out_of_walk and f"{self.mode} walking" in self.rhythms:
            self._start(data, now, f"{self.mode} walking", fade=True)
        else:
            self.rhythm, self.t0, self.steady_from = None, None, now
            self._settle(data, now)

    def _settle(self, data: mujoco.MjData, now: float) -> None:
        """Go to the home pose: at once for a walk kept before, else faded in from the targets of the moment."""
        if self.kept_before:
            self.fade_from = None
        else:
            self._fade(data, now)

    def _steadied(self, data: mujoco.MjData, steadying: float) -> bool:
        if self.kept_before:
            return self._still(data) or steadying >= SETTLE_MAX
        return self._calm(data) or steadying >= CALM_MAX

    def _start(self, data: mujoco.MjData, now: float, rhythm: str, fade: bool) -> None:
        self.rhythm, self.t0 = rhythm, now
        if rhythm == "forward":
            self.aim = trunk_yaw(data, self.body)
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
