"""A walk a robot learned for itself, kept with its world and played on command.

The RL II lesson teaches any freely moving robot a rhythm per joint (control/cpg.py). Keeping one writes
it to walks/<robot>.json in the world's folder: the rhythm's numbers, the joint each belongs to (without
the robot's name, so the file does not care what the robot is called in the lesson), the joint angles it swings
around and how fast it went. From then on `move` and the keyboard drive that robot with it, even when no gait code
knows its body.

A rhythm is learned going forward, at its own pace, from standing still. After keeping it the robot learns more on
its own, each saved in the same file once learned: to hold its heading and keep to the line it was asked to walk
along (`learn_steering`), and to turn on the spot,
one rhythm each way from standing still, then each tuned again to begin straight out of its walk (`learn_turns`).
Standing still, a rhythm starts exactly as it was learned; a turn asked for mid-stride, once the walk has played
WALKED_MIN, starts faded in from the stride as its walking version learned to, at once or at the next point of the
stride it was checked to start from safely (a stop asked just before is let go of); any other switch while moving
first steadies the robot on its home pose, the way checked to tip it over least (`learn_switches`): holding home until
it stands still and starting the new rhythm as learned, or fading to home until it is calm and fading the new rhythm
in. It cannot back up or change speed, and the status says so.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
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
    # Where in its stride (0 to 1, from where the walk's rhythm starts) each walking turn may begin; None: anywhere.
    turn_left_walking_at: list[float] | None = None
    turn_right_walking_at: list[float] | None = None
    # How it steadies leaving the walk (and any rhythm whose way out is not given), checked when its turns are learned:
    # "fade" to home and wait until it is calm, then fade the next rhythm in; "hold" home at once and wait until it
    # stands still, then start the next rhythm as learned. None, as files kept before: as "hold".
    switching: Literal["fade", "hold"] | None = None
    # How the robot steadies leaving each turn, checked after it is learned: "fade" its joints to home and wait until
    # it is calm, or "hold" home at once until it stands still; None: as `switching` says.
    turn_left_exit: Literal["fade", "hold"] | None = None
    turn_right_exit: Literal["fade", "hold"] | None = None
    # rad/s: what a turn asked for right after walking turns in its first second, each way, as move and the keys play
    # it (on average over the points of the stride it is asked at), measured once the switching was checked. turn_rate
    # is the rate from standing still.
    turn_left_rate_from_walk: float | None = Field(None, ge=0)
    turn_right_rate_from_walk: float | None = Field(None, ge=0)

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
        for at in (self.turn_left_walking_at, self.turn_right_walking_at):
            if at is not None and (not at or not _finite(at) or not all(0.0 <= v < 1.0 for v in at)):
                raise ValueError("a walking turn begins at stride phases from 0 up to 1")
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

    @property
    def turn_rates_from_walk(self) -> dict[str, float] | None:
        """rad/s each way in a turn's first second right after walking, if measured."""
        left, right = self.turn_left_rate_from_walk, self.turn_right_rate_from_walk
        return None if left is None or right is None else {"left": left, "right": right}

    def describe(self) -> dict[str, Any]:
        """What `robots` reports for a robot that has one."""
        out: dict[str, Any] = {"speed": round(self.speed, 3), "turns": self.turns, "walker": "rhythm"}
        if self.steer is not None:
            out["holds_heading"] = True
        if self.turns and self.turn_rate is not None:
            out["turn_deg_s"] = round(math.degrees(self.turn_rate))
        from_walk = self.turn_rates_from_walk
        if self.turns and from_walk is not None:
            out["turn_deg_s_from_walk"] = {side: round(math.degrees(rate)) for side, rate in from_walk.items()}
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
# Search rounds per side tuning a turn learned from standing to begin out of the walk. With 20, a 1 s left turn out
# of the walk reached 67% of the rate on average over 90 tries on 6 template dog walks and 70% on 6 A1 walks; with 40,
# 72% and 84% (scripts/rhythm_bench.py switch --trials 15).
WALKING_ROUNDS = 40
# A turn begins straight out of the walk only once the walk has played WALKED_MIN; asked sooner, the robot steadies
# first, as before turns learned to begin out of the walk. In random runs of 60 commands (each 0.3 to 1.5 s, the next
# at once), where a walk is often short and follows steadying out of a turn, 6 template dog walks with no such gate fell
# 16 times in 14400 commands against 7 for the same walks switched as before, 10 of them in or after a turn begun out
# of a walk 0.06 to 1.04 s old. Gated at 1 s (the earliest moment the walking turns were checked from), 80 times in
# 86400 commands against 66 for the walks kept and switched as before; leaving out one walk whose turn from standing
# tips it over either way, 21 against 10, 8 of the 11 more in or after a turn begun out of a walk 1.0 to 1.47 s old.
# Gated at 1.5 s, 75 (13 leaving that walk out), none of them in or after a turn begun out of the walk.
WALKED_MIN = 1.5  # s
# A turn's walking version is checked from more moments of the walk as move plays it, from WALKED_MIN to 5 s at every
# update (at 2 Hz, each of 25 points of the stride 7 times), each for TURN_CHECK seconds. Of the tuned turn and the one
# from standing (faded in from the stride too), the one expected to turn the most in FIRST_SECOND begins out of the
# walk, at any point of the stride if it fell from none of the moments, else only at the points it never fell from (at
# least MIN_SAFE of them), the robot walking on until the next one. If neither, the robot steadies first. Checked on
# 21 moments of the walk as the trainer played it, an A1 left turn kept for 1 of the 5 points fell in 4 of 15 tries
# straight out of the walk; checked like this, the left turns of 6 A1 walks fell in 1 of 90. Checked a tenth of a
# second apart (5 points of the stride), a template dog walk's left turn was kept for any point and fell in 3 of its
# 100 bench sequences, begun at 0.28 of its stride; every update, it fell from 4 of 176 moments, at 0.28 to 0.36.
WALK_CHECKS = tuple(round(WALKED_MIN + k / CONTROL_HZ, 2) for k in range(round((5.0 - WALKED_MIN) * CONTROL_HZ) + 1))
MIN_SAFE = 0.5  # share of the points of the stride checked
TURN_CHECK = 2.0  # s
FIRST_SECOND = 1.0  # s: what a short turn command gets
# After keeping, the walk learns to hold its heading: STEER_ROUNDS rounds aiming STEER_AIMS (rad, 17 degrees to either
# side, so it learns to correct both ways), losing HOLD_WEIGHT of its progress per rad it strays on average. Its
# steering is kept if, played as move plays it for STEER_CHECK seconds toward each aim and straight ahead, it falls
# in none, keeps nearer the line toward the aim on average than the walk alone and loses at most STEER_COST of its
# distance. With tries as long as the walk's (4 s), 6 A1 walks ended 20 degrees off on average after 20 s of walking,
# the worst 46 (its steering fell in the check); with 8 s tries, 4 and 6 (the template dog's and the starfish's: 2
# either way).
STEER_ROUNDS = 30
STEER_SECONDS = 8.0  # s per try
STEER_AIMS = (0.3, -0.3)
HOLD_WEIGHT = 1.0
STEER_CHECK = 20.0  # s
STEER_COST = 0.15  # share of the distance
# Playing it, the steering comes in over the walk's first STEER_RAMP seconds. At full strength from the start it
# threw an A1 still swinging out of a turn from side to side until it fell: of 810 tries of each switch on its own (6
# A1 walks, scripts/rhythm_bench.py switch --trials 15), 8 fell at once, 7 eased in over 1 s and 4 over 2 s. It aims
# back at the heading the walk started with, but never further than STEER_LEASH (rad) from where the robot faces, so
# a robot spun or pushed far off corrects fully without its aim running away, and the aim comes back by at most
# STEER_RELAX (rad/s). With the aim following the robot for good once it was 17 degrees off, an A1 walk that drifted
# while its steering eased in held -26 to -35 degrees for the rest of a 20 s walk.
STEER_LEASH = max(abs(a) for a in STEER_AIMS)
STEER_RAMP = 2.0  # s
STEER_RELAX = 0.1  # rad/s
# Holding the heading is not going straight: a walk that holds it can still walk a little crabwise, and its heading
# swings before the steering is in. It steers back toward the line it started on, aiming at a point LINE_AHEAD
# further along it, at most LINE_CUT (rad) off the course.
LINE_AHEAD = 1.0  # m
LINE_CUT = 0.3  # rad
# Standing still (trunk and joint speeds under these), a rhythm starts exactly as it was learned.
STILL_SPEED = 0.02  # m/s
STILL_SPIN = 0.1  # rad/s
STILL_JOINT = 0.1  # rad/s
# Before a switch it never learned (out of a turn, or into a turn without a walking version) the robot steadies, one
# of two ways (learn_switches checks which tips it over less): its joints fade to the home pose and hold it until its
# trunk turns slower than CALM_SPIN and leans less than CALM_TILT, at most CALM_MAX; or they hold home at once until
# the robot stands still, SETTLE_MAX at most, as walks kept before do. Stop goes home the same way. Holding took the
# whole of a 1 s command, and in 810 tries of each switch on its own 6 A1 walks fell 119 times; steadying until calm
# and fading, 4 (6 template dog walks: none and 8). Steadying until the trunk turned slower than 2 rad/s, 3 template
# dog walks fell 4 times in 405 tries of each switch and 3 A1 walks 3; slower than 1 rad/s, 1 and 0.
CALM_SPIN = 1.0  # rad/s, the trunk's angular speed
CALM_TILT = math.radians(25.0)  # a walking A1 rocks up to 20 degrees
CALM_MAX = 0.6  # s
SETTLE_MAX = 1.0  # s
# A stop asked while the walk plays takes STOP_GRACE to come: a turn asked by then begins straight out of the walk, as
# if no stop came between, and the walk asked again plays on. By hand, W let go and A pressed a moment later send a stop
# between the two; taken at once, the robot steadied first, and a 1 s turn one update after letting go of W turned the
# template dog 27 to 70% of its rate (median per walk and side, 6 walks, scripts/rhythm_bench.py leads, path quick);
# with the stop let go of, 25 to 127% (the lowest on a side whose walking turn was dropped), as with no gap at all. The
# longer it waits, the later a robot really let go of settles: waiting 0.1 s, a 1 s turn asked 0.16 to 0.3 s after
# letting go of W got 54 to 77% of the rate (median by gap; 70 to 86% stopping at once), waiting 0.06 s as stopping at
# once (path gap: 198 of 300 tries at 60% or more against 199).
STOP_GRACE = 0.06  # s: three updates
# How the walk switches is checked after the turns are learned (learn_switches): a walk of SWITCH_WALK seconds and
# then some, ending at each of SWITCH_TRIES points of the stride (every update at 2 Hz) in a stride early in the walk or
# later on; then the turn for FIRST_SECOND and up to a stride more (ending at SWITCH_ENDS points of its own stride),
# asked at once or SWITCH_GAPS after letting go of the walk; then walking on for SWITCH_ON or standing for EXIT_WAIT.
# And turns from standing. The bench (scripts/rhythm_bench.py leads) walks 2 to 4.5 s, other gaps, 1 s turns. A turn
# asked within STOP_GRACE of letting go plays as one asked at once from a later point of the stride, so the gaps are
# longer: the walk has stopped, and the robot steadies first.
SWITCH_TRIES = 25
SWITCH_WALK = (WALKED_MIN, 4.5)  # s
SWITCH_ENDS = 5
SWITCH_GAPS = (0.12, 0.18, 0.28)  # s, from asking to stop: 0.06 to 0.22 s after the walk stopped
SWITCH_ON = 2.0  # s
EXIT_WAIT = 2.0  # s
# Leaving the walk (SavedWalk.switching), holding first: "hold" home at once and wait until the robot stands still,
# then start the next rhythm as learned; "fade" to home and wait until it is calm, then fade the next rhythm in.
STYLES = ("hold", "fade")
EXITS = ("fade", "hold")  # leaving a turn, the same two ways
# A turn's walking version that falls DROP_MARGIN times or more in the turn's switch_checks is dropped (the robot
# steadying first, as before turns began out of the walk), unless steadying first falls more; one fall in about 85
# sequences is noise. Dropped on any fall more than steadying first, the turns of 3 of 6 imported A1 walks went (1 to 4
# falls against 0 to 3) and their 1 s turns after walking slowed to about half the rate (one from 121% and 95% of it to
# 52% and 44%), while the bench's falls stayed the same: 35 in 1200 sequences either way (scripts/rhythm_bench.py
# leads). Dropped only on 2 or more falls than steadying first, a template dog walk's left turn that fell twice, as
# often as steadying first, was kept and fell in 4 of its 100 bench sequences (left by holding home instead, 8),
# steadying first in none; those of 3 other dog walks, dropped on 2 to 4 falls more, fell 22 times in the bench when
# put back.
DROP_MARGIN = 2  # falls


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

    def walk_rhythm(self) -> Rhythm:
        walk = self.walk
        return Rhythm(self.rhythm(walk.theta, walk.reflexes), walk.freq, walk.reflexes)


def _walker_alone(model: mujoco.MjModel, robot: str, walk: SavedWalk) -> tuple[mujoco.MjData, RhythmWalker, int]:
    """The robot alone in `model` (never the live world) standing at home, a RhythmWalker asked to walk it forward
    with `walk` as move and the keyboard do, and the physics steps per update."""
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    walker = RhythmWalker(model, robot, walk, data.qpos.copy())
    walker.command(1.0, 0.0)  # any speed forward: it walks at its own pace
    return data, walker, max(1, round(1.0 / (CONTROL_HZ * model.opt.timestep)))


def walk_alone(
    model: mujoco.MjModel, robot: str, walk: SavedWalk, seconds: float, aim: float = 0.0
) -> dict[str, float]:
    """Walk forward with `walk` as move and the keyboard play it (a RhythmWalker), from standing at home, the robot
    alone in `model` (never the live world), for `seconds`, holding a heading `aim` rad from the one it starts with:
    how far it got along the aim, how far its heading turned (rad, positive to the left, whole turns counted), how far
    it strayed from the aim on average (rad) and from the line it started on toward the aim (m, `off`), and whether it
    fell."""
    data, walker, substeps = _walker_alone(model, robot, walk)
    trunk = walker.body.trunk
    origin, heading, turned, stray, fell = data.xpos[trunk][:2].copy(), trunk_yaw(data, walker.body), 0.0, 0.0, False
    across, off = np.array([-math.sin(heading + aim), math.cos(heading + aim)]), 0.0
    steps = round(seconds * CONTROL_HZ)
    for k in range(steps):
        walker.update(model, data)
        if k == 0:  # the walk has started where the robot faces: aim off to the side of that
            walker.course += aim
            walker.aim += aim
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        now = trunk_yaw(data, walker.body)
        turned += _wrapped(now - heading)
        heading = now
        stray += abs(aim - turned) / steps
        off += abs(float((data.xpos[trunk][:2] - origin) @ across)) / steps
        walker.tilt_cos = float(data.xmat[trunk][8])
        if walker.fallen:
            fell = True
            break
    moved = data.xpos[trunk][:2] - origin
    along = math.cos(aim) * float(moved[0]) + math.sin(aim) * float(moved[1])
    return {"distance": along, "turned": turned, "stray": stray, "off": off, "fell": float(fell)}


def walk_moments(model: mujoco.MjModel, robot: str, walk: SavedWalk, times: Sequence[float]) -> list[Start]:
    """Moments `times` seconds into walking forward with `walk` as move plays it (walk_alone), to start other rhythms
    from as they would be asked for mid-stride; none past a fall."""
    data, walker, substeps = _walker_alone(model, robot, walk)
    out: list[Start] = []
    for k in range(round(max(times, default=0.0) * CONTROL_HZ) + 1):
        if any(abs(k / CONTROL_HZ - when) < 0.5 / CONTROL_HZ for when in times):
            out.append(Start(data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()))
        walker.update(model, data)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        walker.tilt_cos = float(data.xmat[walker.body.trunk][8])
        if walker.fallen:
            break
    return out


def heading_drift(model: mujoco.MjModel, robot: str, walk: SavedWalk) -> float | None:
    """How far the walk's heading turns (rad, positive to the left) in STEER_CHECK seconds from standing without
    steering, the robot alone in `model`; None if it falls."""
    loose = walk_alone(model, robot, walk.model_copy(update={"steer": None}), STEER_CHECK)
    return None if loose["fell"] else loose["turned"]


def train_steering(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """The walk with steering learned for it (alone in `model`) by the same random search, the walk itself fixed
    (cpg.Steering), whether or not it helps."""
    joints = _Joints(model, robot, walk)
    steering = Steering.still(joints.walk_rhythm(), joints.body.joints)
    reward = Reward(hold=HOLD_WEIGHT, aims=STEER_AIMS)
    Trainer(model, robot, reward, STEER_SECONDS, walk.freq, home=joints.home()).train(
        steering, iterations=STEER_ROUNDS, seed=seed + 7
    )
    return walk.model_copy(update={"steer": joints.to_saved(steering.theta)})


def holds_heading(model: mujoco.MjModel, robot: str, held: SavedWalk, loose: SavedWalk) -> bool:
    """Whether `held` (a walk with steering) beats `loose` (the same without) played as move plays it for STEER_CHECK
    seconds toward each of STEER_AIMS and straight ahead: it falls in none, keeps nearer the line toward the aim on
    average, and loses at most STEER_COST of the distance. Its heading may stray more: keeping to the line, a walk
    that goes a little crabwise faces a little off it."""
    aims = (0.0, *STEER_AIMS)
    steered = [walk_alone(model, robot, held, STEER_CHECK, aim) for aim in aims]
    alone = [walk_alone(model, robot, loose, STEER_CHECK, aim) for aim in aims]
    if any(r["fell"] for r in steered):
        return False
    off, off_alone = (float(np.mean([r["off"] for r in rs])) for rs in (steered, alone))
    distance, distance_alone = (float(np.mean([r["distance"] for r in rs])) for rs in (steered, alone))
    return off < off_alone and distance >= (1.0 - STEER_COST) * distance_alone


def learn_steering(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """Teach the walk (alone in `model`, never the live world) to hold its heading, and return it with the steering
    added if that holds it better (holds_heading), else as it was. The reward's heading term straightened most walks,
    but not the imported A1's: open loop, its walks turned 46 degrees in 4 s on average over 20 trainings, with or
    without the term. And a walk that goes straight from standing can still swing off course starting out of a turn
    (a starfish walk that turned 2 degrees in 20 s turned 48 in the first 4 s after a right turn)."""
    loose = walk.model_copy(update={"steer": None})
    held = train_steering(model, robot, loose, seed)
    return held if holds_heading(model, robot, held, loose) else loose


def learn_turns(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """Teach the robot (alone in `model`, never the live world) to turn on the spot each way, by the same random
    search as the walk with a reward for the angle turned, from standing still; then tune each again to begin
    straight out of the walk (learn_walking_turns). Returns the walk with the turns added."""
    joints = _Joints(model, robot, walk)
    learned: dict[str, Any] = {}
    rates = []
    for side, sign in (("turn_left", 1.0), ("turn_right", -1.0)):
        rhythm, result = _standing_turn(model, robot, joints, Reward(turn=sign), seed)
        if result["fell"]:
            # A fall costs a point while a fast spin earns many, so the search can keep a turn that ends on its back
            # (an imported A1's left turn did on 1 of 6 walks, in all of 15 tries); it learns again, a fall
            # forfeiting the turn. Forfeiting always would slow the turns that stay up.
            rhythm, result = _standing_turn(model, robot, joints, Reward(turn=sign, forfeit=True), seed)
        learned[side] = joints.to_saved(rhythm.theta)
        rates.append(max(0.0, sign * result["turned"]) / walk.seconds)
    turned = learn_walking_turns(model, robot, walk.model_copy(update={**learned, "turn_rate": min(rates)}), seed)
    return learn_switches(model, robot, turned)


def learn_switches(model: mujoco.MjModel, robot: str, walk: SavedWalk) -> SavedWalk:
    """How the walk switches, checked by driving the robot (alone in `model`) the way move and the keyboard do: how it
    steadies leaving the walk (`switching`, STYLES), and for each turn whether it begins straight out of the walk with
    its walking version or steadies first, and how it is left (EXITS), on its switch_checks (_turn_switching). Of the
    two styles, fading is kept only if it tips the robot over less than holding: a check that finds no fall does not
    show a rare one cannot happen, and holding was the safer one wherever a robot steadies when held. Also returns how
    fast a turn asked for right after walking turns in its first second each way, as move then plays it. Reporting the
    slower side alone said 26 deg/s for a template dog walk whose 1 s turns right after walking went 70 degrees left
    and 159 right.

    Leaving each turn was first checked only for stops, from standing and after 1.5 s of walking: a template dog walk
    whose right turn was kept to fade then fell in 2 of 25 walk, 1 s turn, walk sequences (asked at every point of the
    stride), one kept to hold in 10 of 25. Fading everywhere, 6 dog walks still fell 12 times in 2100 sequences of
    move, gap and idle; holding (and beginning the turns that do not fall out of the walk), in none, but 6 imported A1
    walks fell 520 times (fading, 71)."""
    if not walk.turns:
        return walk
    best: tuple[int, dict[str, Any]] | None = None
    for style in STYLES:
        styled = walk.model_copy(update={"switching": style})
        falls = _falls(model, robot, styled, walk_checks(walk.freq), None if best is None else best[0])
        kept: dict[str, Any] = {"switching": style}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            if best is not None and falls >= best[0]:
                break
            left_over = None if best is None else best[0] - falls
            side_falls, update, first = _turn_switching(model, robot, styled, side, sign, left_over)
            falls += side_falls
            kept.update(update)
            # None if the walk falls before every turn: nothing to say
            kept[f"turn_{side}_rate_from_walk"] = max(0.0, float(np.mean(first))) / FIRST_SECOND if first else None
        if best is None or falls < best[0]:
            best = (falls, kept)
    assert best is not None
    return walk.model_copy(update=best[1])


def _turn_switching(
    model: mujoco.MjModel, robot: str, walk: SavedWalk, side: str, sign: float, enough: int | None
) -> tuple[int, dict[str, Any], list[float]]:
    """The way into and out of one turn, checked on its switch_checks (counting up to `enough` falls): it begins with
    its walking version straight out of the walk if that falls fewer than DROP_MARGIN times, or fewer than steadying
    first; else the robot steadies first. Either way it is left the way that tips it over least (EXITS, the style's
    own first, winning a tie). Its falls, its fields, and its first-second angles."""
    walking = getattr(walk, f"turn_{side}_walking")
    if walking is None:
        return _turn_exit(model, robot, walk, side, sign, None, None, enough)
    best = _turn_exit(model, robot, walk, side, sign, walking, getattr(walk, f"turn_{side}_walking_at"), enough)
    if best[0] < DROP_MARGIN:  # one fall is noise
        return best
    steady = _turn_exit(model, robot, walk, side, sign, None, None, best[0] + 1)  # counting until it falls more
    return best if steady[0] > best[0] else steady


def _turn_exit(
    model: mujoco.MjModel,
    robot: str,
    walk: SavedWalk,
    side: str,
    sign: float,
    walking: list[float] | None,
    at: list[float] | None,
    enough: int | None,
) -> tuple[int, dict[str, Any], list[float]]:
    """Of the ways of leaving one turn (EXITS, the style's own first), the one that tips the robot over least in its
    switch_checks with the turn beginning as `walking` and `at` say (None: steadying first), the first winning a tie,
    counting up to `enough` falls: its falls, its fields, and its first-second angles."""
    exits = sorted(EXITS, key=lambda e: e != walk.switching)
    best: tuple[int, dict[str, Any], list[float]] | None = None
    for exit_ in exits:
        update = {f"turn_{side}_walking": walking, f"turn_{side}_walking_at": at, f"turn_{side}_exit": exit_}
        limit = enough if best is None else best[0] if enough is None else min(enough, best[0])
        first: list[float] = []
        falls = _falls(model, robot, walk.model_copy(update=update), switch_checks(walk.freq, sign), limit, first)
        if best is None or falls < best[0]:
            best = (falls, update, first)
        if falls == 0:
            break
    assert best is not None
    return best


def walk_checks(freq: float) -> list[list[tuple[float, float, float]]]:
    """The sequences leaving the walk is checked on (speed, turn, seconds): stopping out of SWITCH_TRIES moments of
    the walk, one at each point of a stride of `freq` Hz, and stopping for a moment and walking on."""
    out = []
    for k, lead in enumerate(_leads(freq)):
        out.append([(1.0, 0.0, lead), (0.0, 0.0, EXIT_WAIT)])
        out.append([(1.0, 0.0, lead), (0.0, 0.0, SWITCH_GAPS[k % len(SWITCH_GAPS)]), (1.0, 0.0, SWITCH_ON)])
    return out


def switch_checks(freq: float, sign: float) -> list[list[tuple[float, float, float]]]:
    """The sequences a turn's switching is checked on (speed, turn, seconds; `sign` the turn's side): out of
    SWITCH_TRIES moments of the walk, one at each point of a stride of `freq` Hz, into a turn of FIRST_SECOND to about
    one and a half (ending at different points of its own stride), then walking on or stopping; the same turns asked a
    moment after letting go of the walk, then walking on a moment after letting go of the turn; and from standing into
    turns as long, then stopping or walking. Each turn command is given as its first second and the rest, the same mode
    (no switch), so its first second can be told apart."""
    out = []
    for k, lead in enumerate(_leads(freq)):
        turn = [(0.0, sign, FIRST_SECOND), (0.0, sign, (k % SWITCH_ENDS) / (SWITCH_ENDS * freq))]
        gap = (0.0, 0.0, SWITCH_GAPS[k % len(SWITCH_GAPS)])
        out.append([(1.0, 0.0, lead), *turn, (1.0, 0.0, SWITCH_ON), (0.0, 0.0, EXIT_WAIT)])
        out.append([(1.0, 0.0, lead), *turn, (0.0, 0.0, EXIT_WAIT)])
        out.append([(1.0, 0.0, lead), gap, *turn, gap, (1.0, 0.0, SWITCH_ON), (0.0, 0.0, EXIT_WAIT)])
    for j in range(SWITCH_ENDS):
        turn = [(0.0, sign, FIRST_SECOND), (0.0, sign, j / (SWITCH_ENDS * freq))]
        out.append([*turn, (0.0, 0.0, EXIT_WAIT)])
        out.append([*turn, (1.0, 0.0, SWITCH_ON)])
    return out


def _leads(freq: float) -> list[float]:
    """Seconds of walking before a switch is checked: SWITCH_TRIES of them, each at another point of the stride."""
    return [SWITCH_WALK[k % len(SWITCH_WALK)] + k / (SWITCH_TRIES * freq) for k in range(SWITCH_TRIES)]


def _falls(
    model: mujoco.MjModel,
    robot: str,
    walk: SavedWalk,
    checks: Sequence[Sequence[tuple[float, float, float]]],
    enough: int | None,
    first: list[float] | None = None,
) -> int:
    """How often the robot falls in `checks` after their first command (a fall while walking up to a switch, or
    turning from standing, is no switch's), counting up to `enough`. Into `first`, the angle (rad, the turn's way) of
    the first second of each turn straight out of the walk that it stayed up through."""
    falls = 0
    for commands in checks:
        if enough is not None and falls >= enough:
            break
        played = play_alone(model, robot, walk, commands)
        falls += played[-1].fell and len(played) > 1
        out_of_walk = commands[0][0] > 0 and len(commands) > 1 and commands[1][1] != 0.0
        if first is not None and out_of_walk and len(played) > 1 and not (len(played) == 2 and played[-1].fell):
            first.append(math.copysign(1.0, commands[1][1]) * played[1].turned)
    return falls


@dataclass(frozen=True)
class Played:
    """One command of a sequence played alone: the angle the heading turned while it lasted (rad, to the left), and
    whether the robot fell then (the sequence ends there)."""

    turned: float
    fell: bool


def play_alone(
    model: mujoco.MjModel, robot: str, walk: SavedWalk, commands: Sequence[tuple[float, float, float]]
) -> list[Played]:
    """Drive the robot (alone in `model`, never the live world) with `walk` as move and the keyboard do, from standing
    at home: each command (speed, turn, seconds) in turn, the next given at once (a stop between two is skipped, as
    the move tool's is). One entry per command played, up to a fall."""
    data, walker, substeps = _walker_alone(model, robot, walk)
    out: list[Played] = []
    for speed, angle, seconds in commands:
        walker.command(speed, angle)
        heading, turned = trunk_yaw(data, walker.body), 0.0
        for _ in range(round(seconds * CONTROL_HZ)):
            walker.update(model, data)
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            now = trunk_yaw(data, walker.body)
            turned, heading = turned + _wrapped(now - heading), now
            walker.tilt_cos = float(data.xmat[walker.body.trunk][8])
            if walker.fallen:
                return [*out, Played(turned, True)]
        out.append(Played(turned, False))
    return out


def _standing_turn(
    model: mujoco.MjModel, robot: str, joints: _Joints, reward: Reward, seed: int
) -> tuple[Rhythm, dict[str, float]]:
    """A turn on the spot learned from standing still for `reward`, and how its rollout went."""
    walk = joints.walk
    trainer = Trainer(model, robot, reward, walk.seconds, walk.freq, home=joints.home())
    # Random search lands somewhere different from every start (one tripod turn came out at 16 deg/s, the next
    # at 87): two starts each way, the better one kept.
    best: tuple[float, Rhythm, dict[str, float]] | None = None
    for start in range(TURN_STARTS):
        rhythm = Rhythm.still(joints.body.joints, walk.freq)
        trainer.train(rhythm, iterations=TURN_ROUNDS, seed=seed + 10 * start + (1 if reward.turn > 0 else 2))
        result = trainer.rollout(rhythm)
        if best is None or result["reward"] > best[0]:
            best = (result["reward"], rhythm, result)
    assert best is not None
    return best[1], best[2]


def learn_walking_turns(model: mujoco.MjModel, robot: str, walk: SavedWalk, seed: int = 0) -> SavedWalk:
    """The walk's turns, learned from standing, tuned again to begin straight out of the walk (the robot alone in
    `model`), and checked from more moments of it (_walking_turn). Learned only from standing still, a turn asked for
    mid-stride either waited for the robot to stand still (a 1 s turn right after walking turned the imported A1 16
    degrees of the 88 it promised) or started from a pose and speed it never met. They are kept apart: tuning the
    turns themselves on both left them slower to start from standing (5 of 6 walks turned less in a 1 s command
    after the robot had stood still)."""
    cleared = {f"turn_{side}_walking{part}": None for side in ("left", "right") for part in ("", "_at")}
    walk = walk.model_copy(update=cleared)
    if not walk.turns:
        return walk
    joints = _Joints(model, robot, walk)
    home = joints.home()
    walking = walk_moments(model, robot, walk, WALK_STARTS)
    checks = walk_moments(model, robot, walk, WALK_CHECKS)
    if not walking or not checks:  # it falls before its first moments: nothing to begin from
        return walk
    learned: dict[str, Any] = {}
    for side, sign in (("turn_left", 1.0), ("turn_right", -1.0)):
        standing = Rhythm(joints.rhythm(getattr(walk, side)), walk.freq)
        # Tuned twice: rewarded for the angle it turns in all its tries, and for the angle of its first second only
        # (staying up all its tries). Either can come out ahead: the second took one A1 walk's left turn from 50 to
        # 80% of its rate in a first second out of the walk, another's from 161% to 55%.
        tuned = []
        for k, early in enumerate((0.0, FIRST_SECOND)):
            reward = Reward(turn=sign, early=early)
            turn = Rhythm(standing.theta.copy(), walk.freq)
            Trainer(model, robot, reward, walk.seconds, walk.freq, home=home).train(
                turn, iterations=WALKING_ROUNDS, seed=seed + 100 * (k + 1) + (1 if sign > 0 else 2), starts=walking
            )
            tuned.append(turn)
        check = Trainer(model, robot, Reward(turn=sign), TURN_CHECK, walk.freq, home=home)
        first = Trainer(model, robot, Reward(turn=sign), FIRST_SECOND, walk.freq, home=home)
        chosen = _walking_turn((*tuned, standing), sign, check, first, checks, WALK_CHECKS)
        if chosen is not None:
            learned[f"{side}_walking"] = joints.to_saved(chosen[0].theta)
            learned[f"{side}_walking_at"] = chosen[1]
    return walk.model_copy(update=learned)


def stride_phase(t: float, freq: float) -> float:
    """Where in its stride (0 up to 1) a rhythm is `t` seconds after it started, to a thousandth."""
    return round((t * freq) % 1.0, 3) % 1.0


def _mean_wait(phases: Sequence[float]) -> float:
    """How long (in strides) the next of `phases` is away from a point of the stride picked at random, on average."""
    ring = sorted(phases)
    gaps = [((ring[(i + 1) % len(ring)] - p) % 1.0) or 1.0 for i, p in enumerate(ring)]
    return sum(g * g for g in gaps) / 2.0


def _walking_turn(
    turns: Sequence[Rhythm],
    sign: float,
    check: Trainer,
    first: Trainer,
    moments: Sequence[Start],
    times: Sequence[float],
) -> tuple[Rhythm, list[float] | None] | None:
    """Of `turns`, the one to begin out of the walk, and the points of the stride it may begin at (None: any). Each is
    started from `moments` (`times` s into the walk) for `check`'s seconds; a point of the stride it fell from at any
    of them is left out. Of the turns with at least MIN_SAFE of the points left, the one that turns the most in
    `first`'s seconds wins, counting the wait for its next point. None if each falls from more."""
    freq = check.freq
    phases = [stride_phase(t, freq) for t in times[: len(moments)]]
    best: tuple[float, Rhythm, list[float] | None] | None = None
    for turn in turns:
        unsafe = {p for p, m in zip(phases, moments, strict=True) if check.rollout(turn, start=m)["fell"]}
        safe = sorted(set(phases) - unsafe)
        if not safe or len(safe) < MIN_SAFE * len(set(phases)):  # fell from most: too close to falling anywhere
            continue
        angles = [sign * first.rollout(turn, start=m)["turned"] for p, m in zip(phases, moments, strict=True)]
        angle = float(np.mean([a for a, p in zip(angles, phases, strict=True) if p not in unsafe]))
        wait = _mean_wait(safe) / freq if unsafe else 0.0
        score = angle * max(0.0, 1.0 - wait / first.seconds)
        if best is None or score > best[0]:
            best = (score, turn, safe if unsafe else None)
    return None if best is None else (best[1], best[2])


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
    """Drives a robot with a kept walk: forward plays the walk (keeping to the line it started on, if it learned to
    steer), a turn plays a turn (once learned), each from its start as in training, standing still at home; stop goes
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
        # before): a faded stop leaves its feet elsewhere, and its fast turn from there tips it over. A walk kept now
        # was checked for which of the two tips it over less ("hold" or "fade").
        self.holds = walk.switching != "fade"
        # How steadying goes: holding home at once until still or fading to it until calm. What it leaves decides.
        self.holding = self.holds
        self.exits = {"left": walk.turn_left_exit, "right": walk.turn_right_exit}
        forward = Rhythm(np.array(walk.theta), walk.freq, walk.reflexes)
        self.rhythms: dict[str, Rhythm | Steering] = {
            "forward": forward if walk.steer is None else Steering(forward, np.array(walk.steer))
        }
        self.walking_at: dict[str, list[float] | None] = {}
        if walk.turns:
            self.rhythms["left"] = Rhythm(np.array(walk.turn_left), walk.freq)
            self.rhythms["right"] = Rhythm(np.array(walk.turn_right), walk.freq)
            for side, theta, at in (
                ("left", walk.turn_left_walking, walk.turn_left_walking_at),
                ("right", walk.turn_right_walking, walk.turn_right_walking_at),
            ):
                if theta is not None:
                    self.rhythms[f"{side} walking"] = Rhythm(np.array(theta), walk.freq)
                    self.walking_at[side] = at
        # Only a walk with a turn to begin out of it waits STOP_GRACE to stop: one without stops at once, as before.
        self.grace = STOP_GRACE if self.walking_at else 0.0
        self.max_speed = walk.speed
        self.mode: str | None = None  # forward, left, right, or None: standing
        self.asked: tuple[float, float] = (0.0, 0.0)
        # The mode asked for last that the joints follow, or will once steadied or at the next point of the stride
        # its turn may begin at (`turn_at`).
        self.playing: str | None = None
        self.rhythm: str | None = None  # the rhythm playing (a key of `rhythms`); None while standing or steadying
        self.t0: float | None = None  # when it started
        self.course = 0.0  # rad: the heading the walk holds, where the robot faced when asked to walk
        self.origin = np.zeros(2)  # m: where the trunk was then; the walk keeps to the line through it along `course`
        self.aim = 0.0  # rad: the heading it steers for now, at most STEER_LEASH from where it faces
        self.steady_from: float | None = None  # steadying on home since then, before the asked rhythm starts
        self.turn_at: float | None = None  # sim time the asked turn begins out of the walk, which plays on until then
        self.stop_at: float | None = None  # sim time an asked stop comes (STOP_GRACE), the walk playing on until then
        self.fade_from: np.ndarray | None = None  # the joint targets of the last switch, fading out
        self.fade_t0 = 0.0
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
        dt = now - self.clock if now >= self.clock > -math.inf else 0.0
        if now < self.clock:  # the clock was turned back (a reset): times kept from before lie in the future
            self.playing, self.rhythm, self.t0, self.steady_from, self.fade_from = None, None, None, None, None
            self.turn_at, self.stop_at = None, None
        self.clock = now
        if self.mode != self.playing:
            self._switch(data, now)
        if self.stop_at is not None and now >= self.stop_at - 0.5 / CONTROL_HZ:
            self.stop_at, self.rhythm, self.t0 = None, None, None
            self._settle(data, now)
        if self.steady_from is not None and self._steadied(data, now - self.steady_from):
            self.steady_from = None
            assert self.mode is not None
            self._start(data, now, self.mode, fade=not self.holding)
        if self.turn_at is not None and now >= self.turn_at - 0.5 / CONTROL_HZ:
            self.turn_at = None
            self._start(data, now, f"{self.mode} walking", fade=True)
        if self.rhythm is None or self.t0 is None:  # standing, or steadying before a rhythm starts
            target = self.home
        else:
            rhythm = self.rhythms[self.rhythm]
            senses = trunk_senses(data, self.body) if rhythm.reflexes else None
            error = 0.0
            if self.rhythm == "forward":
                error = self._heading_error(data, now - self.t0, dt)
            target = rhythm.targets(self.home, now - self.t0, senses=senses, error=error)
        if self.fade_from is not None:
            target = fade_in(self.fade_from, target, now - self.fade_t0)
            if now - self.fade_t0 >= FADE:
                self.fade_from = None
        data.ctrl[self.actuators] = np.clip(target, self.lower, self.upper)

    def _heading_error(self, data: mujoco.MjData, walked: float, dt: float) -> float:
        """How far (rad) the walk should steer back, `walked` seconds after it started: toward an aim that comes back
        by STEER_RELAX to the course, turned toward the line the walk started on if it is beside it (LINE_AHEAD,
        LINE_CUT), never more than STEER_LEASH away, with the correction coming in over STEER_RAMP."""
        yaw = trunk_yaw(data, self.body)
        across = np.array([-math.sin(self.course), math.cos(self.course)])
        side = float((data.xpos[self.body.trunk][:2] - self.origin) @ across)  # m to the left of the line
        cut = max(-LINE_CUT, min(LINE_CUT, math.atan2(side, LINE_AHEAD)))
        back = _wrapped(self.course - cut - self.aim)
        self.aim += max(-STEER_RELAX * dt, min(STEER_RELAX * dt, back))
        error = _wrapped(self.aim - yaw)
        if abs(error) > STEER_LEASH:
            error = math.copysign(STEER_LEASH, error)
            self.aim = yaw + error
        return max(-STEER_SPAN, min(STEER_SPAN, error)) * min(1.0, walked / STEER_RAMP)

    def _switch(self, data: mujoco.MjData, now: float) -> None:
        """The asked mode takes over. Each rhythm was learned from standing still, and the turns' walking versions
        (in files that have them) right out of the walk; those start at once, or at the next point of the stride they
        were checked to begin at safely, walking on until then. Any other switch while moving steadies on home first.
        When every switch waited until the robot stood still, a 1 s turn right after walking turned the imported A1
        16 degrees of the 88 it promised. A walk kept to hold home before other switches begins its walking turns
        straight out of the walk all the same, once the walk has played WALKED_MIN. A stop asked while the walk plays
        comes STOP_GRACE later, so what is asked by then switches straight out of the walk too."""
        # Only out of the walk as it plays (or walks on to a turn, or to a stop): a robot stopped a moment ago is
        # fading to home, a pose and a point of the stride its walking turns were never checked from. Any switch within
        # a quarter second of the walk's last update used to count, so a turn soon after a stop began at once wherever
        # the stride was.
        out_of_walk = self.rhythm == "forward"
        self.playing, self.steady_from, self.turn_at, self.stop_at = self.mode, None, None, None
        if self.rhythm is not None:  # what it leaves decides how it steadies (already steadying: as it is)
            exit_ = self.exits.get(self.rhythm.split()[0])
            self.holding = self.holds if exit_ is None else exit_ == "hold"
        if self.mode == "forward" and self.rhythm != "forward":
            # Where it faces now, not once steadied: an imported A1 coming out of a fast turn spun on by 47 degrees
            # while it steadied, and then walked that way.
            self.course = self.aim = trunk_yaw(data, self.body)
            self.origin = data.xpos[self.body.trunk][:2].copy()
        if self.mode is None and out_of_walk and self.grace:  # stop: the walk plays on for STOP_GRACE first
            self.stop_at = now + self.grace
        elif self.mode is None:  # stop
            self.rhythm, self.t0 = None, None
            self._settle(data, now)
        elif self.mode == "forward" and self.rhythm == "forward":  # it was walking on to a turn, now not asked for
            return
        elif self._still(data):
            self._start(data, now, self.mode, fade=False)
        elif self.mode != "forward" and out_of_walk and self._walked(now) and f"{self.mode} walking" in self.rhythms:
            wait = self._wait_for_stride(now)
            if wait > 0.0:
                self.turn_at = now + wait
            else:
                self._start(data, now, f"{self.mode} walking", fade=True)
        else:
            self.rhythm, self.t0, self.steady_from = None, None, now
            self._settle(data, now)

    def _walked(self, now: float) -> bool:
        """Whether the walk has played long enough for a turn to begin straight out of it (WALKED_MIN)."""
        return self.t0 is not None and now - self.t0 >= WALKED_MIN - 0.5 / CONTROL_HZ

    def _wait_for_stride(self, now: float) -> float:
        """How long (s) the walk plays on before the asked turn may begin out of it: until the next point of the
        stride the turn was checked to begin at safely, within half an update; 0 if it may begin anywhere."""
        at = self.walking_at.get(self.mode or "")
        if at is None or self.rhythm != "forward" or self.t0 is None:
            return 0.0
        freq = self.walk.freq
        phase = ((now - self.t0) * freq) % 1.0
        slack = 0.5 * freq / CONTROL_HZ  # half an update, in strides
        ahead = min((p - phase + slack) % 1.0 for p in at) - slack
        return max(0.0, ahead) / freq

    def _settle(self, data: mujoco.MjData, now: float) -> None:
        """Go to the home pose: at once when holding, else faded in from the targets of the moment."""
        if self.holding:
            self.fade_from = None
        else:
            self._fade(data, now)

    def _steadied(self, data: mujoco.MjData, steadying: float) -> bool:
        if self.holding:
            return self._still(data) or steadying >= SETTLE_MAX
        return self._calm(data) or steadying >= CALM_MAX

    def _start(self, data: mujoco.MjData, now: float, rhythm: str, fade: bool) -> None:
        self.rhythm, self.t0 = rhythm, now
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
