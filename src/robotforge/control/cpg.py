"""Learning to walk for any body: a rhythm per joint, tuned by random search.

Every servo joint swings around its home angle, target = home + offset + amplitude * sin(2 pi f t + phase),
so the rhythm is built in and only 3 numbers per joint are learned. Augmented Random Search (the same
recipe as control/ars.py: try nudges both ways, step toward the best) tunes them for a reward: how far the
robot gets in a few seconds, counted for less the more its heading turned, minus drifting sideways and, if
asked, the energy it spends. No model of the robot and no neural network: it works on a dog, a three-legged
starfish built from parts, or whatever a learner made, in seconds on a laptop.

The rhythm alone is open loop: it never reads a sensor. Reflexes add feedback, 4 more numbers per joint
that turn the trunk's roll, pitch and their rates into joint corrections, learned by the same search.
Training can also shove the robot at random and count surviving as reward, so the search has a reason to
learn them.

A rhythm can also be learned to start mid-stride of another (a turn right out of a walk): its tries then begin
from moments of that one, the new rhythm fading in from the targets the joints had, as RhythmWalker plays it.

A walk can also learn to hold its heading (Steering): the heading error moves its offsets, amplitudes and phases
by 3 more numbers per joint, learned by the same search with the walk itself fixed, aiming a little to either side.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.sim.actuators import drives_joint, is_position_servo

CONTROL_HZ = 50.0
FELL = 0.5  # cosine of the trunk's tilt below which the robot is down
FALL_PENALTY = 1.0  # m of progress a fall costs
DRIFT_WEIGHT = 0.5  # m of progress per m of sideways drift
# Share of a forward walk's distance lost per rad its heading turned by the end. Over 20 trainings each (30 rounds,
# scripts/rhythm_bench.py forward), 0.15 cut the mean heading change in 4 s from 40 to 23 degrees (template dog), 20
# to 12 (starfish) and 29 to 21 (humanoid), for 3 to 9 percent less distance. It never straightened the imported A1
# (46 degrees at 0, 45 at 0.15, 46 at 0.25), and 0.25 cost it and the humanoid a sixth of their distance: a kept walk
# that still curves learns to steer instead (Steering).
HEADING_WEIGHT = 0.15
WANDER_WEIGHT = 3.0  # rad of turning per m the trunk wanders (at its farthest) while learning to turn on the spot


class CPGError(ValueError):
    pass


@dataclass(frozen=True)
class Body:
    """Where one robot's moving parts are in a model: its trunk and the servo joints the rhythm drives."""

    robot: str
    trunk: int
    actuators: np.ndarray
    qadr: np.ndarray  # the driven joints' qpos addresses
    dadr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    qpos: np.ndarray  # every qpos address of the robot (free joint and all joints)
    qvel: np.ndarray

    @classmethod
    def find(cls, model: mujoco.MjModel, robot: str) -> Body:
        prefix = f"{robot}/"
        joints = [j for j in range(model.njnt) if model.body(int(model.jnt_bodyid[j])).name.startswith(prefix)]
        free = [j for j in joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            raise CPGError(f"{robot} is fixed to the world; only robots that move freely can learn to walk")
        hinges = [
            a
            for a in range(model.nu)
            if (model.actuator(a).name or "").startswith(prefix)
            and drives_joint(model, a)
            # Not velocity servos: an imported wheel's speed command is bounded too, but it is no angle to aim at.
            and is_position_servo(model, a)
            and model.jnt_type[model.actuator_trnid[a, 0]] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        bounded = {a: span for a in hinges if (span := _servo_range(model, a)) is not None}
        servos = list(bounded)
        if len(servos) < 2:
            raise CPGError(f"{robot} has {len(servos)} servo joints; walking by rhythm needs at least 2")
        driven = [int(model.actuator_trnid[a, 0]) for a in servos]
        ranges = np.array(list(bounded.values()))
        qpos, qvel = [], []
        for j in joints:
            width = {0: 7, 1: 4}.get(int(model.jnt_type[j]), 1)
            qpos += range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + width)
            qvel += range(
                int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + {0: 6, 1: 3}.get(int(model.jnt_type[j]), 1)
            )
        return cls(
            robot=robot,
            trunk=int(model.jnt_bodyid[free[0]]),
            actuators=np.array(servos),
            qadr=np.array([model.jnt_qposadr[j] for j in driven]),
            dadr=np.array([model.jnt_dofadr[j] for j in driven]),
            lower=ranges[:, 0],
            upper=ranges[:, 1],
            qpos=np.array(qpos),
            qvel=np.array(qvel),
        )

    @property
    def joints(self) -> int:
        return len(self.actuators)


def _servo_range(model: mujoco.MjModel, actuator: int) -> tuple[float, float] | None:
    """Where a position servo can put its joint: the joint's limits and the servo's command range, whichever
    are set (both: the overlap). A URDF joint without limits imports unlimited, its servo commanded within half
    a turn either way, so the servo's range is what bounds it. None when nothing bounds it."""
    joint = int(model.actuator_trnid[actuator, 0])
    lo, hi = -math.inf, math.inf
    if model.jnt_limited[joint]:
        lo, hi = (float(v) for v in model.jnt_range[joint])
    if model.actuator_ctrllimited[actuator]:
        low, high = model.actuator_ctrlrange[actuator]
        lo, hi = max(lo, float(low)), min(hi, float(high))
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) and lo < hi else None


SENSES = 4  # what reflexes read: trunk roll, pitch, roll rate, pitch rate


def trunk_yaw(data: mujoco.MjData, body: Body) -> float:
    """Which way the trunk faces (rad, counter-clockwise from +x)."""
    rot = data.xmat[body.trunk]
    return math.atan2(float(rot[3]), float(rot[0]))


def trunk_senses(data: mujoco.MjData, body: Body) -> np.ndarray:
    """Roll and pitch of the trunk (rad) and their rates (rad/s, body frame), what an IMU gives."""
    rot = data.xmat[body.trunk].reshape(3, 3)
    roll = math.atan2(rot[2, 1], rot[2, 2])
    pitch = math.asin(-max(-1.0, min(1.0, rot[2, 0])))
    spin = data.qvel[body.qvel[3:5]]  # the free joint's angular velocity is in the body frame
    return np.array([roll, pitch, spin[0], spin[1]])


@dataclass
class Rhythm:
    """The learned numbers: an offset, an amplitude and a phase per joint and the shared frequency, and with
    reflexes a gain per joint for each of the trunk's senses."""

    theta: np.ndarray  # [offsets, amplitudes, phases] per joint, then the reflex gains [joint x sense]
    freq: float = 2.0
    reflexes: bool = False

    @classmethod
    def still(cls, joints: int, freq: float = 2.0, reflexes: bool = False) -> Rhythm:
        return cls(np.zeros((3 + (SENSES if reflexes else 0)) * joints), freq, reflexes)

    def targets(
        self,
        home: np.ndarray,
        t: float,
        theta: np.ndarray | None = None,
        senses: np.ndarray | None = None,
        error: float = 0.0,
    ) -> np.ndarray:
        """The joint targets `t` seconds in. `error` (the heading's, for a Steering) does not change a rhythm."""
        th = self.theta if theta is None else theta
        n = len(home)
        out = home + th[:n] + th[n : 2 * n] * np.sin(2 * math.pi * self.freq * t + th[2 * n : 3 * n])
        if self.reflexes and senses is not None:
            out = out + th[3 * n :].reshape(n, SENSES) @ senses
        return out


# Heading error (rad, 14 degrees) at which a Steering corrects fully; it corrects in proportion below that. The 6
# walks the lesson's first trainings give the imported A1 turned 44 degrees in 4 s open loop on average; steering
# with 0.25 (not tuned further), 7.
STEER_SPAN = 0.25


def steer_input(error: float) -> float:
    """How hard a Steering corrects for a heading `error` (rad, aim minus heading): -1 to 1."""
    return max(-1.0, min(1.0, error / STEER_SPAN))


@dataclass
class Steering:
    """How a walk holds its heading. Its offsets, amplitudes and phases move by steer_input(error) times `theta`
    (3 numbers per joint), so a heading error turns the walk back toward its aim. The walk stays as it was learned;
    the search tunes only `theta`, the way it tunes a rhythm (Trainer.train), so it plays wherever a rhythm does."""

    walk: Rhythm
    theta: np.ndarray  # added to the walk's [offsets, amplitudes, phases] when correcting fully to the left

    @classmethod
    def still(cls, walk: Rhythm, joints: int) -> Steering:
        return cls(walk, np.zeros(3 * joints))

    @property
    def freq(self) -> float:
        return self.walk.freq

    @property
    def reflexes(self) -> bool:
        return self.walk.reflexes

    def targets(
        self,
        home: np.ndarray,
        t: float,
        theta: np.ndarray | None = None,
        senses: np.ndarray | None = None,
        error: float = 0.0,
    ) -> np.ndarray:
        gains = self.theta if theta is None else theta
        walk = self.walk.theta.copy()
        walk[: len(gains)] += steer_input(error) * gains
        return self.walk.targets(home, t, walk, senses)


FADE = 0.25  # s: a rhythm started mid-stride fades in from the joint targets of the moment, over this long


def fade_in(source: np.ndarray, target: np.ndarray, elapsed: float) -> np.ndarray:
    """`target`, faded in from `source` over FADE seconds (eased at both ends), `elapsed` seconds after the switch."""
    x = min(1.0, max(0.0, elapsed / FADE))
    w = x * x * (3.0 - 2.0 * x)
    return target if w >= 1.0 else (1.0 - w) * source + w * target


@dataclass(frozen=True)
class Start:
    """A moment to begin a rollout from instead of standing still at home: the whole model's positions, velocities
    and joint targets, say mid-stride of a walk. The rhythm tried from it fades in from those targets."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray


@dataclass(frozen=True)
class Reward:
    energy: float = 0.0  # weight on energy spent, per (m g): 0 ignores it
    survive: bool = False  # a fall also costs the seconds it did not stay up
    turn: float = 0.0  # 0: walk forward; +1 or -1: turn on the spot to the left or right (rad turned, less wander)
    heading: float = HEADING_WEIGHT  # walking forward: share of the distance lost per rad turned; 0 ignores it
    # Learning to hold a heading (a Steering): every try walks toward each of `aims` (rad from the heading it starts
    # at) and loses `hold` of its progress that way per rad it strayed from it on average, instead of `heading`.
    hold: float = 0.0
    aims: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class Push:
    at: float  # s into the walk
    velocity: tuple[float, float]  # m/s added to the trunk, (forward, left)


def random_push(rng: np.random.Generator, strength: float, seconds: float) -> Push:
    """A shove of `strength` m/s in a random direction, somewhere in the middle of the walk."""
    angle = rng.uniform(0.0, 2 * math.pi)
    return Push(rng.uniform(0.25, 0.75) * seconds, (strength * math.cos(angle), strength * math.sin(angle)))


@dataclass
class TrainingLog:
    # After each iteration, the current rhythm's progress: metres walked, or radians turned the asked way.
    distances: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    seconds: float = 0.0
    attempts: int = 0


class Trainer:
    """Rollouts of one robot alone in its own model (never the live world) and the search over the rhythm."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        reward: Reward,
        seconds: float = 4.0,
        freq: float = 2.0,
        home: np.ndarray | None = None,
    ):
        self.model = model
        self.body = Body.find(model, robot)
        self.reward = reward
        self.seconds = seconds
        self.freq = freq
        # The joint angles rhythms swing around (the driven joints'): the model's home pose unless told otherwise.
        if home is not None:
            self.home = np.asarray(home, dtype=float).copy()
        elif model.nkey:
            self.home = model.key("home").qpos[self.body.qadr].copy()
        else:
            self.home = model.qpos0[self.body.qadr].copy()
        self.mass = float(sum(model.body_mass))
        self.substeps = max(1, round(1.0 / (CONTROL_HZ * model.opt.timestep)))

    def rollout(
        self,
        rhythm: Rhythm | Steering,
        theta: np.ndarray | None = None,
        push: Push | None = None,
        start: Start | None = None,
        aim: float = 0.0,
    ) -> dict[str, float]:
        """Walk for `seconds` from the home pose, or from `start` (shoved once if `push`): distance, sideways drift,
        energy, how long it stayed up, fell, and the reward. A Steering aims `aim` rad from the starting heading,
        and distance and drift are then measured that way."""
        m, b = self.model, self.body
        d = mujoco.MjData(m)
        if start is not None:
            d.qpos[:], d.qvel[:], d.ctrl[:] = start.qpos, start.qvel, start.ctrl
        elif m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        before = d.ctrl[b.actuators].copy()  # what a rhythm started mid-stride fades in from
        origin = d.xpos[b.trunk][:2].copy()
        energy, fell, up = 0.0, False, self.seconds
        heading, turned, wander, stray = trunk_yaw(d, b), 0.0, 0.0, 0.0
        steps = round(self.seconds * CONTROL_HZ)
        for k in range(steps):
            t = k / CONTROL_HZ
            if push is not None and push.at <= t < push.at + 1.0 / CONTROL_HZ:
                d.qvel[b.qvel[0]] += push.velocity[0]
                d.qvel[b.qvel[1]] += push.velocity[1]
            senses = trunk_senses(d, b) if rhythm.reflexes else None
            target = rhythm.targets(self.home, t, theta, senses, error=aim - turned)
            if start is not None:
                target = fade_in(before, target, t)
            d.ctrl[b.actuators] = np.clip(target, b.lower, b.upper)
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
                energy += float(np.abs(d.actuator_force[b.actuators] * d.qvel[b.dadr]).sum()) * m.opt.timestep
            now = trunk_yaw(d, b)
            turned += math.atan2(math.sin(now - heading), math.cos(now - heading))  # unwrapped: whole turns count
            heading = now
            stray += abs(aim - turned) / steps
            # The farthest it got, not where it ended: circling round a point comes back to the start.
            wander = max(wander, float(np.linalg.norm(d.xpos[b.trunk][:2] - origin)))
            if float(d.xmat[b.trunk][8]) < FELL:
                fell, up = True, t
                break
        moved = d.xpos[b.trunk][:2] - origin
        if aim:  # measured along the aim and across it
            moved = np.array([[math.cos(aim), math.sin(aim)], [-math.sin(aim), math.cos(aim)]]) @ moved
        distance, drift = float(moved[0]), abs(float(moved[1]))
        cost = self.reward.energy * energy / (self.mass * 9.81)
        if self.reward.turn:
            # Turning on the spot: the angle turned the asked way, less how far the trunk wandered off. A fall
            # forfeits the turn: a fast spin that ends on its back scored more than a slower one that stays up.
            turn = 0.0 if fell else self.reward.turn * turned
            reward = turn - WANDER_WEIGHT * wander - cost
        else:
            # Forward: the distance counts for less the more the heading turned, so a walk that curves round
            # (an imported A1 went 1.46 m but turned 42 degrees in 4 s) loses to one that goes straight. Never less
            # than nothing: past 1 / HEADING_WEIGHT rad, walking further would score worse. Holding a heading counts
            # how far it strayed on average instead: the end alone may come back after a swing.
            straying = self.reward.hold * stray if self.reward.hold else self.reward.heading * abs(turned)
            reward = distance - min(straying, 1.0) * max(distance, 0.0) - DRIFT_WEIGHT * drift - cost
        if fell:
            reward -= FALL_PENALTY + (self.seconds - up if self.reward.survive else 0.0)
        return {
            "distance": distance,
            "drift": drift,
            "turned": turned,
            "stray": stray,
            "wander": wander,
            "energy": energy,
            "up": up,
            "fell": float(fell),
            "reward": reward,
        }

    def train(
        self,
        rhythm: Rhythm | Steering,
        iterations: int = 30,
        directions: int = 8,
        top: int = 4,
        noise: float = 0.08,
        step_size: float = 0.04,
        seed: int = 0,
        pushes: float = 0.0,
        callback: Callable[[int, dict[str, float]], None] | None = None,
        stop: Callable[[], bool] | None = None,
        starts: Sequence[Start | None] | None = None,
    ) -> TrainingLog:
        """ARS on the rhythm, in place (a live controller may be playing it). With `pushes` (m/s) every try is
        shoved once at random, the same shove for both sides of a direction; with `starts`, it begins from one of
        them picked at random (None: standing still at home), the same for both sides. After each iteration the
        current rhythm walks once more, from home and unshoved, for the log and the callback."""
        if not 1 <= top <= directions:
            raise CPGError("top must be between 1 and the number of directions")
        rng = np.random.default_rng(seed)
        log = TrainingLog()
        started = time.perf_counter()
        for it in range(iterations):
            if stop is not None and stop():
                break
            deltas = rng.standard_normal((directions, len(rhythm.theta)))
            shoves = [random_push(rng, pushes, self.seconds) if pushes > 0 else None for _ in deltas]
            begins = [starts[int(rng.integers(len(starts)))] for _ in deltas] if starts else [None] * len(deltas)
            base = rhythm.theta.copy()
            tries = list(zip(deltas, shoves, begins, strict=True))
            plus = np.array([self._score(rhythm, base + noise * dl, p, s) for dl, p, s in tries])
            minus = np.array([self._score(rhythm, base - noise * dl, p, s) for dl, p, s in tries])
            best = np.argsort(-np.maximum(plus, minus))[:top]
            spread = np.concatenate([plus[best], minus[best]]).std() + 1e-8
            rhythm.theta = base + step_size / (top * spread) * np.tensordot(
                plus[best] - minus[best], deltas[best], axes=1
            )
            walk = self.rollout(rhythm)
            log.attempts += 2 * directions + 1
            log.distances.append(self.reward.turn * walk["turned"] if self.reward.turn else walk["distance"])
            log.energies.append(walk["energy"])
            if callback is not None:
                callback(it + 1, walk)
        log.seconds = time.perf_counter() - started
        return log

    def _score(self, rhythm: Rhythm | Steering, theta: np.ndarray, push: Push | None, start: Start | None) -> float:
        """A try's reward: its mean over the reward's aims (one, straight ahead, unless learning to steer)."""
        aims = self.reward.aims
        return sum(self.rollout(rhythm, theta, push, start, aim)["reward"] for aim in aims) / len(aims)

    def moments(self, rhythm: Rhythm | Steering, times: Sequence[float]) -> list[Start]:
        """Play `rhythm` from the home pose (a Steering holding the heading it starts with) and keep the moments
        `times` seconds into it, to start other rhythms from; none past a fall."""
        m, b = self.model, self.body
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        out: list[Start] = []
        heading, turned = trunk_yaw(d, b), 0.0
        for k in range(round(max(times, default=0.0) * CONTROL_HZ) + 1):
            t = k / CONTROL_HZ
            if float(d.xmat[b.trunk][8]) < FELL:
                break
            if any(abs(t - when) < 0.5 / CONTROL_HZ for when in times):
                out.append(Start(d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()))
            senses = trunk_senses(d, b) if rhythm.reflexes else None
            target = rhythm.targets(self.home, t, senses=senses, error=-turned)
            d.ctrl[b.actuators] = np.clip(target, b.lower, b.upper)
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
            now = trunk_yaw(d, b)
            turned += math.atan2(math.sin(now - heading), math.cos(now - heading))
            heading = now
        return out

    def push_test(self, rhythm: Rhythm, strength: float, trials: int = 12, seed: int = 99) -> dict[str, float]:
        """Walk `trials` times, each shoved once at random: how many stayed up, and how far they got."""
        rng = np.random.default_rng(seed)
        walks = [self.rollout(rhythm, push=random_push(rng, strength, self.seconds)) for _ in range(trials)]
        return {
            "survived": float(sum(1 for w in walks if not w["fell"])),
            "trials": float(trials),
            "distance": float(np.mean([w["distance"] for w in walks])),
        }


class RhythmController:
    """Plays a rhythm (and its reflexes) on a robot in the live world; `restart` puts the clock back to zero."""

    rate_hz = CONTROL_HZ

    def __init__(self, model: mujoco.MjModel, robot: str, rhythm: Rhythm, home: np.ndarray) -> None:
        self.body = Body.find(model, robot)
        self.rhythm = rhythm
        self.home = home
        self.t0 = 0.0

    def restart(self, t0: float) -> None:
        self.t0 = t0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        b = self.body
        senses = trunk_senses(data, b) if self.rhythm.reflexes else None
        target = self.rhythm.targets(self.home, float(data.time) - self.t0, senses=senses)
        data.ctrl[b.actuators] = np.clip(target, b.lower, b.upper)

    def status(self) -> dict[str, object]:
        return {"controller": "rhythm", "robot": self.body.robot, "reflexes": self.rhythm.reflexes}
