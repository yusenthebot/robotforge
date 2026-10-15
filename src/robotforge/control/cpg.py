"""Learning to walk for any body: a rhythm per joint, tuned by random search.

Every servo joint swings around its home angle, target = home + offset + amplitude * sin(2 pi f t + phase),
so the rhythm is built in and only 3 numbers per joint are learned. Augmented Random Search (the same
recipe as control/ars.py: try nudges both ways, step toward the best) tunes them for a reward: how far the
robot gets in a few seconds, minus drifting sideways and, if asked, the energy it spends. No model of the
robot and no neural network: it works on a dog, a three-legged starfish built from parts, or whatever a
learner made, in seconds on a laptop. What it cannot do is balance on two legs: the rhythm is open loop,
it never looks at a sensor.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

CONTROL_HZ = 50.0
FELL = 0.5  # cosine of the trunk's tilt below which the robot is down
FALL_PENALTY = 1.0  # m of progress a fall costs
DRIFT_WEIGHT = 0.5  # m of progress per m of sideways drift


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
        servos = [
            a
            for a in range(model.nu)
            if (model.actuator(a).name or "").startswith(prefix)
            and model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE  # position servos
            and model.jnt_type[model.actuator_trnid[a, 0]] == mujoco.mjtJoint.mjJNT_HINGE
            and model.jnt_limited[model.actuator_trnid[a, 0]]
        ]
        if len(servos) < 2:
            raise CPGError(f"{robot} has {len(servos)} servo joints; walking by rhythm needs at least 2")
        driven = [int(model.actuator_trnid[a, 0]) for a in servos]
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
            lower=np.array([model.jnt_range[j][0] for j in driven]),
            upper=np.array([model.jnt_range[j][1] for j in driven]),
            qpos=np.array(qpos),
            qvel=np.array(qvel),
        )

    @property
    def joints(self) -> int:
        return len(self.actuators)


@dataclass
class Rhythm:
    """The learned numbers: an offset, an amplitude and a phase per joint, and the shared frequency."""

    theta: np.ndarray  # [offsets..., amplitudes..., phases...]
    freq: float = 2.0

    @classmethod
    def still(cls, joints: int, freq: float = 2.0) -> Rhythm:
        return cls(np.zeros(3 * joints), freq)

    def targets(self, home: np.ndarray, t: float, theta: np.ndarray | None = None) -> np.ndarray:
        th = self.theta if theta is None else theta
        n = len(home)
        return home + th[:n] + th[n : 2 * n] * np.sin(2 * math.pi * self.freq * t + th[2 * n :])


@dataclass(frozen=True)
class Reward:
    energy: float = 0.0  # weight on energy spent, per (m g): 0 ignores it


@dataclass
class TrainingLog:
    distances: list[float] = field(default_factory=list)  # after each iteration, the current rhythm's walk
    energies: list[float] = field(default_factory=list)
    seconds: float = 0.0
    attempts: int = 0


class Trainer:
    """Rollouts of one robot alone in its own model (never the live world) and the search over the rhythm."""

    def __init__(self, model: mujoco.MjModel, robot: str, reward: Reward, seconds: float = 4.0, freq: float = 2.0):
        self.model = model
        self.body = Body.find(model, robot)
        self.reward = reward
        self.seconds = seconds
        self.freq = freq
        self.home = model.key("home").qpos[self.body.qadr].copy() if model.nkey else model.qpos0[self.body.qadr].copy()
        self.mass = float(sum(model.body_mass))
        self.substeps = max(1, round(1.0 / (CONTROL_HZ * model.opt.timestep)))

    def rollout(self, rhythm: Rhythm, theta: np.ndarray | None = None) -> dict[str, float]:
        """Walk for `seconds` from the home pose: distance, sideways drift, energy, fell, and the reward."""
        m, b = self.model, self.body
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        start = d.xpos[b.trunk][:2].copy()
        energy, fell = 0.0, False
        for k in range(round(self.seconds * CONTROL_HZ)):
            target = rhythm.targets(self.home, k / CONTROL_HZ, theta)
            d.ctrl[b.actuators] = np.clip(target, b.lower, b.upper)
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
                energy += float(np.abs(d.actuator_force[b.actuators] * d.qvel[b.dadr]).sum()) * m.opt.timestep
            if float(d.xmat[b.trunk][8]) < FELL:
                fell = True
                break
        moved = d.xpos[b.trunk][:2] - start
        distance, drift = float(moved[0]), abs(float(moved[1]))
        reward = distance - DRIFT_WEIGHT * drift - self.reward.energy * energy / (self.mass * 9.81)
        if fell:
            reward -= FALL_PENALTY
        return {"distance": distance, "drift": drift, "energy": energy, "fell": float(fell), "reward": reward}

    def train(
        self,
        rhythm: Rhythm,
        iterations: int = 30,
        directions: int = 8,
        top: int = 4,
        noise: float = 0.08,
        step_size: float = 0.04,
        seed: int = 0,
        callback: Callable[[int, dict[str, float]], None] | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> TrainingLog:
        """ARS on the rhythm, in place (a live controller may be playing it). After each iteration the
        current rhythm walks once more for the log and the callback."""
        if not 1 <= top <= directions:
            raise CPGError("top must be between 1 and the number of directions")
        rng = np.random.default_rng(seed)
        log = TrainingLog()
        started = time.perf_counter()
        for it in range(iterations):
            if stop is not None and stop():
                break
            deltas = rng.standard_normal((directions, len(rhythm.theta)))
            base = rhythm.theta.copy()
            plus = np.array([self.rollout(rhythm, base + noise * dl)["reward"] for dl in deltas])
            minus = np.array([self.rollout(rhythm, base - noise * dl)["reward"] for dl in deltas])
            best = np.argsort(-np.maximum(plus, minus))[:top]
            spread = np.concatenate([plus[best], minus[best]]).std() + 1e-8
            rhythm.theta = base + step_size / (top * spread) * np.tensordot(
                plus[best] - minus[best], deltas[best], axes=1
            )
            walk = self.rollout(rhythm)
            log.attempts += 2 * directions + 1
            log.distances.append(walk["distance"])
            log.energies.append(walk["energy"])
            if callback is not None:
                callback(it + 1, walk)
        log.seconds = time.perf_counter() - started
        return log


class RhythmController:
    """Plays a rhythm on a robot in the live world; `restart` puts the clock back to zero."""

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
        target = self.rhythm.targets(self.home, float(data.time) - self.t0)
        data.ctrl[b.actuators] = np.clip(target, b.lower, b.upper)

    def status(self) -> dict[str, object]:
        return {"controller": "rhythm", "robot": self.body.robot}
