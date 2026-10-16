"""Behaviour cloning with numpy: demonstrations of a pick, two learners, and roll-outs on real physics.

The imitation lesson (learn/imitation.py) has a learner steer a desk arm through a pick: the gripper to a red
cube, close, lift, over a bin, open. Every 0.1 s it keeps what the arm could sense (the cube's position, the
gripper's position and how far its fingers are open: the observation) and what the learner did next (where they
steered the gripper and the grip: the action). Two learners copy that: k nearest neighbours looks up the closest
moments of the demonstrations, a small neural network fits one smooth function through all of them. Both are
rolled out on the same physics as the live arm, from the same start pose, at cube spots the lesson picks; the
hand-written pick (behaviors/arm.py) runs on the same spots for comparison.

How the actions are labelled matters more than the learner (measured on 32 scripted demonstrations, 40 test
spots, 2026-10-17): with "where the target was 0.1 s later" as the action, pauses taught the network to wait
for ever at the cube (0 of 40 with a 1 s pause before each move), and a teacher who drags faster one time
than the next made the labels disagree (5 to 25% at 8 demonstrations). Labelled as "where the learner's path
goes 3 cm ahead of the gripper, or the next grip change", the same human-like demonstrations (speeds 0.12 to
0.4 m/s, pauses up to 1.5 s, 6 mm aiming errors) gave the network 88 to 95% at 8 demonstrations. The constants
below say what each of the later fixes (ARRIVED, SURE, JITTER) was measured to do.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.behaviors.arm import DOWN, ArmMotion
from robotforge.design.scenes import DESK_TOP, WOOD
from robotforge.design.world import World, compile_world

ARM = "arm"
DESK = DESK_TOP  # m: the desk's top, 0.75 as in the desk lab scene
CUBE = 0.04  # m: the cube's edge, the size pick_and_place is tested with (tests/test_behaviors.py)
BIN_XY = (0.05, 0.28)  # m from the arm's base: beside the cube's region, inside the arm's reach
BIN_SIZE = (0.14, 0.14, 0.05)  # m: length, width, height
# Where cubes appear: a ring sector in front of the arm. The default arm (reach 0.45 m) holds its gripper
# pointing down 10 cm over the desk out to 0.34 m from its base, and to 0.10 m at the inside (measured with
# ArmMotion.reach_error); 2 cm in from both ends, and clear of the bin on the left.
REGION_R = (0.16, 0.32)  # m from the base
REGION_DEG = (-60.0, 35.0)  # heading from straight ahead, left positive
HOME_SPOT = (0.25, -0.05)  # m: the first cube, straight in front where it is easy to reach
DT = 0.1  # s between samples, and between a learned policy's decisions
CONTROL_HZ = 50.0  # the arm's controller: one IK step toward the goal per tick
SPEED = 0.25  # m/s: the gripper follows its target no faster (a real arm's teleoperation speed limit)
GRIP_HOLD = 0.3  # s the arm holds still after the grip changes: the fingers close in under 0.2 s (measured)
LOOK = 0.03  # m along the learner's path: the action is where the path goes this far ahead of the gripper
# m: a grip change is the action only from this close to where the learner made it. The arm settles within
# 3 mm of its target (gravity compensated); 6 mm high, the fingertips still reach 7 mm below the cube's middle.
ARRIVED = 0.006
CARRY = DESK + 0.10  # m: high enough to carry the cube over the bin's 5 cm rim with room to spare
TOP = DESK + 0.30  # m: the highest target allowed
FLOOR = DESK + 0.012  # m: the lowest; the fingertips are 1.2 cm under the tool point (templates/arm.py)
TRY_SECONDS = 12.0  # a roll-out that has not finished by then has failed (demonstrations take 4 to 9 s)
STUCK_SECONDS = 2.5  # a roll-out whose gripper has not moved 1 cm nor changed its grip for this long has failed
STUCK_MOVE = 0.01  # m
SETTLE = 0.5  # s after the cube lands in the bin with the gripper open, before it counts
# The network's grip is a number, 0 open to 1 closed; the arm switches only once it is this far from the grip it
# has (closes above 0.7, opens below 0.3). Before the ARRIVED rule it closed a centimetre or two over the cube
# at 0.5 (71% at 8 demonstrations, 83% at 0.7); with it the threshold matters little (12 sets of 5
# demonstrations: 83.0% at 0.5, worst set 54%; 83.3% at 0.7, worst set 62%).
SURE = 0.7
# MuJoCo's soft contacts let a held cube creep down between the fingers: lifted 10 cm through the sliders and held
# still, it slid 3.6 mm a second and was back on the desk 7.5 s after the lift, so a learner carrying at their own
# pace (one slider at a time, a pause between moves: 10 to 15 s) lost it. The no-slip pass stops the creep: the
# gripper-to-cube gap held at 5 mm for 20 s (an elliptic cone with impratio 10 still crept 0.15 mm a second).
# It costs about a third more time per roll-out (an IK pick 0.16 -> 0.21 s). Measured 2026-10-17.
NOSLIP = 5
# Samples kept of a stretch in which nothing moves (see Recording): before the learner's first move, 1 s; after
# it, 5 s. The pauses of a pick are recorded whole: at the cube before closing they are the samples labelled
# "close", and cutting them to 1 s took the network from 79% to 38% on five slow demonstrations (0.05 m/s, 2 s
# pauses, 2026-10-17), whose many moving samples then outweighed the close.
PAUSE_KEEP = 10
PAUSE_CAP = 50
STILL = 0.001  # m: less than this between two samples is not a move (the held arm sags 0.05 mm a second)
MAX_ROWS = 1500  # samples, 150 s of moving: a demonstration takes 5 to 40 s; past this the recording stops

# The network: three small nets whose answers are averaged. One net's success swung between 20 and 57% with
# its random start (4 demonstrations); three, with weight decay, held 57 to 75% (and 88 to 95% at 8).
NETS = 3
HIDDEN = 64
STEPS = 4000
LEARNING_RATE = 3e-3
DECAY = 3e-3
BATCH = 128
# m: in training the gripper's position is shaken this much and the target kept where it was, so the network
# also learns to come back to the path from a little off it: a fix for covariate shift (as DART does). On 11
# sets of 5 demonstrations (4 recorded through the viewer, 7 scripted with human-like speeds, pauses and aim,
# 2026-10-17) it took success at the 24 spots from 74% (25 to 95) to 83% (62 to 95), and beside a
# demonstration from 96% (60 to 100) to 100% in every set.
JITTER = 0.01
NEIGHBOURS = 3


def _text(zh: str, en: str) -> dict[str, str]:
    return {"zh": zh, "en": en}


# ---- the scene -------------------------------------------------------------------------------------------------


def build_world() -> World:
    return World.model_validate(
        {
            "name": "imitation",
            "floor": {"style": "tile"},
            "robots": [{"name": ARM, "template": "arm", "params": {"color": "teal"}, "pos": [0.0, 0.0, DESK]}],
            "objects": [
                {"name": "desk", "shape": "table", "pos": [0.2, 0.05], "size": [0.8, 1.0, DESK], "color": WOOD},
                {
                    "name": "cube",
                    "shape": "box",
                    "size": [CUBE, CUBE, CUBE],
                    "pos": [*HOME_SPOT, DESK + CUBE / 2],
                    "color": "red",
                },
                {
                    "name": "bin",
                    "shape": "bin",
                    "size": list(BIN_SIZE),
                    "pos": [*BIN_XY, DESK],
                    "color": "blue",
                    "fixed": True,
                },
            ],
        }
    )


def build_model() -> mujoco.MjModel:
    """The desk, the arm, the cube and the bin. The arm's motors cancel gravity on its links, as a real arm's
    controller does: without it the hobby servos sag 1.3 cm at the cube and 2.5 cm stretched out, and a
    learner would have to aim under every point. MuJoCo applies gravity compensation only if some body had it
    when the model was compiled, so it is set in the spec, not switched on afterwards."""
    compiled = compile_world(build_world())
    spec = compiled.spec
    for body in spec.bodies:
        if body.name.startswith(f"{ARM}/"):
            body.gravcomp = 1.0
    for joint in spec.joints:
        if joint.name.startswith(f"{ARM}/"):
            joint.actgravcomp = True
    model = spec.compile()
    model.opt.noslip_iterations = NOSLIP
    return model


def in_region(xy: Sequence[float]) -> bool:
    r = math.hypot(xy[0], xy[1])
    deg = math.degrees(math.atan2(xy[1], xy[0]))
    return REGION_R[0] <= r <= REGION_R[1] and REGION_DEG[0] <= deg <= REGION_DEG[1]


def random_spot(rng: np.random.Generator) -> np.ndarray:
    while True:
        p = rng.uniform(-REGION_R[1], REGION_R[1], 2)
        if in_region(p):
            return p


def test_spots(count: int = 24, seed: int = 2026) -> np.ndarray:
    """The same spread-out spots for every test, so runs compare: one random spot in each of `count` equal
    slices of the region (4 rings by count/4 headings)."""
    rng = np.random.default_rng(seed)
    rings, slices = 4, count // 4
    out = []
    for i in range(rings):
        for j in range(slices):
            # equal areas: radius by the square root
            r2 = rng.uniform(i, i + 1) / rings * (REGION_R[1] ** 2 - REGION_R[0] ** 2) + REGION_R[0] ** 2
            deg = REGION_DEG[0] + (j + rng.uniform()) / slices * (REGION_DEG[1] - REGION_DEG[0])
            out.append([math.sqrt(r2) * math.cos(math.radians(deg)), math.sqrt(r2) * math.sin(math.radians(deg))])
    return np.array(out)


def far_spot(demo_spots: Sequence[np.ndarray]) -> np.ndarray:
    """The spot of the region furthest from every demonstrated one (on a 1 cm grid)."""
    grid = [np.array([x, y]) for x in np.arange(0.0, 0.33, 0.01) for y in np.arange(-0.3, 0.3, 0.01)]
    grid = [p for p in grid if in_region(p)]
    if not demo_spots:
        return np.array(HOME_SPOT)
    demos = np.asarray(demo_spots)
    return max(grid, key=lambda p: float(np.min(np.linalg.norm(demos - p, axis=1))))


def nearest(spot: Sequence[float], demo_spots: Sequence[np.ndarray]) -> float:
    """m from `spot` to the closest demonstrated cube spot (inf without demonstrations)."""
    if not len(demo_spots):
        return math.inf
    return float(np.min(np.linalg.norm(np.asarray(demo_spots) - np.asarray(spot)[:2], axis=1)))


class Scene:
    """Addresses in one model of the scene, and putting the arm and the cube where a run starts."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.home = model.key("home").qpos.copy()
        cube = model.joint("cube")
        self.cube_q = int(cube.qposadr[0])
        self.cube_v = int(cube.dofadr[0])
        self.cube_body = int(model.body("cube").id)
        self.finger = int(model.joint(f"{ARM}/finger_left").qposadr[0])

    def start(self, data: mujoco.MjData, spot: Sequence[float]) -> None:
        """The arm in its home pose, still, gripper open; the cube resting at `spot` (m from the arm's base)."""
        data.qpos[:] = self.home
        data.qvel[:] = 0.0
        self.place_cube(data, spot)
        mujoco.mj_forward(self.model, data)

    def place_cube(self, data: mujoco.MjData, spot: Sequence[float]) -> None:
        data.qpos[self.cube_q : self.cube_q + 3] = [spot[0], spot[1], DESK + CUBE / 2 + 0.0005]
        data.qpos[self.cube_q + 3 : self.cube_q + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[self.cube_v : self.cube_v + 6] = 0.0

    def cube(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.cube_body].copy()

    def opening(self, data: mujoco.MjData) -> float:
        """m between the fingers' travel: both fingers slide together, so twice one finger's."""
        return 2.0 * float(data.qpos[self.finger])

    def in_bin(self, data: mujoco.MjData) -> bool:
        p = self.cube(data)
        half = np.array(BIN_SIZE[:2]) / 2 - 0.005  # inside the 5 mm walls
        return bool(np.all(np.abs(p[:2] - BIN_XY) <= half) and p[2] < DESK + BIN_SIZE[2] + 0.01)

    def fell(self, data: mujoco.MjData) -> bool:
        return bool(self.cube(data)[2] < DESK - 0.05)


# ---- steering the arm ------------------------------------------------------------------------------------------


class Follow:
    """The gripper's goal chases the target at up to SPEED; resolved-rate IK keeps the gripper over it,
    pointing down. A change of grip holds the arm still while the fingers move."""

    def __init__(self, model: mujoco.MjModel, scene: Scene) -> None:
        self.scene = scene
        self.arm = ArmMotion(model, ARM, scene.home)
        self.home_q = self.arm.q.copy()
        self.home_point = self.arm.ik.point(self.arm.chain, self.home_q)
        self.reset()

    def reset(self) -> None:
        self.q = self.home_q.copy()
        self.goal = self.home_point.copy()
        self.target = self.home_point.copy()
        self.grip = 0.0
        self.busy_until = -1.0
        self.last: float | None = None

    def set(self, target: Sequence[float], grip: float, now: float) -> None:
        target = clamp(target)
        grip = 1.0 if grip >= 0.5 else 0.0
        if grip != self.grip:
            self.grip = grip
            self.busy_until = now + GRIP_HOLD
        self.target = target

    def busy(self, now: float) -> bool:
        return now < self.busy_until

    def sync(self, data: mujoco.MjData) -> None:
        """Take over an arm something else has been driving: aim where it is, hold the grip it has."""
        self.q = data.qpos[self.arm.chain.qpos_adr].copy()
        self.goal = self.hand(data)
        self.target = self.goal.copy()
        self.grip = 1.0 if data.ctrl[self.arm.gripper] < 0.5 * self.arm.gripper_open else 0.0
        self.busy_until = -1.0
        self.last = None

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        now = float(data.time)
        dt = 0.0 if self.last is None else max(0.0, now - self.last)
        self.last = now
        if not self.busy(now):
            step = self.target - self.goal
            n = float(np.linalg.norm(step))
            if n > SPEED * dt:
                step *= SPEED * dt / n
            self.goal = self.goal + step
        arm = self.arm
        self.q, _ = arm.ik.solve(arm.chain, self.goal, self.q, iterations=3, damping=0.01, axis=(arm.tool_axis, DOWN))
        data.ctrl[arm.actuators] = self.q
        data.ctrl[arm.gripper] = 0.0 if self.grip else arm.gripper_open

    def hand(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.arm.site].copy()


def clamp(target: Sequence[float]) -> np.ndarray:
    """Keep a target over the desk, above its top and within a sensible height."""
    x, y, z = (float(v) for v in target)
    return np.array([min(max(x, -0.15), 0.55), min(max(y, -0.4), 0.5), min(max(z, FLOOR), TOP)])


def observe(scene: Scene, follow: Follow, data: mujoco.MjData) -> np.ndarray:
    """What the arm senses: the cube's position, the gripper's position (m) and its finger opening (m)."""
    return np.concatenate([scene.cube(data), follow.hand(data), [scene.opening(data)]])


# ---- demonstrations ---------------------------------------------------------------------------------------------


@dataclass
class Demo:
    spot: np.ndarray  # where the cube started, m from the arm's base
    X: np.ndarray  # observations, one row per kept sample
    Y: np.ndarray  # actions: target x, y, z (m) and grip (0 open, 1 closed)
    path: list[list[float]]  # the gripper every DT, for the drawing
    seconds: float
    samples: int  # taken, before the ones while the fingers moved were left out
    rows: list[tuple[np.ndarray, np.ndarray, bool]] = field(default_factory=list)  # as recorded, before labelling
    correction: bool = False  # taken over from a try that went wrong, from where the arm had got to

    def describe(self, index: int) -> dict[str, object]:
        return {
            "n": index + 1,
            "spot": [round(100 * float(v)) for v in self.spot],
            "kept": int(len(self.X)),
            "seconds": round(self.seconds, 1),
        }


@dataclass
class Recording:
    """Samples of one demonstration as it happens: (observation, the command just given, fingers moving).

    A recording opens with the lesson, with New cube and with Reset, so the time a learner spends reading before
    their first move would be recorded too: two minutes of it made 1250 samples of the arm standing at home,
    four times the real samples of all five demonstrations, and took the network from 83% to 71% at the test
    spots (the scripted learner, 2026-10-17). Before the first move a still stretch keeps PAUSE_KEEP samples:
    they all see the same and are labelled with the same move (label), so the rest added weight and nothing else
    (trimmed, the same two minutes gave 62 samples and 79%), and that time is not counted in the demonstration's
    seconds. Later pauses keep up to PAUSE_CAP, so a break in the middle cannot flood it either."""

    spot: np.ndarray
    started: float
    correction: bool = False
    rows: list[tuple[np.ndarray, np.ndarray, bool]] = field(default_factory=list)
    still: int = 0  # samples in a row in which nothing moved
    moved: bool = False  # the learner has made their first move
    full: bool = False  # MAX_ROWS reached: the recording takes no more

    def add(self, obs: np.ndarray, target: np.ndarray, grip: float, busy: bool, now: float | None = None) -> None:
        row = (obs.copy(), np.array([*target, grip], float), busy)
        # the learner's first command, not the arm settling into its start pose
        self.moved = self.moved or bool(self.rows and not np.array_equal(self.rows[-1][1], row[1]))
        if self.rows and _still(self.rows[-1], row):
            self.still += 1
            if self.still > (PAUSE_CAP if self.moved else PAUSE_KEEP):
                if not self.moved and now is not None:
                    self.started = now - PAUSE_KEEP * DT  # the demonstration has not begun yet
                return
        else:
            self.still = 0
        if len(self.rows) >= MAX_ROWS:
            self.full = True
            return
        self.rows.append(row)

    def finish(self, now: float) -> Demo:
        X, Y = label(self.rows)
        path = [[round(float(v), 3) for v in obs[3:6]] for obs, _, _ in self.rows[::2]]
        rows = list(self.rows)
        return Demo(self.spot.copy(), X, Y, path, now - self.started, len(rows), rows, self.correction)


def label(rows: Sequence[tuple[np.ndarray, np.ndarray, bool]]) -> tuple[np.ndarray, np.ndarray]:
    """Pair every sample with what the learner did next: the point their path reaches LOOK metres ahead of the
    gripper, or, if they changed the grip first, the grip change; a grip change counts only once the gripper is
    where it happened (within ARRIVED), and until then the action is to get there. Samples taken while the
    fingers moved (the arm holds still then, whatever the learner does) are left out.

    Looking a distance ahead rather than a time ahead makes the action the same whether the learner dragged
    fast or slow; and a sample during a pause gets the move that ended the pause, so the copy does not learn
    to wait (see the module notes for what the other labels did). Without the ARRIVED rule the last 3 cm of
    the way down to the cube were labelled "close": the copies closed 2 to 3 cm over the cube's middle and
    pinched its top or missed it (found replaying demonstrations recorded through the viewer, 2026-10-17)."""
    X, Y = [], []
    if not rows:
        return np.zeros((0, 7)), np.zeros((0, 4))
    # Every sample looks ahead along the commands after it: as arrays, so a long demonstration labels in
    # milliseconds (one row at a time, 3000 samples held the physics up for 3 s).
    cmds = np.array([cmd for _, cmd, _ in rows], float)
    for i, (obs, cmd, busy) in enumerate(rows):
        if busy:
            continue
        hand = obs[3:6]
        grip_now = cmds[i - 1, 3] if i else cmd[3]
        later = cmds[i:]
        changed = later[:, 3] != grip_now
        far = np.linalg.norm(later[:, :3] - hand, axis=1) >= LOOK
        first_change = int(np.argmax(changed)) if changed.any() else len(later)
        first_far = int(np.argmax(far)) if far.any() else len(later)
        if first_change < len(later) and first_change <= first_far:
            there = np.linalg.norm(later[first_change, :3] - hand) <= ARRIVED
            action = later[first_change] if there else np.array([*later[first_change, :3], grip_now])
        elif first_far < len(later):
            action = np.array([*later[first_far, :3], grip_now])
        else:
            action = cmds[-1]
        X.append(obs)
        Y.append(action)
    return np.array(X, float).reshape(-1, 7), np.array(Y, float).reshape(-1, 4)


def _still(before: tuple[np.ndarray, np.ndarray, bool], after: tuple[np.ndarray, np.ndarray, bool]) -> bool:
    """Nothing moved between two samples: the same command, and the cube, the gripper and its fingers within
    STILL of where they were."""
    return bool(np.array_equal(before[1], after[1]) and np.max(np.abs(after[0] - before[0])) < STILL)


def stack(demos: Sequence[Demo]) -> tuple[np.ndarray, np.ndarray]:
    if not demos:
        return np.zeros((0, 7)), np.zeros((0, 4))
    return np.concatenate([d.X for d in demos]), np.concatenate([d.Y for d in demos])


# ---- the learners -----------------------------------------------------------------------------------------------

Policy = Callable[[np.ndarray], np.ndarray]


class Copier:
    """k nearest neighbours: the actions of the NEIGHBOURS closest samples (by distance in metres, the finger
    opening counted like a position), the closest weighted most. Nothing to train: it keeps the samples."""

    kind = "knn"

    def __init__(self, X: np.ndarray, Y: np.ndarray, k: int = NEIGHBOURS) -> None:
        if not len(X):
            raise ValueError("no demonstrations to copy yet")
        self.X, self.Y, self.k = X, Y, min(k, len(X))
        self.seconds = 0.0
        self.loss = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self.X - obs, axis=1)
        idx = np.argpartition(d, self.k - 1)[: self.k]
        w = 1.0 / (d[idx] + 1e-3)
        out = (self.Y[idx] * w[:, None]).sum(0) / w.sum()
        out[3] = float(out[3] >= 0.5)  # the grip by the neighbours' weighted vote: open or closed, never between
        return out


class Net:
    """One multilayer perceptron, 7 -> HIDDEN -> HIDDEN -> 4 with tanh, trained with Adam on squared error.
    It predicts the move from the gripper (target minus gripper) and the grip, both scaled to unit spread."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        T = Y.copy()
        T[:, :3] -= X[:, 3:6]
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-6
        self.y_mean, self.y_std = T.mean(0), T.std(0) + 1e-6
        Tn = (T - self.y_mean) / self.y_std
        sizes = (X.shape[1], HIDDEN, HIDDEN, T.shape[1])
        self.W = [rng.normal(0.0, 1.0 / math.sqrt(a), (a, b)) for a, b in zip(sizes, sizes[1:], strict=False)]
        self.b = [np.zeros(b) for b in sizes[1:]]
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for it in range(1, STEPS + 1):
            idx = rng.integers(0, len(X), min(BATCH, len(X)))
            x, t = X[idx], Tn[idx]
            if JITTER:
                shake = rng.normal(0.0, JITTER, (len(idx), 3))
                x, t = x.copy(), t.copy()
                x[:, 3:6] += shake
                t[:, :3] -= shake / self.y_std[:3]
            h = [(x - self.x_mean) / self.x_std]
            for W, b in zip(self.W[:-1], self.b[:-1], strict=True):
                h.append(np.tanh(h[-1] @ W + b))
            g = 2.0 * (h[-1] @ self.W[-1] + self.b[-1] - t) / len(idx)
            gW: list[np.ndarray] = [np.zeros(0)] * len(self.W)
            gb: list[np.ndarray] = [np.zeros(0)] * len(self.b)
            for k in range(len(self.W) - 1, -1, -1):
                gW[k] = h[k].T @ g + DECAY * self.W[k]
                gb[k] = g.sum(0)
                if k:
                    g = (g @ self.W[k].T) * (1.0 - h[k] ** 2)
            for i, (p, grad) in enumerate(zip(params, gW + gb, strict=True)):
                m[i] = 0.9 * m[i] + 0.1 * grad
                v[i] = 0.999 * v[i] + 0.001 * grad**2
                p -= LEARNING_RATE * (m[i] / (1 - 0.9**it)) / (np.sqrt(v[i] / (1 - 0.999**it)) + 1e-8)
        self.loss = float(np.mean((self._forward((X - self.x_mean) / self.x_std) - Tn) ** 2))

    def _forward(self, xn: np.ndarray) -> np.ndarray:
        h = xn
        for W, b in zip(self.W[:-1], self.b[:-1], strict=True):
            h = np.tanh(h @ W + b)
        return h @ self.W[-1] + self.b[-1]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self._forward((obs - self.x_mean) / self.x_std) * self.y_std + self.y_mean
        out[:3] += obs[3:6]
        return out


class Network:
    """NETS small nets trained from different random starts; the action is their average."""

    kind = "mlp"

    def __init__(self, X: np.ndarray, Y: np.ndarray, seed: int = 0) -> None:
        if len(X) < 2:
            raise ValueError("no demonstrations to learn from yet")
        t0 = time.perf_counter()
        self.nets = [Net(X, Y, seed=seed * NETS + k) for k in range(NETS)]
        self.seconds = time.perf_counter() - t0
        self.loss = float(np.mean([n.loss for n in self.nets]))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.mean([net(obs) for net in self.nets], axis=0)


def learn(kind: str, demos: Sequence[Demo]) -> Copier | Network:
    X, Y = stack(demos)
    return Copier(X, Y) if kind == "knn" else Network(X, Y)


# ---- runs ---------------------------------------------------------------------------------------------------------


@dataclass
class Outcome:
    spot: np.ndarray
    success: bool
    seconds: float
    path: list[list[float]]
    why: dict[str, str]  # how it ended, in words


class Pilot:
    """A learned policy at the controls: every DT it looks (observe) and sets the target and the grip. While the
    fingers move the arm holds still and the policy is not asked."""

    def __init__(self, scene: Scene, follow: Follow, policy: Policy, now: float, spot: Sequence[float]) -> None:
        self.scene, self.follow, self.policy = scene, follow, policy
        self.started = now
        self.next = now
        self.spot = np.asarray(spot, float)
        self.path: list[list[float]] = []
        # where the fingers first closed: the gripper's distance from the cube across the desk, and its height over
        # the cube's middle (m)
        self.closed_at: list[float] | None = None
        self.done_at: float | None = None
        self.outcome: Outcome | None = None
        self.anchor: tuple[np.ndarray, float, float] | None = None  # (gripper, grip, time) of the last progress

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        now = float(data.time)
        if self.outcome is None and now + 1e-9 >= self.next:
            self.next += DT
            obs = observe(self.scene, self.follow, data)
            self.path.append([round(float(v), 3) for v in obs[3:6]])
            if not self.follow.busy(now):
                action = self.policy(obs)
                grip = self.follow.grip
                if abs(float(action[3]) - grip) > SURE:
                    grip = 1.0 - grip
                # Only a close that happens counts: a network hovering at 0.6 over the cube never closed, and was
                # reported as having closed there and missed.
                if grip >= 0.5 and self.follow.grip < 0.5 and self.closed_at is None:
                    self.closed_at = [float(np.linalg.norm(obs[:2] - obs[3:5])), float(obs[5] - obs[2])]
                self.follow.set(action[:3], grip, now)
            self._judge(data, now, obs[3:6])
        self.follow.update(model, data)

    def _judge(self, data: mujoco.MjData, now: float, hand: np.ndarray) -> None:
        scene, follow = self.scene, self.follow
        anchor = self.anchor
        if anchor is None or np.linalg.norm(hand - anchor[0]) > STUCK_MOVE or follow.grip != anchor[1]:
            self.anchor = anchor = (hand.copy(), follow.grip, now)
        if self.done_at is None and scene.in_bin(data) and follow.grip < 0.5 and not follow.busy(now):
            self.done_at = now
        if self.done_at is not None and now - self.done_at >= SETTLE:
            self._end(scene.in_bin(data), now, _text("放进了盒子", "put it in the bin"))
        elif scene.fell(data):
            self._end(False, now, _text("方块掉到了桌子下面", "the cube fell off the desk"))
        elif now - self.started >= TRY_SECONDS or now - anchor[2] >= STUCK_SECONDS:
            self._end(False, now, self._why_not(data))

    def _why_not(self, data: mujoco.MjData) -> dict[str, str]:
        held = self.scene.cube(data)[2] > DESK + CUBE
        if self.closed_at is None:
            return _text("一直没合上手爪", "never closed the gripper")
        off = 100 * self.closed_at[0]
        if not held:
            return _text(
                f"在离方块 {off:.0f} 厘米的地方合上了手爪，没抓到",
                f"closed the gripper {off:.0f} cm from the cube and missed it",
            )
        return _text("抓起来了，却没放进盒子", "picked it up but never put it in the bin")

    def _end(self, success: bool, now: float, why: dict[str, str]) -> None:
        self.outcome = Outcome(self.spot.copy(), success, now - self.started, self.path, why)


def rollout(model: mujoco.MjModel, policy: Policy, spot: Sequence[float]) -> Outcome:
    """One try of `policy` from the home pose with the cube at `spot`, on a fresh MjData of `model` (a private
    copy: the live arm is not touched). Controllers run as the session runs them: first, then mj_step."""
    scene = Scene(model)
    data = mujoco.MjData(model)
    scene.start(data, spot)
    follow = Follow(model, scene)
    pilot = Pilot(scene, follow, policy, float(data.time), spot)
    due = 0.0
    while pilot.outcome is None:
        if data.time + 1e-9 >= due:
            pilot.update(model, data)
            due = float(data.time) + 1.0 / CONTROL_HZ
        mujoco.mj_step(model, data)
    return pilot.outcome


class ScriptedPick:
    """The hand-written pick for comparison: behaviors/arm.py plans the grasp from the cube's position and the
    bin's, and inverse kinematics does the rest. It needs no demonstrations."""

    def __init__(self, model: mujoco.MjModel, scene: Scene, data: mujoco.MjData, spot: Sequence[float]) -> None:
        self.scene = scene
        self.arm = ArmMotion(model, ARM, scene.home)
        self.started = float(data.time)
        self.spot = np.asarray(spot, float)
        self.path: list[list[float]] = []
        self.next = self.started
        self.done_at: float | None = None
        self.outcome: Outcome | None = None
        place = np.array([*BIN_XY, DESK + 0.005])
        self.arm.pick_and_place(scene.cube(data), CUBE / 2, place, DESK + BIN_SIZE[2])

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.arm.update(model, data)
        now = float(data.time)
        if self.outcome is not None or now + 1e-9 < self.next:
            return
        self.next += DT
        self.path.append([round(float(v), 3) for v in data.site_xpos[self.arm.site]])
        if self.arm.done and self.done_at is None:
            self.done_at = now
        if self.done_at is not None and now - self.done_at >= SETTLE:
            ok = self.scene.in_bin(data)
            why = _text("放进了盒子", "put it in the bin") if ok else _text("没放进盒子", "missed the bin")
            self.outcome = Outcome(self.spot.copy(), ok, now - self.started, self.path, why)
        elif now - self.started >= 2 * TRY_SECONDS:
            self.outcome = Outcome(self.spot.copy(), False, now - self.started, self.path, _text("超时", "timed out"))


def scripted_rollout(model: mujoco.MjModel, spot: Sequence[float]) -> Outcome:
    scene = Scene(model)
    data = mujoco.MjData(model)
    scene.start(data, spot)
    pick = ScriptedPick(model, scene, data, spot)
    due = 0.0
    while pick.outcome is None:
        if data.time + 1e-9 >= due:
            pick.update(model, data)
            due = float(data.time) + 1.0 / pick.arm.rate_hz
        mujoco.mj_step(model, data)
    return pick.outcome


def private(model: mujoco.MjModel) -> mujoco.MjModel:
    """A copy of the model for runs off the live world (tests on another thread)."""
    return copy.deepcopy(model)


# ---- grading -------------------------------------------------------------------------------------------------------

BINS_CM = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, math.inf)  # distance to the nearest demonstration
NEAR_CM = 3.0  # a test spot this close to a demonstrated one is "near"
FAR_CM = 6.0  # and from here on "far"


def by_distance(results: Sequence[tuple[float, bool]]) -> tuple[list[float], list[float | None]]:
    """Success in % per distance band (cm to the nearest demonstration): the bands' middles and their rates
    (None where no spot fell)."""
    xs, ys = [], []
    for lo, hi in zip(BINS_CM, BINS_CM[1:], strict=False):
        inside = [ok for d, ok in results if lo <= d < hi]
        xs.append(lo + (hi - lo) / 2 if math.isfinite(hi) else lo + 2.0)
        ys.append(round(100.0 * sum(inside) / len(inside), 1) if inside else None)
    return xs, ys


def rate(flags: Sequence[bool]) -> float | None:
    return round(100.0 * sum(flags) / len(flags), 1) if flags else None
