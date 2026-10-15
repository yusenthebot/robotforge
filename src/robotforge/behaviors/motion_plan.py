"""Planning arm motions around obstacles: RRT-Connect in joint space, then shortcuts.

A move straight from one joint configuration to another sweeps the arm through whatever is in the way.
RRT-Connect (Kuffner and LaValle, 2000) grows two trees of collision-free configurations, one from the
start and one from the goal: each round it picks a random configuration, grows one tree a step toward it,
then lets the other tree reach for the new node as far as it can without touching anything. When they
meet, the path is the branch back to each root. Random paths zigzag, so shortcutting then tries to join
two random points of the path directly and keeps the joint whenever that stretch is free too.

Collisions are checked in a private copy of the world, every geom the arm moves against everything else
(obstacles, the floor, the rest of the robot), leaving out pairs that already touch at home.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain

PENETRATION = 0.0005  # m: contacts within the robot shallower than this are a graze, not a collision
RESOLUTION = 0.03  # rad: segments are checked every this much joint motion
# Room kept from obstacles: a real arm lags its plan (servos give a little under load), so a path that
# only just clears a fence in planning scrapes it when followed. The floor gets less: targets sit low.
CLEARANCE = 0.045  # m
FLOOR_CLEARANCE = 0.005  # m


class ArmSpace:
    """An arm's joint space and its collision check, in its own copy of the world."""

    def __init__(self, model: mujoco.MjModel, chain: Chain, base_qpos: np.ndarray) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.chain = chain
        self.base_qpos = base_qpos.copy()
        self.lower, self.upper = chain.lower.copy(), chain.upper.copy()
        first = int(model.dof_bodyid[chain.dof_adr[0]])
        self.moving = {b for b in range(model.nbody) if _descends_from(model, b, first)}
        root = int(model.body_rootid[first])
        # Fixed things (props and the ground, geoms of the world body) are kept CLEARANCE away, measured as
        # distances: contact margins changed at run time were not reliable for every pair in a big world.
        # Everything else (other robots, loose objects) only must not overlap.
        own = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in self.moving and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        fixed = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0 and model.geom_contype[g]]
        self.near: list[tuple[int, int, float]] = []  # (arm geom, fixed geom, room it needs)
        for o in fixed:
            ground = int(model.geom_type[o]) in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
            self.near += [(g, o, FLOOR_CLEARANCE if ground else CLEARANCE) for g in own]
        self.checks = 0
        # Shapes of the robot itself already overlapping at home are the model (coarse imported shapes),
        # not a crash. Only the robot's own: an obstacle that happens to overlap the home pose still counts.
        self.ignored: set[tuple[int, int]] = set()
        self.ignored = {
            (g1, g2)
            for g1, g2 in self.clashes(base_qpos[chain.qpos_adr])
            if all(int(model.body_rootid[model.geom_bodyid[g]]) == root and model.geom_bodyid[g] != 0 for g in (g1, g2))
        }

    def pose(self, q: np.ndarray) -> None:
        d = self.data
        d.qpos[:] = self.base_qpos
        d.qpos[self.chain.qpos_adr] = q
        mujoco.mj_kinematics(self.model, d)

    def clashes(self, q: np.ndarray) -> set[tuple[int, int]]:
        """Geom pairs that overlap at `q` with a moving part of the arm on at least one side (MuJoCo already
        leaves out a link touching its own parent)."""
        self.checks += 1
        self.pose(q)
        m, d = self.model, self.data
        mujoco.mj_collision(m, d)
        pairs = set()
        for con in d.contact[: d.ncon]:
            if con.dist > -PENETRATION:
                continue
            g1, g2 = int(con.geom1), int(con.geom2)
            if int(m.geom_bodyid[g1]) in self.moving or int(m.geom_bodyid[g2]) in self.moving:
                pairs.add((min(g1, g2), max(g1, g2)))
        for g, o, room in self.near:
            if mujoco.mj_geomDistance(m, d, g, o, room, None) < room:
                pairs.add((min(g, o), max(g, o)))
        return pairs

    def free(self, q: np.ndarray) -> bool:
        if np.any(q < self.lower - 1e-9) or np.any(q > self.upper + 1e-9):
            return False
        return not (self.clashes(q) - self.ignored)

    def segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Every configuration on the straight joint-space line from a to b, at RESOLUTION spacing."""
        steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / RESOLUTION)))
        return all(self.free(a + (b - a) * (k / steps)) for k in range(1, steps + 1))

    def first_clash(self, a: np.ndarray, b: np.ndarray) -> float | None:
        """Where along the straight line from a to b (0 to 1) the arm first hits something, if it does."""
        steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / RESOLUTION)))
        for k in range(1, steps + 1):
            if not self.free(a + (b - a) * (k / steps)):
                return k / steps
        return None

    def tool(self, q: np.ndarray) -> np.ndarray:
        """Where the tool point is at `q` (world frame)."""
        self.pose(q)
        d, c = self.data, self.chain
        return d.xpos[c.body] + d.xmat[c.body].reshape(3, 3) @ c.local_point


@dataclass
class Tree:
    nodes: list[np.ndarray] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q)
        self.parents.append(parent)
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(np.asarray(self.nodes) - q, axis=1)))

    def branch(self, index: int) -> list[np.ndarray]:
        """From the root to node `index`."""
        out = []
        while index >= 0:
            out.append(self.nodes[index])
            index = self.parents[index]
        return out[::-1]


@dataclass
class Plan:
    path: list[np.ndarray] | None  # joint configurations from start to goal, or None if none was found
    trees: tuple[Tree, Tree]
    seconds: float
    checks: int

    @property
    def nodes(self) -> int:
        return len(self.trees[0].nodes) + len(self.trees[1].nodes)


def rrt_connect(
    space: ArmSpace,
    start: np.ndarray,
    goal: np.ndarray,
    step: float = 0.2,
    max_nodes: int = 4000,
    seed: int = 0,
) -> Plan:
    """Grow trees from both ends until they meet; `step` is the longest joint-space hop (rad)."""
    rng = np.random.default_rng(seed)
    started, checks0 = time.perf_counter(), space.checks
    a, b = Tree(), Tree()
    a.add(np.asarray(start, float), -1)
    b.add(np.asarray(goal, float), -1)
    if not (space.free(a.nodes[0]) and space.free(b.nodes[0])):
        return Plan(None, (a, b), time.perf_counter() - started, space.checks - checks0)
    if space.segment_free(a.nodes[0], b.nodes[0]):
        return Plan([a.nodes[0], b.nodes[0]], (a, b), time.perf_counter() - started, space.checks - checks0)
    grow, other = a, b
    while len(a.nodes) + len(b.nodes) < max_nodes:
        sample = rng.uniform(space.lower, space.upper)
        new = _extend(space, grow, sample, step)
        if new is not None:
            meet = _connect(space, other, grow.nodes[new], step)
            if meet is not None:
                ends = grow.branch(new), other.branch(meet)
                path = ends[0] + ends[1][::-1] if grow is a else ends[1] + ends[0][::-1]
                return Plan(path, (a, b), time.perf_counter() - started, space.checks - checks0)
        grow, other = other, grow
    return Plan(None, (a, b), time.perf_counter() - started, space.checks - checks0)


def _extend(space: ArmSpace, tree: Tree, toward: np.ndarray, step: float) -> int | None:
    """One hop from the nearest node toward `toward`; the new node's index, or None if blocked."""
    near = tree.nearest(toward)
    origin = tree.nodes[near]
    gap = toward - origin
    length = float(np.linalg.norm(gap))
    target = toward if length <= step else origin + gap * (step / length)
    if not space.segment_free(origin, target):
        return None
    return tree.add(target, near)


def _connect(space: ArmSpace, tree: Tree, toward: np.ndarray, step: float) -> int | None:
    """Hop toward `toward` until it is reached (its index) or something is in the way (None)."""
    while True:
        new = _extend(space, tree, toward, step)
        if new is None:
            return None
        if np.allclose(tree.nodes[new], toward):
            return new


def shortcut(space: ArmSpace, path: list[np.ndarray], tries: int = 60, seed: int = 0) -> list[np.ndarray]:
    """Join two random points of the path directly whenever that stretch is free; repeat."""
    rng = np.random.default_rng(seed)
    path = [np.asarray(q, float) for q in path]
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i < 2:
            continue
        if space.segment_free(path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return path


def joint_length(path: list[np.ndarray]) -> float:
    """How far the joints turn along the path, in total (rad, summed over the hops)."""
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:], strict=False)))


def resample(path: list[np.ndarray], spacing: float = 0.05) -> list[np.ndarray]:
    """The path with configurations every `spacing` rad of joint motion, for drawing and for following."""
    out = [path[0]]
    for a, b in zip(path, path[1:], strict=False):
        steps = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / spacing)))
        out += [a + (b - a) * (k / steps) for k in range(1, steps + 1)]
    return out


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return body == ancestor
