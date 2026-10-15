"""Find a pose a legged robot can stand in: feet under the hips, body over its feet, symmetric and tall."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robotforge.design.importing.geometry import Geometry, clusters, compile_spec, pose_model
from robotforge.design.importing.plan import MOVABLE, build_spec
from robotforge.design.importing.urdf import JointInfo, StagedUrdf

STANCE_SAMPLES = 4000
STANCE_CANDIDATES = 6
STANCE_CLIMB_STEPS = 600
MAX_FOOT_SPREAD = 0.05  # m: all feet must reach the same floor
MIN_LEG_DROP = 0.04  # m: and hang at least this far below their shoulders
STANCE_REACH = 1.6  # rad: how far a joint may stray from its zero pose while looking for a stance


@dataclass
class _Limb:
    joints: list[JointInfo]
    bodies: set[int]
    hip_joint: str


@dataclass
class StanceResult:
    """Candidate stances, best first, and whether the time budget cut the search short."""

    poses: list[dict[str, float]]
    stopped_early: bool = False


def find_stances(staged: StagedUrdf, urdf: Path, home: dict[str, float], deadline: float = math.inf) -> StanceResult:
    """Joint poses, best first, where the feet sit under the hips and the body is over its feet.

    The search stops refining once `deadline` (a time.monotonic() value) has passed.
    """
    if time.monotonic() > deadline:
        return StanceResult([], stopped_early=True)
    spec = build_spec(urdf)
    model, data = compile_spec(spec)
    pose_model(model, data, home)
    geo = Geometry(model, data)
    limbs = _limbs(staged, geo)
    if len(limbs) < 2:
        return StanceResult([])
    search = _StanceSearch(model, data, geo, limbs, geo.body_id(staged.root), home, deadline)
    poses = search.run()
    return StanceResult(poses, search.stopped_early)


class _StanceSearch:
    """Random sampling followed by hill climbing on a stance score (kinematics only, no simulation)."""

    def __init__(
        self, model, data, geo: Geometry, limbs: list[_Limb], root: int, home: dict[str, float], deadline: float
    ):
        self.model, self.data, self.geo, self.limbs, self.root, self.home = model, data, geo, limbs, root, home
        self.deadline = deadline
        self.stopped_early = False
        self.rng = np.random.default_rng(1)
        self.names: list[str] = []
        self.adr: list[int] = []
        lows, highs = [], []
        for limb in limbs:
            for j in limb.joints:
                lo, hi = (j.lower, j.upper) if j.lower is not None and j.upper is not None else (-math.pi, math.pi)
                center = home.get(j.name, 0.0)  # stay within a natural reach of the zero pose
                lo, hi = max(lo, center - STANCE_REACH), min(hi, center + STANCE_REACH)
                self.names.append(j.name)
                self.adr.append(int(model.jnt_qposadr[geo.joint_id(j.name)]))
                lows.append(lo)
                highs.append(hi)
        self.lo, self.hi = np.array(lows), np.array(highs)
        self.start = np.array([home.get(n, 0.0) for n in self.names])
        self.base_q = data.qpos.copy()
        self.reference_height = max(float(data.xpos[root][2] - geo.lowest), 0.05)

    def pose(self, values: np.ndarray) -> dict[str, float]:
        out = dict(self.home)
        out.update({n: float(v) for n, v in zip(self.names, values, strict=True)})
        return out

    def score(self, values: np.ndarray, strict: bool = False) -> float:
        q = self.base_q.copy()
        q[self.adr] = values
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        deviation = float(np.abs(values - self.start).sum())
        return _stance_score(
            self.model, self.data, self.geo, self.limbs, self.root, deviation, self.reference_height, strict
        )

    def run(self) -> list[dict[str, float]]:
        n = len(self.names)
        samples = []
        for k in range(STANCE_SAMPLES):
            if k % 64 == 0 and self._out_of_time():
                break
            if k % 2:
                values = self.rng.uniform(self.lo, self.hi)
            else:
                values = np.clip(self.start + self.rng.normal(0, 0.5, n), self.lo, self.hi)
            samples.append((self.score(values), values))
        samples.sort(key=lambda s: s[0])
        seeds: list[np.ndarray] = []
        for score, values in samples:
            if score == math.inf:
                break
            if all(float(np.abs(values - other).max()) > 0.25 for other in seeds):
                seeds.append(values)
            if len(seeds) >= STANCE_CANDIDATES:
                break
        refined = [self._climb(v) for v in seeds]
        refined.sort(key=lambda s: s[0])
        # Only poses with every foot on the same floor, below its shoulder, count as stances at all.
        return [self.pose(v) for score, v in refined if score != math.inf and self.score(v, strict=True) != math.inf]

    def _out_of_time(self) -> bool:
        if time.monotonic() > self.deadline:
            self.stopped_early = True
        return self.stopped_early

    def _climb(self, values: np.ndarray) -> tuple[float, np.ndarray]:
        best, best_score = values, self.score(values)
        step = 0.2
        for i in range(STANCE_CLIMB_STEPS):
            if i % 50 == 0 and self._out_of_time():
                break
            trial = np.clip(
                best + self.rng.normal(0, step, len(best)) * (self.rng.random(len(best)) < 0.4), self.lo, self.hi
            )
            trial_score = self.score(trial)
            if trial_score < best_score:
                best, best_score = trial, trial_score
            step = max(0.01, step * 0.997)
        return best_score, best


def _limbs(staged: StagedUrdf, geo: Geometry) -> list[_Limb]:
    """Movable subtrees hanging off the base that reach down toward the ground."""
    cluster = clusters(staged)
    root_cluster = cluster[staged.root]
    child_joints: dict[str, list[JointInfo]] = {}
    for j in staged.joints:
        child_joints.setdefault(j.parent, []).append(j)

    def collect(link: str) -> list[str]:
        out = [link]
        for j in child_joints.get(link, []):
            out += collect(j.child)
        return out

    depth: dict[str, int] = {staged.root: 0}
    stack = [staged.root]
    while stack:
        link = stack.pop()
        for j in child_joints.get(link, []):
            depth[j.child] = depth[link] + 1
            stack.append(j.child)

    top = max(geo.low.values(), default=0.0)
    reach_limit = geo.lowest + 0.3 * max(top - geo.lowest, 0.2)
    limbs = []
    for link in [n for n, label in cluster.items() if label == root_cluster]:
        for j in child_joints.get(link, []):
            if j.type not in MOVABLE or j.mimic_of:
                continue
            links = collect(j.child)
            bodies = {geo.body_id(n) for n in links if geo.body_id(n) >= 0}
            lows = [geo.low[b] for b in bodies if b in geo.low]
            if not lows or min(lows) > reach_limit:
                continue
            joints = [x for x in staged.joints if x.child in links and x.type in MOVABLE and not x.mimic_of]
            # The feet belong under the leg's own shoulder: with a hip-abduction joint first, that is the second
            # joint (the first sits at the body's edge, and chasing it would draw the feet in under the trunk).
            order = sorted(joints, key=lambda x: depth[x.child])
            reference = order[1] if len(order) >= 3 else order[0]
            limbs.append(_Limb(joints, bodies, reference.name))
    return limbs


def _stance_score(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geo: Geometry,
    limbs: list[_Limb],
    root: int,
    deviation: float,
    reference_height: float,
    strict: bool = False,
) -> float:
    feet, hips = [], []
    for limb in limbs:
        lowest = (0.0, 0.0, math.inf)
        for g in geo.collision_geoms:
            if int(model.geom_bodyid[g]) in limb.bodies:
                point = geo.geom_low(g)
                if point[2] < lowest[2]:
                    lowest = point
        if lowest[2] == math.inf:
            return math.inf
        feet.append(lowest)
        hips.append(data.xanchor[geo.joint_id(limb.hip_joint)])
    feet_arr, hips_arr = np.array(feet), np.array(hips)
    com = data.subtree_com[root][:2]
    com_offset = float(np.linalg.norm(com - feet_arr[:, :2].mean(axis=0)))
    spread = float(feet_arr[:, 2].max() - feet_arr[:, 2].min())
    foot_offsets = feet_arr[:, :2] - hips_arr[:, :2]
    hip_offset = float(np.linalg.norm(foot_offsets, axis=1).mean())
    body_height = float(data.xpos[root][2] - feet_arr[:, 2].mean())
    if body_height < 0.02:
        return math.inf
    leg_drop = float((hips_arr[:, 2] - feet_arr[:, 2]).min())  # every foot must hang below its own shoulder
    if strict and (spread > MAX_FOOT_SPREAD or leg_drop < MIN_LEG_DROP):
        return math.inf
    asymmetry = _asymmetry(foot_offsets, hips_arr[:, :2] - data.xpos[root][:2])
    margin = _support_margin(feet_arr[:, :2], com)
    # Prefer a tall, left-right symmetric stance close to the zero pose over a crouch that merely balances.
    return (
        4.0 * com_offset
        + 12.0 * spread
        + 20.0 * max(0.0, MIN_LEG_DROP * 1.5 - leg_drop)
        + 2.0 * hip_offset
        + 4.0 * asymmetry
        - 3.0 * min(body_height / reference_height, 0.9)
        + 5.0 * max(0.0, 0.5 - body_height / reference_height)
        - 8.0 * min(margin, 0.08)
        + 0.01 * deviation
    )


def _support_margin(feet: np.ndarray, com: np.ndarray) -> float:
    """Distance from the center of mass to the nearest edge of the support polygon (negative = outside)."""
    pts = sorted({(round(float(x), 5), round(float(y), 5)) for x, y in feet})
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    margin = math.inf
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        edge = np.array([b[0] - a[0], b[1] - a[1]])
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        margin = min(margin, float((edge[0] * (com[1] - a[1]) - edge[1] * (com[0] - a[0])) / length))
    return 0.0 if margin == math.inf else margin


def _asymmetry(foot_offsets: np.ndarray, hips: np.ndarray) -> float:
    """How far mirrored limbs (same x, opposite y hips) are from mirrored foot placements."""
    total, pairs = 0.0, 0
    for a in range(len(hips)):
        for b in range(a + 1, len(hips)):
            if abs(hips[a, 0] - hips[b, 0]) < 0.02 and abs(hips[a, 1] + hips[b, 1]) < 0.02 and abs(hips[a, 1]) > 0.01:
                total += abs(foot_offsets[a, 0] - foot_offsets[b, 0]) + abs(foot_offsets[a, 1] + foot_offsets[b, 1])
                pairs += 1
    return total / pairs if pairs else 0.0
