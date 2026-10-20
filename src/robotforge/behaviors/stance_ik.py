"""A pose a two-legged robot can stand in: feet flat on the floor, the centre of mass over the middle of them.

A keyframe is often drawn for how the robot looks, not for balance: Menagerie's H1 stands with its weight 13 cm behind
the middle of its feet, on the very edge of its heels, and Fourier's GR1 has its zero pose with the toes up. A robot in
such a pose stands only as long as its servos are stiff and nothing touches it. `balanced_pose` moves it by damped
least squares on a copy of the model: the feet go flat on the floor where they are, the trunk stays level, the centre
of mass goes over the middle of the feet, and every joint stays as near its starting angle as that allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.sim.geometry import collides

FOOT_BAND = 0.05  # m above the lowest point of the robot within which a body can be a foot
SOLE_BAND = 0.006  # m above the lowest point of a flat foot that still counts as its sole
CENTRAL_SHARE = 0.3  # of the feet's extent either way: where the centre of mass is already balanced enough
FLAT_TOLERANCE = math.radians(3.0)  # how far from flat a foot may be and still be called flat
ANKLE_SHARE = 0.35  # of the way from the middle of the feet to the ankles that the centre of mass is brought
ITERATIONS = 120
STEP = 0.05  # rad (or m) a joint moves per iteration at most
JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class Foot:
    """A foot as it stands flat: the body, the rotation it has then, where its origin is, and its sole."""

    body: int
    rot: np.ndarray  # 3x3, world frame, with the sole level
    pos: np.ndarray  # the body's origin when the sole is on the floor
    sole: np.ndarray  # (n, 3) points on the underside, in the body's own frame

    def soles(self) -> np.ndarray:
        """The sole points in the world, with the foot flat."""
        return self.pos + self.sole @ self.rot.T

    @staticmethod
    def here(data: mujoco.MjData, body: int, world_points: np.ndarray) -> Foot:
        """A foot as it stands now (flat or not), from the world points it touches the floor with."""
        rot = data.xmat[body].reshape(3, 3).copy()
        return Foot(body, rot, data.xpos[body].copy(), (world_points - data.xpos[body]) @ rot)


def end_bodies(model: mujoco.MjModel, data: mujoco.MjData, tree: np.ndarray) -> dict[int, np.ndarray]:
    """The end bodies of a robot's limbs that reach the floor (`tree`: a boolean per body; posed in `data`), each with
    the world points of its shapes: the bodies whose lowest points are within FOOT_BAND of the lowest point of the
    robot, leaving out one that has another such body below it (the ankle above a foot)."""
    from robotforge.behaviors import _geom_points

    points: dict[int, list[np.ndarray]] = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if tree[b] and collides(model, g):
            rot = data.geom_xmat[g].reshape(3, 3)
            points.setdefault(b, []).append(data.geom_xpos[g] + _geom_points(model, g) @ rot.T)
    if not points:
        return {}
    lowest = {b: float(min(p[:, 2].min() for p in ps)) for b, ps in points.items()}
    floor = min(lowest.values())
    near = {b for b, z in lowest.items() if z <= floor + FOOT_BAND}
    near = {b for b in near if not any(_descends(model, c, b) for c in near if c != b)}
    return {b: np.vstack(points[b]) for b in sorted(near)}


def current_soles(ends: dict[int, np.ndarray]) -> list[np.ndarray]:
    """The points of each end body within SOLE_BAND of its lowest, as it stands (a tilted foot: its heel or toe)."""
    return [p[p[:, 2] <= p[:, 2].min() + SOLE_BAND] for p in ends.values()]


def plan_feet(model: mujoco.MjModel, data: mujoco.MjData, tree: np.ndarray) -> list[Foot]:
    """The feet of a robot (`tree`: a boolean per body) as they would stand flat on the floor under them (see
    end_bodies for which bodies)."""
    ends = end_bodies(model, data, tree)
    if not ends:
        return []
    floor = min(float(p[:, 2].min()) for p in ends.values())
    feet = []
    for b, world in ends.items():
        origin, rot = data.xpos[b], data.xmat[b].reshape(3, 3)
        local = (world - origin) @ rot  # the body's own frame
        level = _level(rot)
        flat = local @ level.T  # about the body origin, with the sole down
        for band in (SOLE_BAND, 2 * SOLE_BAND, 4 * SOLE_BAND):
            sole = local[flat[:, 2] <= flat[:, 2].min() + band]
            if len(sole) >= 3:
                break
        # The body's origin stands where it is across the floor, at the height that puts the sole on the floor.
        pos = np.array([origin[0], origin[1], floor - float((sole @ level.T)[:, 2].min())])
        feet.append(Foot(b, level, pos, sole))
    return feet


def places(soles: list[np.ndarray], gap: float) -> list[list[int]]:
    """Which of the soles (arrays of world points) make one place to stand: those that come within `gap` of each
    other (across the floor) do. Returns the indices of each place."""
    soles = [s[:, :2] for s in soles]
    group = list(range(len(soles)))

    def find(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for i in range(len(soles)):
        for j in range(i + 1, len(soles)):
            nearest = float(np.min(np.linalg.norm(soles[i][:, None, :] - soles[j][None, :, :], axis=2)))
            if nearest <= gap:
                group[find(j)] = find(i)
    out: dict[int, list[int]] = {}
    for i in range(len(soles)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _descends(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        body = int(model.body_parentid[body])
        if body == ancestor:
            return True
    return False


def _level(rot: np.ndarray) -> np.ndarray:
    """`rot` turned the least that puts the body's most downward axis straight down: its sole is level, and it faces
    as it faced."""
    axes = np.concatenate([rot.T, -rot.T])  # the body's six axes in the world
    down = axes[int(np.argmax(axes @ DOWN))]
    axis = np.cross(down, DOWN)
    sine, cosine = float(np.linalg.norm(axis)), float(down @ DOWN)
    if sine < 1e-9:
        return rot.copy()
    axis = axis / sine
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    turn = np.eye(3) + sine * k + (1 - cosine) * (k @ k)
    return turn @ rot


def middle_of(feet: list[Foot]) -> np.ndarray:
    """Where the centre of mass goes: the centre of the area the feet cover (the centroid of the convex hull of all
    their sole points; a mean of the points would lean toward the side of a foot that has more of them), a little of
    the way toward the ankles. A robot with weak ankles (the Unitree H1's give 40 N*m for 51 kg) holds its weight
    over the middle of a long foot only just: over the ankle, no ankle torque is needed at all."""
    return toward_ankles(hull_centre(np.vstack([f.soles() for f in feet])), np.mean([f.pos for f in feet], axis=0))


def toward_ankles(middle: np.ndarray, ankles: np.ndarray) -> np.ndarray:
    """`middle` moved ANKLE_SHARE of the way to `ankles` across the floor."""
    out = middle.copy()
    out[:2] += ANKLE_SHARE * (ankles[:2] - middle[:2])
    return out


def needs_balancing(com: np.ndarray, feet: list[Foot], data: mujoco.MjData) -> bool:
    """Whether a robot is not yet standing balanced: a foot off flat, or the centre of mass off the middle of the
    area the feet cover."""
    for foot in feet:
        now = data.xmat[foot.body].reshape(3, 3)
        if float(np.linalg.norm(rotation_error(foot.rot, now))) > FLAT_TOLERANCE:
            return True
    points = np.vstack([f.soles() for f in feet])
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    centre, extent = (low + high) / 2, high - low
    return bool(np.any(np.abs(com[:2] - centre) > CENTRAL_SHARE * np.maximum(extent, 0.02)))


def hull_indices(points: np.ndarray) -> list[int]:
    """Indices of the points (x, y) on the convex hull, counter-clockwise; every index when there are fewer than 3."""
    n = len(points)
    if n < 3:
        return list(range(n))
    order = sorted(range(n), key=lambda i: (points[i][0], points[i][1]))

    def half(seq: list[int]) -> list[int]:
        out: list[int] = []
        for i in seq:
            while len(out) >= 2 and _turn(points[out[-2]], points[out[-1]], points[i]) <= 0:
                out.pop()
            out.append(i)
        return out

    hull = half(order)[:-1] + half(order[::-1])[:-1]
    return hull or order[:1]


def hull_centre(points: np.ndarray) -> np.ndarray:
    """The centroid of the convex hull of `points` (x, y taken, z averaged); their mean when the hull has no area."""
    mean = points.mean(axis=0)
    flat = np.unique(np.round(points[:, :2], 5), axis=0)
    if len(flat) < 3:
        return mean
    poly = flat[hull_indices(flat)]
    if len(poly) < 3:
        return mean
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return mean
    cx = ((x + x2) * cross).sum() / (6 * area)
    cy = ((y + y2) * cross).sum() / (6 * area)
    return np.array([cx, cy, mean[2]])


def balanced_pose(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    trunk: int,
    tree: np.ndarray,
    feet: list[Foot],
    reference: np.ndarray,
    level: np.ndarray,
    iterations: int = ITERATIONS,
) -> np.ndarray:
    """`qpos` moved until the feet stand as `feet` say and the centre of mass of the tree under `trunk` is over the
    middle of them (see the module doc). `reference` is the qpos the joints stay near, `level` the rotation matrix
    the trunk keeps."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    dofs = np.flatnonzero(tree[model.dof_bodyid])
    joint_rows = np.array([i for i, dof in enumerate(dofs) if int(model.jnt_type[model.dof_jntid[dof]]) in JOINTS])
    qadr = np.array([int(model.jnt_qposadr[model.dof_jntid[dofs[i]]]) for i in joint_rows])
    middle = middle_of(feet)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    post = np.zeros((len(joint_rows), len(dofs)))
    post[np.arange(len(joint_rows)), joint_rows] = 1.0
    for _ in range(iterations):
        rows, errs, weights = [], [], []
        for foot in feet:
            mujoco.mj_jacBody(model, data, jacp, jacr, foot.body)
            rows += [jacp[:, dofs].copy(), jacr[:, dofs].copy()]
            errs += [foot.pos - data.xpos[foot.body], rotation_error(foot.rot, data.xmat[foot.body].reshape(3, 3))]
            weights += [100.0, 100.0]
        mujoco.mj_jacSubtreeCom(model, data, jacp, trunk)
        rows.append(jacp[:2, dofs].copy())
        errs.append(middle[:2] - data.subtree_com[trunk][:2])
        weights.append(30.0)
        mujoco.mj_jacBody(model, data, jacp, jacr, trunk)
        rows.append(jacr[:, dofs].copy())
        errs.append(rotation_error(level, data.xmat[trunk].reshape(3, 3)))
        weights.append(10.0)
        rows.append(post)
        errs.append(0.2 * (reference[qadr] - data.qpos[qadr]))
        weights.append(1.0)
        a = np.vstack([w * r for w, r in zip(weights, rows, strict=True)])
        b_vec = np.concatenate([w * e for w, e in zip(weights, errs, strict=True)])
        step = np.linalg.solve(a.T @ a + 1e-4 * np.eye(len(dofs)), a.T @ b_vec)
        full = np.zeros(model.nv)
        full[dofs] = np.clip(step, -STEP, STEP)
        mujoco.mj_integratePos(model, data.qpos, full, 1.0)
        _clip_to_limits(model, data, dofs)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
    return data.qpos.copy()


def _turn(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Positive when a, b, c turn left (the z of the cross product)."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """The rotation vector (world frame) that turns `current` into `target`."""
    delta = target @ current.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, delta.reshape(-1))
    vec = np.zeros(3)
    mujoco.mju_quat2Vel(vec, quat, 1.0)
    return vec


def _clip_to_limits(model: mujoco.MjModel, data: mujoco.MjData, dofs: np.ndarray) -> None:
    for dof in dofs:
        j = int(model.dof_jntid[dof])
        if model.jnt_limited[j] and int(model.jnt_type[j]) in JOINTS:
            a = int(model.jnt_qposadr[j])
            data.qpos[a] = float(np.clip(data.qpos[a], *model.jnt_range[j]))
