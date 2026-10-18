"""Fetch and carry: an arm that rides on a robot that drives or walks gets the robot to go where the arm reaches.

pick_and_place uses this for such an arm when the object or the place is out of the arm's reach from where the robot
stands (BehaviorManager.pick_and_place): the robot goes next to the object, picks it up, holds it over its own back
with the arm tucked in while it goes next to the place, and puts it down there.

Where to stand: around the object, spots on the floor at a few distances and bearings from where the arm's base would
be, with the robot turned so that the object lies ahead of the arm or to one side of it. A spot is kept if the robot's
map calls it free floor it can get to and its shapes, posed there in a copy of the world, touch nothing; the most
promising are checked by IK, with the robot moved there in the IK's own copy, and a spot is kept only if the arm
reaches from it a few cm and degrees off too, as navigation stops within 8 cm and 12 degrees. The last of that the
robot creeps out (Errand._align), and the arm checks its reach again from where the robot really stands.

Things go round the robot on the way (navigate.py), the load in its gripper with them: left out of the obstacles and
counted in its footprint (nav_map.with_load).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from robotforge.behaviors import nav_map
from robotforge.behaviors.arm import REACH_TOL, ArmMotion, OutOfReach
from robotforge.behaviors.navigate import NoWay
from robotforge.sim.geometry import collides

if TYPE_CHECKING:
    from robotforge.behaviors.manager import BehaviorManager
    from robotforge.behaviors.navigate import Navigator

log = logging.getLogger(__name__)

KEY = "__fetch__"  # session controller key of a robot's errand: KEY + robot
RATE_HZ = 20.0
TIME_LIMIT = 180.0  # s of sim time pick_and_place waits for an errand (a Go2 crossing the kitchen twice takes ~60)
# Spots tried round a point: directions from it every THETA_STEP, the robot turned so that the point lies at one of
# BEARINGS from its arm's base, at these shares of how far the arm reaches out at that height (Stand.out_at).
THETA_STEP = math.radians(15.0)
BEARINGS = tuple(math.radians(b) for b in (0, 45, -45, 90, -90, 135, -135, 180))
SHARES = (0.65, 0.5, 0.8, 0.35, 0.92)
MIN_OUT = 0.1  # m from the arm's base, at the least
MAX_CHECKS = 40  # spots checked by IK in one search
# A spot counts when the arm reaches from it shifted this far each way and turned this much each way too.
SHIFT = 0.03  # m
TWIST = math.radians(5.0)
PENETRATE = 0.002  # m: deeper overlaps of the robot with something, posed at a spot, rule it out
POS_TOL, YAW_TOL = 0.015, math.radians(2.0)  # where creeping into a spot stops
CREEP_SPEED, CREEP_TURN = 0.08, 0.5  # m/s and rad/s at most, creeping
CREEP_SECONDS = 12.0
STAND_TRIES = 2  # times it creeps or goes again into a spot from which the arm did not reach
HELD = 0.03  # m: a load whose middle is within its half height and this of the tool point is in the gripper
SETTLE = 0.6  # s it stands still after it arrives, before the arm moves (a gait's sway dies down)
EDGE_INSET = 0.02  # m kept between a load put down and the edge of what it goes on
SPOT_STEP = 0.04  # m between the spots tried on a table or in a bin
FREE_GAP = 0.015  # m kept between a load put down and what already stands there
WEIGHT_SHARE = 0.8  # of a joint's torque limit that holding a load may take, with the arm's own weight
GRIP_SAFETY = 2.0  # a gripper holds a load whose weight is at most 1 / this of what its friction can hold


class FetchError(ValueError):
    """Why the robot cannot fetch it, before it moves; says what to do instead."""


@dataclass(frozen=True)
class Spot:
    """Where the robot stands to work: its middle (its root) and heading, and the point it works on from there."""

    xy: tuple[float, float]
    heading: float  # rad
    target: tuple[float, float, float]

    def face(self) -> tuple[float, float]:
        """A point 1 m ahead of the robot standing here, for navigation to turn it to."""
        return self.xy[0] + math.cos(self.heading), self.xy[1] + math.sin(self.heading)


@dataclass(frozen=True)
class Place:
    """Where a load can go: spots for its bottom (best first), the highest thing round them (a bin's rim), and the
    region that counts as there (BehaviorManager.placed)."""

    spots: list[np.ndarray]
    top: float
    region: dict[str, Any]
    name: str


# ---- the robot as it would stand elsewhere ------------------------------------------------------------------


class Stand:
    """Posing the robot that carries an arm elsewhere, in copies of the world: does it fit there, and does its arm
    reach a point from there?"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot: str, arm: ArmMotion, hold_q: np.ndarray):
        self.model, self.robot, self.arm = model, robot, arm
        self.root = nav_map.root_body(model, robot)
        joint = int(model.body_jntadr[self.root])
        if joint < 0 or int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE:
            raise FetchError(f"{robot} is fixed to the floor, so it cannot go anywhere")
        self.adr = int(model.jnt_qposadr[joint])
        self.qpos = data.qpos.copy()
        self.qpos[arm.chain.qpos_adr] = hold_q  # the arm tucked in, as it is while the robot goes
        rot = data.xmat[self.root].reshape(3, 3)
        self.heading = math.atan2(rot[1, 0], rot[0, 0])
        self.xy = np.array(data.xpos[self.root][:2], float)
        c, s = math.cos(self.heading), math.sin(self.heading)
        frame = np.array([[c, s], [-s, c]])
        self.arm_base = frame @ (arm.base_xy - self.xy)  # the arm's base in the robot's heading frame
        self.probe = mujoco.MjData(model)
        self._out: dict[float, float] = {}
        self.robot_bodies = {b for b in range(model.nbody) if int(model.body_rootid[b]) == self.root}
        self.skip: set[int] = set()  # roots of what may overlap the robot (a load it carries)

    def qpos_at(self, xy, heading: float) -> np.ndarray:
        """The world's qpos with the robot moved to `xy`, turned to `heading`, the arm tucked."""
        q = self.qpos.copy()
        q[self.adr : self.adr + 2] = xy
        turn = np.array([math.cos((heading - self.heading) / 2), 0.0, 0.0, math.sin((heading - self.heading) / 2)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, self.qpos[self.adr + 3 : self.adr + 7])
        q[self.adr + 3 : self.adr + 7] = quat
        return q

    def arm_base_at(self, xy, heading: float) -> np.ndarray:
        c, s = math.cos(heading), math.sin(heading)
        return np.asarray(xy, float) + np.array(
            [c * self.arm_base[0] - s * self.arm_base[1], s * self.arm_base[0] + c * self.arm_base[1]]
        )

    def fits(self, xy, heading: float) -> bool:
        """Whether the robot's shapes, posed at `xy` and `heading` with the arm tucked, overlap nothing but the floor
        (and what it carries)."""
        m, d = self.model, self.probe
        d.qpos[:] = self.qpos_at(xy, heading)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for con in d.contact[: d.ncon]:
            if con.dist > -PENETRATE:
                continue
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            mine1, mine2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if mine1 == mine2:
                continue
            other_geom, other = (con.geom2, b2) if mine1 else (con.geom1, b1)
            if other == 0 and int(m.geom_type[other_geom]) in (
                mujoco.mjtGeom.mjGEOM_PLANE,
                mujoco.mjtGeom.mjGEOM_HFIELD,
            ):
                continue
            if int(m.body_rootid[other]) in self.skip:
                continue
            return False
        return True

    def out_at(self, z: float) -> float:
        """How far from its base (m) the arm points its gripper down at height z: the furthest of ahead of the robot
        and to either side (measured once per cm of height, as the robot stands now). An arm on a Go2 reaches the
        floor beside the dog, not past its head."""
        key = round(z, 2)
        if key not in self._out:
            yaw = self.heading
            self._out[key] = max(_reach_out(self.arm, z, yaw + turn) for turn in (0.0, math.pi / 2, -math.pi / 2))
        return self._out[key]

    def reaches(self, xy, heading: float, points: list[np.ndarray], quick: bool = True) -> bool:
        """Whether the arm, with the robot at `xy` and `heading`, points its gripper down at every point."""
        with self.arm.carried_to(self.qpos_at(xy, heading)):
            check = self.arm.quick_error if quick else self.arm.reach_error
            return all(check(p) <= REACH_TOL for p in points)


def spots_for(
    stand: Stand,
    nav: nav_map.NavMap,
    targets: list[np.ndarray],
    lift: float,
    start: np.ndarray,
    grid_reach: np.ndarray,
    below: list[tuple[np.ndarray, np.ndarray, float]] = (),
) -> tuple[Spot | None, int]:
    """The best spot from which the arm works on one of `targets` (points, best first) and on each `lift` m above it:
    near the robot's start, the point at a comfortable distance from the arm's base and ahead of it rather than
    behind, with room round it; and how many spots were checked by IK. `grid_reach`: the cells the robot can get to.
    `below`: boxes (low xy, high xy, top z) of what the targets stand on, which the arm's base must not be under."""
    checked = 0
    fallback: Spot | None = None
    for target in targets:
        candidates = _candidates(stand, nav, target, start, grid_reach, below)
        for _, xy, heading in candidates:
            if checked >= MAX_CHECKS:
                return fallback, checked
            if not stand.fits(xy, heading):
                continue
            checked += 1
            points = [target, target + [0.0, 0.0, lift]]
            if not stand.reaches(xy, heading, points):
                continue
            spot = Spot((float(xy[0]), float(xy[1])), float(heading), tuple(float(v) for v in target))
            # Navigation stops within 8 cm and 12 degrees, and creeping in gets within 2.5 cm and 3 degrees: a spot
            # on the edge of where the arm reaches is reached in the plan and missed from where it really stops.
            if _robust(stand, xy, heading, target):
                return spot, checked
            fallback = fallback or spot
    return fallback, checked


def _robust(stand: Stand, xy: np.ndarray, heading: float, target: np.ndarray) -> bool:
    c, s = math.cos(heading), math.sin(heading)
    shifts = [
        np.array([c, s]) * SHIFT,
        np.array([-c, -s]) * SHIFT,
        np.array([-s, c]) * SHIFT,
        np.array([s, -c]) * SHIFT,
    ]
    poses = [(xy + d, heading) for d in shifts] + [(xy, heading + TWIST), (xy, heading - TWIST)]
    return all(stand.reaches(p, h, [target]) for p, h in poses)


def _candidates(
    stand: Stand,
    nav: nav_map.NavMap,
    target: np.ndarray,
    start: np.ndarray,
    grid_reach: np.ndarray,
    below: list[tuple[np.ndarray, np.ndarray, float]],
) -> list[tuple[float, np.ndarray, float]]:
    """(score, robot xy, heading) of the spots round `target` on free floor it can get to, best (lowest) first."""
    grid = nav.grid
    reach = stand.out_at(float(target[2]))  # how far it reaches out at that height
    if reach <= MIN_OUT:
        return []
    out = []
    for theta in np.arange(0.0, 2 * math.pi, THETA_STEP):
        for share in SHARES:
            r = max(MIN_OUT, share * reach)
            base = target[:2] + r * np.array([math.cos(theta), math.sin(theta)])
            if any(np.all(base >= lo) and np.all(base <= hi) and top > target[2] - 0.2 for lo, hi, top in below):
                continue  # the arm's base under the table it works on: it could not get up past the edge
            for bearing in BEARINGS:
                heading = theta + math.pi - bearing  # the target at `bearing` off its heading, from the arm
                xy = base - (stand.arm_base_at((0.0, 0.0), heading))
                cell = grid.cell(*xy)
                if not (grid.inside(*cell) and nav.free[cell] and grid_reach[cell]):
                    continue
                score = (
                    float(np.linalg.norm(xy - start))
                    + 0.6 * abs(share - SHARES[0]) * reach
                    + 0.15 * abs(bearing) / (math.pi / 2)
                    - 0.5 * min(float(nav.room[cell]) - nav.body.need, 0.2)
                )
                out.append((score, xy, float(_wrap(heading))))
    out.sort(key=lambda c: c[0])
    return out


# ---- what goes where ----------------------------------------------------------------------------------------


def load_size(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> tuple[float, float, float]:
    """(half height, radius seen from above, narrowest width across) of a loose object's collision shapes."""
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body and collides(model, g)]
    if not geoms:
        return 0.02, 0.02, 0.04
    lows, highs, widths = [], [], []
    for g in geoms:
        low, high = nav_map.world_extent(model, data, g)
        lows.append(low)
        highs.append(high)
        size = model.geom_size[g]
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            widths.append(2 * float(min(size[0], size[1])))
        else:
            widths.append(2 * float(size[0]))
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    center = data.xpos[body]
    far = np.maximum(np.abs(low[:2] - center[:2]), np.abs(high[:2] - center[:2]))  # its furthest corner
    radius = float(np.hypot(far[0], far[1]))
    return float((high[2] - low[2]) / 2), radius, float(min(widths))


def gripper_opening(model: mujoco.MjModel, arm: ArmMotion) -> float | None:
    """How wide the gripper opens (m): the gap between its fingers' shapes with the fingers as far apart as their
    joints go. None when that cannot be told (no gripper, fingers not on slides)."""
    if arm.gripper < 0:
        return None
    joints = _finger_joints(model, arm.gripper)
    if not joints:
        return None
    sides: dict[int, list[int]] = {}
    for j in joints:
        body = int(model.jnt_bodyid[j])
        sides[j] = [
            g for g in range(model.ngeom) if _descends(model, int(model.geom_bodyid[g]), body) and collides(model, g)
        ]
    pairs = [sides[j] for j in joints if sides[j]]
    if len(pairs) < 2:
        return None
    d = mujoco.MjData(model)
    d.qpos[:] = arm.ik.base_qpos
    for j in joints:
        if int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_SLIDE:
            return None
        lo, hi = model.jnt_range[j]
        d.qpos[model.jnt_qposadr[j]] = hi if abs(hi) >= abs(lo) else lo
    mujoco.mj_kinematics(model, d)
    gap = min(mujoco.mj_geomDistance(model, d, a, b, 1.0, None) for a in pairs[0] for b in pairs[1])
    return float(gap)


def _finger_joints(model: mujoco.MjModel, actuator: int) -> list[int]:
    joints: set[int] = set()
    kind = int(model.actuator_trntype[actuator])
    target = int(model.actuator_trnid[actuator, 0])
    if kind == mujoco.mjtTrn.mjTRN_JOINT:
        joints.add(target)
    elif kind == mujoco.mjtTrn.mjTRN_TENDON:
        for w in range(model.tendon_adr[target], model.tendon_adr[target] + model.tendon_num[target]):
            if int(model.wrap_type[w]) == mujoco.mjtWrap.mjWRAP_JOINT:
                joints.add(int(model.wrap_objid[w]))
    for e in range(model.neq):  # a finger that follows the driven one
        if int(model.eq_type[e]) == mujoco.mjtEq.mjEQ_JOINT and (
            model.eq_obj1id[e] in joints or model.eq_obj2id[e] in joints
        ):
            joints |= {int(model.eq_obj1id[e]), int(model.eq_obj2id[e])}
    return sorted(joints)


def heaviest(model: mujoco.MjModel, arm: ArmMotion, q: np.ndarray, friction: float) -> tuple[float, str]:
    """The heaviest load (kg) the arm holds with its gripper at `q`, and what limits it: its joints' torque limits less
    what holding up its own weight takes there, and how hard its gripper squeezes times the friction of its fingers."""
    m, d = model, arm.ik.data
    arm.ik.point(arm.chain, q)
    d.qvel[:] = 0.0
    mujoco.mj_comVel(m, d)
    bias = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 0, bias)
    jacp = np.zeros((3, m.nv))
    tool = d.site_xpos[arm.site].copy()
    mujoco.mj_jac(m, d, jacp, None, tool, arm.chain.body)
    g = float(np.linalg.norm(m.opt.gravity)) or 9.81
    best, why = math.inf, ""
    for j, a in zip(arm.joints, arm.actuators, strict=True):
        if not m.actuator_forcelimited[a]:
            continue
        limit = float(np.abs(m.actuator_forcerange[a]).max() * abs(m.actuator_gear[a, 0]))
        dof = int(m.jnt_dofadr[j])
        lever = abs(float(jacp[2, dof]))
        if lever < 1e-4 or limit <= 0:
            continue
        spare = (WEIGHT_SHARE * limit - abs(float(bias[dof]))) / (g * lever)
        if spare < best:
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").rsplit("/", 1)[-1]
            unit = "N*m" if int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_HINGE else "N"
            best, why = spare, f"its {name} joint gives at most {limit:.0f} {unit}"
    if arm.gripper >= 0 and m.actuator_forcelimited[arm.gripper]:
        squeeze = float(np.abs(m.actuator_forcerange[arm.gripper]).max() * abs(m.actuator_gear[arm.gripper, 0]))
        grip = 2 * friction * squeeze / (g * GRIP_SAFETY)
        if grip < best:
            best, why = grip, f"its gripper squeezes with at most {squeeze:.0f} N"
    return max(best, 0.0), why


def _descends(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---- where a load can go ------------------------------------------------------------------------------------


def place_for(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: str | list[float],
    kind: str | None,
    load: int,
    radius: float,
) -> Place:
    """Where a load (body `load`, `radius` m round seen from above) can go on `target`: an object's name (`kind` is its
    shape: a table, a bin, a box ...) or a terrain patch's, or a point [x, y] or [x, y, z]. Spots on a surface keep
    clear of its edges and of what stands there already, nearest the middle first; FetchError when there is none
    (the place is full)."""
    if not isinstance(target, str):
        return _point_place(model, data, target, load)
    geoms = _named_geoms(model, target)
    boxes = [g for g in geoms if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX and data.geom_xmat[g][8] > 0.99]
    if not boxes:
        raise FetchError(f"the {target} has no flat top to put things on")
    tops = {g: float(data.geom_xpos[g][2] + model.geom_size[g][2]) for g in boxes}
    if kind == "bin":  # its floor, inside its walls
        surface = min(boxes, key=lambda g: tops[g])
        inset, rim = radius + EDGE_INSET + 0.005, max(tops.values())
    else:  # a table's top, a box's, a ramp's landing
        surface = max(boxes, key=lambda g: tops[g])
        inset, rim = radius + EDGE_INSET, tops[surface]
    top = tops[surface]
    rot = data.geom_xmat[surface].reshape(3, 3)
    half = model.geom_size[surface][:2]
    center = data.geom_xpos[surface]
    us = np.arange(-half[0] + inset, half[0] - inset + 1e-9, SPOT_STEP)
    vs = np.arange(-half[1] + inset, half[1] - inset + 1e-9, SPOT_STEP)
    spots = [center + rot @ np.array([u, v, model.geom_size[surface][2]]) for u in us for v in vs]
    extent = np.abs(rot[:2, :2]) @ half
    region = {"center": [float(center[0]), float(center[1]), top], "half_xy": extent.tolist(), "min_z": top - 0.01}
    if not spots:
        raise FetchError(f"the {target} is too small to put it on: {2 * half[0]:.2f} by {2 * half[1]:.2f} m")
    standing = _standing_on(model, data, center[:2], extent, top, load)
    free = [p for p in spots if all(math.dist(p[:2], xy) >= radius + r + FREE_GAP for _, xy, r in standing)]
    if not free:
        names = sorted({name for name, _, _ in standing})
        raise FetchError(f"the {target} is full: {' and '.join(names)} leave no room for it")
    free.sort(key=lambda p: float(np.linalg.norm(p[:2] - center[:2])))
    return Place(free, rim, region, target)


def _point_place(model: mujoco.MjModel, data: mujoco.MjData, target: list[float], load: int) -> Place:
    x, y = float(target[0]), float(target[1])
    z = float(target[2]) if len(target) == 3 else _ground_under(model, data, x, y, load)
    region = {"center": [x, y, z], "half_xy": [0.04, 0.04], "min_z": z - 0.01}
    return Place([np.array([x, y, z])], z, region, f"[{x:.2f}, {y:.2f}]")


def _ground_under(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float, load: int) -> float:
    """The height of what a load put at (x, y) would stand on: the floor, terrain or furniture (looking past the load
    itself and robots, which move); FetchError when another loose object stands there."""
    top = 10.0
    for _ in range(8):
        hit = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(model, data, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]), None, 1, -1, hit)
        if dist < 0:
            return 0.0
        body = int(model.geom_bodyid[hit[0]])
        if body == 0 or int(model.body_weldid[body]) == 0:
            return float(top - dist)  # the floor, terrain or something fixed
        root = int(model.body_rootid[body])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ""
        if root != load and "/" not in name:
            raise FetchError(f"[{x:.2f}, {y:.2f}] is taken: the {name} stands there")
        top = top - dist - 1e-4  # the load itself, or a robot: look under it
    return 0.0


def _named_geoms(model: mujoco.MjModel, name: str) -> list[int]:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b > 0:
        return [g for g in range(model.ngeom) if _descends(model, int(model.geom_bodyid[g]), b) and collides(model, g)]
    found = []
    for g in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        ours = gname == name or gname.startswith((f"{name}/", f"{name}_"))
        if int(model.geom_bodyid[g]) == 0 and ours and collides(model, g):
            found.append(g)
    return found


def _standing_on(
    model: mujoco.MjModel, data: mujoco.MjData, center: np.ndarray, extent: np.ndarray, top: float, load: int
) -> list[tuple[str, np.ndarray, float]]:
    """Loose things that stand on a surface (its middle, half extents and height): (name, x y, radius)."""
    out = []
    for b in range(1, model.nbody):
        if int(model.body_rootid[b]) != b or b == load or not model.body_dofnum[b]:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "/" in name:  # a robot
            continue
        geoms = [
            g for g in range(model.ngeom) if int(model.body_rootid[model.geom_bodyid[g]]) == b and collides(model, g)
        ]
        if not geoms:
            continue
        lows, highs = zip(*(nav_map.world_extent(model, data, g) for g in geoms), strict=True)
        low, high = np.min(lows, axis=0), np.max(highs, axis=0)
        mid = (low[:2] + high[:2]) / 2
        if abs(float(low[2]) - top) > 0.03 or np.any(np.abs(mid - center) > extent + 0.02):
            continue
        out.append((name, mid, float(np.hypot(*((high[:2] - low[:2]) / 2)))))
    return out


# ---- planning an errand -------------------------------------------------------------------------------------


def plan(manager: BehaviorManager, arm: ArmMotion, obj: str, target: str | list[float], kind: str | None) -> Errand:
    """An errand for `arm` (riding on a robot that moves) to bring `obj` to `target` (see place_for for `kind`), or
    FetchError saying why it cannot: the object is too big or too heavy for the gripper, there is nowhere to stand
    to reach it or a free spot on the place, or no way there."""
    session = manager.session
    model, data = session.model, session.data
    robot = arm.robot.split("/", 1)[0]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    half, radius, width = load_size(model, data, body)
    opening = gripper_opening(model, arm)
    if opening is not None and width > opening - 0.004:
        raise FetchError(
            f"the {obj} is {width:.2f} m across where it would be held, and {arm.robot}'s gripper opens only "
            f"{opening:.2f} m"
        )
    arm.sync(data)
    hold = arm.hold_point(half, radius)
    if hold is None:
        raise FetchError(
            f"{arm.robot} cannot fold back over {robot} with something in its gripper, so it cannot carry anything "
            f"while {robot} goes; move {robot} next to the {obj} and to the place yourself (move with to=...)"
        )
    hold_q, _ = arm.pose_for(hold)
    root = manager.root(robot)
    rot = data.xmat[root].reshape(3, 3)
    heading = math.atan2(rot[1, 0], rot[0, 0])
    c, s = math.cos(heading), math.sin(heading)
    off = hold - data.xpos[root]
    hold_local = np.array([c * off[0] + s * off[1], -s * off[0] + c * off[1], off[2]])
    stand = Stand(model, data, robot, arm, hold_q)
    start = np.array(data.xpos[root][:2], float)
    tucked = _posed(model, data, arm, hold_q)
    holds = holding(data, arm, body, half)
    # Where it picks it up.
    center = np.array(data.xpos[body], float)
    pick: Spot | None = None
    if not holds and not all(arm.reach_error(p) <= REACH_TOL for p in (center, center + [0, 0, APPROACH_UP])):
        nav = manager.floor_map(robot, tucked)
        reach = _reachable_cells(nav, start)
        pick, _ = spots_for(stand, nav, [center], APPROACH_UP, start, reach, _supports(model, data, body))
        if pick is None:
            raise FetchError(_nowhere(manager, stand, nav, reach, robot, center, f"the {obj}", body))
    # Where it puts it down.
    place = place_for(model, data, target, kind, body, radius)
    loaded = _posed(model, data, arm, hold_q, body, hold)
    nav = manager.floor_map(robot, loaded, carrying=obj)
    reach = _reachable_cells(nav, start)
    stand.skip = {body}
    lift = max(APPROACH_UP, place.top + 2 * half + 0.03 - (min(p[2] for p in place.spots) + half + 0.01))
    tools = [p + [0.0, 0.0, half + 0.01] for p in place.spots[:PLACE_SPOTS]]
    begin = np.array(pick.xy) if pick is not None else start
    drop, _ = spots_for(stand, nav, tools, lift, begin, reach, [])
    stand.skip = set()
    if drop is None:
        raise FetchError(
            _nowhere(manager, stand, nav, reach, robot, tools[0], f"a free spot on the {place.name}", body)
        )
    drop = Spot(drop.xy, drop.heading, tuple(float(v) for v in np.array(drop.target) - [0.0, 0.0, half + 0.01]))
    # Not more than it can hold.
    mass = float(model.body_subtreemass[body])
    q = hold_q
    if not holds:
        at = stand.qpos_at(pick.xy, pick.heading) if pick is not None else data.qpos
        with arm.carried_to(at):
            q, _ = arm.pose_for(center)
    limit, why = heaviest(model, arm, q, _grip_friction(model, arm, body))
    arm.sync(data)
    if mass > limit:
        raise FetchError(
            f"the {obj} weighs {mass:.2f} kg, more than {arm.robot} can hold out there (about {limit:.2f} kg: {why})"
        )
    return Errand(manager, robot, arm, obj, body, half, radius, hold_local, pick, drop, place, held=holds)


APPROACH_UP = 0.07  # m: the arm comes down onto a load from this high (arm.APPROACH)
PLACE_SPOTS = 12  # spots on a place tried, best first


def _posed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmMotion,
    q: np.ndarray,
    load: int | None = None,
    at: np.ndarray | None = None,
) -> mujoco.MjData:
    """A copy of the world with the arm at `q` (tucked in), and the load in its gripper at `at` if given."""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qpos[arm.chain.qpos_adr] = q
    if load is not None and at is not None:
        joint = int(model.body_jntadr[load])
        if joint >= 0 and int(model.jnt_type[joint]) == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[model.jnt_qposadr[joint] : model.jnt_qposadr[joint] + 3] = at
    mujoco.mj_kinematics(model, d)
    return d


def holding(data: mujoco.MjData, arm: ArmMotion, body: int, half: float | None = None) -> bool:
    """Whether the arm's gripper is closed on the object `body` (its middle within its half height and HELD of the tool
    point)."""
    if half is None:
        half = _half_height(arm.ik.model, data, body)
    gripper = arm.gripper
    closed = gripper < 0 or float(data.ctrl[gripper]) < 0.5 * arm.gripper_open
    return closed and float(np.linalg.norm(data.xpos[body] - data.site_xpos[arm.site])) <= half + HELD


def _half_height(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> float:
    return load_size(model, data, body)[0]


def _reachable_cells(nav: nav_map.NavMap, start: np.ndarray) -> np.ndarray:
    """The cells the robot can get to from where it stands (from the nearest free cell, if it stands too close to
    something)."""
    from robotforge.behaviors import nav_grid

    grid = nav.grid
    cell = grid.cell(*start)
    if not (grid.inside(*cell) and nav.free[cell]):
        xs, ys = grid.centers()
        dist = np.where(nav.free, np.hypot(xs - start[0], ys - start[1]), np.inf)
        if not np.isfinite(dist).any():
            return np.zeros(grid.shape, bool)
        k = np.unravel_index(int(np.argmin(dist)), grid.shape)
        cell = (int(k[0]), int(k[1]))
    return nav_grid.reachable(grid, nav.free, cell)


def _supports(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """What an object stands on, as boxes seen from above (low xy, high xy, top z): the arm's base must not be under
    them."""
    own = [g for g in range(model.ngeom) if int(model.body_rootid[model.geom_bodyid[g]]) == body]
    if not own:
        return []
    low = min(float(nav_map.world_extent(model, data, g)[0][2]) for g in own)
    xy = data.xpos[body][:2]
    out = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if int(model.body_rootid[b]) == body or not collides(model, g) or int(model.body_weldid[b]) != 0:
            continue
        if int(model.geom_type[g]) in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
            continue
        lo, hi = nav_map.world_extent(model, data, g)
        if abs(float(hi[2]) - low) < 0.03 and np.all(xy >= lo[:2] - 0.01) and np.all(xy <= hi[:2] + 0.01):
            out.append((lo[:2], hi[:2], float(hi[2])))
    return out


def _grip_friction(model: mujoco.MjModel, arm: ArmMotion, body: int) -> float:
    """The friction between the gripper's fingers and the load: the fingers' own where they take priority."""
    fingers = [
        g
        for j in _finger_joints(model, arm.gripper)
        for g in range(model.ngeom)
        if _descends(model, int(model.geom_bodyid[g]), int(model.jnt_bodyid[j])) and collides(model, g)
    ]
    own = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
    pad = max((float(model.geom_friction[g][0]) for g in fingers), default=1.0)
    thing = max((float(model.geom_friction[g][0]) for g in own), default=1.0)
    first = max((int(model.geom_priority[g]) for g in fingers), default=0)
    return pad if first > max((int(model.geom_priority[g]) for g in own), default=0) else max(pad, thing)


def _nowhere(
    manager: BehaviorManager,
    stand: Stand,
    nav: nav_map.NavMap,
    reach: np.ndarray,
    robot: str,
    point: np.ndarray,
    what: str,
    load: int,
) -> str:
    """Why there is no spot to work on `point` from: too high or too low for the arm anywhere, no way to where it
    could stand, or its arm's base gets no nearer than the arm reaches out at that height."""
    arm = stand.arm
    at = arm.down_reach(arm.ik.data, point)
    if at.reach <= MIN_OUT:
        where = "high" if point[2] > at.base[2] else "low"
        return (
            f"{what} is too {where} for {arm.robot}, at z {point[2]:.2f} m: it points its gripper down only between "
            f"{_heights(arm)} m high"
        )
    everywhere = np.ones(nav.grid.shape, bool)
    out = stand.out_at(float(point[2]))
    near = _nearest_base(stand, nav, reach, point)
    if near is None:
        far = _nearest_base(stand, nav, everywhere, point)
        if far is not None:
            return f"there is no way for {robot} to where its arm would reach {what}: walls or furniture close it off"
        return f"there is no room for {robot} anywhere near {what}"
    return (
        f"nowhere {robot} can stand does its arm reach {what}: the nearest its arm's base gets is {near:.2f} m from it "
        f"(seen from above), and at that height (z {point[2]:.2f} m) {arm.robot} points its gripper down at most "
        f"{out:.2f} m from its base; put it nearer the edge or lower down, or use a robot that reaches further"
    )


def _nearest_base(stand: Stand, nav: nav_map.NavMap, reach: np.ndarray, point: np.ndarray) -> float | None:
    """How near (m, seen from above) the arm's base gets to a point, with the robot on free floor it can get to and
    fitting there, turned any way (every 15 degrees); None if it fits nowhere within 2 m."""
    grid = nav.grid
    xs, ys = grid.centers()
    ok = nav.free & reach & (np.hypot(xs - point[0], ys - point[1]) <= 2.0)
    best = None
    cells = np.argwhere(ok)
    order = np.argsort(np.hypot(xs[ok] - point[0], ys[ok] - point[1]))
    for k in order[:400]:
        r, col = cells[k]
        xy = np.array(grid.center(int(r), int(col)))
        for heading in np.arange(-math.pi, math.pi, math.radians(15.0)):
            gap = float(np.linalg.norm(stand.arm_base_at(xy, heading) - point[:2]))
            if (best is None or gap < best) and stand.fits(xy, heading):
                best = gap
    return best


def _reach_out(arm: ArmMotion, z: float, way: float | None = None) -> float:
    """How far from its base (m) the arm points its gripper down at height z, along `way` (rad, world; by default
    where the arm's zero points)."""
    way = arm.base_yaw if way is None else way
    ahead = np.array([math.cos(way), math.sin(way)])

    def at(r: float) -> np.ndarray:
        return np.array([*(arm.base_xy + r * ahead), z])

    def scan(first: float) -> float:
        best = 0.0
        for r in np.arange(first, 1.5, 0.02):
            if arm.quick_error(at(r)) <= REACH_TOL:
                best = float(r)
            elif best > 0:
                break
        return best

    best = scan(0.05)
    if best == 0.0:  # nothing from the pose it is in (a lift folded low): find one point the long way, go on from it
        first = next((r for r in WARM_OUT if arm.warm_up(at(r))), None)
        best = scan(first) if first is not None else 0.0
    return best


WARM_OUT = (0.2, 0.3, 0.4, 0.5, 0.12, 0.6)  # m from the base where _reach_out looks the long way


def _heights(arm: ArmMotion) -> str:
    """The heights (m) the arm points its gripper down at, a little way out from its base: "0.15 and 0.62"."""
    found = []
    ahead = np.array([math.cos(arm.base_yaw), math.sin(arm.base_yaw)])
    for z in np.arange(0.0, 1.6, 0.05):
        if any(arm.quick_error(np.array([*(arm.base_xy + r * ahead), z])) <= REACH_TOL for r in (0.15, 0.25, 0.35)):
            found.append(float(z))
    return f"{found[0]:.2f} and {found[-1]:.2f}" if found else "no heights at all"


# ---- the errand ---------------------------------------------------------------------------------------------


@dataclass
class Leg:
    """One trip of the errand: where to, and how it went (Navigator.result())."""

    to: str
    spot: Spot
    trip: Navigator | None = None
    result: dict[str, Any] | None = None


@dataclass
class Errand:
    """A controller that fetches: go to the object, pick it up, carry it to the place, put it down. Runs at RATE_HZ
    on top of the robot's walker, the trips (navigate.Navigator) and the arm (ArmMotion), which do the moving."""

    manager: BehaviorManager
    robot: str
    arm: ArmMotion
    obj: str
    body: int
    half: float  # m, half the load's height
    radius: float  # m, the load seen from above
    hold: np.ndarray  # the tool point over the robot, in the robot's frame: (ahead, left, up) from its root
    pick: Spot | None  # None: the arm reaches the object from where the robot stands
    drop: Spot
    place: Place
    rate_hz: float = RATE_HZ
    step: str = "starting"
    state: str = "working"  # working, done, failed
    why: str | None = None
    note: str | None = None
    held: bool = False
    legs: list[Leg] = field(default_factory=list)
    started: float = 0.0
    ended: float | None = None
    picked_at: list[float] | None = None
    placed_at: list[float] | None = None
    _leg: Leg | None = None
    _until: float = 0.0
    _tries: int = 0
    _creep: tuple[float, Spot] | None = None  # (since when, into which spot)

    @property
    def done(self) -> bool:
        return self.state != "working"

    @property
    def time_limit(self) -> float:
        return TIME_LIMIT

    # -- controller interface --

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.done:
            return
        try:
            self._update(data)
        except Exception as exc:  # an error here must end the errand, not every step of the simulation
            log.exception("%s's errand failed", self.robot)
            self._fail(data, "error", f"the errand broke off on an internal error: {exc}")

    def status(self) -> dict[str, Any]:
        return {"behavior": "fetch", "robot": self.robot, "object": self.obj, "step": self.step, "state": self.state}

    def _update(self, data: mujoco.MjData) -> None:
        t = float(data.time)
        if self.step == "starting":
            self.started = t
            if self._holds(data):  # already in its gripper: on to the place
                self.held = True
                self._go(data, "to_place")
            else:
                self._tuck(data, self.hold_world(data))
                self.step = "tucking"
            return
        if not self.manager.pose(self.robot)["upright"]:
            self._fail(data, "fell", "it fell over")
            return
        if self.held and self.step not in ("placing", "tucking_after") and not self._holds(data):
            where = [round(float(v), 2) for v in data.xpos[self.body][:2]]
            self.manager.stop(self.robot)
            self._fail(data, "dropped", f"it dropped the {self.obj} on the way, at {where}")
            return
        handler = getattr(self, f"_on_{self.step}")
        handler(t, data)

    # -- the steps --

    def _on_tucking(self, t: float, data: mujoco.MjData) -> None:
        if self.arm.done:
            if self.pick is None:
                self._start_pick(t, data)
            else:
                self._go(data, "to_object")

    def _on_to_object(self, t: float, data: mujoco.MjData) -> None:
        self._on_leg(t, data, "creep_object")

    def _on_to_place(self, t: float, data: mujoco.MjData) -> None:
        self._on_leg(t, data, "creep_place")

    def _on_leg(self, t: float, data: mujoco.MjData, then: str) -> None:
        leg = self._leg
        if leg is None or leg.trip is None or not leg.trip.done:
            return
        leg.result = leg.trip.result()
        if not leg.trip.arrived:
            note = leg.result.get("note") or "it did not get there"
            self._fail(data, leg.result.get("why") or "stuck", f"on its way to the {leg.to}: {note}")
            return
        self.step, self._creep, self._tries = then, (t, leg.spot), 0

    def _on_creep_object(self, t: float, data: mujoco.MjData) -> None:
        if self._creep_in(t, data):
            self._until = t + SETTLE
            self.step = "settle_object"

    def _on_settle_object(self, t: float, data: mujoco.MjData) -> None:
        if t >= self._until:
            self._start_pick(t, data)

    def _on_creep_place(self, t: float, data: mujoco.MjData) -> None:
        if self._creep_in(t, data):
            self._until = t + SETTLE
            self.step = "settle_place"

    def _on_settle_place(self, t: float, data: mujoco.MjData) -> None:
        if t >= self._until:
            self._start_place(t, data)

    def _on_picking(self, t: float, data: mujoco.MjData) -> None:
        if not self.arm.done:
            return
        if not self._holds(data):
            self._fail(data, "missed", f"its gripper closed but did not hold the {self.obj}")
            return
        self.held = True
        self.picked_at = [round(float(v), 3) for v in self.arm.ik.data.site_xpos[self.arm.site]]
        self._go(data, "to_place")

    def _on_placing(self, t: float, data: mujoco.MjData) -> None:
        if not self.arm.done:
            return
        self.held = False
        self.placed_at = [round(float(v), 3) for v in data.xpos[self.body]]
        self._tuck(data, self.hold_world(data))
        self.step = "tucking_after"

    def _on_tucking_after(self, t: float, data: mujoco.MjData) -> None:
        if self.arm.done:
            self._finish(data)

    # -- helpers --

    def hold_world(self, data: mujoco.MjData) -> np.ndarray:
        """The tool point it holds the load at, for the robot as it stands now."""
        root = self.manager.root(self.robot)
        rot = data.xmat[root].reshape(3, 3)
        heading = math.atan2(rot[1, 0], rot[0, 0])
        c, s = math.cos(heading), math.sin(heading)
        x, y, z = self.hold
        return np.array(
            [data.xpos[root][0] + c * x - s * y, data.xpos[root][1] + s * x + c * y, data.xpos[root][2] + z]
        )

    def _holds(self, data: mujoco.MjData) -> bool:
        tool = data.site_xpos[self.arm.site]
        return float(np.linalg.norm(data.xpos[self.body] - tool)) <= self.half + HELD

    def _tuck(self, data: mujoco.MjData, hold: np.ndarray) -> None:
        """Back over the robot: across at the height the tool is at, if it can, then down."""
        self.arm.sync(data)
        start = self.arm.tool_target_end()
        over = np.array([hold[0], hold[1], max(start[2], hold[2])])
        if float(np.linalg.norm(start[:2] - hold[:2])) > 0.05 and self.arm.reach_error(over) <= REACH_TOL:
            self.arm.move_to(over)
        if float(np.linalg.norm(start - hold)) > 0.02:
            self.arm.move_to(hold)

    def _go(self, data: mujoco.MjData, step: str) -> None:
        spot = self.pick if step == "to_object" else self.drop
        to = self.obj if step == "to_object" else self.place.name
        try:
            trip = self.manager.navigate(
                self.robot, list(spot.xy), face=spot.face(), carrying=self.obj if self.held else None
            )
        except NoWay as exc:
            self._fail(data, exc.cause, f"no way to the {to}: {exc}")
            return
        self._leg = Leg(to, spot, trip)
        self.legs.append(self._leg)
        self.step = step

    def _creep_in(self, t: float, data: mujoco.MjData) -> bool:
        """Creep the last few cm and degrees into the spot (navigation stops within 8 cm and 12 degrees); True when
        it is there or has tried long enough."""
        since, spot = self._creep
        walker = self.manager.session.controllers.get(self.robot)
        root = self.manager.root(self.robot)
        xy = np.array(data.xpos[root][:2])
        rot = data.xmat[root].reshape(3, 3)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        off = np.array(spot.xy) - xy
        dist = float(np.linalg.norm(off))
        twist = _wrap(spot.heading - yaw)
        if walker is None or not hasattr(walker, "command"):
            return True
        if (dist <= POS_TOL and abs(twist) <= YAW_TOL) or t - since > CREEP_SECONDS:
            walker.command(0.0, 0.0)
            return True
        backs = getattr(walker, "max_reverse", 1.0) > 0 and type(walker).__name__ != "RhythmWalker"
        if dist > POS_TOL:
            alpha = _wrap(math.atan2(off[1], off[0]) - yaw)
            way = 1.0
            if backs and abs(alpha) > math.pi / 2:
                way, alpha = -1.0, _wrap(alpha - math.pi)
            if abs(alpha) > math.radians(30.0) and dist > 2 * POS_TOL:
                walker.command(0.0, _turn(alpha))
            else:
                walker.command(way * min(CREEP_SPEED, max(0.03, 1.5 * dist)), float(np.clip(1.5 * alpha, -0.3, 0.3)))
            return False
        walker.command(0.0, _turn(twist))
        return False

    def _start_pick(self, t: float, data: mujoco.MjData) -> None:
        self.arm.sync(data)
        center = np.array(data.xpos[self.body], float)
        try:
            self.arm.pick_up(center, self.half, self.hold_world(data))
        except (OutOfReach, ValueError) as exc:
            if self._again(t, data, self.pick, "to_object"):
                return
            self._fail(data, "out_of_reach", f"from where it stopped its arm could not pick up the {self.obj}: {exc}")
            return
        self.step = "picking"

    def _start_place(self, t: float, data: mujoco.MjData) -> None:
        self.arm.sync(data)
        target = np.array(self.drop.target, float)
        try:
            self.arm.put_down(target, self.half, self.place.top)
        except (OutOfReach, ValueError) as exc:
            if self._again(t, data, self.drop, "to_place"):
                return
            self._fail(
                data, "out_of_reach", f"from where it stopped its arm could not put it on the {self.place.name}: {exc}"
            )
            return
        self.step = "placing"

    def _again(self, t: float, data: mujoco.MjData, spot: Spot | None, leg: str) -> bool:
        """Creep into the spot once more (it stopped short of it); True if it does."""
        if spot is None or self._tries >= STAND_TRIES:
            return False
        self._tries += 1
        self._creep = (t, spot)
        self.step = "creep_object" if leg == "to_object" else "creep_place"
        return True

    def _fail(self, data: mujoco.MjData, why: str, note: str) -> None:
        self.state, self.why, self.note = "failed", why, note
        self.ended = float(data.time)
        self._leave()

    def _finish(self, data: mujoco.MjData) -> None:
        self.state = "done"
        self.ended = float(data.time)
        self._leave()

    def _leave(self) -> None:
        self.manager.errand_ended(self)

    def interrupt(self, note: str, time: float) -> None:
        if not self.done:
            self.state, self.why, self.note, self.ended = "failed", "interrupted", note, time

    def result(self) -> dict[str, Any]:
        """What it did: the trips (path_m, distance_m, time_s each), where it picked the load up and put it down."""
        out: dict[str, Any] = {"went": []}
        for leg in self.legs:
            r = leg.result or (leg.trip.result() if leg.trip is not None else {})
            out["went"].append(
                {"to": leg.to, **{k: r[k] for k in ("arrived", "path_m", "distance_m", "time_s") if k in r}}
            )
        out["driven_m"] = round(sum(w.get("distance_m", 0.0) for w in out["went"]), 2)
        if self.picked_at is not None:
            out["picked_up_at"] = self.picked_at
        if self.placed_at is not None:
            out["put_down_at"] = self.placed_at
        if self.ended is not None:
            out["errand_s"] = round(self.ended - self.started, 2)
        if self.state != "done":
            out["why"] = self.why or "out_of_time"
            out["step"] = self.step
            if self.note:
                out["note"] = self.note
        return out


def _turn(alpha: float) -> float:
    rate = float(np.clip(1.5 * alpha, -CREEP_TURN, CREEP_TURN))
    return math.copysign(max(abs(rate), 0.15), alpha)
