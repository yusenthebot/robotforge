"""An arm working a mechanism: open and close a drawer or a door, press a button, flip a switch or a lever, turn a
valve wheel (design/mechanisms.py).

The hand takes hold of the mechanism's handle and moves the only way the handle can go: along a straight line for a
drawer, round the hinge for a door, a lever or a wheel. Every point of that way is solved before anything moves,
with the tool pointing down and, for a handle the hand holds round a bar, turning with the bar as the bar turns: IK
constrained to the joint's own path. A way the arm cannot follow is cut where its reach ends (the result says how
far it went and why), or refused when it cannot even take hold. A valve turned further than one hold allows is
turned a quarter at a time, the hand letting go and taking hold again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.arm import DOWN, JOINT_SPEED, PAD_PRESS, REACH_TOL, ArmMotion
from robotforge.design import mechanisms
from robotforge.sim.actuators import is_position_servo

UP = -DOWN
HANDLE_SPEED = 0.05  # m/s the handle moves at: a drawer opens 20 cm in 4 s
ABOVE = 0.06  # m above the handle the hand comes down from and goes back up to
PATH_STEP = 0.01  # m between the points of a way that are solved before moving
TURN_PER_HOLD = math.radians(90)  # a wheel turned further is let go and taken hold of again
# m a press aims past the button's bottom stop. A servo pushes only as hard as it is kept from its target: aimed at
# the bottom, the template arm stopped 9 mm short of it, held off by the spring; a real hand pushes firmly too.
PRESS_DEEPER = 0.015
TILT_OK = math.radians(12)  # how far off pointing down an answer may leave the tool
GRASP_SECONDS = 0.7
RELEASE_SECONDS = 0.5
HOLD_PRESS_SECONDS = 0.3
SUCCESS_FRACTION = 0.8  # of the way asked: less than this, the result says it fell short
MAX_JOINT_STEP = 0.35  # rad a joint may move between two points of a way 1 cm apart: more is a jump to another pose
THERE_M, THERE_RAD = 0.002, math.radians(1.0)  # closer than this to the asked place is there already
HOLD_STARTS = 12  # random starts tried for a hold, besides the arm's pose now and its own solver's answer
PRE_OPEN = 0.5  # of their full opening the fingers open to take a bar: 3.5 cm for our 7 cm gripper, 4 cm on a Panda


class OperateError(ValueError):
    """User-facing: what cannot be done, and what to do instead."""


@dataclass
class Leg:
    """One stretch of an operation: a joint move to `q` (`joints`), or a tool path through poses, or a pause."""

    label: str
    duration: float
    q: np.ndarray | None = None  # end of a joint move
    path: list[np.ndarray] = field(default_factory=list)  # joint values along a tool path, solved beforehand
    grip: float | None = None  # gripper control set when the leg starts


@dataclass
class Way:
    """Where the handle goes, as the joint moves by `delta` from where it is now."""

    joint: int
    hinge: bool
    axis: np.ndarray  # world
    anchor: np.ndarray  # world, a point on the axis
    handle: np.ndarray  # world, where the hand holds (at delta 0)
    rot: np.ndarray  # the handle's frame at delta 0 (x along the bar)
    knob: float = 0.0  # m, the radius of a ball to hold (a switch's, a lever's)

    def at(self, delta: float) -> tuple[np.ndarray, np.ndarray]:
        """(the hold point, the handle's frame) once the joint has moved by `delta`."""
        if not self.hinge:
            return self.handle + self.axis * delta, self.rot
        turn = _rotation(self.axis, delta)
        return self.anchor + turn @ (self.handle - self.anchor), turn @ self.rot

    def length(self, delta: float) -> float:
        """Metres the hold point travels for a move of `delta`."""
        if not self.hinge:
            return abs(delta)
        radius = np.linalg.norm(np.cross(self.axis, self.handle - self.anchor))
        return float(radius * abs(delta))


class Operation:
    """The controller that works one mechanism with one arm (it takes the arm's place among the controllers)."""

    rate_hz = 100.0

    def __init__(self, arm: ArmMotion, legs: list[Leg], report: dict[str, Any], watch: tuple[int, str]):
        self.arm = arm
        self.robot = arm.robot
        self.legs = legs
        self.report = report  # what was planned; the result adds what happened
        self._joint_adr, self._shape = watch
        self._leg: tuple[Leg, float, np.ndarray] | None = None  # (leg, start time, q at its start)
        self.extreme: float | None = None  # lowest joint value seen (a button's press)
        self.current_label = "start"
        model = arm.ik.model
        # Servo stiffness of each arm joint (0 for anything that is not a position servo), for gravity's share of the
        # targets (see _targets).
        self._kp = np.array(
            [float(model.actuator_gainprm[a, 0]) if is_position_servo(model, a) else 0.0 for a in arm.actuators]
        )

    def _targets(self, data: mujoco.MjData) -> np.ndarray:
        """The servo targets that hold the arm at its joint values: each aimed past them by what holds its weight up.
        A position servo pushes only in proportion to how far it is kept from its target, so aimed at the values
        themselves the template arm hung 1.6 cm low over a switch, closed under the knob and missed it; a real
        servo's controller adds that push itself (its integral term)."""
        bias = data.qfrc_bias[self.arm.chain.dof_adr]
        lift = np.divide(bias, self._kp, out=np.zeros_like(bias), where=self._kp > 0)
        return self.arm.q + lift

    @property
    def done(self) -> bool:
        return not self.legs and self._leg is None

    def status(self) -> dict[str, Any]:
        return {"behavior": "operate", "robot": self.robot, "step": self.current_label, "done": self.done}

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        arm = self.arm
        arm.sync(data)
        q_mech = float(data.qpos[self._joint_adr])
        self.extreme = q_mech if self.extreme is None else min(self.extreme, q_mech)
        t = float(data.time)
        if self._leg is None and self.legs:
            leg = self.legs.pop(0)
            self._leg = (leg, t, arm.q.copy())
            self.current_label = leg.label
            if leg.grip is not None and arm.gripper >= 0:
                data.ctrl[arm.gripper] = leg.grip
        if self._leg is None:
            self.current_label = "done"
            data.ctrl[arm.actuators] = self._targets(data)
            return
        leg, t0, q0 = self._leg
        u = min(1.0, (t - t0) / leg.duration)
        s = u * u * (3 - 2 * u)
        if leg.q is not None:
            arm.q = q0 + (leg.q - q0) * s
        elif leg.path:
            k = s * (len(leg.path) - 1)
            i = min(int(k), len(leg.path) - 2) if len(leg.path) > 1 else 0
            f = k - i
            arm.q = leg.path[i] if len(leg.path) == 1 else leg.path[i] + (leg.path[i + 1] - leg.path[i]) * f
        data.ctrl[arm.actuators] = self._targets(data)
        if u >= 1.0:
            self._leg = None


# ---- planning ---------------------------------------------------------------------------------------


def plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmMotion,
    target: str,
    shape: str,
    action: str,
    amount: float | None,
) -> Operation:
    """The operation that does `action` on mechanism `target` (a world object of `shape`), or OperateError."""
    info = mechanisms.MECHANISMS[shape]
    if action not in info.actions:
        can = " or ".join(info.actions) or "nothing operate does (set_controls sets a conveyor's speed)"
        raise OperateError(f"{target} is a {shape}: it can {can}, not {action}")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{target}/{info.part}")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{target}/handle")
    if joint < 0 or site < 0:
        raise OperateError(f"{target} has no handle to take hold of; reload the world (open_world)")
    if arm.gripper < 0 and action != "press":
        raise OperateError(f"{arm.robot} has no gripper or tool to take hold of {target}'s handle with")
    q_now = float(data.qpos[model.jnt_qposadr[joint]])
    lo, hi = (float(v) for v in model.jnt_range[joint])
    delta, asked = _delta(shape, action, amount, q_now, lo, hi, info.unit)
    report = {
        "robot": arm.robot,
        "target": target,
        "action": action,
        "unit": info.unit,
        "asked": asked,
        "start": _shown(q_now, info.unit),
        "notes": [],
    }
    hinge = int(model.jnt_type[joint]) == mujoco.mjtJoint.mjJNT_HINGE
    knob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{target}/knob")
    way = Way(
        joint,
        hinge,
        data.xaxis[joint].copy(),
        data.xanchor[joint].copy(),
        data.site_xpos[site].copy(),
        data.site_xmat[site].reshape(3, 3).copy(),
        float(model.geom_size[knob][0]) if knob >= 0 else 0.0,
    )
    watch = (int(model.jnt_qposadr[joint]), shape)
    if action == "press" and shape == "button":
        legs = _press_legs(model, data, arm, target, way, delta)
        return Operation(arm, legs, report, watch)
    if abs(delta) < (THERE_M if info.unit == "m" else THERE_RAD):  # within 2 mm or 1 degree: nothing to do
        report["notes"].append(f"{target} is already there: {_said(q_now, info.unit)}")
        return Operation(arm, [], report, watch)
    if shape == "valve":
        way = _nearest_hold(way, arm)
    legs, done = _hold_and_move(model, data, arm, target, info.grip, way, delta)
    if abs(done) < abs(delta) - 1e-6:
        went, asked = _said(abs(done), info.unit), _said(abs(delta), info.unit)
        report["notes"].append(
            f"{arm.robot} can follow the handle only {went} of the {asked} asked before it leaves the reach of its "
            f"gripper pointing down; move the arm or the {shape} nearer each other to go further"
        )
    report["planned"] = _shown(abs(done), info.unit)
    return Operation(arm, legs, report, watch)


def _delta(
    shape: str, action: str, amount: float | None, q: float, lo: float, hi: float, unit: str
) -> tuple[float, float]:
    """How far the joint moves (its own units, signed) and how far that was asked (m or degrees, as asked)."""
    if amount is not None and (not math.isfinite(amount) or amount < 0 and action != "turn"):
        raise OperateError(f"amount must be a positive number ({'m' if unit == 'm' else 'degrees'}), got {amount}")
    to_joint = (lambda v: v) if unit == "m" else math.radians
    if action == "press":  # a switch or a lever flips to the other end; a button goes down and comes back
        if shape == "button":
            return -(mechanisms.BUTTON_TRAVEL + PRESS_DEEPER), 0.0
        end = 0.9 * (hi if q < 0 else lo)
        return end - q, round(abs(math.degrees(end - q)), 1)
    if action == "turn":
        if amount is None:
            if shape == "valve":
                raise OperateError(
                    "turn needs an amount in degrees (positive turns counter-clockwise, seen from above)"
                )
            end = 0.9 * (hi if q < 0 else lo)
            return end - q, round(math.degrees(end - q), 1)
        want = to_joint(amount)
        return float(np.clip(q + want, lo, hi) - q), amount
    sign = 1.0 if action == "open" else -1.0
    if amount is None and (shape == "valve" or shape == "cabinet" and action == "open"):
        amount = 90.0  # a quarter turn; a door opened square to its cabinet
    goal = (hi if sign > 0 else lo) if amount is None else q + sign * to_joint(amount)
    delta = float(np.clip(goal, lo, hi) - q)
    return delta, (amount if amount is not None else abs(_shown(delta, unit)))


def _shown(q: float, unit: str) -> float:
    return round(q, 3) if unit == "m" else round(math.degrees(q), 1)


def _said(q: float, unit: str) -> str:
    return f"{q:.2f} m" if unit == "m" else f"{math.degrees(q):.0f} degrees"


def _nearest_hold(way: Way, arm: ArmMotion) -> Way:
    """A wheel is held anywhere on its rim: hold it at the point nearest the arm's base, where the arm reaches best
    for a quarter turn either way."""
    radius = way.handle - way.anchor
    radius -= way.axis * (radius @ way.axis)
    toward = np.append(arm.base_xy, way.anchor[2]) - way.anchor
    toward -= way.axis * (toward @ way.axis)
    if np.linalg.norm(toward) < 1e-6:
        return way
    a = math.atan2(float(np.cross(radius, toward) @ way.axis), float(radius @ toward))
    turn = _rotation(way.axis, a)
    return Way(
        way.joint, True, way.axis, way.anchor, way.anchor + turn @ (way.handle - way.anchor), turn @ way.rot, way.knob
    )


def _hold_and_move(
    model: mujoco.MjModel, data: mujoco.MjData, arm: ArmMotion, target: str, grip: str, way: Way, delta: float
) -> tuple[list[Leg], float]:
    """Legs that take hold of the handle and move it by `delta` (in holds of at most TURN_PER_HOLD for a wheel),
    and how far of `delta` the arm can follow."""
    legs: list[Leg] = []
    done = 0.0
    q = arm.q.copy()
    wheel = way.hinge and _is_wheel(target, model)
    holds = max(1, math.ceil(abs(delta) / TURN_PER_HOLD - 1e-9)) if wheel else 1
    step = delta / holds
    for k in range(holds):
        # A wheel is taken hold of again at the same place on its rim's way round (any point of a rim is one to
        # hold); anything else is held once.
        segment, reached, q = _one_hold(model, data, arm, target, grip, way, 0.0, step, q, first=k == 0)
        legs += segment
        done += reached
        if abs(reached) < abs(step) - 1e-6:
            break
    if not legs:
        raise OperateError(
            f"{arm.robot} cannot take hold of {target}'s handle with its gripper pointing down: it is "
            f"{_distance(arm, way.handle):.2f} m from the arm's base, at height {way.handle[2]:.2f} m; move the arm "
            f"or the {target} so the handle is within reach"
        )
    return legs, done


def _is_wheel(target: str, model: mujoco.MjModel) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{target}/wheel") >= 0


def _distance(arm: ArmMotion, point: np.ndarray) -> float:
    return float(np.linalg.norm(point[:2] - arm.base_xy))


def _one_hold(
    model, data, arm: ArmMotion, target: str, grip: str, way: Way, start: float, step: float, q: np.ndarray, first: bool
) -> tuple[list[Leg], float, np.ndarray]:
    """Take hold at the handle as it is once the joint has moved by `start`, move it on by `step`, let go and back
    off: (legs, how far of `step` it gets, the arm's joints at the end)."""
    point, frame = way.at(start)
    tip = _hold_point(arm, grip, point, way.knob)
    rot0 = _hand_rotation(model, data, arm, grip, frame, q, way.axis if way.hinge else None)
    turning = way.hinge and grip == "bar" and rot0 is not None  # a bar on a door or a wheel: the hand turns with it
    above = tip + UP * ABOVE

    def follow(q_hold: np.ndarray) -> tuple[list[np.ndarray], float]:
        """The joint values along the handle's way from a hold, and how far of `step` they go."""
        path, reached, qs = [q_hold], 0.0, q_hold
        n = max(2, math.ceil(way.length(step) / PATH_STEP))
        for i in range(1, n + 1):
            d = start + step * i / n
            p, f = way.at(d)
            rot = (_rotation(way.axis, d - start) @ rot0) if turning else rot0
            q_next = _solve(arm, _hold_point(arm, grip, p, way.knob), rot, qs, iterations=30)
            if q_next is None or np.max(np.abs(q_next - qs)) > MAX_JOINT_STEP:  # out of reach, or another pose
                break
            path.append(q_next)
            qs, reached = q_next, step * i / n
        return path, reached

    # The hold decides the arm's pose for the whole way (elbow up or down, wrist over or under): of the holds the
    # arm can take, the one whose way goes furthest, then the nearest to how the arm is now. Solving the hold from
    # a pose above it first, the template arm took the drawer's bar elbow down and ran into the floor 1 cm later.
    best: tuple[float, float, np.ndarray, list[np.ndarray]] | None = None
    for q_hold in _holds(arm, tip, rot0, q, fresh=first):
        path, reached = follow(q_hold)
        rank = (abs(reached), -float(np.max(np.abs(q_hold - q))))
        if best is None or rank > (abs(best[0]), best[1]):
            best = (reached, rank[1], q_hold, path)
    if best is None:
        return [], 0.0, q
    reached, _, q_hold, path = best
    down = _line(arm, tip, above, rot0, q_hold)[::-1]  # solved up from the hold, so it ends exactly there
    if len(down) < 2:
        return [], 0.0, q
    open_grip, close_grip = arm._grip("open"), arm._grip("close")
    if arm.tool is None and open_grip is not None:
        # Opened all the way, the finger behind the bar landed on the drawer's front: open only as far as the bar
        # needs, as a hand does reaching for a handle.
        open_grip *= PRE_OPEN
    legs = [
        Leg("approach", _joint_time(q, down[0]), q=down[0], grip=open_grip),
        Leg("reach", 1.2, path=down),
        Leg("take hold", GRASP_SECONDS, path=[q_hold], grip=close_grip),
    ]
    qs = path[-1]
    if len(path) > 1:
        legs.append(Leg("move", max(1.0, way.length(reached) / HANDLE_SPEED), path=path))
    end_tip = arm.ik.point(arm.chain, qs)
    end_rot = (_rotation(way.axis, reached) @ rot0) if turning else rot0
    retreat = _line(arm, end_tip, end_tip + UP * ABOVE, end_rot, qs)
    legs += [Leg("let go", RELEASE_SECONDS, path=[qs], grip=open_grip), Leg("back off", 1.0, path=retreat)]
    return legs, reached, retreat[-1]


def _press_legs(model, data, arm: ArmMotion, target: str, way: Way, delta: float) -> list[Leg]:
    """Come down over the button with the gripper closed (a fingertip pusher) or a cup off, push it to the bottom
    and past, hold, and come back up."""
    reach = _tip_reach(model, data, arm)
    top = way.handle + UP * reach
    q0 = arm.q.copy()
    above = top + UP * ABOVE
    q_above = _solve(arm, above, None, q0, fresh=True)
    if q_above is None:
        raise OperateError(
            f"{arm.robot} cannot reach above {target} with its gripper pointing down: it is "
            f"{_distance(arm, top):.2f} m from the arm's base; move the arm or the button nearer each other"
        )
    pushed = top + way.axis * delta
    down = _line(arm, above, pushed, None, q_above)
    if len(down) < 2 or _solve(arm, pushed, None, down[-1]) is None:
        raise OperateError(f"{arm.robot} cannot push {target} down from there; move the arm or the button")
    pusher = arm._grip("close") if arm.tool is None else arm._grip("open")
    return [
        Leg("approach", _joint_time(q0, q_above), q=q_above, grip=pusher),
        Leg("press", 1.4, path=down),
        Leg("hold", HOLD_PRESS_SECONDS, path=[down[-1]]),
        Leg("let go", 1.0, path=down[::-1]),
    ]


def _tip_reach(model: mujoco.MjModel, data: mujoco.MjData, arm: ArmMotion) -> float:
    """How far the hand reaches past its tool point along the tool axis (fingertips past the point between them; 0
    for a cup's face), measured on the posed arm's collision shapes."""
    site = arm.site
    axis = data.site_xmat[site].reshape(3, 3)[:, 2]
    tcp = data.site_xpos[site]
    hand = int(model.site_bodyid[site])
    furthest = 0.0
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if not (model.geom_contype[g] or model.geom_conaffinity[g]) or not _below(model, body, hand):
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g] + rot @ model.geom_aabb[g, :3]
        extent = float(np.abs(axis @ rot) @ model.geom_aabb[g, 3:])
        furthest = max(furthest, float(axis @ (center - tcp)) + extent)
    return furthest


def _below(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _hold_point(arm: ArmMotion, grip: str, point: np.ndarray, knob: float) -> np.ndarray:
    """Where the tool point goes to hold a handle at `point`: between the fingers round a bar or a knob, or on top of
    it for a suction cup or a magnet (pressed on a little; a ball's top is the same height over its middle however
    its stick is tipped)."""
    if arm.tool is None:
        return point.copy()
    if grip == "knob":
        return point + UP * (knob - PAD_PRESS)
    return point + UP * (mechanisms.BAR / 2 - PAD_PRESS)


def _hand_rotation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmMotion,
    grip: str,
    frame: np.ndarray,
    q: np.ndarray,
    hinge: np.ndarray | None = None,
) -> np.ndarray | None:
    """The hand's frame for holding a handle, tool pointing down: fingers closing across a bar, or along the way a
    knob on a stick about a level `hinge` goes, so they push it both ways (closing across that way, a switch's knob
    slipped 7 mm in the grip and the switch stopped 11 degrees short). A cup or a magnet on a bar keeps the hand
    turned as it is now. None where only pointing down matters (a knob held by a cup, a button)."""
    if grip not in ("bar", "knob"):
        return None
    arm.ik.point(arm.chain, q)
    now = arm.ik.data.xmat[arm.chain.body].reshape(3, 3)
    a = arm.tool_axis / np.linalg.norm(arm.tool_axis)
    closing = _closing_axis(model, arm, now)
    if grip == "knob" and (closing is None or hinge is None):
        return None
    across = np.cross(UP, hinge if grip == "knob" else frame[:, 0])  # level, square to the bar or the hinge
    if np.linalg.norm(across) < 1e-6:
        across = np.array([1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    if closing is None:  # a cup or a magnet: any turn holds, so keep the hand's own
        c_local = _perpendicular(a)
        seen = now @ c_local
        seen -= UP * (seen @ UP)
        if np.linalg.norm(seen) < 1e-6:
            seen = across
        return _frame(a, c_local, DOWN, seen / np.linalg.norm(seen))
    c_local = closing - a * (closing @ a)
    c_local /= np.linalg.norm(c_local)
    options = [_frame(a, c_local, DOWN, across), _frame(a, c_local, DOWN, -across)]
    return min(options, key=lambda r: _angle(r, now))  # the one the wrist turns least to


def _closing_axis(model: mujoco.MjModel, arm: ArmMotion, hand_rot: np.ndarray) -> np.ndarray | None:
    """The way the fingers close, in the hand's frame: the axis of the slide their actuator drives."""
    if arm.tool is not None or arm.gripper < 0:
        return None
    if int(model.actuator_trntype[arm.gripper]) != mujoco.mjtTrn.mjTRN_JOINT:
        return np.array([0.0, 1.0, 0.0])  # a tendon-driven pair (a Panda): ours and Franka's close along y
    j = int(model.actuator_trnid[arm.gripper, 0])
    d = arm.ik.data
    world = d.xaxis[j] if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_SLIDE else np.array([0.0, 1.0, 0.0])
    return hand_rot.T @ world


def _frame(a_local: np.ndarray, c_local: np.ndarray, a_world: np.ndarray, c_world: np.ndarray) -> np.ndarray:
    """The rotation taking hand axes a and c (orthonormal) onto world directions a and c."""
    b_local = np.cross(a_local, c_local)
    c_world = c_world - a_world * (c_world @ a_world)
    c_world /= np.linalg.norm(c_world)
    b_world = np.cross(a_world, c_world)
    return np.column_stack([a_world, c_world, b_world]) @ np.column_stack([a_local, c_local, b_local]).T


def _perpendicular(a: np.ndarray) -> np.ndarray:
    other = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p = other - a * (other @ a)
    return p / np.linalg.norm(p)


def _angle(r1: np.ndarray, r2: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(r1.T @ r2) - 1) / 2, -1.0, 1.0)))


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, float), float(angle))
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    return rot.reshape(3, 3)


def _solve(
    arm: ArmMotion, point: np.ndarray, rot: np.ndarray | None, q0: np.ndarray, iterations: int = 80, fresh: bool = False
) -> np.ndarray | None:
    """Joint values putting the tool at `point` pointing down (and the hand at `rot`, when given), or None when the
    arm cannot: too far off, tipped, or into itself or what carries it. `fresh` also tries the arm's own solver's
    starts (its base turned toward the point, then random starts) for the first pose of a move."""
    starts = [q0]
    if fresh:
        seeded, err, _ = arm._solve(point)
        if err <= REACH_TOL:
            starts.insert(0, seeded)
    for start in starts:
        if rot is None:
            q, err = arm.ik.solve(arm.chain, point, start, iterations=iterations, axis=(arm.tool_axis, DOWN))
        else:
            q, err = arm.ik.solve(arm.chain, point, start, iterations=iterations, rotation=rot)
        if err > REACH_TOL:
            continue
        arm.ik.point(arm.chain, q)
        hand = arm.ik.data.xmat[arm.chain.body].reshape(3, 3)
        tilt = math.acos(float(np.clip((hand @ arm.tool_axis) @ DOWN / np.linalg.norm(arm.tool_axis), -1, 1)))
        if tilt > TILT_OK or (rot is not None and _angle(hand, rot) > 2 * TILT_OK) or arm._collides(q):
            continue
        return q
    return None


def _holds(arm: ArmMotion, point: np.ndarray, rot: np.ndarray | None, q: np.ndarray, fresh: bool) -> list[np.ndarray]:
    """Different ways the arm can hold the tool at `point` (and the hand at `rot`): from how it is now, from the arm's
    own solver's answer and from fixed random starts with the base turned toward the point."""
    starts = [q]
    if fresh:
        seeded, err, _ = arm._solve(point)
        if err <= REACH_TOL:
            starts.append(seeded)
        rng = np.random.default_rng(0)
        base = arm._seed(point)
        for _ in range(HOLD_STARTS):
            start = rng.uniform(arm.chain.lower, arm.chain.upper)
            if arm.yaw is not None:
                start[arm.yaw] = base[arm.yaw]
            starts.append(start)
    found: list[np.ndarray] = []
    for start in starts:
        answer = _solve(arm, point, rot, start)
        if answer is not None and all(np.max(np.abs(answer - other)) > 0.2 for other in found):
            found.append(answer)
    return found


def _line(arm: ArmMotion, a: np.ndarray, b: np.ndarray, rot: np.ndarray | None, q0: np.ndarray) -> list[np.ndarray]:
    """Joint values along the straight tool path from `a` to `b`, solved from `q0` on (stops where it fails)."""
    n = max(2, math.ceil(float(np.linalg.norm(b - a)) / PATH_STEP))
    out, q = [q0], q0
    for i in range(1, n + 1):
        nxt = _solve(arm, a + (b - a) * i / n, rot, q, iterations=30)
        if nxt is None:
            break
        out.append(nxt)
        q = nxt
    return out


def _joint_time(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.8, float(np.max(np.abs(b - a))) / JOINT_SPEED)


def start(manager: Any, robot: str, target: str, action: str, amount: float | None) -> tuple[Operation, str, str]:
    """Plan `action` on `target` for the arm `robot` names and set it going in its place among the controllers:
    (the operation, the target as the world spells it, its shape). Raises OperateError or BehaviorError."""
    from robotforge import wording  # light, but kept with the one use

    session = manager.session
    with session.lock:
        name = manager._arm_name(robot)
        world = manager._world()
        objects = {o.name: o for o in world.objects} if world is not None else {}
        found = wording.match(target, list(objects))
        if found is None:
            raise OperateError(wording.unknown("object", target, list(objects)))
        shape = objects[found].shape
        if shape not in mechanisms.MECHANISMS or not mechanisms.MECHANISMS[shape].actions:
            things = ", ".join(s for s, m in mechanisms.MECHANISMS.items() if m.actions)
            raise OperateError(
                f"{found} is a {shape}, nothing to operate: operate works {things}"
                + ("; pick_and_place moves it" if shape in ("box", "sphere", "cylinder") else "")
            )
        try:
            arm = ArmMotion(session.model, name, manager.home_qpos())
        except ValueError as exc:
            raise OperateError(str(exc)) from exc
        arm.sync(session.data)
        arm.q = session.data.qpos[arm.chain.qpos_adr].copy()
        op = plan(session.model, session.data, arm, found, shape, action, amount)
        session.set_controller(name, op)
    return op, found, shape


def result(op: Operation, model: mujoco.MjModel, data: mujoco.MjData, state: dict[str, Any]) -> dict[str, Any]:
    """What an operation did: how far the mechanism moved (in m or degrees), its state now, and whether that was
    what was asked."""
    report = dict(op.report)
    unit = report["unit"]
    now = float(data.qpos[op._joint_adr])
    start = report["start"] if unit == "m" else math.radians(report["start"])
    moved = now - start
    report["moved"] = _shown(moved, unit)
    report["state"] = state
    if report["action"] == "press" and op._shape == "button":
        low = op.extreme if op.extreme is not None else now
        report["pressed"] = low < -mechanisms.PRESSED_AT * mechanisms.BUTTON_TRAVEL
        report["success"] = report["pressed"]
        report.pop("moved")
    elif report["action"] == "press":  # a switch or a lever flipped: it is on the other side now
        report["success"] = bool(np.sign(now) != np.sign(start) and abs(now) > math.radians(2))
    else:
        asked = report["asked"]
        asked_joint = asked if unit == "m" else math.radians(asked)
        report["success"] = bool(abs(moved) >= SUCCESS_FRACTION * abs(asked_joint) - 1e-6) if asked else True
        if not report["success"] and "planned" in report and abs(report["planned"]) >= SUCCESS_FRACTION * abs(asked):
            report["notes"].append(
                "the hand could not keep hold of the handle on the way (it slipped); try again, or with the other "
                "end effector"
            )
    if not report["notes"]:
        report.pop("notes")
    report.pop("start")
    report.pop("planned", None)
    return report
