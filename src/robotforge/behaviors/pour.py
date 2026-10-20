"""Pouring: take a cup (or a mug, a bowl) by its grasp, carry it over another container, tilt it about its lip until
what is in it runs out, tip it back and put it down where it stood.

The tilt is worked out beforehand in joint space. The lip of the cup is held still above the target while the cup
turns about it (so what runs over the lip falls where it was aimed), the tool point follows from that, and the arm's
inverse kinematics finds the joints for the tool point and the way the tool points at every few degrees of tilt. A tilt
the arm cannot make (a joint's limit, the arm running into itself or the floor) is tried again toward another side;
when none works the pour is refused before the arm moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotforge.behaviors.arm import APPROACH, REACH_TOL, ArmMotion, OutOfReach, Waypoint

TILT_STEP = math.radians(6.0)
MAX_TILT = math.radians(118.0)
ABOVE = 0.03  # m the upright cup's base is held over the target's rim
HOLD = 2.4  # s held tilted, shaking
KNOT_TIME = 0.5  # s per knot of the tilt: slow enough for what is in the cup to roll out and not be thrown
SIDES = 12  # directions tried to pour toward
JUMP = 0.7  # rad a joint may change between two knots: more is a jump from one branch of the arm to another


@dataclass
class Pour:
    """What a pour needs about the two containers: the cup's height (rim over base), mouth radius and how high it is
    held over its base, and the middle of the target's mouth at its rim height."""

    height: float
    mouth: float
    held_at: float
    target: np.ndarray
    target_radius: float


def directions(arm: ArmMotion, target: np.ndarray) -> list[np.ndarray]:
    """Horizontal directions to pour toward, the one that points away from the arm's base first, then the opposite and
    the rest round the circle by turns."""
    out = []
    outward = np.asarray(target[:2], float) - np.asarray(arm.base_xy, float)
    if np.linalg.norm(outward) < 1e-6:
        outward = np.array([1.0, 0.0])
    start = math.atan2(outward[1], outward[0])
    order = [0]
    for k in range(1, SIDES // 2 + 1):
        order += [k, -k]
    for k in order[:SIDES]:
        a = start + k * 2 * math.pi / SIDES
        out.append(np.array([math.cos(a), math.sin(a), 0.0]))
    return out


def tool_at(pour: Pour, lip: np.ndarray, d: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """The tool point and the way it points when the cup is tilted `theta` about its lip toward `d`."""
    up = np.array([0.0, 0.0, 1.0])
    a = math.cos(theta) * up + math.sin(theta) * d  # the cup's axis, base to rim
    b = math.cos(theta) * d - math.sin(theta) * up  # from its axis to its lip
    return lip - a * (pour.height - pour.held_at) - b * pour.mouth, -a


def lip_at(pour: Pour, theta: float) -> np.ndarray:
    """Where the cup's lip is held when it is tilted `theta`: over the middle of the target, as low as the cup's base
    allows (the upright cup's base is ABOVE over the target's rim; as it tips, its base lifts and the lip comes down,
    so what runs over it has a short way to fall)."""
    lift = max(pour.height * math.cos(theta), 0.0)
    return np.array([pour.target[0], pour.target[1], float(pour.target[2]) + ABOVE + lift])


def tilt_knots(arm: ArmMotion, q0: np.ndarray, pour: Pour, d: np.ndarray) -> list[np.ndarray] | None:
    """Joint values from upright to the full tilt with the lip kept over the target, or None when the arm cannot
    follow."""
    knots = [q0]
    q = q0
    for theta in np.arange(TILT_STEP, MAX_TILT + 1e-9, TILT_STEP):
        tool, axis = tool_at(pour, lip_at(pour, float(theta)), d, float(theta))
        q2, err = arm.ik.solve(arm.chain, tool, q, iterations=80, axis=(arm.tool_axis, axis))
        q2 = arm._unwound(q2)
        if err > REACH_TOL or arm._collides(q2) or float(np.max(np.abs(q2 - q))) > JUMP:
            return None
        q = q2
        knots.append(q)
    return knots


def pick_and_pour(
    arm: ArmMotion,
    grasp: np.ndarray,
    below: float,
    faces: np.ndarray | None,
    pour: Pour,
    back: np.ndarray,
    back_top: float | None = None,
) -> np.ndarray:
    """Queue taking the cup at `grasp` (the tool point; `below` is how far the cup reaches under it), carrying it over
    the target, pouring, and putting it down at `back` (where its base goes). Returns the direction it pours toward.
    Raises OutOfReach or ValueError, saying why, before anything is queued."""
    chosen = None
    for d in directions(arm, pour.target):
        tool0, _ = tool_at(pour, lip_at(pour, 0.0), d, 0.0)
        reach = arm.key_point_error("place", tool0)
        if reach > REACH_TOL:
            continue
        q0, err, _ = arm._solve(tool0)
        if err > REACH_TOL:
            continue
        knots = tilt_knots(arm, arm._unwound(q0), pour, d)
        if knots is not None:
            chosen = (d, tool0, knots)
            break
    if chosen is None:
        raise ValueError(
            f"{arm.robot} cannot tilt the cup over there: no way of turning it that its joints allow from any side; "
            "put the target nearer the arm"
        )
    d, tool0, knots = chosen
    travel_z = max(float(grasp[2]), float(tool0[2])) + APPROACH
    above = np.array([grasp[0], grasp[1], travel_z])
    above_over = np.array([tool0[0], tool0[1], travel_z])
    points = {"cup": grasp, "above cup": above, "over target": tool0, "above target": above_over}
    bad = arm._out_of_reach(points)
    if bad:
        raise OutOfReach(arm.robot, bad, points)
    across = arm._across(grasp, tool0, travel_z, below, points, "from the cup to over the target")
    above[2] = above_over[2] = across.z
    taking = arm._taking(grasp, above, above_over, across, below, faces)
    down, _ = arm._way_down(above_over, tool0)
    if not down:
        raise ValueError(arm._no_way_down("target", float(across.z)))
    path = np.array(knots)
    time = KNOT_TIME * len(knots)
    arm.queue.extend(taking)
    arm.queue.append(Waypoint(tool0, None, arm._time(above_over, tool0) + 0.3, "over target", turn=0.0))
    arm.queue.append(Waypoint(None, None, time, "pour", joints=path[-1], path=path))
    shake = np.array([path[-1], path[-2], path[-1], path[-3], path[-1], path[-2], path[-1]])  # a wrist shake: a heap
    arm.queue.append(Waypoint(None, None, HOLD, "drain", joints=path[-1], path=shake))  # of balls can arch and jam
    arm.queue.append(Waypoint(None, None, time * 0.7, "upright", joints=path[0], path=path[::-1]))
    arm.put_down(back, below, back_top)
    return d
