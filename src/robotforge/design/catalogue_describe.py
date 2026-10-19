"""What a loaded catalogue robot is and which tools work on it, read from the compiled world.

catalogue_inspect.finish prepares the file's robot (an arm's tool point); this module looks at the result in the
world: it classifies the robot by structure (arm, quadruped, humanoid, wheeled, hand, aircraft, other) rather than by
what its makers tagged it, and says in a line what it is and which of the tools fit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from robotforge.design import catalogue
from robotforge.sim.geometry import lowest_point

MOVING = frozenset({int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)})  # joints that move
FOOT_BAND = 0.08  # m above the lowest point of the robot at home that still counts as standing on it
FOOT_APART = 0.1  # of the robot's height between feet: closer contact points belong to one foot (toe and heel)
SMALL_HAND = 0.45  # m, the largest extent of a hand or gripper
REACH_SAMPLES = 150


@dataclass(frozen=True)
class Described:
    kind: str
    line: str
    can: list[str]
    legs: int = 0
    reach: float | None = None
    mass: float = 0.0
    joints: int = 0
    motors: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "about": self.line, "can": self.can}
        out.update(mass_kg=round(self.mass, 2), joints=self.joints, motors=self.motors)
        if self.legs:
            out["legs"] = self.legs
        if self.reach is not None:
            out["reach_m"] = round(self.reach, 2)
        return out


def describe(
    model: mujoco.MjModel, robot: str, home_qpos: np.ndarray, entry: catalogue.Entry, manager: Any = None
) -> Described:
    """What the loaded robot `robot` is and which tools work on it."""
    kind, legs = classify(model, robot, home_qpos)
    bodies = _bodies(model, robot)
    mass = float(sum(model.body_mass[b] for b in bodies))
    joints = sum(1 for b in bodies for j in _joints(model, b) if int(model.jnt_type[j]) in MOVING)
    motors = sum(1 for a in range(model.nu) if _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a).startswith(f"{robot}/"))
    reach = _reach(model, robot, home_qpos) if kind == "arm" else None
    what = {
        "arm": f"a {joints}-joint arm" + (f" reaching about {reach:.2f} m" if reach else ""),
        "quadruped": "a four-legged robot",
        "humanoid": "a walking robot on two legs",
        "wheeled": "a wheeled robot",
        "hand": f"a hand or gripper with {joints} joints",
        "aircraft": "an aircraft",
        "other": f"a {joints}-joint robot" + (f" with {legs} legs" if legs else ""),
    }[kind]
    line = f"{entry.title}: {what}, {mass:.1f} kg, {motors} motors ({entry.format.upper()}, {entry.license or '?'})"
    return Described(kind, line, _abilities(model, robot, kind, manager), legs, reach, mass, joints, motors)


def classify(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> tuple[str, int]:
    """(arm, quadruped, humanoid, wheeled, hand, aircraft or other; number of legs) from the structure of `robot`."""
    from robotforge.behaviors import Kind, robot_kind

    kind = robot_kind(model, robot)
    known = {
        Kind.ARM: ("arm", 0),
        Kind.AIRCRAFT: ("aircraft", 0),
        Kind.QUADRUPED: ("quadruped", 4),
        Kind.HEXAPOD: ("other", 6),
        Kind.ROVER: ("wheeled", 0),
        Kind.BALANCER: ("wheeled", 0),
        Kind.HUMANOID: ("humanoid", 2),
    }
    if kind in known and not (kind == Kind.ARM and not _moves_a_tool_point(model, robot)):
        return known[kind]
    bodies = _bodies(model, robot)
    if not bodies:
        return "other", 0
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_kinematics(model, data)
    floating = any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for b in bodies for j in _joints(model, b))
    feet = _feet(model, data, bodies) if floating else 0
    if feet == 2:
        return "humanoid", 2
    if feet == 4:
        return "quadruped", 4
    if not floating and _is_hand(model, data, bodies):
        return "hand", 0
    return "other", feet


def _moves_a_tool_point(model: mujoco.MjModel, robot: str) -> bool:
    """Whether the robot's "tcp" site is moved by powered joints: a hand whose file names a site "tcp" is no arm."""
    from robotforge.behaviors.arm import arm_joints

    prefix = f"{robot}/"
    sites = [s for s in range(model.nsite) if _name(model, mujoco.mjtObj.mjOBJ_SITE, s).startswith(prefix)]
    return any(_name(model, mujoco.mjtObj.mjOBJ_SITE, s).endswith("tcp") and arm_joints(model, s) for s in sites)


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or ""


def _joints(model: mujoco.MjModel, body: int) -> range:
    return range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body]))


def _bodies(model: mujoco.MjModel, robot: str) -> list[int]:
    prefix = f"{robot}/"
    return [b for b in range(1, model.nbody) if _name(model, mujoco.mjtObj.mjOBJ_BODY, b).startswith(prefix)]


def _colliders(model: mujoco.MjModel, bodies: list[int]) -> list[int]:
    wanted = set(bodies)
    return [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in wanted and model.geom_contype[g] + model.geom_conaffinity[g] > 0
    ]


def _feet(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]) -> int:
    """How many separate places the robot stands on at home: the lowest points of its bodies within a hand's width of
    the lowest, merged where they are close."""
    points = []
    for body in bodies:  # a foot is often not a leaf: it carries a toe, a sensor or a wheel hub
        geoms = _colliders(model, [body])
        if geoms:
            points.append(min((lowest_point(model, data, g) for g in geoms), key=lambda p: p[2]))
    if not points:
        return 0
    floor = min(p[2] for p in points)
    height = max(float(data.xpos[b][2]) for b in bodies) - floor
    apart = max(0.03, FOOT_APART * height)
    places: list[np.ndarray] = []
    for p in (p for p in points if p[2] <= floor + FOOT_BAND * max(height, 0.2) / 0.6):
        if all(np.linalg.norm(p[:2] - q[:2]) > apart for q in places):
            places.append(p)
    return len(places)


def _is_hand(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]) -> bool:
    """A small robot of several moving parts that end in separate tips: a hand or a gripper."""
    points = np.array([data.xpos[b] for b in bodies])
    extent = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.0
    moving = [b for b in bodies if any(int(model.jnt_type[j]) in MOVING for j in _joints(model, b))]
    parents = {int(model.body_parentid[b]) for b in bodies}
    tips = [b for b in bodies if b not in parents]
    return extent <= SMALL_HAND and len(moving) >= 2 and len(tips) >= 2


def _reach(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> float | None:
    """How far the tool point gets from the arm's first joint over random poses within the joint limits."""
    from robotforge.behaviors.arm import arm_joints

    prefix = f"{robot}/"
    sites = [s for s in range(model.nsite) if _name(model, mujoco.mjtObj.mjOBJ_SITE, s).startswith(prefix)]
    site = next((s for s in sites if _name(model, mujoco.mjtObj.mjOBJ_SITE, s).endswith("tcp")), None)
    if site is None:
        return None
    joints = arm_joints(model, site)
    if not joints:
        return None
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_kinematics(model, data)
    origin = data.xanchor[joints[0]].copy()
    rng = np.random.default_rng(0)
    farthest = 0.0
    for _ in range(REACH_SAMPLES):
        for j in joints:
            low, high = model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi)
            data.qpos[model.jnt_qposadr[j]] = rng.uniform(low, high)
        mujoco.mj_kinematics(model, data)
        farthest = max(farthest, float(np.linalg.norm(data.site_xpos[site] - origin)))
    return farthest


def _abilities(model: mujoco.MjModel, robot: str, kind: str, manager: Any) -> list[str]:
    if kind == "arm":
        arms = manager.arms(robot) if manager is not None else [robot]
        names = " or ".join(f"'{a}'" for a in arms)
        out = [f"move_arm({names}, [x, y, z]) steers a tool point by inverse kinematics"]
        grippers = [a for a in arms if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a}/gripper") >= 0]
        out.append("gripper and pick_and_place work" if grippers else "no gripper yet: mount one on its flange")
        return out
    if kind in ("quadruped", "humanoid") and manager is not None:
        info = next((r for r in manager.robots() if r["name"] == robot), {})
        if "max_speed" in info:
            return [
                f"move starts a gait found from its legs (up to {info['max_speed']} m/s); the gait is tuned for the "
                "library's robots, so check pose 'upright' after a move"
            ]
        return [info.get("walk_hint") or "stands; move cannot walk it and it cannot learn a walk here"]
    if kind == "wheeled":
        return ["drives with move and navigate"]
    if kind == "hand":
        return ["its joints move with set_controls; it is a hand: it grasps once it is on an arm"]
    if kind == "aircraft":
        return ["flies with move (heave for height)"]
    if model.nu and any(_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a).startswith(f"{robot}/") for a in range(model.nu)):
        return ["its joints move with set_controls and pose"]
    return [
        "a rigid body: the file has no motors or joints, so it rests or is pushed (the library has drones with rotors)"
    ]
