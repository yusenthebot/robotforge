"""What a loaded catalogue robot is and which tools work on it, read from the compiled world.

catalogue_inspect.finish prepares the file's robot (an arm's tool point); this module looks at the result in the
world: it classifies the robot by structure (arm, quadruped, humanoid, wheeled, hand, aircraft, other) rather than by
what its makers tagged it, and says in a line what it is and which of the tools fit it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import stance
from robotforge.design import catalogue, catalogue_probe

MOVING = frozenset({int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)})  # joints that move
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
    checked: dict[str, Any] | None = None  # what a short trial of its legs showed (catalogue_probe.py)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "about": self.line, "can": self.can}
        out.update(mass_kg=round(self.mass, 2), joints=self.joints, motors=self.motors)
        if self.legs:
            out["legs"] = self.legs
        if self.reach is not None:
            out["reach_m"] = round(self.reach, 2)
        if self.checked is not None:
            out["checked"] = {**self.checked, "summary": catalogue_probe.describe(self.checked)}
        return out

    def with_trial(self, checked: dict[str, Any]) -> Described:
        """This description after a trial of the robot's legs: what the trial showed, and what it makes of the promises
        (a robot that collapsed at home is not "a walking robot", a gait that fell is offered with that warning)."""
        what = {"humanoid": "a walking robot on two legs", "quadruped": "a four-legged robot"}.get(self.kind)
        line = self.line
        can = list(self.can)
        if not checked["stands"]:
            collapsed = (
                "a two-legged robot that collapses at home"
                if self.kind == "humanoid"
                else ("a four-legged robot that collapses at home")
            )
            line = line.replace(what, collapsed) if what else line
            can = [
                f"it {catalogue_probe.describe(checked)}: no controller here keeps it up. "
                "set_controls and pose move its joints; do not expect move to walk it"
            ]
        elif checked["walk"] == "fell":
            can = [
                f"{c.split(';')[0]}; in a trial it fell over after {checked['fell_after_m']} m at "
                f"{checked['walk_speed']} m/s, so keep moves short and slow and check pose 'upright'"
                if c.startswith("move starts a gait")
                else c
                for c in can
            ]
        elif checked["walk"] == "stalled":
            can = [
                f"{c.split(';')[0]}; in a trial it stood but got only {checked['walked_m']} m in "
                f"{checked['walk_s']:.0f} s at {checked['walk_speed']} m/s, so do not count on it to travel"
                if c.startswith("move starts a gait")
                else c
                for c in can
            ]
        elif checked["walk"] == "walked":
            can = [
                f"{c} (a trial walked {checked['walked_m']} m upright)" if c.startswith("move starts") else c
                for c in can
            ]
        elif checked["walk"] == "no gait":
            can = [f"{c} (it stands at home)" for c in can]
        return Described(self.kind, line, can, self.legs, self.reach, self.mass, self.joints, self.motors, checked)


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
    chains = _arm_chains(model, robot) if kind == "arm" else []
    arm = "a rigid arm"
    if chains:
        arm = f"a {chains[0]}-joint arm" if len(chains) == 1 else f"two arms of {max(chains)} joints each"
    what = {
        "arm": arm + (f" reaching about {reach:.2f} m" if reach else ""),
        "quadruped": "a four-legged robot",
        "humanoid": "a walking robot on two legs",
        "wheeled": "a wheeled robot",
        "hand": f"a hand or gripper with {joints} joints",
        "aircraft": "an aircraft",
        "other": "a single rigid body"
        if not joints
        else f"a {joints}-joint robot" + (f" with {legs} legs" if legs > 1 else ""),
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
    bodies = _bodies(model, robot)
    # Wheels as feet make a wheeled robot, however many legs carry them (a wheel-legged robot is no walker)
    on_wheels = bool(bodies) and kind not in (Kind.ARM, Kind.AIRCRAFT) and stance.is_floating(model, bodies)
    if on_wheels and kind not in (Kind.ROVER, Kind.BALANCER) and stance.count_wheels(model, bodies) >= 2:
        return "wheeled", 0
    if kind in known and not (kind == Kind.ARM and not _moves_a_tool_point(model, robot)):
        return known[kind]
    if not bodies:
        return "other", 0
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_kinematics(model, data)
    floating = stance.is_floating(model, bodies)
    feet = stance.count_feet(model, data, bodies) if floating else 0
    if feet == 2:
        return "humanoid", 2
    if feet == 4:
        return "quadruped", 4
    if not floating and _is_hand(model, data, bodies):
        return "hand", 0
    return "other", feet


GRIPPER_NAME = re.compile(r"gripper|finger|jaw|claw|hand", re.IGNORECASE)


def _arm_chains(model: mujoco.MjModel, robot: str) -> list[int]:
    """How many joints each arm of the robot has between its base and its tool point."""
    from robotforge.behaviors.arm import arm_joints

    prefix = f"{robot}/"
    sites = [s for s in range(model.nsite) if _name(model, mujoco.mjtObj.mjOBJ_SITE, s).startswith(prefix)]
    tools = [s for s in sites if _name(model, mujoco.mjtObj.mjOBJ_SITE, s).endswith("tcp")]
    chains = [list(arm_joints(model, s)) for s in tools]
    for chain in chains:  # the tool point can sit on a finger: the gripper's joints are not the arm's
        while chain and GRIPPER_NAME.search(_name(model, mujoco.mjtObj.mjOBJ_JOINT, chain[-1])):
            chain.pop()
    return [len(c) for c in chains if c]


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
        info = next((r for r in manager.robots() if r["name"] == robot), {}) if manager is not None else {}
        if info.get("kind") in ("rover", "balancer") or "max_speed" in info:
            return ["drives with move and navigate"]
        return [
            "its wheels and joints move with set_controls and pose; move and navigate need a differential-drive or "
            "balance controller that fits its wheels, and none does"
        ]
    if kind == "hand":
        return ["its joints move with set_controls; it is a hand: it grasps once it is on an arm"]
    if kind == "aircraft":
        return ["flies with move (heave for height)"]
    if model.nu and any(_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a).startswith(f"{robot}/") for a in range(model.nu)):
        return ["its joints move with set_controls and pose"]
    return [
        "a rigid body: the file has no motors or joints, so it rests or is pushed (the library has drones with rotors)"
    ]
