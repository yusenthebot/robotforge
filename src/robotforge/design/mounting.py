"""Robots carrying robots: an arm on a robot dog's back, a gripper on an arm's flange.

A robot entry in world.yaml can list `mounts`: whole robots from the library or the templates, each fixed to one
of the carrier's bodies, the way a part's `robot` is in the builder (see builder/assemble.py). The mounted robot
loses its free joint and rides along; its names get its mount name as a prefix, so an arm called "arm" on a Go2
named "go2" is "go2/arm" in the world, and move_arm and pick_and_place work it as they do an arm on a rover.

Where it goes (`at`):
- "top" (the default): on top of the carrier's main body, in the middle of it. A camera, an arm on a dog's back.
- "tool": an arm's flange (the "flange" site library.Tool puts on arms without a gripper). A gripper mounted there
  becomes the arm's own: its grasp point and actuator are the arm's "tcp" and "gripper", so a UR5e with a Robotiq
  2F-85 works like a Franka Panda. This is the default for a gripper on such an arm.
- the name of any body of the carrier, as in its own file: the mount goes on that body's origin.
`shift` moves it from there (m, in that frame) and `rotate` turns it ([roll, pitch, yaw] in degrees, as in the builder).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import mujoco
import numpy as np

from robotforge import wording
from robotforge.design.templates.base import BuiltRobot
from robotforge.sim.geometry import collides, lowest_point

OVERLAP_ERROR = 0.004  # m: deeper than this, the mount and its carrier are inside each other (as the builder's check)
OPTIONS = ("timestep", "integrator", "cone", "impratio")  # physics options a mounted robot takes from its carrier
# m: where rays look for the surface a mount stands on, around the middle of the carrier's top: an arm's base is about
# 7 cm across (PiPER), and resting on the highest point under it, it neither floats over a dip nor sinks into a ridge.
FOOTPRINT = [(0.0, 0.0), (0.03, 0.0), (-0.03, 0.0), (0.0, 0.03), (0.0, -0.03)]


class Mount(Protocol):
    """A world.yaml mount entry (world.MountEntry)."""

    name: str | None
    library: str | None
    template: str | None
    params: dict[str, Any]
    at: str | None
    shift: list[float]
    rotate: list[float]


def default_name(mount: Mount) -> str:
    """A mount's name when none is given: what it is ("arm", "gripper")."""
    from robotforge.design import library

    if mount.name:
        return mount.name
    if mount.library is not None:
        return library.get(mount.library).kind
    return str(mount.template)


def is_gripper(mount: Mount) -> bool:
    from robotforge.design import library

    if mount.library is not None:
        return library.get(mount.library).kind == "gripper"
    return mount.template == "gripper"


def what(mount: Mount) -> str:
    """The mounted robot in words: "AgileX PiPER", "arm template"."""
    from robotforge.design import library

    return library.get(mount.library).title if mount.library is not None else f"{mount.template} template"


def build_child(mount: Mount) -> BuiltRobot:
    from robotforge.design import library, templates

    if mount.library is not None:
        return library.build(mount.library, mount.params)
    return templates.build(str(mount.template), mount.params)


def ride_on(child: mujoco.MjSpec, carrier: mujoco.MjSpec) -> None:
    """Make a robot ready to be attached to a carrier's body: it loses its free joint, and the physics options (which
    are global) follow the carrier, except that elliptic friction cones from the child win, as in a world."""
    root = child.worldbody.first_body()
    for joint in list(root.joints) if root is not None else []:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:  # it rides on the carrier now instead of moving freely
            child.delete(joint)
    if child.option.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        carrier.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        carrier.option.impratio = max(carrier.option.impratio, child.option.impratio)
    for name in OPTIONS:
        setattr(child.option, name, getattr(carrier.option, name))


def attach(carrier: BuiltRobot, mounts: Sequence[Mount], robot: str) -> BuiltRobot:
    """`carrier` with every mount attached, its home pose extended, and checked: a mount that overlaps the carrier is
    refused with the overlap in millimetres, and a floating carrier rises if a mount hangs below its feet."""
    if not mounts:
        return carrier
    spec = carrier.spec
    home_qpos, home_ctrl = dict(carrier.home_qpos), dict(carrier.home_ctrl)
    notes = list(carrier.notes)
    names: list[str] = []
    before = _lowest(spec, home_qpos) if carrier.floating else 0.0
    for mount in mounts:
        name = default_name(mount)
        if name in names:
            raise ValueError(f"robot {robot!r}: two mounts are called {name!r}; give each a name")
        names.append(name)
        try:
            child = build_child(mount)
        except ValueError as exc:
            raise ValueError(f"robot {robot!r}, mount {name!r}: cannot build {what(mount)}: {exc}") from exc
        ride_on(child.spec, spec)
        at = mount.at or ("tool" if is_gripper(mount) and _site(spec, "flange") is not None else "top")
        body, pos, quat, where = _place(spec, carrier.home_qpos, at, robot, name)
        rot = np.zeros(4)
        mujoco.mju_mat2Quat(rot, _euler(mount.rotate).ravel())
        turned = np.zeros(4)
        mujoco.mju_mulQuat(turned, quat, rot)
        offset = _rotate(quat, np.asarray(mount.shift, float))
        if at == "top":  # its base stands on that surface, as it would on the floor (an SO-100's reaches below 0)
            offset += _rotate(turned, np.array([0.0, 0.0, child.spawn_height]))
        frame = body.add_frame(pos=(pos + offset).tolist(), quat=turned.tolist())
        spec.attach(child.spec, prefix=f"{name}/", frame=frame)
        home_qpos.update({f"{name}/{k}": v for k, v in child.home_qpos.items()})
        home_ctrl.update({f"{name}/{k}": v for k, v in child.home_ctrl.items()})
        if at == "tool" and _site(spec, f"{name}/tcp") is not None:
            _complete_the_arm(spec, name, home_ctrl)
            notes.append(f"{name}: {what(mount)} on the arm's flange; it is now the arm's gripper")
        else:
            notes.append(f"{name}: {what(mount)} mounted {where}")
        notes += [f"{name}: {n}" for n in child.notes]
    _check_overlaps(spec, home_qpos, robot, names)
    spawn = carrier.spawn_height
    if carrier.floating:
        spawn += max(0.0, before - _lowest(spec, home_qpos))  # a mount hanging below the feet lifts the robot
    return BuiltRobot(
        spec=spec,
        home_qpos=home_qpos,
        home_ctrl=home_ctrl,
        spawn_height=spawn,
        floating=carrier.floating,
        notes=notes,
        doctor=carrier.doctor,
    )


# ---- where a mount goes --------------------------------------------------------------------------------------------


def _place(
    spec: mujoco.MjSpec, home_qpos: dict[str, float], at: str, robot: str, name: str
) -> tuple[mujoco.MjsBody, np.ndarray, np.ndarray, str]:
    """(body, position, orientation in it, words for the notes) of a mount point."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    if at == "tool":
        flange = _site(spec, "flange")
        if flange is None:
            raise ValueError(
                f"robot {robot!r}, mount {name!r}: at 'tool' needs an arm with a bare flange (a library arm without a "
                "gripper, such as ur5e); give at a body name or 'top'"
            )
        return flange.parent, np.array(flange.pos, float), np.array(flange.quat, float), "on the arm's flange"
    if at == "top":
        root = spec.worldbody.first_body()
        return root, _top(spec, home_qpos), identity, f"on top of {root.name or 'its main body'}"
    body = spec.body(at)
    if body is None or body == spec.worldbody:
        bodies = [b.name for b in spec.bodies if b.name and b != spec.worldbody and "/" not in b.name]
        close = wording.near(at, bodies)
        hint = f"did you mean {', '.join(close)}? " if close else ""
        raise ValueError(
            f"robot {robot!r}, mount {name!r}: no body {at!r} to mount on; {hint}at is 'top', 'tool' or one of: "
            f"{wording.choices(bodies)}"
        )
    return body, np.zeros(3), identity, f"on {at}"


def _top(spec: mujoco.MjSpec, home_qpos: dict[str, float]) -> np.ndarray:
    """The middle of the main (root) body's top, in its frame: above the middle of its collision shapes, where rays
    dropped onto its shapes (the visible ones too, so nothing mounted sinks into a shell) meet the highest surface
    under a mount's footprint."""
    model = spec.compile()
    data = _posed(model, home_qpos)
    body = 1  # the first body of the world, the robot's root
    rot, origin = data.xmat[body].reshape(3, 3), data.xpos[body]
    shapes = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
    if not shapes:
        return np.zeros(3)
    boxes = {g: (_corners(model, data, g) - origin) @ rot for g in shapes}
    solid = [boxes[g] for g in shapes if collides(model, g)] or list(boxes.values())
    points = np.vstack(solid)
    middle = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
    above = float(np.vstack(list(boxes.values()))[:, 2].max()) + 0.1
    down = rot @ np.array([0.0, 0.0, -1.0])
    heights = []
    for dx, dy in FOOTPRINT:
        start = origin + rot @ np.array([middle[0] + dx, middle[1] + dy, above])
        hits = [_ray(model, data, g, start, down) for g in shapes]
        hits = [h for h in hits if h >= 0]
        if hits:
            heights.append(above - min(hits))
    top = max(heights) if heights else float(points[:, 2].max())
    return np.array([middle[0], middle[1], top])


def _ray(model: mujoco.MjModel, data: mujoco.MjData, g: int, start: np.ndarray, down: np.ndarray) -> float:
    """How far down from `start` the ray meets geom `g`, or -1."""
    if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        return float(mujoco.mj_rayMesh(model, data, g, start, down))
    return float(
        mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], start, down, model.geom_type[g])
    )


def _corners(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> np.ndarray:
    center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    rot = data.geom_xmat[g].reshape(3, 3)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return data.geom_xpos[g] + (center + signs * half) @ rot.T


# ---- a gripper on an arm -------------------------------------------------------------------------------------------


def _complete_the_arm(spec: mujoco.MjSpec, name: str, home_ctrl: dict[str, float]) -> None:
    """The gripper's grasp point and actuator become the arm's "tcp" and "gripper"; its bare flange is gone."""
    for site in ("tcp", "flange"):
        found = _site(spec, site)
        if found is not None:
            spec.delete(found)
    _site(spec, f"{name}/tcp").name = "tcp"
    actuator = next((a for a in spec.actuators if a.name == f"{name}/gripper"), None)
    if actuator is not None:
        actuator.name = "gripper"
        if f"{name}/gripper" in home_ctrl:
            home_ctrl["gripper"] = home_ctrl.pop(f"{name}/gripper")


def _site(spec: mujoco.MjSpec, name: str) -> mujoco.MjsSite | None:
    return next((s for s in spec.sites if s.name == name), None)


# ---- checks ----------------------------------------------------------------------------------------------------------


def _check_overlaps(spec: mujoco.MjSpec, home_qpos: dict[str, float], robot: str, names: list[str]) -> None:
    """Refuse mounts that sit inside their carrier (or each other) at the home pose: they would fly apart."""
    try:
        model = spec.compile()
    except ValueError as exc:
        raise ValueError(f"robot {robot!r}: MuJoCo could not compile it with its mounts: {exc}") from exc
    data = _posed(model, home_qpos)
    mujoco.mj_collision(model, data)

    def owner(geom: int) -> str:
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        prefix = body.split("/", 1)[0] if "/" in body else ""
        return prefix if prefix in names else robot

    worst: dict[tuple[str, str], float] = {}
    for c in data.contact[: data.ncon]:
        a, b = owner(int(c.geom[0])), owner(int(c.geom[1]))
        if a != b and c.dist < -OVERLAP_ERROR:
            pair = (min(a, b), max(a, b))
            worst[pair] = max(worst.get(pair, 0.0), -float(c.dist))
    # A mount and the body it is fixed to move as one (a weld) and never touch in the simulation, so the contacts above
    # miss a mount sunk into that body; it would look wrong. Measure those pairs directly.
    solid = [g for g in range(model.ngeom) if collides(model, g)]
    for g1 in solid:
        for g2 in solid:
            a, b = owner(g1), owner(g2)
            same_weld = model.body_weldid[model.geom_bodyid[g1]] == model.body_weldid[model.geom_bodyid[g2]]
            if g1 < g2 and a != b and same_weld:
                depth = -float(mujoco.mj_geomDistance(model, data, g1, g2, OVERLAP_ERROR * 4, None))
                if depth > OVERLAP_ERROR:
                    pair = (min(a, b), max(a, b))
                    worst[pair] = max(worst.get(pair, 0.0), depth)
    if worst:
        (a, b), depth = max(worst.items(), key=lambda kv: kv[1])
        if robot in (a, b):
            what = f"mount {b if a == robot else a!r} sits {depth * 1000:.0f} mm inside {robot}"
        else:
            what = f"mounts {a!r} and {b!r} overlap by {depth * 1000:.0f} mm"
        raise ValueError(
            f"robot {robot!r}: {what} at the home pose; move it with shift (e.g. [0, 0, {depth + 0.005:.3f}] lifts "
            "it) or mount it somewhere else with at"
        )


def _lowest(spec: mujoco.MjSpec, home_qpos: dict[str, float]) -> float:
    """The lowest point of a robot's collision shapes at home, its root where the spec has it."""
    model = spec.compile()
    data = _posed(model, home_qpos)
    geoms = [g for g in range(model.ngeom) if collides(model, g)]
    return min((float(lowest_point(model, data, g)[2]) for g in geoms), default=0.0)


def _posed(model: mujoco.MjModel, home_qpos: dict[str, float]) -> mujoco.MjData:
    data = mujoco.MjData(model)
    for joint, value in home_qpos.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_kinematics(model, data)
    return data


def _rotate(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, np.asarray(quat, float))
    return out


def _euler(rpy_deg: Sequence[float]) -> np.ndarray:
    from robotforge.design.builder.expand import euler

    return euler(list(rpy_deg))
