"""Robots carrying robots: an arm on a robot dog's back, a gripper on an arm's flange.

A robot entry in world.yaml can list `mounts`: whole robots from the library or the templates, each fixed to one
of the carrier's bodies, the way a part's `robot` is in the builder (see builder/assemble.py). The mounted robot
loses its free joint and rides along; its names get its mount name as a prefix, so an arm called "arm" on a Go2
named "go2" is "go2/arm" in the world, and move_arm and pick_and_place work it as they do an arm on a rover.

Where it goes (`at`):
- "top" (the default): on the carrier's top as you see it, on a flat spot near its centre of mass (see SPOTS): a dog's
  back, a humanoid's head (its main body, the pelvis, lies inside its torso). A camera, an arm on a dog's back.
- "tool": an arm's flange (the "flange" site library.Tool puts on arms without a gripper): the carrier's own, or else
  that of the one arm mounted on it before ("<mount>/tool" names one). A gripper mounted there becomes the arm's own:
  its grasp point and actuator are the arm's "tcp" and "gripper", so a UR5e with a Robotiq 2F-85 works like a Franka
  Panda, on the floor or on Spot's back. This is the default for a gripper where there is such a flange.
- the name of any body of the carrier, as in its own file: the mount goes on that body's origin.
`shift` moves it from there (m, in that frame) and `rotate` turns it ([roll, pitch, yaw] in degrees, as in the builder).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mujoco
import numpy as np

from robotforge import wording
from robotforge.design.templates.base import BuiltRobot
from robotforge.sim.geometry import collides, lowest_point, ray, visible

OVERLAP_ERROR = 0.004  # m: deeper than this, the mount and its carrier are inside each other (as the builder's check)
OPTIONS = ("timestep", "integrator", "cone", "impratio")  # physics options a mounted robot takes from its carrier
# Where a mount on top stands: on the carrier's top near its centre of mass (the gait carries it best there), on the
# flattest spot among these offsets (m, in the main body's frame, nearest first). A mount's base is a disk (see
# _footprint) and rests on the highest point under it, so it never cuts into the carrier, and on a flat spot it does
# not hover either. Right at the centre of mass a Z1 perched on the payload knob of Spot's back, 13 mm over the back
# itself, and a PiPER on a 27 mm bump on ANYmal's. Spots whose middle height is more than TOP_DROP (m) off that of the
# spot nearest the centre of mass are not the top any more: a humanoid's shoulders beside its head, ANYmal's lidar.
SPOTS = sorted(
    ((dx, dy) for dx in np.arange(-0.08, 0.081, 0.02) for dy in (-0.02, 0.0, 0.02)), key=lambda o: math.hypot(*o)
)
TOP_DROP = 0.03
# Where rays look under a base of radius 1: its middle and two rings, close enough to catch a 2 cm handle.
UNDER_BASE = [(0.0, 0.0)] + [
    (r * math.cos(a), r * math.sin(a)) for r in (0.5, 1.0) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
]
FOOTPRINT_RANGE = (0.015, 0.06)  # m: the radius of the disk a mount's base stands on
FLAT = 0.002  # m: spots this much flatter than another are no flatter
# A mount sunk into its carrier is refused with the shift that lifts it clear, found by trying: lifted by how deep it
# sits, then again, at most this many times and this high (m). Lifting by the depth alone left a G1's PiPER, pressed
# into the torso from below, stuck 117 mm deep whatever it was told.
LIFT_TRIES = 6
LIFT_MAX = 0.15
PAYLOAD_SLACK = 0.05  # kg: the catalog's masses are rounded; the arm a limit was measured with stays within it


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


@dataclass
class _Placed:
    """A mount as placed: the frame it hangs from, the way it is lifted (that frame's z, in the frame's body) and the
    shift it was given, for the overlap check's hint."""

    name: str
    body: str  # the carrier's body it hangs from
    frame: mujoco.MjsFrame
    up: np.ndarray
    shift: list[float]
    radius: float  # of its base (see _footprint)


def attach(carrier: BuiltRobot, mounts: Sequence[Mount], robot: str, model: str | None = None) -> BuiltRobot:
    """`carrier` with every mount attached, its home pose extended, and checked: a mount that overlaps the carrier is
    refused with the overlap in millimetres and the shift that lifts it clear, and a floating carrier rises if a mount
    hangs below its feet. `model` names the carrier in the library, whose Payload says how much it carries."""
    if not mounts:
        return carrier
    spec = carrier.spec
    home_qpos, home_ctrl = dict(carrier.home_qpos), dict(carrier.home_ctrl)
    notes = list(carrier.notes)
    placed: list[_Placed] = []
    carried: list[tuple[str, float]] = []  # (what, kg)
    before = _lowest(spec, home_qpos) if carrier.floating else 0.0
    for mount in mounts:
        name = default_name(mount)
        if name in (p.name for p in placed):
            raise ValueError(f"robot {robot!r}: two mounts are called {name!r}; give each a name")
        try:
            child = build_child(mount)
        except ValueError as exc:
            raise ValueError(f"robot {robot!r}, mount {name!r}: cannot build {what(mount)}: {exc}") from exc
        radius, mass = _footprint(child)
        carried.append((what(mount), mass))
        ride_on(child.spec, spec)
        at = mount.at or _default_at(spec, mount, robot, name, [p.name for p in placed])
        body, pos, quat, where = _place(spec, home_qpos, at, robot, name, [p.name for p in placed], radius)
        rot = np.zeros(4)
        mujoco.mju_mat2Quat(rot, _euler(mount.rotate).ravel())
        turned = np.zeros(4)
        mujoco.mju_mulQuat(turned, quat, rot)
        offset = _rotate(quat, np.asarray(mount.shift, float))
        if at == "top":  # its base stands on that surface, as it would on the floor (an SO-100's reaches below 0)
            offset += _rotate(turned, np.array([0.0, 0.0, child.spawn_height]))
        frame = body.add_frame(pos=(pos + offset).tolist(), quat=turned.tolist())
        spec.attach(child.spec, prefix=f"{name}/", frame=frame)
        up = _rotate(quat, np.array([0.0, 0.0, 1.0]))
        placed.append(_Placed(name, body.name, frame, up, list(mount.shift), radius))
        home_qpos.update({f"{name}/{k}": v for k, v in child.home_qpos.items()})
        home_ctrl.update({f"{name}/{k}": v for k, v in child.home_ctrl.items()})
        arm = _flange_owner(at, spec, [p.name for p in placed])
        if arm is not None and _site(spec, f"{name}/tcp") is not None:
            _complete_the_arm(spec, name, arm, home_ctrl)
            whose = f"{robot}/{arm.rstrip('/')}'s" if arm else "the arm's"
            notes.append(f"{name}: {what(mount)} on {whose} flange; it is now {whose} gripper")
        else:
            if is_gripper(mount) and _site(spec, f"{name}/tcp") is not None:
                _rename_site(spec, f"{name}/tcp", f"{name}/grasp")  # no arm moves it: not an arm to work
            notes.append(f"{name}: {what(mount)} mounted {where}")
        notes += [f"{name}: {n}" for n in child.notes]
    notes += _check_payload(robot, model, carried)
    _check_overlaps(spec, home_qpos, robot, placed)
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


def _default_at(spec: mujoco.MjSpec, mount: Mount, robot: str, name: str, mounted: list[str]) -> str:
    """ "tool" for a gripper where there is a bare flange (see _flange), else "top". A gripper anywhere else would be a
    dead weight on a dog's back or on top of another gripper's arm, so that needs an `at` of its own."""
    if not is_gripper(mount):
        return "top"
    flanges = _flanges(spec, mounted)
    if len(flanges) == 1:
        return "tool"
    if flanges:
        tools = " or ".join(f"'{arm}tool'" for arm in flanges)
        raise ValueError(f"robot {robot!r}, mount {name!r}: {robot} has {len(flanges)} bare flanges; give at {tools}")
    gripper = mount.library or "robotiq_2f85"
    raise ValueError(
        f"robot {robot!r}, mount {name!r}: {what(mount)} goes on the bare flange of an arm without a gripper, such "
        f"as a ur5e, and {robot} has none; carry one first and the gripper after it, e.g. "
        f'mounts=[{{"library": "ur5e"}}, {{"library": "{gripper}"}}], or give at'
    )


def _flanges(spec: mujoco.MjSpec, mounted: list[str]) -> dict[str, mujoco.MjsSite]:
    """The bare flanges a gripper can go on, by the name prefix of their arm: "" for the robot's own (an arm without a
    gripper), "arm/" for that of the arm mounted as "arm". The robot's own comes first and alone."""
    own = _site(spec, "flange")
    if own is not None:
        return {"": own}
    return {f"{m}/": site for m in mounted if (site := _site(spec, f"{m}/flange")) is not None}


def _flange_owner(at: str, spec: mujoco.MjSpec, mounted: list[str]) -> str | None:
    """For a mount at "tool" or "<mount>/tool": the name prefix of the arm whose flange that is (see _flanges)."""
    if at == "tool":
        return next(iter(_flanges(spec, mounted)), None)
    arm, slash, rest = at.partition("/")
    return f"{arm}/" if slash and rest == "tool" and arm in mounted else None


def _place(
    spec: mujoco.MjSpec,
    home_qpos: dict[str, float],
    at: str,
    robot: str,
    name: str,
    mounted: list[str],
    radius: float,
) -> tuple[mujoco.MjsBody, np.ndarray, np.ndarray, str]:
    """(body, position, orientation in it, words for the notes) of a mount point. `mounted` names the mounts already
    on the robot: a second one does not stand on the first. `radius` is the mount's base's (see _footprint)."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    if at == "tool" or at.endswith("/tool"):
        flanges = _flanges(spec, mounted)
        arm = _flange_owner(at, spec, mounted)
        if arm is None or arm not in flanges:
            there = "on it" if at == "tool" else f"on {at.removesuffix('/tool')!r}"
            known = " or ".join(f"'{a}tool'" for a in flanges)
            raise ValueError(
                f"robot {robot!r}, mount {name!r}: at {at!r} needs an arm with a bare flange {there} (a library arm "
                f"without a gripper, such as ur5e); give at {known or 'a body name'} or 'top'"
            )
        if at == "tool" and len(flanges) > 1:
            raise ValueError(
                f"robot {robot!r}, mount {name!r}: {robot} has {len(flanges)} bare flanges; give at "
                + " or ".join(f"'{a}tool'" for a in flanges)
            )
        flange = flanges[arm]
        whose = f"{robot}/{arm.rstrip('/')}'s" if arm else "the arm's"
        return flange.parent, np.array(flange.pos, float), np.array(flange.quat, float), f"on {whose} flange"
    if at == "top":
        return _on_top(spec, home_qpos, mounted, radius)
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


def _on_top(
    spec: mujoco.MjSpec, home_qpos: dict[str, float], mounted: list[str], radius: float
) -> tuple[mujoco.MjsBody, np.ndarray, np.ndarray, str]:
    """Where a mount with a base of this radius stands on top of the carrier (see SPOTS), found with rays dropped onto
    what it looks like (see _surface). That is on the main body for a dog, and on the head of a humanoid, whose main
    body (the pelvis) lies inside its torso. The mount stands upright as the main body does."""
    model = spec.compile()
    data = _posed(model, home_qpos)
    root = 1  # the first body of the world, the robot's root
    rot, origin = data.xmat[root].reshape(3, 3), data.xpos[root]
    own = [b for b in range(1, model.nbody) if not _mounted_body(model, b, mounted)]
    shapes = _surface(model, own)
    mass = model.body_mass[own]
    com = (mass @ data.xipos[own]) / mass.sum() if shapes and mass.sum() > 0 else origin
    middle = (rot.T @ (com - origin))[:2]
    above = max((float(((_corners(model, data, g) - origin) @ rot)[:, 2].max()) for g in shapes), default=0.0) + 0.1
    down = rot @ np.array([0.0, 0.0, -1.0])

    def height(x: float, y: float) -> tuple[float, int] | None:
        start = origin + rot @ np.array([x, y, above])
        hits = [(h, g) for g in shapes if (h := ray(model, data, g, start, down)) >= 0]
        return (above - min(hits)[0], min(hits)[1]) if hits else None

    spots = []  # (the base's height there, the median height under it, offset, the geom at the top)
    for dx, dy in SPOTS:
        x, y = middle[0] + dx, middle[1] + dy
        under = [h for cx, cy in UNDER_BASE if (h := height(x + radius * cx, y + radius * cy)) is not None]
        # On the carrier: its middle and at least half of it. A base wider than its carrier hangs over the sides.
        if height(x, y) is not None and 2 * len(under) >= len(UNDER_BASE):
            rest = max(under)
            spots.append((rest[0], float(np.median([h for h, _ in under])), (x, y), rest[1]))
    if not spots:  # nothing under its middle near the centre of mass (a ring, say): on the main body's origin
        return spec.body(model.body(root).name), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), "on its main body"
    level = spots[0][1]
    on_top = [s for s in spots if abs(s[1] - level) <= TOP_DROP]
    flattest = min(rest - median for rest, median, *_ in on_top)
    rest, _, (x, y), geom = next(s for s in on_top if s[0] - s[1] <= flattest + FLAT)  # the nearest of the flattest
    body = int(model.geom_bodyid[geom])
    world = origin + rot @ np.array([x, y, rest])
    body_rot = data.xmat[body].reshape(3, 3)
    pos = body_rot.T @ (world - data.xpos[body])
    quat, inverse = np.zeros(4), np.zeros(4)  # the root's orientation in the body's frame
    mujoco.mju_negQuat(inverse, data.xquat[body])
    mujoco.mju_mulQuat(quat, inverse, data.xquat[root])
    name = model.body(body).name
    return spec.body(name), pos, quat, f"on top of {name or 'its main body'}"


def _footprint(child: BuiltRobot) -> tuple[float, float]:
    """The radius (m) of the disk a robot's base stands on: how far its fixed shapes reach from its origin, sideways,
    in the narrower direction (see FOOTPRINT_RANGE); and the robot's mass (kg)."""
    model = child.spec.compile()
    mass = float(model.body_subtreemass[0])
    data = _posed(model, child.home_qpos)
    root = 1
    welded = [b for b in range(1, model.nbody) if model.body_weldid[b] == model.body_weldid[root]]
    shapes = _surface(model, welded)
    if not shapes:
        return FOOTPRINT_RANGE[0], mass
    rot = data.xmat[root].reshape(3, 3)
    points = np.vstack([(_corners(model, data, g) - data.xpos[root]) @ rot for g in shapes])
    reach = min(-points[:, 0].min(), points[:, 0].max(), -points[:, 1].min(), points[:, 1].max())
    return float(np.clip(reach, *FOOTPRINT_RANGE)), mass


def carried_mass(model: mujoco.MjModel, robot: str, mounts: Sequence[Mount]) -> float:
    """The mass (kg) of the robots mounted on `robot` in a compiled world."""
    prefixes = tuple(f"{robot}/{default_name(m)}/" for m in mounts)
    return float(sum(model.body_mass[b] for b in range(model.nbody) if model.body(b).name.startswith(prefixes)))


def loaded_speed(model: str | None, kg: float) -> float | None:
    """The top speed (m/s) of a library robot carrying `kg` more than it walks with at full speed (see
    library.Payload), or None."""
    from robotforge.design import library

    payload = library.get(model).payload if model is not None else None
    return payload.slow if payload is not None and kg > payload.full + PAYLOAD_SLACK else None


def _check_payload(robot: str, model: str | None, carried: list[tuple[str, float]]) -> list[str]:
    """Refuse mounts heavier than a library carrier carries (see library.Payload), naming lighter arms that it does;
    a note when it carries them at a lower speed."""
    from robotforge.design import library

    payload = library.get(model).payload if model is not None else None
    total = sum(kg for _, kg in carried)
    if payload is None or total <= payload.full + PAYLOAD_SLACK:
        return []
    title = library.get(model).title
    if total > payload.most + PAYLOAD_SLACK:
        load = (
            f"{carried[0][0]} weighs {total:.1f} kg"
            if len(carried) == 1
            else f"{' and '.join(w for w, _ in carried)} weigh {total:.1f} kg together"
        )
        lighter = sorted(  # one-armed: ALOHA is a two-armed station, not something to carry
            (
                r
                for r in library.CATALOG.values()
                if r.kind == "arm" and len(r.tools) == 1 and r.mass <= payload.most + PAYLOAD_SLACK
            ),
            key=lambda r: -r.mass,
        )
        hint = (
            f": try {', '.join(f'{r.name} ({r.mass:g} kg)' for r in lighter[:3])}"
            if lighter
            else "; no arm in the library is that light"
        )
        how = f" ({payload.describe()})" if payload.most > payload.full else ""
        raise ValueError(f"robot {robot!r}: {load}, and a {title} carries at most {payload.most:g} kg{how}{hint}")
    return [
        f"carrying {total:.1f} kg, more than the {payload.full:g} kg it walks with at full speed, {robot} walks at "
        f"{payload.slow:g} m/s at most"
    ]


def _mounted_body(model: mujoco.MjModel, body: int, mounted: list[str]) -> bool:
    return any(model.body(body).name.startswith(f"{m}/") for m in mounted)


def _surface(model: mujoco.MjModel, bodies: Sequence[int]) -> list[int]:
    """The shapes that make up how these bodies look: their visible geoms, or the colliding ones of a body that shows
    none. Spot's collision box rises 9 mm over its visible shell, and an arm resting on the box hovered over it."""
    out = []
    for body in bodies:
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
        shown = [g for g in geoms if visible(model, g)]
        out += shown or [g for g in geoms if collides(model, g)]
    return out


def _corners(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> np.ndarray:
    center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    rot = data.geom_xmat[g].reshape(3, 3)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return data.geom_xpos[g] + (center + signs * half) @ rot.T


# ---- a gripper on an arm -------------------------------------------------------------------------------------------


def _complete_the_arm(spec: mujoco.MjSpec, name: str, arm: str, home_ctrl: dict[str, float]) -> None:
    """The gripper's grasp point and actuator become the arm's "tcp" and "gripper" ("arm/tcp" for an arm mounted as
    "arm", see _flanges); its bare flange is gone."""
    for site in (f"{arm}tcp", f"{arm}flange"):
        found = _site(spec, site)
        if found is not None:
            spec.delete(found)
    _rename_site(spec, f"{name}/tcp", f"{arm}tcp")
    actuator = next((a for a in spec.actuators if a.name == f"{name}/gripper"), None)
    if actuator is not None:
        actuator.name = f"{arm}gripper"
        if f"{name}/gripper" in home_ctrl:
            home_ctrl[f"{arm}gripper"] = home_ctrl.pop(f"{name}/gripper")


def _site(spec: mujoco.MjSpec, name: str) -> mujoco.MjsSite | None:
    return next((s for s in spec.sites if s.name == name), None)


def _rename_site(spec: mujoco.MjSpec, old: str, new: str) -> None:
    """Rename a site and the sensors that name it: our gripper template measures its "tcp" with a sensor, and a UR5e
    given that gripper failed to compile ("unrecognized name 'gripper/tcp' of sensorized object")."""
    _site(spec, old).name = new
    for sensor in spec.sensors:
        if sensor.objtype == mujoco.mjtObj.mjOBJ_SITE and sensor.objname == old:
            sensor.objname = new
        if sensor.reftype == mujoco.mjtObj.mjOBJ_SITE and sensor.refname == old:
            sensor.refname = new


# ---- checks ----------------------------------------------------------------------------------------------------------


def _check_overlaps(spec: mujoco.MjSpec, home_qpos: dict[str, float], robot: str, placed: list[_Placed]) -> None:
    """Refuse mounts that sit inside their carrier (or each other) at the home pose: they would fly apart, or, where
    they are welded together, look wrong. The refusal says how deep, and which shift lifts the mount clear."""
    names = [p.name for p in placed]
    worst = _overlaps(spec, home_qpos, robot, placed)
    if not worst:
        return
    (a, b), (depth, into) = max(worst.items(), key=lambda kv: kv[1][0])
    if robot in (a, b):
        mount = b if a == robot else a
        what = f"mount {mount!r} sits {depth * 1000:.0f} mm inside {robot} ({into})"
    else:  # two mounts: the later one was put where the first already is
        mount = max(a, b, key=names.index)
        what = f"mounts {a!r} and {b!r} overlap by {depth * 1000:.0f} mm"
    moved = next(p for p in placed if p.name == mount)
    lift = _lift(spec, home_qpos, robot, placed, moved, depth)
    if lift is None:
        raise ValueError(
            f"robot {robot!r}: {what} at the home pose, and lifting it does not get it clear; mount it somewhere else "
            "with at (a body name, or 'top')"
        )
    shift = [round(v, 3) for v in moved.shift[:2]] + [math.ceil((moved.shift[2] + lift) * 1000) / 1000]
    raise ValueError(
        f"robot {robot!r}: {what} at the home pose; shift {shift} lifts it clear, or mount it somewhere else with at"
    )


def _overlaps(
    spec: mujoco.MjSpec, home_qpos: dict[str, float], robot: str, placed: list[_Placed]
) -> dict[tuple[str, str], tuple[float, str]]:
    """The deepest overlap between each two of the robot and its mounts at the home pose, deeper than OVERLAP_ERROR:
    {(one, other): (depth in m, the body it runs into)}."""
    try:
        model = spec.compile()
    except ValueError as exc:
        raise ValueError(f"robot {robot!r}: MuJoCo could not compile it with its mounts: {exc}") from exc
    data = _posed(model, home_qpos)
    mujoco.mj_collision(model, data)
    names = [p.name for p in placed]

    def owner(body: int) -> str:
        prefix = model.body(body).name.split("/", 1)[0] if "/" in model.body(body).name else ""
        return prefix if prefix in names else robot

    worst: dict[tuple[str, str], tuple[float, str]] = {}

    def note(a: str, b: str, depth: float, into: int) -> None:
        pair = (min(a, b), max(a, b))
        if a != b and depth > max(OVERLAP_ERROR, worst.get(pair, (0.0, ""))[0]):
            worst[pair] = (depth, model.body(into).name.split("/")[-1])

    def first(a: str, b: str) -> bool:  # a was there before b: the carrier, or a mount listed earlier
        return a == robot or (b != robot and names.index(a) < names.index(b))

    for c in data.contact[: data.ncon]:
        b1, b2 = (int(model.geom_bodyid[g]) for g in c.geom)
        a, b = owner(b1), owner(b2)
        note(a, b, -float(c.dist), b1 if first(a, b) else b2)
    # A mount and the body it is fixed to move as one (a weld) and never touch in the simulation, so the contacts above
    # miss a mount sunk into that body; there only the looks matter. Convex hulls of what is drawn would not do: Go2's
    # shell dips under its hull in the middle, and a PiPER resting on it read as 17 mm deep. See _sunk.
    for i, mount in enumerate(placed):
        weld = int(model.body_weldid[model.body(mount.body).id])
        welded = [b for b in range(1, model.nbody) if int(model.body_weldid[b]) == weld]
        mine = _surface(model, [b for b in welded if owner(b) == mount.name])
        body = model.body(mount.body).id
        rot = data.xmat[body].reshape(3, 3)
        origin, up = data.xpos[body] + rot @ np.array(mount.frame.pos, float), rot @ mount.up
        for other in [robot, *names[:i]]:
            theirs = _surface(model, [b for b in welded if owner(b) == other])
            depth, geom = _sunk(model, data, mine, theirs, origin, up, mount.radius)
            if geom >= 0:
                note(mount.name, other, depth, int(model.geom_bodyid[geom]))
    return worst


def _sunk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mine: list[int],
    theirs: list[int],
    origin: np.ndarray,
    up: np.ndarray,
    radius: float,
) -> tuple[float, int]:
    """How far a mount's shapes (`mine`) must rise along `up` to clear the shapes they are welded to (`theirs`), and the
    geom at the top of those. On lines parallel to `up` through the mount's base (see _footprint and UNDER_BASE),
    wherever the stretch the mount fills overlaps the stretch the others fill, its bottom must reach their
    top. (0, -1) when nothing overlaps."""
    if not mine or not theirs:
        return 0.0, -1
    across = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    across /= np.linalg.norm(across)
    sideways = np.cross(up, across)
    need, top = 0.0, -1
    for cx, cy in UNDER_BASE:
        point = origin + radius * (cx * across + cy * sideways)
        own, others = _span(model, data, mine, point, up), _span(model, data, theirs, point, up)
        if own is None or others is None:
            continue
        (low, high, _), (their_low, their_high, geom) = own, others
        if low < their_high and high > their_low and their_high - low > need:
            need, top = their_high - low, geom
    return need, top


def _span(
    model: mujoco.MjModel, data: mujoco.MjData, geoms: list[int], point: np.ndarray, up: np.ndarray
) -> tuple[float, float, int] | None:
    """Where the line through `point` along `up` enters and leaves these shapes (m from `point`), and the geom it
    leaves through at the top; None if it misses them."""
    far = 2.0  # m, from well outside any robot
    downward = [(d, g) for g in geoms if (d := ray(model, data, g, point + far * up, -up)) >= 0]
    upward = [d for g in geoms if (d := ray(model, data, g, point - far * up, up)) >= 0]
    if not downward or not upward:
        return None
    distance, geom = min(downward)
    return min(upward) - far, far - distance, geom


def _lift(
    spec: mujoco.MjSpec, home_qpos: dict[str, float], robot: str, placed: list[_Placed], mount: _Placed, depth: float
) -> float | None:
    """How far up (m, along the mount's shift z) a mount must go to sit clear, found by trying (see LIFT_TRIES); None
    when lifting does not get it clear: it runs into something above it, or it is deeper than LIFT_MAX."""
    start = np.array(mount.frame.pos, float)
    lift = 0.0
    try:
        for _ in range(LIFT_TRIES):
            lift += depth
            if lift > LIFT_MAX:
                return None
            mount.frame.pos = (start + mount.up * lift).tolist()
            overlaps = _overlaps(spec, home_qpos, robot, placed)
            left = max((d for pair, (d, _) in overlaps.items() if mount.name in pair), default=0.0)
            if left <= OVERLAP_ERROR:
                return lift
            if left >= depth - 1e-4:  # no better for it
                return None
            depth = left
        return None
    finally:
        mount.frame.pos = start.tolist()


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
