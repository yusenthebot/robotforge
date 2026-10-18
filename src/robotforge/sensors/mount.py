"""Put a robot's sensors on its bodies, in its own spec, before it joins the world.

Any robot's spec works the same way (a template, a library robot, an imported file, one built from parts):
the spec is compiled once on its own to see its bodies and shapes, each sensor is placed on the body it names
(the base when it names none), and its site, camera, MuJoCo sensors and drawn module are added there. Anchor
words find the body's real surface: "top" is where a ray coming down onto the body's middle first meets one
of its shapes, so a lidar stands on a Go2's back rather than floating at the corner of its bounding box.

The modules drawn round a sensor (a lidar puck, a board camera) weigh nothing and collide with nothing, so
adding a sensor never changes how the robot moves; rays pass through them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from robotforge import wording
from robotforge.design.builder import looks
from robotforge.design.visuals import DEFAULT_RGBA, VISUAL_GROUP, Visuals
from robotforge.sensors import declare
from robotforge.sensors.spec import SensorEntry, axis_vector, base_name, resolved, unique_name

S, OBJ = mujoco.mjtSensor, mujoco.mjtObj
RANGE_DIST = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)  # MuJoCo 3.15 rangefinders report just the distance
HIDDEN = [0.0, 0.0, 0.0, 0.0]
SITE_GROUP = 5  # never drawn: the viewer shows the modules, not the sites
CAMERA_FRONT = looks.CAMERA_PROUD + looks.CAMERA_BARREL * 1.5 + 0.0015  # m from the housing's seat to the eye
LOOKS_ALONG = 0.5  # |cos| between a module's view and its face's normal below which it looks along the face
FACES = {
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}
TOUCH_OUTSIDE = 0.01  # m a touch zone reaches out from the surface, so a contact a little off it still counts
GEOM_PRIMITIVES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_BOX),
}


@dataclass(frozen=True)
class Target:
    """What a sensor sits on: a body, the shapes whose surface its anchors are on, and a natural spot."""

    body: int
    geoms: tuple[int, ...]
    point: np.ndarray | None = None  # a site's or a joint's position in the body frame, used when `at` is left out
    child: bool = False  # force_torque `between`: the body is the child of the pair


@dataclass(frozen=True)
class Place:
    """Where a sensor is in its body's frame: the point, the face's outward normal (None inside or at numbers),
    its frame, whose columns are where it looks (x), its left (y) and its up (z), and the robot's own axes
    (forward, left, up as it stands at home) in that body's frame."""

    point: np.ndarray
    normal: np.ndarray | None
    frame: np.ndarray
    axes: np.ndarray

    @property
    def look(self) -> np.ndarray:
        return self.frame[:, 0]

    @property
    def up(self) -> np.ndarray:
        return self.frame[:, 2]


def mount(
    spec: mujoco.MjSpec, entries: Sequence[SensorEntry], robot: str, home: dict[str, float] | None = None
) -> None:
    """Add `entries` to the robot's spec. Words for places and directions (top, bottom, forward, down) mean the
    robot's own as it stands in its `home` pose (joint values by name), whatever the body's own axes are: the
    bottom of a Go2's foot is the side that meets the floor, though its calf's frame is tilted, and "down" from
    a Panda's hand is down. Raises ValueError naming the sensor when one cannot be placed."""
    if not entries:
        return
    try:
        model = spec.compile()
    except ValueError as exc:
        raise ValueError(
            f"robot {robot!r}: cannot place sensors, the robot does not compile on its own: {exc}"
        ) from exc
    data = mujoco.MjData(model)
    for joint, value in (home or {}).items():
        j = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, joint)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_kinematics(model, data)
    taken = declare.declared_names(spec) | {c.name for c in spec.cameras if c.name}
    for entry in entries:
        opts = resolved(entry)
        try:
            target = _target(model, entry, robot)
            if entry.type == "touch":  # contacts happen on the shapes that collide, not on the ones drawn over them
                target = Target(target.body, _colliding(model, target), target.point)
            name = entry.name or unique_name(base_name(entry.type, opts.get("mode"), _label(model, entry)), taken)
            if name in taken:
                raise ValueError(f"it already has a sensor or camera named {name!r}; give another name")
            taken.add(name)
            place = _place(model, data, target, entry, opts)
            body = _spec_body(spec, model, target.body)
            vis = Visuals(spec, owner=f"{name}_sensor", rays=False)
            definition = ADD[entry.type](spec, body, name, opts, place, vis, model, target)
            vis.finish()
        except ValueError as exc:
            what = entry.name or entry.type
            raise ValueError(f"robot {robot!r}, sensor {what!r}: {exc}") from None
        knobs = {k: opts.get(k) for k in ("rate_hz", "latency_s", "noise", "bias", "drift", "dropout", "seed")}
        body_name = mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, target.body) or ""
        declare.declare(
            spec,
            {
                "name": name,
                "type": entry.type,
                "body": body_name,
                "at": [round(float(v), 4) for v in place.point],
                **definition,
                **{k: v for k, v in knobs.items() if v is not None},
            },
        )


def _spec_body(spec: mujoco.MjSpec, model: mujoco.MjModel, body: int) -> mujoco.MjsBody:
    name = mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, body)
    found = spec.body(name) if name else None
    if found is None:
        raise ValueError(f"body {body} has no name to put a sensor on; name it in the robot's file")
    return found


def _label(model: mujoco.MjModel, entry: SensorEntry) -> str | None:
    """What a default name says the sensor is on ("fl" for a touch on Go2's FL foot); nothing for the base."""
    on = entry.between[1] if entry.between else entry.on
    if on is None:
        return None
    return on.rsplit("/", 1)[-1]


# ---- what a sensor sits on -----------------------------------------------------------------------------


def _names(model: mujoco.MjModel, kind: mujoco.mjtObj, count: int) -> list[str]:
    return [n for i in range(count) if (n := mujoco.mj_id2name(model, kind, i))]


def _target(model: mujoco.MjModel, entry: SensorEntry, robot: str) -> Target:
    if entry.between:
        return _between(model, entry.between, robot)
    if entry.on is None:
        if model.nbody < 2:
            raise ValueError("the robot has no bodies to put a sensor on")
        root = 1  # the robot's first body: its base
        return Target(root, _surface_geoms(model, root))
    on = entry.on
    finders = (
        (OBJ.mjOBJ_BODY, model.nbody),
        (OBJ.mjOBJ_GEOM, model.ngeom),
        (OBJ.mjOBJ_SITE, model.nsite),
        (OBJ.mjOBJ_JOINT, model.njnt),
    )
    for kind, count in finders:
        names = _names(model, kind, count)
        found = on if on in names else wording.match(on, names)
        if found is None:
            continue
        i = mujoco.mj_name2id(model, kind, found)
        if kind == OBJ.mjOBJ_BODY:
            if i == 0:
                raise ValueError("a sensor goes on one of the robot's bodies, not on the world")
            return Target(i, _surface_geoms(model, i))
        if kind == OBJ.mjOBJ_GEOM:
            return Target(int(model.geom_bodyid[i]), (i,))
        if kind == OBJ.mjOBJ_SITE:
            body = int(model.site_bodyid[i])
            return Target(body, _surface_geoms(model, body), np.array(model.site_pos[i], float))
        body = int(model.jnt_bodyid[i])
        return Target(body, _surface_geoms(model, body), np.array(model.jnt_pos[i], float))
    bodies = _names(model, OBJ.mjOBJ_BODY, model.nbody)
    text = wording.unknown("body", on, bodies, where=f"on {robot}")
    raise ValueError(f"{text} (a part, geom, site or joint name works too)")


def _between(model: mujoco.MjModel, pair: list[str], robot: str) -> Target:
    """The child of two neighboring bodies: MuJoCo measures the force between a body and the one it hangs from."""
    bodies = _names(model, OBJ.mjOBJ_BODY, model.nbody)
    ids = []
    for name in pair:
        found = name if name in bodies else wording.match(name, bodies)
        if found is None:
            raise ValueError(wording.unknown("body", name, bodies, where=f"on {robot}"))
        ids.append(mujoco.mj_name2id(model, OBJ.mjOBJ_BODY, found))
    a, b = ids
    if int(model.body_parentid[b]) == a:
        child = b
    elif int(model.body_parentid[a]) == b:
        child = a
    else:
        parent_a = mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, int(model.body_parentid[a])) or "the world"
        raise ValueError(
            f"{pair[0]} and {pair[1]} are not neighbors ({pair[0]} hangs from {parent_a}); a force-torque sensor "
            "measures between a body and the one it hangs from"
        )
    return Target(child, _surface_geoms(model, child), np.zeros(3), child=True)


def _colliding(model: mujoco.MjModel, target: Target) -> tuple[int, ...]:
    """The target's shapes that collide (a foot geom named itself stays the target)."""
    if len(target.geoms) == 1 and (model.geom_contype[target.geoms[0]] or model.geom_conaffinity[target.geoms[0]]):
        return target.geoms
    own = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == target.body]
    hits = tuple(g for g in own if model.geom_contype[g] or model.geom_conaffinity[g])
    return hits or target.geoms


def _surface_geoms(model: mujoco.MjModel, body: int) -> tuple[int, ...]:
    """The shapes a body shows (its visual meshes where it has them), else whatever shapes it has."""
    own = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]

    def seen(g: int) -> bool:
        mat = int(model.geom_matid[g])
        default = np.allclose(model.geom_rgba[g], DEFAULT_RGBA)  # drawn in its material's color then
        alpha = model.mat_rgba[mat][3] if mat >= 0 and default else model.geom_rgba[g][3]
        return int(model.geom_group[g]) <= VISUAL_GROUP and alpha > 0

    visible = [g for g in own if seen(g)]
    return tuple(visible or own)


# ---- where on it ---------------------------------------------------------------------------------------


def _place(
    model: mujoco.MjModel, data: mujoco.MjData, target: Target, entry: SensorEntry, opts: dict[str, Any]
) -> Place:
    """`at` as given; left out, a site's or joint's own spot, else the type's default anchor. Numbers are meters
    from the body's origin along the robot's axes."""
    root = int(model.body_rootid[target.body]) if model.body_rootid[target.body] else 1
    body_rot = data.xmat[target.body].reshape(3, 3)
    axes = body_rot.T @ data.xmat[root].reshape(3, 3)  # columns: the robot's forward, left, up in the body frame
    at = entry.at if entry.at is not None or target.point is not None else opts.get("at")
    low, high = _box(model, target)
    center = (low + high) / 2
    normal = None
    if isinstance(at, list):
        point = axes @ np.asarray(at, float)
    elif at is None and target.point is not None:
        point = target.point
    elif at == "origin" or (at is None and target.point is None):
        point = np.zeros(3)
    elif at == "center":
        point = center
    else:
        normal = axes @ np.asarray(FACES[at], float)
        point = _surface(model, data, target, center, normal, low, high)
    look = axes @ np.asarray(axis_vector(opts.get("direction") or "forward"), float)
    return Place(point, normal, _frame(look, axes[:, 2], axes[:, 0]), axes)


def _box(model: mujoco.MjModel, target: Target) -> tuple[np.ndarray, np.ndarray]:
    """The bounding box of the target's shapes, in its body's frame."""
    if not target.geoms:
        return np.zeros(3), np.zeros(3)
    pts = np.concatenate([_corners(model, g) for g in target.geoms])
    return pts.min(axis=0), pts.max(axis=0)


def _corners(model: mujoco.MjModel, g: int) -> np.ndarray:
    """The corners of a geom's bounding box in its body's frame (a ball's box is as wide as the ball)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    middle, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return model.geom_pos[g] + (middle + half * signs) @ rot.reshape(3, 3).T


def _surface(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: Target,
    center: np.ndarray,
    normal: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """The middle of the target's outermost surface along `normal` (body frame): the points of its shapes that
    reach furthest that way, and those within a few millimetres of them. The bottom of a block is the middle of
    its bottom face, the bottom of a tilted shin is its foot, the top of a Go2 is the crown of its back. The box's
    face when it has no shapes."""
    points = np.concatenate([_extremes(model, g, normal) for g in target.geoms]) if target.geoms else np.zeros((0, 3))
    if not len(points):
        reach = float(np.abs(normal) @ ((high - low) / 2))
        return center + normal * reach
    reach = points @ normal
    near = reach >= reach.max() - max(0.004, 0.02 * float(np.linalg.norm(high - low)))
    return points[near].mean(axis=0)


def _extremes(model: mujoco.MjModel, g: int, normal: np.ndarray) -> np.ndarray:
    """Points of a geom's surface (body frame) among which are the furthest along `normal`: a mesh's vertices, a
    box's corners, the rims of a cylinder's ends, the point of a ball or a capsule's end that leads."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    rot = rot.reshape(3, 3)
    pos = np.array(model.geom_pos[g], float)
    size = model.geom_size[g]
    local = rot.T @ normal  # the direction in the geom's own frame
    kind = int(model.geom_type[g])
    G = mujoco.mjtGeom
    if kind == int(G.mjGEOM_MESH):
        mesh = int(model.geom_dataid[g])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        return pos + model.mesh_vert[start : start + count].astype(float) @ rot.T
    if kind == int(G.mjGEOM_BOX):
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
        return pos + corners @ rot.T
    if kind == int(G.mjGEOM_SPHERE):
        return (pos + normal * size[0])[None]
    if kind == int(G.mjGEOM_CAPSULE):
        ends = np.array([[0, 0, size[1]], [0, 0, -size[1]]]) @ rot.T
        return pos + ends + normal * size[0]
    if kind == int(G.mjGEOM_CYLINDER):
        angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        rim = np.stack([np.cos(angles) * size[0], np.sin(angles) * size[0], np.zeros(16)], axis=-1)
        sides = [rim + [0, 0, size[1]], rim - [0, 0, size[1]]]
        return pos + np.concatenate(sides) @ rot.T
    if kind == int(G.mjGEOM_ELLIPSOID):
        lead = size**2 * local / max(float(np.sqrt(np.sum(size**2 * local**2))), 1e-9)
        return (pos + rot @ lead)[None]
    return np.zeros((0, 3))  # planes and height fields: no surface of a robot's own


def _frame(look: np.ndarray, up: np.ndarray, front: np.ndarray | None = None) -> np.ndarray:
    """Columns: where it looks, its left and its up (the robot's up made square to the look; looking straight
    down or up, the robot's front is the top of its picture)."""
    x = look / np.linalg.norm(look)
    z = up - x * float(up @ x)
    if np.linalg.norm(z) < 1e-6:
        ahead = front if front is not None else np.cross(up, [0.0, 1.0, 0.0])
        z = ahead if float(x @ up) < 0 else -ahead
    z = z / np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def _quat(frame: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(frame).ravel())
    return [float(v) for v in q]


def _site(body: mujoco.MjsBody, name: str, point: np.ndarray, frame: np.ndarray, **kwargs: Any) -> str:
    body.add_site(
        name=name,
        pos=[float(v) for v in point],
        quat=_quat(frame),
        size=kwargs.pop("size", [0.004, 0, 0]),
        rgba=HIDDEN,
        group=SITE_GROUP,
        **kwargs,
    )
    return name


def _on_face(place: Place) -> bool:
    return place.normal is not None


def _looks_along(place: Place) -> bool:
    return place.normal is not None and abs(float(place.look @ place.normal)) < LOOKS_ALONG


# ---- one adder per type: the model objects, the module drawn, and what the runtime needs ------------------


def _camera(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    look, up = place.look, place.up
    seat = place.point
    if _looks_along(place):  # a camera on top looking forward stands on the top and sees over it
        seat = seat + place.normal * looks.CAMERA_HALF
    eye = seat + look * CAMERA_FRONT if _on_face(place) else seat
    if _on_face(place):
        looks.camera(vis, body, seat, look, up)
    left = place.frame[:, 1]
    body.add_camera(name=name, pos=[float(v) for v in eye], xyaxes=[*(-left).tolist(), *up.tolist()], fovy=opts["fov"])
    keys = ("mode", "width", "height", "fov", "range")
    return {"camera": name, **{k: opts[k] for k in keys}}


def _lidar(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    up = place.up
    half = looks.LIDAR_HEIGHT / 2
    if place.normal is not None and float(place.normal @ up) > 0.7:
        foot = place.point  # stands on the face
    elif place.normal is not None:
        foot = place.point + place.normal * looks.LIDAR_RADIUS - up * half  # beside it
    else:
        foot = place.point - up * half  # its middle where it was put
    origin = foot + up * half
    if _on_face(place):
        looks.lidar(vis, body, foot, up)
    site = _site(body, f"rf_{name}", origin, place.frame)
    keys = ("channels", "rays", "vfov", "hfov", "range")
    return {"site": site, **{k: opts[k] for k in keys}}


def _rangefinder(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    look, up = place.look, place.up
    frame = looks.rangefinder_frame(look, up)
    board = place.point
    if _looks_along(place):
        board = board + place.normal * looks.RANGE_HALF * looks.RANGE_ASPECT
    if _on_face(place):
        looks.rangefinder(vis, body, board, frame)
    beam = np.column_stack([place.frame[:, 1], place.frame[:, 2], look])  # MuJoCo's rangefinder looks along +z
    site = _site(body, f"rf_{name}", board + look * 0.004, beam)
    ray = spec.add_sensor(
        name=site, type=S.mjSENS_RANGEFINDER, objtype=OBJ.mjOBJ_SITE, objname=site, cutoff=opts["range"]
    )
    ray.intprm[0] = RANGE_DIST
    return {"site": site, "native": {"range": site}, "range": opts["range"]}


def _imu(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    site = _site(body, f"rf_{name}", place.point, place.frame)
    native = {}
    for role, kind in (("accel", S.mjSENS_ACCELEROMETER), ("gyro", S.mjSENS_GYRO), ("quat", S.mjSENS_FRAMEQUAT)):
        native[role] = f"{site}_{role}"
        spec.add_sensor(name=native[role], type=kind, objtype=OBJ.mjOBJ_SITE, objname=site)
    if place.normal is not None:
        looks.imu(vis, body, place.point, place.normal, None)
    return {"site": site, "native": native}


def _encoders(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    moving = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    joints = [n for j in range(model.njnt) if int(model.jnt_type[j]) in moving and (n := model.joint(j).name)]
    wanted = opts.get("joints")
    if wanted:
        chosen = []
        for given in wanted:
            found = given if given in joints else wording.match(given, joints)
            if found is None:
                raise ValueError(wording.unknown("joint", given, joints, where="on it"))
            chosen.append(found)
        joints = chosen
    if not joints:
        raise ValueError("it has no hinge or sliding joints for encoders to read")
    return {"joints": joints, "resolution": opts["resolution"]}


def _force_torque(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    if int(model.body_parentid[target.body]) == 0 and not model.body_jntnum[target.body]:
        raise ValueError("it is bolted to the world; put it on a body that hangs from another (a wrist, an ankle)")
    site = _site(body, f"rf_{name}", place.point, place.frame)
    native = {"force": f"{site}_force", "torque": f"{site}_torque"}
    spec.add_sensor(name=native["force"], type=S.mjSENS_FORCE, objtype=OBJ.mjOBJ_SITE, objname=site)
    spec.add_sensor(name=native["torque"], type=S.mjSENS_TORQUE, objtype=OBJ.mjOBJ_SITE, objname=site)
    parent = mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, int(model.body_parentid[target.body])) or "world"
    return {"site": site, "native": native, "parent": parent}


def _touch(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    """A pad of rows x cols zones on a face (the bottom of a foot by default). A zone counts the contacts of its
    body whose points fall inside it, so each zone is a box reaching a little out of the surface and some way
    into the shape: a ball foot rolling on the floor touches it off its lowest point too."""
    axes = place.axes
    normal = place.normal if place.normal is not None else -axes[:, 2]
    low, high = _box(model, target)

    def across(v: np.ndarray) -> np.ndarray:
        v = v - normal * float(v @ normal)
        return v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else v

    t1 = across(axes[:, 0])  # the robot's forward along the pad, or its up on a front or back face
    if np.linalg.norm(t1) < 1e-6:
        t1 = across(axes[:, 2])
    t2 = np.cross(normal, t1)

    # The pad is as big as the shapes it lies on: those that reach the face (a foot, not the whole shin).
    tips = [g for g in target.geoms if len(_extremes(model, g, normal))]
    reach = {g: float((_extremes(model, g, normal) @ normal).max()) for g in tips}
    top = max(reach.values(), default=0.0)
    leading = [g for g in tips if reach[g] >= top - 0.004] or list(target.geoms)
    corners = np.concatenate([_corners(model, g) for g in leading]) if leading else np.zeros((1, 3))

    def width(t: np.ndarray) -> float:
        along = corners @ t
        return float(along.max() - along.min())

    # A little wider than the shape, so a contact at its very edge (a block's corner) is on the pad.
    size = opts.get("size") or [width(t1) + 2 * TOUCH_OUTSIDE, width(t2) + 2 * TOUCH_OUTSIDE]
    depth = max(0.6 * width(normal) / 2, 0.01)
    rows, cols = int(opts["rows"]), int(opts["cols"])
    cell = [size[0] / rows / 2, size[1] / cols / 2, (TOUCH_OUTSIDE + depth) / 2]
    frame = np.column_stack([t1, t2, normal])
    middle = place.point + normal * (TOUCH_OUTSIDE - depth) / 2
    # Rows run from the pad's front edge (its top edge on a front or back face), columns from its left edge.
    left = across(axes[:, 1])
    if np.linalg.norm(left) < 1e-6:
        left = -t2
    cells = []
    for r in range(rows):
        for c in range(cols):
            u = size[0] / 2 - (r + 0.5) / rows * size[0]
            v = size[1] / 2 - (c + 0.5) / cols * size[1]
            site = f"rf_{name}_{r * cols + c}" if rows * cols > 1 else f"rf_{name}"
            _site(body, site, middle + t1 * u + left * v, frame, type=mujoco.mjtGeom.mjGEOM_BOX, size=cell)
            spec.add_sensor(name=site, type=S.mjSENS_TOUCH, objtype=OBJ.mjOBJ_SITE, objname=site)
            cells.append(site)
    return {"cells": cells, "rows": rows, "cols": cols, "size": [round(s, 4) for s in size]}


def _gps(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    return {"site": _site(body, f"rf_{name}", place.point, place.frame)}


def _odometry(spec, body, name, opts, place: Place, vis: Visuals, model, target) -> dict[str, Any]:
    return {"site": _site(body, f"rf_{name}", place.point, place.frame)}


ADD = {
    "camera": _camera,
    "lidar": _lidar,
    "lidar3d": _lidar,
    "rangefinder": _rangefinder,
    "imu": _imu,
    "encoders": _encoders,
    "force_torque": _force_torque,
    "touch": _touch,
    "gps": _gps,
    "odometry": _odometry,
}
