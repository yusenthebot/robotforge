"""Build a MuJoCo model from a UsdPhysics stage (rigid bodies, colliders, joints, drives, materials).

The stage is read with usd-core, so Isaac Sim is never needed. What PhysX would simulate is mapped
onto MuJoCo as closely as the two engines allow:

- every prim with RigidBodyAPI becomes a body; joints between bodies become the kinematic tree
  (revolute -> hinge, prismatic -> slide, fixed -> a welded child body), and a body with no joint gets
  a free joint; joint values keep the USD zero through MuJoCo's joint `ref`;
- colliders keep their shape: cubes, spheres, cylinders and capsules map one to one, and collision
  meshes become convex pieces (see `hulls`), because MuJoCo collides meshes as convex hulls;
- drives (DriveAPI) are kept as data and applied by `world.World` with PhysX's implicit formula;
- physics materials set friction, `disableGravity` turns into gravity compensation, filtered pairs
  and articulations without self-collision turn into contact excludes.

A fixed joint named in `Options.planar_anchors` (the robot's base anchor) is not welded: its child
gets x, y and yaw joints that `World` drives to the anchor frame each step, as PhysX drags the base.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from robotforge.zenobench import hulls

log = logging.getLogger(__name__)

COLLISION_GROUP = 3
PATCH_RADIUS = 0.015  # m: typical contact patch radius, sets torsional friction
VISUAL_GROUP = 1
_TINY = 5e-4  # m: half thickness given to flat convex pieces so qhull accepts them


@dataclass(frozen=True)
class Options:
    planar_anchors: tuple[str, ...] = ()
    no_collision_bodies: tuple[str, ...] = ()  # body name suffixes whose colliders are dropped
    decompose: bool = True
    visuals: bool = True
    textures: bool = True
    max_hull_vertices: int = 0  # cap for moving bodies' hulls (0: no cap)
    capsule_pads: tuple[str, ...] = ()  # body name suffixes whose box colliders become capsule rows


@dataclass
class Drive:
    stiffness: float
    damping: float
    max_force: float
    target: float  # rad or m
    acceleration: bool


@dataclass
class JointInfo:
    name: str  # MuJoCo joint name (the USD joint prim path)
    prim: str
    kind: str  # "hinge" or "slide"
    child: str  # body name
    drive: Drive | None
    order: int  # creation order (breadth first), Isaac's DOF order inside one articulation
    max_velocity: float = math.inf  # rad/s or m/s (physxJoint:maxJointVelocity)


@dataclass
class Planar:
    """A base held by a moving fixed joint: its pose is origin + (x, y, yaw) joint values."""

    anchor: str  # the fixed joint prim
    body: str
    joints: tuple[str, str, str]  # x, y, yaw joint names
    origin: tuple[float, float, float]  # x, y, yaw (rad) of the body's model frame


@dataclass
class Conversion:
    spec: mujoco.MjSpec
    bodies: set[str] = field(default_factory=set)
    joints: dict[str, JointInfo] = field(default_factory=dict)  # by USD joint prim path
    planar: dict[str, Planar] = field(default_factory=dict)  # by anchor prim path
    articulations: dict[str, list[str]] = field(default_factory=dict)  # root body -> member bodies
    textures: int = 0
    stats: dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ transforms (USD row vectors)
def _mat(m) -> np.ndarray:
    return np.array(m, dtype=np.float64).reshape(4, 4)


def _rigid(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rotation with column vectors, translation, scale) of a row-vector 4x4 matrix."""
    lin = m[:3, :3]
    scale = np.linalg.norm(lin, axis=1)
    scale = np.where(scale < 1e-12, 1.0, scale)
    rows = lin / scale[:, None]
    if np.linalg.det(rows) < 0:
        scale[0] = -scale[0]
        rows[0] = -rows[0]
    u, _, vt = np.linalg.svd(rows)  # re-orthonormalize (sheared or rounded matrices)
    return (u @ vt).T, m[3, :3].copy(), scale


def _quat(r: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r).reshape(9))
    return q


def _gf_quat(q) -> np.ndarray:
    im = q.GetImaginary()
    return np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64)


def _rotation_from_z(axis: np.ndarray) -> np.ndarray:
    """A rotation (column vectors) whose z axis is `axis`."""
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def _quat_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, np.float64) / np.linalg.norm(q))
    return m.reshape(3, 3)


def _apply(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    return points @ m[:3, :3] + m[3, :3]


class _Frames:
    def __init__(self, stage):
        from pxr import Usd, UsdGeom

        self._cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def world(self, prim) -> np.ndarray:
        return _mat(self._cache.GetLocalToWorldTransform(prim))


# ------------------------------------------------------------------ attribute helpers
def _get(prim, name, default=None):
    a = prim.GetAttribute(name)
    if a and a.HasValue():
        v = a.Get()
        return default if v is None else v
    return default


def _enabled(prim, name) -> bool:
    return bool(_get(prim, name, True))


def _triangles(mesh) -> tuple[np.ndarray, np.ndarray]:
    pts, faces, _, _ = _triangles_with_polygons(mesh)
    return pts, faces


def _triangles_with_polygons(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fan-triangulated faces; for each triangle the USD polygon it came from (for GeomSubsets) and its
    three face-vertex corners (positions in faceVertexIndices, for faceVarying primvars such as UVs)."""
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    empty = np.zeros((0, 3), np.int64)
    if not pts or not counts or not idx:
        return np.zeros((0, 3)), empty, np.zeros(0, np.int64), empty
    pts = np.asarray(pts, np.float64)
    counts = np.asarray(counts, np.int64)
    idx = np.asarray(idx, np.int64)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    corners, polys = [], []
    for k in np.unique(counts):
        if k < 3:
            continue
        which = np.flatnonzero(counts == k)
        s = starts[which]
        for j in range(1, k - 1):
            corners.append(np.stack([s, s + j, s + j + 1], axis=1))
            polys.append(which)
    if not corners:
        return pts, empty, np.zeros(0, np.int64), empty
    corners = np.concatenate(corners)
    return pts, idx[corners], np.concatenate(polys), corners


# ------------------------------------------------------------------ the converter
class _Builder:
    def __init__(self, stage, options: Options):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        self.Usd, self.UsdGeom, self.UsdPhysics, self.UsdShade = Usd, UsdGeom, UsdPhysics, UsdShade
        self.stage = stage
        self.opt = options
        self.frames = _Frames(stage)
        self.spec = mujoco.MjSpec()
        self.out = Conversion(self.spec)
        self.mbody: dict[str | None, object] = {None: self.spec.worldbody}
        self.body_rigid: dict[str | None, tuple[np.ndarray, np.ndarray]] = {None: (np.eye(3), np.zeros(3))}
        self.moving: set[str] = set()
        self._piece_volume: dict[str, float] = {}
        self._mesh_id = 0
        self._rgba_cache: dict[str, list[float]] = {}
        self._materials: dict[str, str | None] = {}
        self._textures: dict[str, str | None] = {}
        self._tex_cache: dict[str, list[float]] = {}

    # ---------------------------------------------------------- scan
    def scan(self):
        Usd, UsdGeom, UsdPhysics = self.Usd, self.UsdGeom, self.UsdPhysics
        self.bodies: dict[str, object] = {}
        self.joints: list = []
        self.colliders: list = []
        self.visuals: list = []
        self.filtered: list = []
        self.roots: list = []
        it = Usd.PrimRange.Stage(self.stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
        for prim in it:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and _enabled(prim, "physics:rigidBodyEnabled"):
                self.bodies[str(prim.GetPath())] = prim
            if prim.IsA(UsdPhysics.Joint):
                if _enabled(prim, "physics:jointEnabled"):
                    self.joints.append(prim)
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    self.roots.append(prim)
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                self.roots.append(prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and _enabled(prim, "physics:collisionEnabled"):
                self.colliders.append(prim)
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                self.filtered.append(prim)
            if self.opt.visuals and prim.IsA(UsdGeom.Gprim):
                img = UsdGeom.Imageable(prim)
                if img.ComputePurpose() in ("default", "render") and img.ComputeVisibility() != "invisible":
                    self.visuals.append(prim)
        self.out.bodies = set(self.bodies)

    def owner(self, prim) -> str | None:
        path = prim.GetPath()
        while path and str(path) != "/":
            s = str(path)
            if s in self.bodies:
                return s
            path = path.GetParentPath()
        return None

    def _target_body(self, rel) -> tuple[str | None, object | None]:
        targets = rel.GetTargets() if rel else []
        if not targets:
            return None, None
        prim = self.stage.GetPrimAtPath(targets[0])
        if not prim:
            return None, None
        s = str(targets[0])
        if s in self.bodies:
            return s, prim
        return self.owner(prim), prim

    # ---------------------------------------------------------- tree
    def build_tree(self):
        UsdPhysics = self.UsdPhysics
        edges: dict[str | None, list] = {}
        info = []
        for j in self.joints:
            jt = UsdPhysics.Joint(j)
            b0, p0 = self._target_body(jt.GetBody0Rel())
            b1, p1 = self._target_body(jt.GetBody1Rel())
            if b0 == b1:
                continue
            rec = {"prim": j, "b0": b0, "b1": b1, "p0": p0, "p1": p1}
            info.append(rec)
            edges.setdefault(b0, []).append(rec)
            edges.setdefault(b1, []).append(rec)
        placed: set[str | None] = {None}
        order = 0
        queue: deque[str | None] = deque([None])
        loops = 0

        def grow(queue):
            nonlocal order, loops
            while queue:
                parent = queue.popleft()
                for rec in edges.get(parent, []):
                    child = rec["b1"] if rec["b0"] == parent else rec["b0"]
                    if child in placed:
                        if rec.get("used") is None and rec["b0"] in placed and rec["b1"] in placed:
                            rec["used"] = "loop"
                            loops += 1
                        continue
                    rec["used"] = "tree"
                    flipped = rec["b0"] != parent
                    self._place(child, parent, rec, flipped, order)
                    order += 1
                    placed.add(child)
                    queue.append(child)

        grow(queue)
        for path in self.bodies:
            if path not in placed:  # free body (and whatever hangs on it)
                self._place(path, None, None, False, order)
                placed.add(path)
                grow(deque([path]))
        if loops:
            log.warning("%d joint(s) close kinematic loops and are ignored", loops)
        self.out.stats["loops"] = loops
        self.joint_recs = info

    def _place(self, child: str, parent: str | None, rec, flipped: bool, order: int):
        UsdPhysics = self.UsdPhysics
        prim = self.bodies[child]
        r_c, t_c, _ = _rigid(self.frames.world(prim))
        r_p, t_p = self.body_rigid[parent]
        jprim = rec["prim"] if rec else None
        jpath = str(jprim.GetPath()) if jprim else None
        planar = jpath in self.opt.planar_anchors
        if planar:  # an upright base at the anchor height, keeping its authored x, y and yaw
            z = float(np.asarray(_get(jprim, "physics:localPos0", (0, 0, t_c[2])), float)[2])
            yaw0 = math.atan2(r_c[1, 0], r_c[0, 0])
            c, s = math.cos(yaw0), math.sin(yaw0)
            r_c, t_c = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), np.array([t_c[0], t_c[1], z])
        body = self.mbody[parent].add_body()
        body.name = child
        body.pos = r_p.T @ (t_c - t_p)
        body.quat = _quat(r_p.T @ r_c)
        self.mbody[child] = body
        self.body_rigid[child] = (r_c, t_c)
        if parent in self.moving or planar or rec is None or not jprim.IsA(UsdPhysics.FixedJoint):
            self.moving.add(child)
        if _get(prim, "physxRigidBody:disableGravity", False):
            body.gravcomp = 1.0
        self._mass(body, prim)
        if planar:
            names = (f"{jpath}:x", f"{jpath}:y", f"{jpath}:yaw")
            # slides come first, so their axes stay world aligned while the hinge turns the body
            for name, typ, axis in (
                (names[0], mujoco.mjtJoint.mjJNT_SLIDE, r_c.T @ (1, 0, 0)),
                (names[1], mujoco.mjtJoint.mjJNT_SLIDE, r_c.T @ (0, 1, 0)),
                (names[2], mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1)),
            ):
                j = body.add_joint()
                j.name, j.type, j.axis, j.armature = name, typ, axis, 1.0
            self.out.planar[jpath] = Planar(jpath, child, names, (float(t_c[0]), float(t_c[1]), yaw0))
            return
        if rec is None:
            body.add_freejoint()
            return
        if jprim.IsA(UsdPhysics.FixedJoint):
            return
        kind = None
        if jprim.IsA(UsdPhysics.RevoluteJoint):
            kind = "hinge"
        elif jprim.IsA(UsdPhysics.PrismaticJoint):
            kind = "slide"
        if kind is None:
            log.warning("joint %s (%s) is kept rigid", jpath, jprim.GetTypeName())
            return
        # joint frames in the world (USD: localPos/Rot in each body's frame, scale included)
        f0 = self._joint_frame(rec["p0"], jprim, 0)
        f1 = self._joint_frame(rec["p1"], jprim, 1)
        axis_local = {"X": (1.0, 0, 0), "Y": (0, 1.0, 0), "Z": (0, 0, 1.0)}[_get(jprim, "physics:axis", "X")]
        e = np.asarray(axis_local)
        (r0, p0), (r1, p1) = f0, f1
        if kind == "hinge":
            q = _quat(r0.T @ r1)
            value = 2 * math.atan2(float(np.dot(q[1:], e)), float(q[0]))
            value = (value + math.pi) % (2 * math.pi) - math.pi
        else:
            value = float(np.dot(r0.T @ (p1 - p0), e))
        child_frame_pos, child_axis_world = (p1, r1 @ e) if not flipped else (p0, r0 @ e)
        if flipped:
            child_axis_world = -child_axis_world
        j = body.add_joint()
        j.name = jpath
        j.type = mujoco.mjtJoint.mjJNT_HINGE if kind == "hinge" else mujoco.mjtJoint.mjJNT_SLIDE
        j.pos = r_c.T @ (child_frame_pos - t_c)
        j.axis = r_c.T @ child_axis_world
        j.ref = value
        lo, hi = _get(jprim, "physics:lowerLimit"), _get(jprim, "physics:upperLimit")
        if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
            k = math.pi / 180 if kind == "hinge" else 1.0
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.range = (lo * k, hi * k)
        # a little rotor inertia steadies stiff drives; on a 48 g finger 0.05 doubled the drive force,
        # which scales with the joint inertia (PhysX acceleration drives)
        j.armature = 0.01 if kind == "hinge" else 0.005
        drive = self._drive(jprim, "angular" if kind == "hinge" else "linear", kind)
        vmax = float(_get(jprim, "physxJoint:maxJointVelocity", math.inf))  # deg/s for a hinge, m/s for a slide
        vmax = vmax * math.pi / 180 if kind == "hinge" else vmax
        self.out.joints[jpath] = JointInfo(jpath, jpath, kind, child, drive, order, vmax)

    def _joint_frame(self, body_prim, jprim, side: int):
        pos = np.asarray(_get(jprim, f"physics:localPos{side}", (0, 0, 0)), float)
        rq = _get(jprim, f"physics:localRot{side}")
        rot = _quat_mat(_gf_quat(rq)) if rq is not None else np.eye(3)
        if body_prim is None:
            return rot, pos
        m = self.frames.world(body_prim)
        r, _, _ = _rigid(m)
        return r @ rot, _apply(pos[None, :], m)[0]

    def _drive(self, jprim, which: str, kind: str) -> Drive | None:
        if not jprim.HasAPI(self.UsdPhysics.DriveAPI, which):
            return None
        pre = f"drive:{which}:physics:"
        k = math.pi / 180 if kind == "hinge" else 1.0
        stiff = float(_get(jprim, pre + "stiffness", 0.0)) / (k if kind == "hinge" else 1.0)
        damp = float(_get(jprim, pre + "damping", 0.0)) / (k if kind == "hinge" else 1.0)
        # USD angular gains are per degree; per radian they are 180/pi larger
        return Drive(
            stiffness=stiff,
            damping=damp,
            max_force=float(_get(jprim, pre + "maxForce", math.inf)),
            target=float(_get(jprim, pre + "targetPosition", 0.0)) * k,
            acceleration=_get(jprim, pre + "type", "force") == "acceleration",
        )

    def _mass(self, body, prim):
        if not prim.HasAPI(self.UsdPhysics.MassAPI):
            return
        mass = float(_get(prim, "physics:mass", 0.0))
        inertia = _get(prim, "physics:diagonalInertia")
        com = _get(prim, "physics:centerOfMass")
        if mass > 0 and inertia is not None and min(inertia) > 0 and com is not None and min(com) > -1e30:
            axes = _get(prim, "physics:principalAxes")
            _, _, scale = _rigid(self.frames.world(prim))
            body.explicitinertial = True
            body.mass = mass
            body.ipos = np.asarray(com, float) * np.abs(scale)
            body.inertia = np.asarray(inertia, float)
            body.iquat = _gf_quat(axes) if axes is not None else (1, 0, 0, 0)
        elif mass > 0:
            body.userdata = [mass]  # spread over the collision geoms by volume in finish()

    # ---------------------------------------------------------- geoms
    def add_colliders(self):
        """Collider prims, and the meshes under an Xform that carries CollisionAPI (PhysX cooks those
        with the Xform's approximation, unless a nearer prim on the way has its own CollisionAPI)."""
        UsdGeom, UsdPhysics = self.UsdGeom, self.UsdPhysics
        jobs = []
        for prim in self.colliders:
            owner = self.owner(prim)
            if owner is not None and owner.endswith(self.opt.no_collision_bodies or ("\0",)):
                continue
            material = self._physics_material(prim)
            if prim.IsA(UsdGeom.Gprim):
                jobs.append((prim, owner, _get(prim, "physics:approximation", "none"), material))
                continue
            approx = _get(prim, "physics:approximation", "convexHull")
            for m in self.Usd.PrimRange(prim, self.Usd.TraverseInstanceProxies()):
                if m == prim or not m.IsA(UsdGeom.Mesh) or m.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if self.owner(m) != owner or self._nearer_collider(m, prim):
                    continue
                jobs.append((m, owner, approx, material))
        meshes = {}
        for prim, owner, approx, _ in jobs:
            if prim.IsA(UsdGeom.Mesh):
                pts, faces = _triangles(UsdGeom.Mesh(prim))
                if len(pts) >= 4:
                    decompose = self.opt.decompose and approx in _DECOMPOSE
                    static = owner not in self.moving
                    meshes[prim] = (_apply(pts, self._rel(prim, owner)), faces, decompose, static)
        _warm_cache([(p, f, st) for p, f, dec, st in meshes.values() if dec])
        n_geoms = 0
        for prim, owner, approx, material in jobs:
            if prim.IsA(UsdGeom.Mesh):
                if prim not in meshes:
                    continue
                pts, faces, decompose, static = meshes[prim]
                if approx in ("boundingCube", "boundingSphere"):
                    pieces = [hulls.box_corners(pts)]
                else:
                    pieces = hulls.convex_pieces(pts, faces, decompose=decompose, static=static)
                parent = self.mbody.get(owner, self.spec.worldbody)
                for i, piece in enumerate(pieces):
                    self._hull_geom(parent, _thicken(piece), material, owner in self.moving, f"{prim.GetPath()}#{i}")
                n_geoms += len(pieces)
            else:
                n_geoms += self._collider(prim, owner, approx, material)
        self.out.stats["collision_geoms"] = n_geoms

    def _nearer_collider(self, mesh, top) -> bool:
        p = mesh.GetParent()
        while p and p != top:
            if p.HasAPI(self.UsdPhysics.CollisionAPI):
                return True
            p = p.GetParent()
        return False

    def _rel(self, prim, owner):
        """The prim's matrix (scale included) in its body's frame, row-vector convention."""
        m = self.frames.world(prim)
        r_b, t_b = self.body_rigid.get(owner, self.body_rigid[None])
        inv = np.eye(4)
        inv[:3, :3] = r_b  # row-vector inverse of the rigid body transform
        inv[3, :3] = -t_b @ r_b
        return m @ inv

    def _collider(self, prim, owner, approx, material) -> int:
        """A primitive collider (cube, sphere, cylinder, capsule)."""
        UsdGeom = self.UsdGeom
        parent = self.mbody.get(owner, self.spec.worldbody)
        rel = self._rel(prim, owner)
        dynamic = owner in self.moving
        if not any(prim.IsA(t) for t in (UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Cylinder, UsdGeom.Capsule)):
            log.debug("collider %s (%s) is not supported", prim.GetPath(), prim.GetTypeName())
            return 0
        r, t, s = _rigid(rel)
        if prim.IsA(UsdGeom.Cube) and owner and owner.endswith(self.opt.capsule_pads):
            half = np.maximum(float(_get(prim, "size", 2.0)) / 2 * np.abs(s), 1e-4)
            return self._capsule_pad(parent, str(prim.GetPath()), r, t, half, material, dynamic)
        g = parent.add_geom()
        g.name = str(prim.GetPath())
        g.pos, g.quat = t, _quat(r)
        s = np.abs(s)
        if prim.IsA(UsdGeom.Cube):
            half = float(_get(prim, "size", 2.0)) / 2
            g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, np.maximum(half * s, 1e-4)
        elif prim.IsA(UsdGeom.Sphere):
            g.type, g.size = mujoco.mjtGeom.mjGEOM_SPHERE, (float(_get(prim, "radius", 1.0)) * float(s.max()), 0, 0)
        elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            cap = prim.IsA(UsdGeom.Capsule)
            axis = _get(prim, "axis", "Z")
            i = "XYZ".index(axis)
            radius = float(_get(prim, "radius", 0.5 if cap else 1.0)) * float(max(s[(i + 1) % 3], s[(i + 2) % 3]))
            height = float(_get(prim, "height", 1.0 if cap else 2.0)) * float(s[i])
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE if cap else mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = (radius, height / 2, 0)
            if axis != "Z":
                turn = {
                    "X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
                    "Y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
                }[axis]
                g.quat = _quat(r @ turn.T)
        self._collision_style(g, material, dynamic)
        return 1

    def _capsule_pad(self, parent, name, r, t, half, material, dynamic) -> int:
        """A finger pad box as parallel capsules along its long side, filling its width: the same contact
        face, but smooth. A flat box face on a faceted hull switches its contact points abruptly as the
        object turns, and the impulse flung a pinched bottle out of the fingers."""
        order = np.argsort(half)  # thin, middle, long
        thin, wide, long = order
        radius = float(half[thin])
        length = max(float(half[long]) - radius, 1e-4)
        count = max(1, int(round(half[wide] / radius)))
        offsets = np.linspace(-(half[wide] - radius), half[wide] - radius, count) if count > 1 else [0.0]
        z = np.zeros(3)
        z[long] = 1.0
        axis_world = r @ z
        rot = _rotation_from_z(axis_world)
        for i, off in enumerate(offsets):
            shift = np.zeros(3)
            shift[wide] = off
            g = parent.add_geom()
            g.name = f"{name}#c{i}"
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.size = (radius, length, 0)
            g.pos, g.quat = t + r @ shift, _quat(rot)
            self._collision_style(g, material, dynamic)
        return len(offsets)

    def _hull_geom(self, parent, piece, material, dynamic, name):
        mesh = self.spec.add_mesh()
        mesh.name = f"hull{self._mesh_id}"
        self._mesh_id += 1
        mesh.uservert = np.asarray(piece, np.float32).reshape(-1)
        if dynamic and self.opt.max_hull_vertices:
            mesh.maxhullvert = self.opt.max_hull_vertices
        self._piece_volume[mesh.name] = hulls.hull_volume(piece)
        g = parent.add_geom()
        g.name = name
        g.type, g.meshname = mujoco.mjtGeom.mjGEOM_MESH, mesh.name
        self._collision_style(g, material, dynamic)

    def _collision_style(self, g, material, dynamic):
        g.group = COLLISION_GROUP
        g.rgba = (0.8, 0.3, 0.3, 0.4)
        friction = material[0] if material else 0.5
        # MuJoCo's torsional friction is a length: torque = coefficient x normal force. PhysX contacts
        # between flat faces are patches, which resist twisting by about friction x patch radius; a can
        # pinched by 55 mm finger pads pivoted out of the hand with MuJoCo's default 0.005.
        g.friction = (friction, friction * PATCH_RADIUS, 0.0002)
        g.condim = 4 if dynamic else 3

    def _physics_material(self, prim):
        mat, _ = self.UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if not mat:
            return None
        p = mat.GetPrim()
        s = float(_get(p, "physics:staticFriction", 0.5))
        d = float(_get(p, "physics:dynamicFriction", s))
        return (s, d)

    def add_visuals(self):
        UsdGeom = self.UsdGeom
        n = 0
        for prim in self.visuals:
            owner = self.owner(prim)
            parent = self.mbody.get(owner, self.spec.worldbody)
            rel = self._rel(prim, owner)
            rgba = self._rgba(prim)
            if prim.IsA(UsdGeom.Mesh):
                n += self._visual_mesh(prim, parent, rel, rgba)
                continue
            elif prim.IsA(UsdGeom.Cube):
                r, t, s = _rigid(rel)
                g = parent.add_geom()
                g.type, g.pos, g.quat = mujoco.mjtGeom.mjGEOM_BOX, t, _quat(r)
                g.size = np.maximum(float(_get(prim, "size", 2.0)) / 2 * np.abs(s), 1e-4)
            elif prim.IsA(UsdGeom.Sphere):
                r, t, s = _rigid(rel)
                g = parent.add_geom()
                g.type, g.pos = mujoco.mjtGeom.mjGEOM_SPHERE, t
                g.size = (float(_get(prim, "radius", 1.0)) * float(np.abs(s).max()), 0, 0)
            else:
                continue
            g.group, g.contype, g.conaffinity, g.rgba = VISUAL_GROUP, 0, 0, rgba
            g.mass = 0
            n += 1
        self.out.stats["visual_geoms"] = n

    def _visual_mesh(self, prim, parent, rel, rgba) -> int:
        """One render geom per material: GeomSubsets (Infinigen binds materials per face set) and the
        rest of the mesh with the mesh's own material. Textured materials keep their UVs."""
        UsdGeom = self.UsdGeom
        pts, faces, polys, corners = _triangles_with_polygons(UsdGeom.Mesh(prim))
        if len(faces) == 0:
            return 0
        pts = _apply(pts, rel)
        uv = self._corner_uvs(prim, faces, corners)
        groups = []
        rest = np.ones(len(faces), bool)
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
            if subset.GetElementTypeAttr().Get() != "face":
                continue
            indices = np.asarray(subset.GetIndicesAttr().Get() or [], np.int64)
            mask = np.isin(polys, indices) & rest
            if not mask.any():
                continue
            rest &= ~mask
            groups.append((mask, self._rgba(subset.GetPrim(), fallback=rgba), subset.GetPrim()))
        if rest.any():
            groups.append((rest, rgba, prim))
        n = 0
        for mask, color, bound in groups:
            used, local = np.unique(faces[mask], return_inverse=True)
            v, f, keep = simplify(pts[used], local.reshape(-1, 3))
            if len(f) == 0:
                continue
            mesh = self.spec.add_mesh()
            mesh.name = f"vis{self._mesh_id}"
            self._mesh_id += 1
            mesh.uservert = v.astype(np.float32).reshape(-1)
            mesh.userface = f.astype(np.int32).reshape(-1)
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            g = parent.add_geom()
            g.type, g.meshname = mujoco.mjtGeom.mjGEOM_MESH, mesh.name
            g.group, g.contype, g.conaffinity, g.mass = VISUAL_GROUP, 0, 0, 0
            material = self._texture_material(bound) if uv is not None and self.opt.textures else None
            if material is not None:
                mesh.usertexcoord = uv[mask][keep].astype(np.float32).reshape(-1)
                mesh.userfacetexcoord = np.arange(len(f) * 3, dtype=np.int32)
                g.material = material
            else:
                g.rgba = color
            n += 1
        return n

    def _corner_uvs(self, prim, faces: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
        """(triangles, 3, 2) texture coordinates from the `st` primvar, or None."""
        pv = self.UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if not pv or not pv.HasValue():
            return None
        values = np.asarray(pv.Get(), np.float64)
        if pv.IsIndexed():
            values = values[np.asarray(pv.GetIndices(), np.int64)]
        interp = pv.GetInterpolation()
        if interp == "faceVarying" and len(values) > int(corners.max()):
            return values[corners]
        if interp in ("vertex", "varying") and len(values) > int(faces.max()):
            return values[faces]
        return None

    def _texture_material(self, prim) -> str | None:
        """A MuJoCo material with the prim's diffuse texture (downscaled), or None without one."""
        mat, _ = self.UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not mat:
            return None
        key = str(mat.GetPath())
        if key in self._materials:
            return self._materials[key]
        color, texture, _ = self._shader_info(mat)
        name = None
        tex = self._texture(texture) if texture else None
        if tex is not None:
            name = f"mat{len(self._materials)}"
            m = self.spec.add_material()
            m.name = name
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
            m.rgba = (*(color or (1.0, 1.0, 1.0)), 1.0)
            m.specular, m.shininess = 0.2, 0.2
        self._materials[key] = name
        return name

    def _texture(self, path: str) -> str | None:
        if path in self._textures:
            return self._textures[path]
        name = None
        try:
            from PIL import Image

            with Image.open(Path(path)) as im:
                im = im.convert("RGB")
                if max(im.size) > TEXTURE_SIZE:
                    im = im.resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.Resampling.LANCZOS)
                width, height = im.size
                data = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes()  # UV v=0 is the bottom row
            t = self.spec.add_texture()
            name = t.name = f"tex{len(self._textures)}"
            t.type = mujoco.mjtTexture.mjTEXTURE_2D
            t.width, t.height, t.nchannel = width, height, 3
            t.data = data
            self.out.textures += 1
        except (OSError, ValueError) as exc:
            log.debug("texture %s unreadable: %s", path, exc)
        self._textures[path] = name
        return name

    def _rgba(self, prim, fallback: list[float] | None = None) -> list[float]:
        UsdShade = self.UsdShade
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        key = str(mat.GetPath()) if mat else None
        if key and key in self._rgba_cache:
            return self._rgba_cache[key]
        rgba = None
        if mat:
            rgba = self._shader_rgba(mat)
        if rgba is None and fallback is not None:
            return fallback
        if rgba is None:
            dc = _get(prim, "primvars:displayColor")
            rgba = [float(dc[0][0]), float(dc[0][1]), float(dc[0][2]), 1.0] if dc else [0.7, 0.7, 0.7, 1.0]
        if key:
            self._rgba_cache[key] = rgba
        return rgba

    def _shader_info(self, mat) -> tuple[list[float] | None, str | None, float]:
        """(constant diffuse color, diffuse texture path, opacity) of a UsdPreviewSurface or OmniPBR."""
        UsdShade = self.UsdShade
        color, texture, opacity = None, None, 1.0
        for sp in self.Usd.PrimRange(mat.GetPrim()):
            if not sp.IsA(UsdShade.Shader):
                continue
            for inp in UsdShade.Shader(sp).GetInputs():
                name = inp.GetBaseName()
                v = inp.Get()
                if name in ("diffuseColor", "diffuse_color_constant", "base_color") and v is not None and color is None:
                    if not inp.HasConnectedSource():
                        color = [float(v[0]), float(v[1]), float(v[2])]
                elif name in ("file", "diffuse_texture", "albedo_texture") and v is not None and texture is None:
                    p = getattr(v, "resolvedPath", "") or ""
                    if p and "normal" not in p.lower() and "rough" not in p.lower():
                        texture = p
                elif name == "opacity" and v is not None and not inp.HasConnectedSource():
                    opacity = float(v)
        return color, texture, opacity

    def _shader_rgba(self, mat) -> list[float] | None:
        color, texture, opacity = self._shader_info(mat)
        if texture:
            tex = self._texture_mean(texture)
            if tex is not None:
                color = [tex[i] * (color[i] if color else 1.0) for i in range(3)]
        if color is None:
            return None
        return [*color, max(0.05, min(1.0, opacity))]

    def _texture_mean(self, path: str) -> list[float] | None:
        if path in self._tex_cache:
            return self._tex_cache[path]
        try:
            from PIL import Image

            with Image.open(Path(path)) as im:
                small = np.asarray(im.convert("RGB").resize((16, 16)), np.float64) / 255.0
            lin = small**2.2
            mean = (lin.reshape(-1, 3).mean(0)) ** (1 / 2.2)
            val = [float(x) for x in mean]
        except (OSError, ValueError) as exc:
            log.debug("texture %s unreadable: %s", path, exc)
            val = None
        self._tex_cache[path] = val
        self.out.textures += 1
        return val

    # ---------------------------------------------------------- collision filtering
    def add_excludes(self):
        pairs: set[tuple[str, str]] = set()
        for prim in self.filtered:
            a = self.owner(prim)
            for t in self.UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                tp = self.stage.GetPrimAtPath(t)
                b = self.owner(tp) if tp else None
                if a != b and a is not None and b is not None:
                    pairs.add(tuple(sorted((a, b))))
        # articulations (connected through joints) that switch self collision off
        adj: dict[str, set[str]] = {}
        for rec in self.joint_recs:
            if rec["b0"] and rec["b1"]:
                adj.setdefault(rec["b0"], set()).add(rec["b1"])
                adj.setdefault(rec["b1"], set()).add(rec["b0"])
        for root in self.roots:
            if root.IsA(self.UsdPhysics.Joint):
                start, _ = self._target_body(self.UsdPhysics.Joint(root).GetBody1Rel())
                if start is None:
                    start, _ = self._target_body(self.UsdPhysics.Joint(root).GetBody0Rel())
            else:
                start = str(root.GetPath()) if str(root.GetPath()) in self.bodies else self._first_body_below(root)
            if start is None:
                continue
            members, todo = {start}, [start]
            while todo:
                for nb in adj.get(todo.pop(), ()):
                    if nb not in members:
                        members.add(nb)
                        todo.append(nb)
            self.out.articulations[start] = sorted(members)
            if _get(root, "physxArticulation:enabledSelfCollisions", True):
                continue
            ms = sorted(members)
            for i, a in enumerate(ms):
                for b in ms[i + 1 :]:
                    pairs.add((a, b))
        for a, b in sorted(pairs):
            ex = self.spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = a, b
        self.out.stats["excludes"] = len(pairs)

    def _first_body_below(self, prim) -> str | None:
        for p in self.Usd.PrimRange(prim, self.Usd.TraverseInstanceProxies()):
            if str(p.GetPath()) in self.bodies:
                return str(p.GetPath())
        return None

    # ---------------------------------------------------------- finish
    def finish(self):
        """Bodies with only a mass get it spread over their collision geoms by volume."""
        for name, body in self.mbody.items():
            if name is None or not body.userdata:
                continue
            mass = float(body.userdata[0])
            body.userdata = []
            geoms = [g for g in body.geoms if g.group == COLLISION_GROUP]
            if not geoms:
                body.explicitinertial = True
                body.mass, body.inertia = mass, (1e-4, 1e-4, 1e-4)
                continue
            vols = np.array([_volume(g, self._piece_volume) for g in geoms])
            vols = np.where(vols > 0, vols, max(vols.max(), 1e-9) * 0.1)
            for g, v in zip(geoms, vols, strict=True):
                g.mass = mass * v / vols.sum()
        self.out.stats["bodies"] = len(self.mbody) - 1
        self.out.stats["joints"] = len(self.out.joints)


TEXTURE_SIZE = 512  # px: Infinigen's 1024 px maps, halved, keep the model small
VISUAL_MAX_FACES = 20000  # meshes above this are simplified for rendering
VISUAL_CELLS = 250  # grid cells along the longest side of a simplified mesh


def simplify(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertex clustering for a dense render mesh (Infinigen furniture has 10^5 faces): vertices in one
    grid cell merge into their mean, so detail below 1/250 of the mesh size goes away. Collision is not
    affected; it is built from the full mesh. Also returns which input faces survive, in order."""
    every = np.arange(len(faces))
    if len(faces) <= VISUAL_MAX_FACES:
        return verts, faces, every
    cell = float(np.max(np.ptp(verts, axis=0))) / VISUAL_CELLS
    if cell <= 0:
        return verts, faces, every
    keys = np.floor(verts / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = np.bincount(inverse)
    merged = np.zeros((len(count), 3))
    np.add.at(merged, inverse, verts)
    merged /= count[:, None]
    f = inverse[faces]
    alive = np.flatnonzero((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2]))
    _, first = np.unique(np.sort(f[alive], axis=1), axis=0, return_index=True)  # a face merged twice
    keep = alive[np.sort(first)]
    used, local = np.unique(f[keep], return_inverse=True)
    return merged[used], local.reshape(-1, 3), keep


_DECOMPOSE = ("convexDecomposition", "none", "meshSimplification", "sdf")


def _warm_cache(meshes: list[tuple[np.ndarray, np.ndarray, bool]]) -> None:
    """Decompose the meshes not yet in the cache, in parallel processes (CoACD is single threaded)."""
    todo = [(p, f, st) for p, f, st in meshes if not hulls.cached(p, f, static=st)]
    if not todo:
        return
    import concurrent.futures
    import os

    workers = max(1, min(len(todo), (os.cpu_count() or 2) - 2))
    log.info("decomposing %d collision meshes in %d processes (cached afterwards)", len(todo), workers)
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(hulls.convex_pieces, p, f, static=st) for p, f, st in todo]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()


def _thicken(piece: np.ndarray) -> np.ndarray:
    c = piece - piece.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    if len(s) == 3 and s[2] / max(1, np.sqrt(len(piece))) > _TINY / 4:
        return piece
    n = vt[-1]
    return np.concatenate([piece + n * _TINY, piece - n * _TINY])


def _volume(g, piece_volume: dict[str, float]) -> float:
    if g.type == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.prod(np.asarray(g.size) * 2))
    if g.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return 4 / 3 * math.pi * g.size[0] ** 3
    if g.type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        return math.pi * g.size[0] ** 2 * g.size[1] * 2
    if g.type == mujoco.mjtGeom.mjGEOM_MESH:
        return piece_volume.get(g.meshname, 0.0)
    return 0.0


def convert(stage, options: Options | None = None) -> Conversion:
    """MuJoCo spec (not yet compiled) for the stage, with the maps `World` needs."""
    options = options or Options()
    b = _Builder(stage, options)
    b.scan()
    b.build_tree()
    b.add_colliders()
    if options.visuals:
        b.add_visuals()
    b.add_excludes()
    b.finish()
    spec = b.spec
    spec.compiler.degree = False
    spec.compiler.inertiagrouprange = (COLLISION_GROUP, COLLISION_GROUP)
    spec.compiler.boundmass = 1e-3
    spec.compiler.boundinertia = 1e-7
    return b.out
