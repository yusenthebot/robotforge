"""MuJoCo stand-in for ZenoBench's cuRobo `Motion` (zeno_skills/motion.py, upstream since 4fd4ae5).

Upstream sends every arm IK, collision check and joint-space plan to cuRobo, which needs CUDA.
`install()` swaps `zeno_skills.motion.Motion` for `MujocoMotion`; the rest of that module (sphere
helpers, constants, STATS) stays upstream's.  See docs/ZENOBENCH_MOTION_PLAN.md.

The planning world is a copy of the physics model: the scene colliders the robot really hits in the
MuJoCo run, plus the robot as upstream's cuRobo spheres (robot_sources/curobo/zeno_malo.yml, 283 on
27 links, fitted by Isaac's generator to the PhysX colliders) and a pool of spheres per hand for a
held object.  `add_planning_spheres` puts those spheres into the physics spec before it compiles,
switched off there (no contacts, no mass); the planning copy turns them on against the scene and
everything else off, so one `mj_collision` gives every sphere-to-scene distance.  Self collision is
numpy over the sphere pairs the yml does not ignore.

The collision rules are upstream's (see that module's docstring): held objects and what they carry
are not obstacles to the holding arm, nor objects at a query's TCP target, `ignore_fingers` /
`hand_touches_part` switch hand spheres off, the other arm stays where it was last set, a held
object's box rides on the hand inside `Rig.held_collision`.  Upstream's distance culling (2.5 m) is
left out: it only skips obstacles out of the arm's reach.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from robotforge.zenobench import motion_kin as K
from robotforge.zenobench import motion_plan

ASSET = "/World/ZenoMalo/Asset"
SIDES = ("right", "left")
SPHERE_TAG = "rf_plan_sphere"
HELD_TAG = "rf_plan_held"
SPHERE_GROUP = 5  # never rendered (the viewer shows the visual group only)
PLAN_BIT = 1 << 4  # contype / conaffinity bit of the planning pairs: robot sphere x scene
HELD_IDLE = 1e-3  # m: radius of an unused held-object sphere (at the TCP, switched off)
SEGMENT_RES = 0.05  # rad: dense check spacing on a joint segment (upstream _segments_free)
RANDOM_SEEDS = 28  # IK seeds beyond the 4 a query passes (cuRobo solves 32)
EXTRA_SEEDS = 64  # one more batch when every solution collides
REPAIR_ITERS = 40
REPAIR_CLEAR = 0.01  # m beyond the activation distance a collision repair pushes a sphere to
PARALLEL_MIN = 24  # rows: below this a batch check runs on the calling thread
THREADS = max(1, min(8, (os.cpu_count() or 2) - 2))
ARM_LINKS = tuple(f"arm_link_{i}" for i in range(1, 8)) + (
    "gripper_link",
    "gripper_camera_link",
    "gripper_left_finger_link",
    "gripper_right_finger_link",
)


def _upstream():
    import zeno_skills.motion as U

    return U


def _timed(key):
    """Accumulate the wall time of a query in upstream's STATS[key] (run_episode reports it)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                S = _upstream().STATS
                S[key] = round(S.get(key, 0.0) + time.perf_counter() - t, 3)

        return wrap

    return deco


def _count(key, n=1):
    S = _upstream().STATS
    S[key] = S.get(key, 0) + int(n)


# ------------------------------------------------------------------ robot spheres
@dataclass(frozen=True)
class RobotSpheres:
    """cuRobo's collision spheres for Zeno Malo, in link frames."""

    links: tuple[str, ...]
    centers: np.ndarray
    radii: np.ndarray
    ignore: frozenset = field(default_factory=frozenset)  # unordered link pairs never checked

    @classmethod
    def load(cls, path: Path) -> RobotSpheres:
        import yaml

        k = yaml.safe_load(Path(path).read_text())["robot_cfg"]["kinematics"]
        buffer = k.get("collision_sphere_buffer", 0.0) or 0.0
        links, centers, radii = [], [], []
        for link, spheres in k["collision_spheres"].items():
            pad = float(buffer.get(link, 0.0)) if isinstance(buffer, dict) else float(buffer)
            for s in spheres:
                if float(s["radius"]) <= 0:
                    continue
                links.append(link)
                centers.append([float(v) for v in s["center"]])
                radii.append(float(s["radius"]) + pad)
        ignore = {frozenset((a, b)) for a, bs in (k.get("self_collision_ignore") or {}).items() for b in bs}
        return cls(tuple(links), np.array(centers), np.array(radii), frozenset(ignore))


def add_planning_spheres(spec, conv=None) -> None:
    """World.build hook: the cuRobo spheres and the held-object pools as inert geoms on the robot links.

    They compile with contype PLAN_BIT and conaffinity 0: the physics geoms (contype = conaffinity = 1)
    never pair with them, but MuJoCo puts them in each body's collision tree, which is built once at
    compile from the geoms that can collide (spheres compiled with masks 0 are never found later).
    The held-object pool compiles as tiny spheres at the TCP; MujocoMotion refits the gripper's tree
    in its planning copy when a held object's spheres move in."""
    U = _upstream()
    path = Path(U.ROBOT_CFG)
    if not path.exists() or spec.body(f"{ASSET}/right_gripper_link") is None:
        return
    from zeno_skills.kinematics import TCP_DEPTH

    rs = RobotSpheres.load(path)
    for i, (link, c, r) in enumerate(zip(rs.links, rs.centers, rs.radii, strict=True)):
        body = spec.body(f"{ASSET}/{link}")
        if body is None:
            raise ValueError(f"planning spheres: no body {ASSET}/{link} in the converted robot")
        _inert_sphere(body, f"{SPHERE_TAG}:{i}:{link}", c, r)
    for side in SIDES:
        body = spec.body(f"{ASSET}/{side}_gripper_link")
        for i in range(U.HELD_SPHERES):
            _inert_sphere(body, f"{HELD_TAG}:{side}:{i}", (0.0, 0.0, -TCP_DEPTH), HELD_IDLE)


def _inert_sphere(body, name, pos, radius) -> None:
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = (float(radius), 0.0, 0.0)
    g.pos = [float(v) for v in pos]
    g.contype, g.conaffinity = PLAN_BIT, 0
    g.group = SPHERE_GROUP
    g.density = 0.0
    g.rgba = (0.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------ query state
@dataclass
class _Query:
    side: str
    kin: object
    chain: K.Chain
    base_q: np.ndarray  # planar base joint values for kin's base pose
    lo: np.ndarray
    hi: np.ndarray
    pairs: tuple  # (i, j, link pair of each) sphere pairs checked for self collision
    held_on: bool  # this arm's held-object spheres are attached


class MujocoMotion:
    """Upstream's Motion API (zeno_skills/motion.py) on the MuJoCo world, without CUDA."""

    def __init__(self, stage, ann, robot_cfg=None):
        from zeno_skills.kinematics import FINGER_OPEN, LEFT_ARM_FOLD, TCP_DEPTH, ArmKin

        from robotforge.zenobench import shim

        U = _upstream()
        t0 = time.perf_counter()
        self.ann = ann
        self.world = shim.current_world()
        self.rs = RobotSpheres.load(Path(robot_cfg or U.ROBOT_CFG))
        m0 = self.world.model
        self.model = m = copy.copy(m0)
        name = functools.partial(mujoco.mj_id2name, m, mujoco.mjtObj.mjOBJ_GEOM)
        names = [name(g) or "" for g in range(m.ngeom)]
        self.sph = np.array([g for g, n in enumerate(names) if n.startswith(SPHERE_TAG + ":")], int)
        if len(self.sph) != len(self.rs.links):
            raise RuntimeError(
                "MujocoMotion: the world has no planning spheres; robotforge.zenobench.motion.install() "
                "must run before the simulation is built"
            )
        self.sph_link = np.array([n.split(":", 2)[2] for n in (names[g] for g in self.sph)])
        self.sph_r = m.geom_size[self.sph, 0].copy()
        self.held_geoms = {
            s: np.array([g for g, n in enumerate(names) if n.startswith(f"{HELD_TAG}:{s}:")], int) for s in SIDES
        }
        body_name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        robot_geom = np.array([body_name[b].startswith("/World/ZenoMalo") for b in m.geom_bodyid])
        collides = (m0.geom_contype | m0.geom_conaffinity) != 0
        self.scene = np.nonzero(collides & ~robot_geom)[0]
        self.is_sphere = np.zeros(m.ngeom, bool)
        self.is_sphere[self.sph] = True
        self.sphere_index = np.full(m.ngeom, -1, int)
        self.sphere_index[self.sph] = np.arange(len(self.sph))
        self.held_owner = np.full(m.ngeom, "", object)
        for s in SIDES:
            self.is_sphere[self.held_geoms[s]] = True
            self.held_owner[self.held_geoms[s]] = s
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        m.geom_margin[:] = 0.0
        m.geom_gap[:] = 0.0
        m.geom_conaffinity[self.scene] = PLAN_BIT
        m.geom_contype[self.sph] = PLAN_BIT
        robot_body = np.array([n.startswith("/World/ZenoMalo") for n in body_name])
        m.body_contype[:] = np.where(robot_body, PLAN_BIT, 0)
        m.body_conaffinity[:] = 0
        m.body_conaffinity[np.unique(m.geom_bodyid[self.scene])] = PLAN_BIT
        m.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        self._margin = None
        # owners (upstream scene_geometry.collect): annotated objects and articulated parts by prim prefix
        prefixes = [(o["prim"], f"object:{o['name']}") for o in _objects(ann)]
        prefixes += [(a["part"], f"part:{a['name']}") for a in ann.articulated if a.get("part")]
        prefixes.sort(key=lambda t: -len(t[0]))
        owner = []
        for g in self.scene:
            path = names[g].split("#", 1)[0]
            owner.append(next((own for pre, own in prefixes if path == pre or path.startswith(pre + "/")), "static"))
        self.scene_owner = np.array(owner, object)
        self.owner_geoms = {o: self.scene[self.scene_owner == o] for o in set(owner)}
        # joints
        jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(m.njnt)]
        short = {n.rsplit("/", 1)[-1]: j for j, n in enumerate(jname) if n.startswith(f"{ASSET}/joints/")}
        self.jid = {n: j for j, n in enumerate(jname)}
        adr = lambda n: int(m.jnt_qposadr[short[n]])  # noqa: E731
        planar = [self.jid[n] for n in jname if n.startswith("/World/ZenoMalo/base_anchor:")]
        if len(planar) != 3:
            raise RuntimeError(f"MujocoMotion: expected the planar base anchor (x, y, yaw), found {len(planar)} joints")
        self.base_adr = np.array([m.jnt_qposadr[j] for j in planar], int)
        self.base_axes = m.jnt_axis[planar].copy()
        self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{ASSET}/base_link")
        self.kins = {s: ArmKin(side=s) for s in SIDES}
        self.chains = {s: K.Chain(self.kins[s], TCP_DEPTH) for s in SIDES}
        self.tcp_depth = TCP_DEPTH
        self.side_adr = {s: np.array([adr(n) for n in self.kins[s].names], int) for s in SIDES}
        self.side_dof = {s: np.array([m.jnt_dofadr[short[n]] for n in self.kins[s].names], int) for s in SIDES}
        self.other_names = {s: self.kins["left" if s == "right" else "right"].names[2:] for s in SIDES}
        self.other_adr = {s: np.array([adr(n) for n in self.other_names[s]], int) for s in SIDES}
        fold = dict(zip(self.kins["right"].names, self.kins["right"].posture, strict=True))
        fold.update(LEFT_ARM_FOLD)
        self.lock = {s: np.array([float(fold.get(n, 0.0)) for n in self.other_names[s]]) for s in SIDES}
        fixed = {"head_yaw_joint": 0.0, "head_pitch_joint": 0.0}
        for s in SIDES:
            fixed[f"{s}_gripper_left_finger_axis"] = fixed[f"{s}_gripper_right_finger_axis"] = FINGER_OPEN
        self.fixed_adr = np.array([adr(n) for n in fixed], int)
        self.fixed_val = np.array(list(fixed.values()), float)
        self.robot_adr = np.unique(np.r_[self.base_adr, self.fixed_adr, self.side_adr["right"], self.side_adr["left"]])
        self.part_adr = {
            a["name"]: int(m.jnt_qposadr[self.jid[a["joint"]]]) for a in ann.articulated if a.get("joint") in self.jid
        }
        self._link_bounds()
        self.pairs = {s: self._self_pairs(s) for s in SIDES}
        self.data = mujoco.MjData(m)
        self.qpos_world = self.world.data.qpos.copy()
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(THREADS, thread_name_prefix="rf-motion") if THREADS > 1 else None
        self._calibrate_base()
        self.seeds01 = K.halton(RANDOM_SEEDS + EXTRA_SEEDS, len(self.kins["right"].names))
        self.held = {s: set() for s in SIDES}
        self.held_collision = set()
        self.held_state = {}
        self.other_touching = {s: set() for s in SIDES}
        self._held_box = {}
        self._held_on = {s: False for s in SIDES}
        self._boxes = None
        self._mask_key = None
        U.STATS["init_s"] = round(U.STATS.get("init_s", 0.0) + time.perf_counter() - t0, 3)

    # ------------------------------------------------------------- setup helpers
    def _self_pairs(self, side):
        """Sphere pairs whose links can move against each other in a query on ``side``: one of them on
        that arm, or exactly one on the torso lift (the waist turns everything above it)."""
        arm = {f"{side}_{n}" for n in ARM_LINKS}
        links = self.sph_link
        i, j = np.triu_indices(len(links), 1)
        a, b = links[i], links[j]
        moving = np.isin(a, list(arm)) | np.isin(b, list(arm)) | ((a == "torso_lift_link") != (b == "torso_lift_link"))
        keep = moving & (a != b)
        if self.rs.ignore:
            ign = np.array([frozenset((x, y)) in self.rs.ignore for x, y in zip(a[keep], b[keep], strict=True)])
            idx = np.nonzero(keep)[0][~ign]
        else:
            idx = np.nonzero(keep)[0]
        i, j = i[idx], j[idx]
        # link pair of each sphere pair: a check tests the links' bounding spheres first
        li, lj = self.sph_li[i], self.sph_li[j]
        key = li * len(self.links) + lj
        uniq, lp = np.unique(key, return_inverse=True)
        self.link_pairs[side] = (uniq // len(self.links), uniq % len(self.links))
        return i, j, lp

    def _link_bounds(self):
        """Per link with spheres: its body, and the centre and radius of a sphere around all of them."""
        self.links = sorted(set(self.sph_link))
        self.sph_li = np.array([self.links.index(n) for n in self.sph_link])
        self.link_body = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{ASSET}/{n}") for n in self.links]
        )
        centres, radii = [], []
        for k in range(len(self.links)):
            sel = self.sph_li == k
            c = self.rs.centers[sel].mean(0)
            centres.append(c)
            radii.append(float((np.linalg.norm(self.rs.centers[sel] - c, axis=1) + self.rs.radii[sel]).max()))
        self.link_c, self.link_r = np.array(centres), np.array(radii)
        self.link_pairs = {}

    def _calibrate_base(self):
        """Base pose of the planar anchor at zero joint values; checked against a moved base."""
        d = self.data
        d.qpos[:] = self.qpos_world
        d.qpos[self.base_adr] = 0.0
        mujoco.mj_kinematics(self.model, d)
        self.base_p0 = d.xpos[self.base_body].copy()
        self.base_R0 = d.xmat[self.base_body].reshape(3, 3).copy()
        p, yaw = self.base_p0 + np.array([0.7, -0.4, 0.0]), 0.9
        d.qpos[self.base_adr] = self._base_q(p, yaw)
        mujoco.mj_kinematics(self.model, d)
        R = d.xmat[self.base_body].reshape(3, 3)
        if np.linalg.norm(d.xpos[self.base_body][:2] - p[:2]) > 1e-6 or abs(math.atan2(R[1, 0], R[0, 0]) - yaw) > 1e-6:
            raise RuntimeError("MujocoMotion: the planar base joints do not move base_link as expected")

    def _base_q(self, p, yaw):
        """Planar joint values (x, y, yaw) that put base_link at p (xy) with heading yaw."""
        d = self.base_R0.T @ (np.asarray(p, float) - self.base_p0)
        A = self.base_axes[:2].T  # slide axes in the anchor frame (columns)
        xy = np.linalg.lstsq(A, d, rcond=None)[0]
        yaw0 = math.atan2(self.base_R0[1, 0], self.base_R0[0, 0])
        return np.array([xy[0], xy[1], (yaw - yaw0) * float(np.sign(self.base_axes[2, 2]) or 1.0)])

    # ------------------------------------------------------------- world state
    def _scene_kinematics(self, d):
        d.qpos[:] = self.qpos_world
        mujoco.mj_kinematics(self.model, d)

    def _owner_box(self, owner):
        """Upstream's loose box of an owner: centre +- bounding radius of each collider, unioned."""
        if self._boxes is None:
            self._scene_kinematics(self.data)
            self._boxes = {}
        if owner not in self._boxes:
            g = self.owner_geoms.get(owner)
            if g is None or len(g) == 0:
                self._boxes[owner] = (np.zeros(3), np.zeros(3))
            else:
                aabb = self.model.geom_aabb[g]
                R = self.data.geom_xmat[g].reshape(-1, 3, 3)
                ctr = self.data.geom_xpos[g] + (R @ aabb[:, :3, None])[..., 0]
                r = np.linalg.norm(aabb[:, 3:], axis=1)[:, None]
                self._boxes[owner] = ((ctr - r).min(0), (ctr + r).max(0))
        return self._boxes[owner]

    @staticmethod
    def _inside(a, b, pad=0.0):
        ca = (a[0] + a[1]) / 2
        return bool(np.all(ca > b[0] - pad) and np.all(ca < b[1] + pad))

    def _targets_allow(self, points):
        out = set()
        pad = _upstream().TARGET_PAD
        for owner in self.owner_geoms:
            if not owner.startswith("object:"):
                continue
            lo, hi = self._owner_box(owner)
            if any(np.all(p > lo - pad) and np.all(p < hi + pad) for p in points):
                out.add(owner)
        return out

    @_timed("sync_s")
    def sync(self, rig):
        """Object and part poses from the simulator; held objects (and what they carry) off; the
        links touching the world now; the held object's spheres on or off (Rig.held_collision)."""
        U = _upstream()
        self.qpos_world = self.world.data.qpos.copy()
        self._boxes = None
        self._mask_key = None
        self.held = {s: set() for s in SIDES}
        for side, h in (("right", rig.held), ("left", getattr(rig, "left_held", None))):
            if h is None or h.get("name") not in _object_names(self.ann):
                continue
            owners = {f"object:{h['name']}"}
            box = self._owner_box(f"object:{h['name']}")
            for name in _object_names(self.ann):
                if name != h["name"] and self._inside(self._owner_box(f"object:{name}"), box, pad=0.02):
                    owners.add(f"object:{name}")
            for s in SIDES if h.get("kind") == "bimanual" else (side,):
                self.held[s] |= owners
        q_of = {"right": np.asarray(rig.q_cmd, float), "left": np.r_[rig.q_cmd[:2], rig.left_q_cmd]}
        kin_of = {"right": rig.kin, "left": rig.left_kin}
        for side in SIDES:
            tcp = kin_of[side].tcp(q_of[side])[0]
            for o in self.held[_other(side)] - self.held[side]:
                lo, hi = self._owner_box(o)
                if np.all(tcp > lo - U.HAND_AT_HELD) and np.all(tcp < hi + U.HAND_AT_HELD):
                    self.held[side].add(o)
        touching = {}
        for side in SIDES:
            kin, q = kin_of[side], q_of[side]
            self.lock[side] = np.asarray(q_of[_other(side)], float)[-len(self.other_names[side]) :]
            h = rig.held if side == "right" else getattr(rig, "left_held", None)
            want = side in self.held_collision
            prev = self.held_state.get(side)
            state = None
            if want or prev:
                state = self._attach_held(side, kin, q, h if want else None)
            ctx = self._context(kin, None, {"ignore_fingers": True}, (), True, fingers_only=True)
            hits = self._touching(ctx, q)
            if state is not None and U.held_link(side) in hits:
                self._attach_held(side, kin, q, None)  # still on its support: no free start
                state += " (touching: off)"
            if state != prev:
                rig.log("held_collision", side=side, state=state)
            self.held_state[side] = state
            touching[side] = {link for link in hits if link.startswith(side + "_") and link != U.held_link(side)}
        self.other_touching = {"right": touching["left"], "left": touching["right"]}

    def release_held(self, side):
        """Rig.held_collision ends: the held spheres go now.  Whether something was held."""
        self.held_collision.discard(side)
        was = self.held_state.get(side)
        if was:
            self._attach_held(side, None, None, None)
        self.held_state[side] = None
        return bool(was)

    def _attach_held(self, side, kin, q, h):
        """Spheres filling the held object's box (upstream box_spheres) on this hand, in the TCP frame
        at q; None (or a bimanual hold) empties the pool.  Returns the object's name."""
        U = _upstream()
        geoms = self.held_geoms[side]
        m = self.model
        name = None if h is None or h.get("kind") == "bimanual" else h.get("name")
        m.geom_contype[geoms] = 0
        m.geom_pos[geoms] = (0.0, 0.0, -self.tcp_depth)
        m.geom_size[geoms, 0] = m.geom_rbound[geoms] = HELD_IDLE
        m.geom_aabb[geoms] = (0.0, 0.0, 0.0, HELD_IDLE, HELD_IDLE, HELD_IDLE)
        self._held_on[side] = False
        self._mask_key = None
        if name is not None and len(self.owner_geoms.get(f"object:{name}", ())):
            frame, lo, hi = self._box_of(name)
            self._scene_kinematics(self.data)
            T_obj = np.eye(4)
            T_obj[:3, :3] = self.data.geom_xmat[frame].reshape(3, 3)
            T_obj[:3, 3] = self.data.geom_xpos[frame]
            p, R = kin.tcp(np.asarray(q, float))
            T_tcp = np.eye(4)
            T_tcp[:3, :3], T_tcp[:3, 3] = R, p
            M = np.linalg.inv(T_tcp) @ T_obj
            b = U.box_spheres(lo, hi)[: len(geoms)]
            pos = b[:, :3] @ M[:3, :3].T + M[:3, 3]
            pos[:, 2] -= self.tcp_depth  # TCP frame -> gripper_link frame
            g = geoms[: len(b)]
            m.geom_pos[g] = pos
            m.geom_size[g, 0] = m.geom_rbound[g] = b[:, 3]
            m.geom_aabb[g] = np.c_[np.zeros((len(b), 3)), np.repeat(b[:, 3:4], 3, 1)]
            m.geom_contype[g] = PLAN_BIT
            self._held_on[side] = True
        else:
            name = None
        self._refit_tree(int(m.geom_bodyid[geoms[0]]))
        return name

    def _refit_tree(self, body):
        """Recompute a body's collision-tree boxes (body inertial frame, as MuJoCo compiles them) after
        its geoms moved or changed size: leaves from the geoms, inner nodes the union of their children."""
        from scipy.spatial.transform import Rotation

        m = self.model
        a = int(m.body_bvhadr[body])
        if a < 0:
            return
        Ri = Rotation.from_quat(m.body_iquat[body][[1, 2, 3, 0]]).as_matrix()

        def box(k):
            g = int(m.bvh_nodeid[a + k])
            if g >= 0:
                Rgi = Ri.T @ Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
                c = Ri.T @ (m.geom_pos[g] - m.body_ipos[body]) + Rgi @ m.geom_aabb[g, :3]
                out = np.r_[c, np.abs(Rgi) @ m.geom_aabb[g, 3:]]
            else:
                kids = [box(int(c)) for c in m.bvh_child[a + k] if c >= 0]
                lo = np.min([x[:3] - x[3:] for x in kids], 0)
                hi = np.max([x[:3] + x[3:] for x in kids], 0)
                out = np.r_[(lo + hi) / 2, (hi - lo) / 2]
            m.bvh_aabb[a + k] = out
            return out

        box(0)

    def _box_of(self, name):
        """(frame geom, lo, hi): the object's box over all its colliders, in its first collider's frame."""
        if name not in self._held_box:
            m = self.model
            self._scene_kinematics(self.data)
            g = self.owner_geoms[f"object:{name}"]
            frame = int(g[0])
            Rf, pf = self.data.geom_xmat[frame].reshape(3, 3), self.data.geom_xpos[frame]
            P = []
            for k in g:
                Rk, pk = self.data.geom_xmat[k].reshape(3, 3), self.data.geom_xpos[k]
                P.append(((_geom_points(m, k) @ Rk.T + pk) - pf) @ Rf)
            P = np.vstack(P)
            self._held_box[name] = (frame, P.min(0), P.max(0))
        return self._held_box[name]

    @contextmanager
    def part_at(self, name, q):
        """Articulated part ``name`` at joint value q for the queries inside; the synced pose after."""
        a = self.ann.art(name)
        i = self.part_adr.get(a["name"])
        if i is None:
            yield
            return
        saved = float(self.qpos_world[i])
        self.qpos_world[i] = float(q)
        self._boxes = None
        try:
            yield
        finally:
            self.qpos_world[i] = saved
            self._boxes = None

    # ------------------------------------------------------------- queries: context
    def _context(self, kin, other_q, coll_kw, targets, collision, fingers_only=False):
        U = _upstream()
        coll_kw = coll_kw or {}
        side = kin.side
        if coll_kw.get("hand_touches_part"):
            own = U.HAND_LINKS
        else:
            own = U.FINGER_LINKS if coll_kw.get("ignore_fingers") else ()
        off = {f"{side}_{link}" for link in own} | {f"{_other(side)}_{link}" for link in U.FINGER_LINKS}
        if not fingers_only:
            off |= self.other_touching.get(side, set())
        if other_q is not None:
            self.lock[side] = np.asarray(other_q, float)[-len(self.other_names[side]) :]
        # a pinned joint (lo == hi) gets a +-PIN band, as upstream gives cuRobo
        lo = np.minimum(kin.lo, kin.hi - 2 * U.PIN)
        hi = np.maximum(kin.hi, kin.lo + 2 * U.PIN)
        pinned = (kin.hi - kin.lo) < 2 * U.PIN
        lo[pinned], hi[pinned] = kin.lo[pinned] - U.PIN, kin.hi[pinned] + U.PIN
        yaw = math.atan2(kin.base_R[1, 0], kin.base_R[0, 0])
        allow = self._targets_allow([np.asarray(p, float) for p in targets]) if collision and targets else set()
        excluded = frozenset(self.held[side] | allow)
        key = (side, frozenset(off), excluded, self._held_on[side])
        if key != self._mask_key:
            m = self.model
            m.geom_conaffinity[self.scene] = PLAN_BIT
            for o in excluded:
                m.geom_conaffinity[self.owner_geoms.get(o, [])] = 0
            on = ~np.isin(self.sph_link, list(off))
            m.geom_contype[self.sph] = np.where(on, PLAN_BIT, 0)
            m.geom_contype[self.held_geoms[_other(side)]] = 0
            if not self._held_on[side]:
                m.geom_contype[self.held_geoms[side]] = 0
            self._mask_key = key
        on = self.model.geom_contype[self.sph] != 0
        i, j, lp = self.pairs[side]
        sel = on[i] & on[j]
        pairs = (i[sel], j[sel], lp[sel], *self.link_pairs[side])
        return _Query(side, kin, self.chains[side], self._base_q(kin.base_p, yaw), lo, hi, pairs, self._held_on[side])

    def _set_margin(self, margin):
        if margin != self._margin:
            g = np.r_[self.sph, self.held_geoms["right"], self.held_geoms["left"]]
            self.model.geom_margin[g] = margin
            self._margin = margin

    def _set_robot(self, d, ctx, q):
        d.qpos[:] = self.qpos_world
        d.qpos[self.base_adr] = ctx.base_q
        d.qpos[self.fixed_adr] = self.fixed_val
        d.qpos[self.other_adr[ctx.side]] = self.lock[ctx.side]
        d.qpos[self.side_adr[ctx.side]] = q

    def _data(self):
        d = getattr(self._local, "data", None)
        if d is None:
            d = self._local.data = mujoco.MjData(self.model)
        return d

    # ------------------------------------------------------------- queries: evaluation
    def _row(self, d, ctx, q, mode):
        """One joint vector.  feasible -> bool; self -> bool; clearance / penetration -> float."""
        self._set_robot(d, ctx, q)
        mujoco.mj_kinematics(self.model, d)
        if mode in ("feasible", "self"):
            i, j, lp, la, lb = ctx.pairs
            if len(i):
                body = self.link_body
                C = d.xpos[body] + (d.xmat[body].reshape(-1, 3, 3) @ self.link_c[..., None])[..., 0]
                near = np.linalg.norm(C[la] - C[lb], axis=1) < self.link_r[la] + self.link_r[lb]
                k = near[lp]
                i, j = i[k], j[k]
            if len(i):
                X = d.geom_xpos[self.sph]
                gap = np.linalg.norm(X[i] - X[j], axis=1) - self.sph_r[i] - self.sph_r[j]
                if gap.min() < 0.0:
                    return False
            if mode == "self":
                return True
        mujoco.mj_collision(self.model, d)
        n = d.ncon
        if mode == "feasible":
            return n == 0
        dist = d.contact.dist[:n]
        if mode == "clearance":
            return float(dist.min()) if n else math.inf
        return float(max(0.0, -dist.min())) if n else 0.0  # penetration

    def _eval(self, ctx, Q, mode):
        U = _upstream()
        margin = {"feasible": U.ACTIVATION, "self": 0.0, "clearance": U.CLEAR_CAP, "penetration": 0.0}[mode]
        self._set_margin(margin)
        Q = np.atleast_2d(np.asarray(Q, float))
        if self._pool is None or len(Q) < PARALLEL_MIN:
            d = self.data
            return [self._row(d, ctx, q, mode) for q in Q]
        chunks = np.array_split(np.arange(len(Q)), THREADS)

        def run(idx):
            d = self._data()
            return [self._row(d, ctx, Q[k], mode) for k in idx]

        out = []
        for part in self._pool.map(run, [c for c in chunks if len(c)]):
            out += part
        return out

    def _feasible(self, ctx, Q):
        return np.array(self._eval(ctx, np.clip(np.atleast_2d(Q), ctx.lo, ctx.hi), "feasible"), bool)

    def _segments_free(self, ctx, qs, res=SEGMENT_RES):
        dense = []
        for qa, qb in zip(qs[:-1], qs[1:], strict=True):
            k = max(1, int(np.max(np.abs(qb - qa)) / res))
            dense += [qa + (qb - qa) * u for u in np.linspace(0, 1, k + 1)[1:]]
        return bool(self._feasible(ctx, np.array(dense)).all()) if dense else True

    def _touching(self, ctx, q):
        """{link: depth} of the links (and held object) penetrating the world at q."""
        U = _upstream()
        self._set_margin(0.0)
        d = self.data
        self._set_robot(d, ctx, np.asarray(q, float))
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        hits = {}
        for c in range(d.ncon):
            g1, g2 = d.contact.geom[c]
            g = g1 if self.is_sphere[g1] else g2
            link = self.sph_link[self.sphere_index[g]] if self.sphere_index[g] >= 0 else U.held_link(self.held_owner[g])
            hits[link] = max(hits.get(link, 0.0), float(-d.contact.dist[c]))
        return hits

    # ------------------------------------------------------------- IK
    def _seeds(self, ctx, seeds, local, extra=False):
        U = _upstream()
        seeds = [np.clip(np.asarray(s, float), ctx.lo, ctx.hi) for s in seeds][: U.N_SEEDS]
        posture = np.clip(ctx.kin.posture, ctx.lo, ctx.hi)
        if local and seeds:
            seeds = [seeds[0]] * U.N_SEEDS
        seeds = (seeds + [posture] * U.N_SEEDS)[: U.N_SEEDS]
        rows = self.seeds01[RANDOM_SEEDS:] if extra else self.seeds01[:RANDOM_SEEDS]
        rand = ctx.lo + rows * (ctx.hi - ctx.lo)
        return seeds, np.vstack([np.array(seeds), rand])

    def _solve(self, ctx, p, R, seeds, local, extra=False):
        """Kinematic IK from every seed: (distinct self-collision-free solutions nearest the first seed
        first, at most N_RETURN; the first seeds; the closest failed attempt)."""
        U = _upstream()
        first, S = self._seeds(ctx, seeds, local, extra)
        kin = ctx.kin
        Q, ep, er = K.solve(ctx.chain, kin.base_p, kin.base_R, p, R, S, ctx.lo, ctx.hi, retract=first[0])
        ok = (ep < U.POS_TOL) & (er < U.ROT_TOL)
        best_fail = Q[int(np.argmin(ep + er))]
        if not ok.any():
            return np.empty((0, len(first[0]))), first, best_fail
        Q = Q[ok]
        Q = K.distinct(Q[np.argsort(np.max(np.abs(Q - first[0]), axis=1))])
        free = np.array(self._eval(ctx, Q, "self"), bool)
        return Q[free][: U.N_RETURN], first, best_fail

    def _pick(self, ctx, Q, first, local):
        U = _upstream()
        if len(Q) == 0:
            return None, False
        return K.pick(ctx.kin, Q, first[0], ctx.lo, ctx.hi, local, U.LIMIT_CLEAR, U.IK_MANIP_W, U.MAX_STEP)

    def _ik(self, ctx, p, R, seeds, collision, local=False):
        """(q, ok): upstream's order, kinematic first, then the collision-free solutions."""
        p = np.asarray(p, float)
        Q, first, best_fail = self._solve(ctx, p, R, seeds, local)
        q, ok = self._pick(ctx, Q, first, local)
        if not ok:
            if collision:
                _count("ik_fail_reach")
            return (q if q is not None else best_fail), False
        if not collision or self._feasible(ctx, q[None])[0]:
            return q, True
        qf = q
        free = Q[self._feasible(ctx, Q)] if len(Q) else Q
        q2, ok2 = self._pick(ctx, free, first, local)
        if not ok2:
            others = [c for c in Q if np.max(np.abs(c - qf)) > 1e-9][:3]
            q2, ok2 = self._repair(ctx, p, R, [qf] + others, first, local)
        if not ok2:
            Qx, _, _ = self._solve(ctx, p, R, [qf] + list(seeds), local, extra=True)
            free = Qx[self._feasible(ctx, Qx)] if len(Qx) else Qx
            q2, ok2 = self._pick(ctx, free, first, local)
        if not ok2:
            _count("ik_fail_collision")
            U = _upstream()
            if U.DIAG and U.STATS.get("ik_fail_collision", 0) <= 40:
                why = self._explain(ctx, qf)
                print(f"[motion] {ctx.side} IK blocked at {np.round(p, 3).tolist()}: {why}", flush=True)
            return qf, False
        return q2, True

    def _repair(self, ctx, p, R, cands, first, local):
        """Push colliding spheres out along the contact normals in the null space of the TCP pose
        (cuRobo's collision-aware IK does this with a collision cost)."""
        U = _upstream()
        m, d = self.model, self.data
        target = U.ACTIVATION + REPAIR_CLEAR
        dof = self.side_dof[ctx.side]
        jacp = np.zeros((3, m.nv))
        R = np.asarray(R, float)
        fixed = []
        for q in cands:
            q = np.asarray(q, float).copy()
            for _ in range(REPAIR_ITERS):
                self._set_margin(target)
                self._set_robot(d, ctx, q)
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                tcp, Rq, J = ctx.chain.tcp_jac(q[None], ctx.kin.base_p, ctx.kin.base_R)
                e = np.r_[p - tcp[0], K.rotvec((R @ Rq[0].T)[None])[0]]
                if d.ncon == 0:
                    break
                rows, need = [], []
                worst = {}
                for c in range(d.ncon):
                    g1, g2 = d.contact.geom[c]
                    g, sign = (g1, -1.0) if self.is_sphere[g1] else (g2, 1.0)
                    dist = float(d.contact.dist[c])
                    if g not in worst or dist < worst[g][0]:
                        worst[g] = (dist, sign * d.contact.frame[c][:3].copy())
                for g, (dist, u) in worst.items():
                    mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[g], m.geom_bodyid[g])
                    rows.append(u @ jacp[:, dof])
                    need.append(target - dist)
                A, b = np.array(rows), np.array(need)
                J1 = J[0]
                J1p = J1.T @ np.linalg.inv(J1 @ J1.T + 1e-4 * np.eye(6))
                dq1 = J1p @ e
                N = np.eye(len(q)) - J1p @ J1
                AN = A @ N
                dq2 = N @ (AN.T @ np.linalg.solve(AN @ AN.T + 1e-4 * np.eye(len(b)), b - A @ dq1))
                dq = dq1 + dq2
                n = np.linalg.norm(dq)
                if n > K.STEP_MAX:
                    dq *= K.STEP_MAX / n
                q = np.clip(q + dq, ctx.lo, ctx.hi)
            ep, er = K.pose_error(ctx.chain, q[None], ctx.kin.base_p, ctx.kin.base_R, p, R)
            if ep[0] < U.POS_TOL and er[0] < U.ROT_TOL and self._feasible(ctx, q[None])[0]:
                fixed.append(q)
        if not fixed:
            return None, False
        return self._pick(ctx, np.array(fixed), first, local)

    def _explain(self, ctx, q):
        hits = self._touching(ctx, q)
        if hits:
            worst = sorted(hits.items(), key=lambda t: -t[1])[:4]
            return "world " + ", ".join(f"{k} {v * 1000:.0f} mm" for k, v in worst)
        return "self collision, joint limits or within the 5 mm activation distance"

    # ------------------------------------------------------------- Cartesian paths
    def _path_ik(self, ctx, P, Rs, q0):
        """IK along TCP poses, each waypoint from the previous one; continuous (<= MAX_STEP per joint)
        and free of self collision.  Returns (qs, ok)."""
        U = _upstream()
        kin = ctx.kin
        q = np.clip(np.asarray(q0, float), ctx.lo, ctx.hi)
        out = []
        for p, R in zip(P, Rs, strict=True):
            Q, ep, er = K.solve(ctx.chain, kin.base_p, kin.base_R, p, R, q[None], ctx.lo, ctx.hi, retract=q, iters=60)
            if not (ep[0] < U.POS_TOL and er[0] < U.ROT_TOL and np.max(np.abs(Q[0] - q)) <= U.MAX_STEP):
                S = np.vstack([q, np.clip(q0, ctx.lo, ctx.hi), ctx.lo + self.seeds01[:6] * (ctx.hi - ctx.lo)])
                Q, ep, er = K.solve(ctx.chain, kin.base_p, kin.base_R, p, R, S, ctx.lo, ctx.hi, retract=q)
                ok = (ep < U.POS_TOL) & (er < U.ROT_TOL)
                if not ok.any():
                    return out, False
                Q = Q[ok]
                jump = np.max(np.abs(Q - q), axis=1)
                k = int(np.argmin(jump))
                if jump[k] > U.MAX_STEP:
                    return out, False
                Q = Q[k : k + 1]
            q = Q[0]
            out.append(q)
        if out and not all(self._eval(ctx, np.array(out), "self")):
            return out, False
        return out, True

    def _cart(self, ctx, q0, p_goal, R_goal, step, collision):
        from scipy.spatial.transform import Rotation, Slerp

        kin = ctx.kin
        p0, R0 = kin.tcp(q0)
        n = max(2, int(np.ceil(np.linalg.norm(p_goal - p0) / step)) + 1)
        ang = np.linalg.norm(Rotation.from_matrix(R_goal @ R0.T).as_rotvec())
        n = max(n, int(np.ceil(ang / 0.05)) + 1)
        slerp = Slerp([0, 1], Rotation.from_matrix([R0, R_goal]))
        us = np.linspace(0, 1, n)[1:]
        out, ok = self._path_ik(ctx, [p0 + (p_goal - p0) * u for u in us], [slerp(u).as_matrix() for u in us], q0)
        if not ok:
            return out, False
        if collision and not self._segments_free(ctx, [np.asarray(q0, float)] + out):
            return out, False
        return out, True

    # ------------------------------------------------------------- planning
    def _plan_js(self, ctx, q0, q1):
        """Collision-free joint path q0 -> q1 (waypoints after q0), or (None, status)."""
        U = _upstream()
        q0, q1 = np.clip(q0, ctx.lo, ctx.hi), np.clip(q1, ctx.lo, ctx.hi)
        if not self._feasible(ctx, q0[None])[0]:
            return None, "start state in collision"
        if not self._feasible(ctx, q1[None])[0]:
            return None, "goal state in collision"
        if self._segments_free(ctx, [q0, q1]):
            return _dense(q0, q1, SEGMENT_RES), None
        d = self.data

        def valid(q):
            return self._row(d, ctx, q, "feasible")

        self._set_margin(U.ACTIVATION)
        qs, status = motion_plan.plan(q0, q1, ctx.lo, ctx.hi, valid, SEGMENT_RES, U.PLAN_TIMEOUT, U.PLAN_ATTEMPTS)
        if qs is not None and not self._segments_free(ctx, [q0] + qs):
            return None, "planned path failed the dense check"
        return qs, status

    def _tilt_constraint(self, ctx, R_goal):
        """F(q) = 0: the TCP carries the world z axis exactly as R_goal does (tilt held, yaw free)."""
        kin = ctx.kin
        v = np.asarray(R_goal, float).T @ np.array([0.0, 0.0, 1.0])

        def f(q):
            _, R = ctx.chain.tcp(q[None], kin.base_p, kin.base_R)
            return (R[0] @ v)[:2]

        def jac(q):
            _, R, J = ctx.chain.tcp_jac(q[None], kin.base_p, kin.base_R)
            Rv = R[0] @ v
            return np.cross(J[0, 3:].T, Rv)[:, :2].T

        return motion_plan.Constraint(f, jac, 2)

    def _plan_tilt(self, ctx, q0, q_goal, R_goal):
        """Joint path q0 -> q_goal with the held object's tilt kept at R_goal's.  Both ends are first
        projected onto that tilt (q0 may tilt up to CARRY_TILT away, an IK goal by its tolerance), a
        short straight move each, and OMPL plans on the constraint manifold between them."""
        U = _upstream()
        cons = self._tilt_constraint(ctx, R_goal)
        q0 = np.clip(np.asarray(q0, float), ctx.lo, ctx.hi)
        q_goal = np.clip(np.asarray(q_goal, float), ctx.lo, ctx.hi)
        qa, qb = cons.project(q0, ctx.lo, ctx.hi), cons.project(q_goal, ctx.lo, ctx.hi)
        if qa is None or qb is None:
            return None, "could not level the start or goal onto the goal tilt"
        if not self._segments_free(ctx, [q0, qa]):
            return None, "levelling the start collides"
        if not self._segments_free(ctx, [qb, q_goal]):
            return None, "goal state in collision"
        d = self.data

        def valid(q):
            return self._row(d, ctx, q, "feasible")

        self._set_margin(U.ACTIVATION)
        qs, status = motion_plan.plan(
            qa, qb, ctx.lo, ctx.hi, valid, SEGMENT_RES, U.PLAN_TIMEOUT, U.PLAN_ATTEMPTS, constraint=cons
        )
        if qs is None:
            return None, status
        if np.max(np.abs(qb - q_goal)) > 1e-9:
            qs = qs + [q_goal]
        path = ([qa] if np.max(np.abs(qa - q0)) > 1e-6 else []) + qs
        if not self._segments_free(ctx, [q0] + path):
            return None, "tilt-held path failed the dense check"
        return path, None

    # ------------------------------------------------------------- API (kinematics.ArmKin)
    @_timed("ik_s")
    def ik(self, kin, p, R, seeds, collision, other_q=None, coll_kw=None, local=False):
        _count("ik")
        ctx = self._context(kin, other_q, coll_kw, [p], collision)
        q, ok = self._ik(ctx, p, R, seeds, collision, local)
        _count("ik_ok", ok)
        return q, ok

    @_timed("cart_s")
    def cart_path(self, kin, q0, p_goal, R_goal, step, collision, other_q=None, coll_kw=None):
        """Straight TCP line (slerp orientation) as joint waypoints; continuous and, with collision,
        every dense joint segment free."""
        _count("cart")
        q0, p_goal = np.asarray(q0, float), np.asarray(p_goal, float)
        ctx = self._context(kin, other_q, coll_kw, [kin.tcp(q0)[0], p_goal], collision)
        out, ok = self._cart(ctx, q0, p_goal, R_goal, step, collision)
        _count("cart_ok", ok)
        return out, ok

    @_timed("check_s")
    def free(self, kin, q, other_q=None, coll_kw=None):
        _count("check")
        ctx = self._context(kin, other_q, coll_kw, (), True)
        return bool(self._feasible(ctx, np.asarray(q, float)[None])[0])

    @_timed("check_s")
    def segment_free(self, kin, qa, qb, other_q=None, coll_kw=None):
        _count("check")
        ctx = self._context(kin, other_q, coll_kw, (), True)
        return self._segments_free(ctx, [np.asarray(qa, float), np.asarray(qb, float)])

    @_timed("plan_s")
    def plan(self, kin, q0, q1, other_q=None, coll_kw=None, carry=False):
        """Joint path q0 -> q1 (waypoints after q0) or (None, status).  ``carry``: a path that would
        tilt the held object more than CARRY_TILT is re-planned to q1's TCP pose with the tilt held."""
        from zeno_skills.kinematics import CARRY_TILT, max_tilt, tilt

        _count("plan")
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        ctx = self._context(kin, other_q, coll_kw, (), True)
        qs, status = self._plan_js(ctx, q0, q1)
        R0 = kin.tcp(q0)[1]
        R1 = kin.tcp(q1)[1]
        if carry and tilt(R0, R1) <= CARRY_TILT and (qs is None or max_tilt(kin, R0, qs) > CARRY_TILT):
            _count("plan_carry")
            qc, sc = self._plan_tilt(ctx, q0, q1, R1)
            if qc is not None and max_tilt(kin, R0, qc) <= CARRY_TILT:
                qs, status = qc, None
            else:
                qs, status = None, f"carry tilt held: {sc or f'tilts {max_tilt(kin, R0, qc):.2f} rad'}"
        _count("plan_ok", qs is not None)
        return qs, status

    @_timed("plan_s")
    def plan_to(self, kin, q0, p0, p, R, other_q=None, coll_kw=None, carry=False):
        """A blocked Cartesian leg p0 -> (p, R): carrying, a tilt-held path to an IK solution of the
        goal; otherwise a free path to the start of a straight final approach along the goal-frame
        axis the leg follows (its last APPROACH_FRACTION), then that straight line."""
        from zeno_skills.kinematics import CARRY_TILT, max_tilt, tilt

        U = _upstream()
        _count("plan_to")
        q0, p0, p, R = np.asarray(q0, float), np.asarray(p0, float), np.asarray(p, float), np.asarray(R, float)
        ctx = self._context(kin, other_q, coll_kw, [p0, p], True)
        R0 = kin.tcp(q0)[1]
        if carry and tilt(R0, R) <= CARRY_TILT:
            q_goal, ok = self._ik(ctx, p, R, [q0], True)
            if not ok:
                qs, status = None, "no collision-free IK at the goal"
            else:
                qs, status = self._plan_tilt(ctx, q0, q_goal, R)
            if qs is not None and max_tilt(kin, R0, qs) > CARRY_TILT + tilt(R0, R):
                qs, status = None, "held object tilts on the planned leg"
        else:
            d = R.T @ (p - p0)
            L = float(np.linalg.norm(d))
            axis = int(np.argmax(np.abs(d))) if L > 1e-6 else 2
            if not (L > 1e-6 and abs(d[axis]) / L >= U.APPROACH_ALIGN):
                return None, "leg follows no TCP axis: no constrained approach"
            p_pre = p - R[:, axis] * (float(d[axis]) * U.APPROACH_FRACTION)
            q_pre, ok = self._ik(ctx, p_pre, R, [q0], True)
            if not ok:
                qs, status = None, "no collision-free IK at the approach start"
            else:
                leg1, status = self._plan_js(ctx, q0, q_pre)
                qs = None
                if leg1 is not None:
                    leg2, ok2 = self._cart(ctx, leg1[-1], p, R, 0.01, True)
                    qs, status = (leg1 + leg2, None) if ok2 else (None, "straight approach blocked")
        _count("plan_to_ok", qs is not None)
        return qs, status

    @_timed("check_s")
    def clearance(self, kin, Q, targets=(), other_q=None, coll_kw=None):
        """Per row of Q: the arm's smallest distance to the world, capped at CLEAR_CAP (0 when touching).
        Fingers and the objects at the TCP ``targets`` are left out."""
        U = _upstream()
        _count("check")
        kw = dict(coll_kw or {})
        if not kw.get("hand_touches_part"):
            kw["ignore_fingers"] = True
        ctx = self._context(kin, other_q, kw, [np.asarray(p, float) for p in targets], True)
        Q = np.clip(np.atleast_2d(np.asarray(Q, float)), ctx.lo, ctx.hi)
        dist = np.array(self._eval(ctx, Q, "clearance"), float)
        return np.clip(dist, 0.0, U.CLEAR_CAP)

    def penetration(self, kin, q, other_q=None, coll_kw=None):
        """Deepest penetration (m, >= 0) of the robot's spheres into the world at q."""
        ctx = self._context(kin, other_q, coll_kw, (), True)
        return float(self._eval(ctx, np.asarray(q, float)[None], "penetration")[0])


# ------------------------------------------------------------------ helpers
def _dense(qa, qb, res):
    """Waypoints after qa on the straight joint segment to qb, at most ``res`` apart per joint."""
    k = max(1, int(np.ceil(np.max(np.abs(qb - qa)) / res)))
    return [qa + (qb - qa) * u for u in np.linspace(0, 1, k + 1)[1:]]


def _other(side):
    return "left" if side == "right" else "right"


def _objects(ann):
    objs = ann.objects
    if isinstance(objs, dict):
        return [{"name": k, **v} for k, v in objs.items()]
    return list(objs)


def _object_names(ann):
    objs = ann.objects
    return list(objs) if isinstance(objs, dict) else [o["name"] for o in objs]


def _geom_points(m, g):
    """Points spanning geom g in its own frame (mesh vertices, else the corners of its local AABB)."""
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return np.asarray(m.mesh_vert[a : a + n], float)
    c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    return c + corners * h


def install() -> bool:
    """Replace upstream's cuRobo `Motion` and add the planning spheres to the worlds built from now on;
    False on a checkout from before cuRobo (no zeno_skills.motion)."""
    if importlib.util.find_spec("zeno_skills.motion") is None:
        return False
    from robotforge.zenobench import shim

    U = _upstream()
    U.Motion = MujocoMotion
    if add_planning_spheres not in shim.STATE.spec_hooks:
        shim.STATE.spec_hooks.append(add_planning_spheres)
    return True
