"""A converted USD stage simulated in MuJoCo the way PhysX runs it in Isaac Sim.

- One `step()` is one PhysX step (1/120 s in ZenoBench), split into `substeps` MuJoCo steps.
- Joint drives become MuJoCo position actuators. A PhysX acceleration drive scales its gains by the
  joint's inertia, so the gains here are stiffness and damping times the joint's inertia at the start
  pose (`dof_M0`), and the force limit is the drive's maxForce. MuJoCo's implicitfast integrator treats
  the damping implicitly across coupled joints, as PhysX does.
- A pinched object is held by static friction (`_GraspLocks`), which MuJoCo's soft contacts lack.
- Joint speeds are clamped to PhysX's maxJointVelocity after every MuJoCo step, as PhysX's solver does.
- A planar anchor follows its fixed joint frame in the stage (`physics:localPos0` / `localRot0`), so code
  that moves the base by editing the stage, as ZenoBench does, works unchanged.
- Cameras render with MuJoCo's offscreen renderer; mounted ones ride on their link.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.zenobench import usd_mjcf

log = logging.getLogger(__name__)

NOSLIP_ITERATIONS = 0  # 3 holds grasps but costs 4x per step with ~800 resting contacts
IMPRATIO = 10.0  # friction-to-normal impedance: higher means less creep of pinched objects
CONTACT_TIMECONST = 0.003  # s
SUBSTEPS = 8  # per 1/120 s PhysX step: at 4, a pinched can slid out while the arm swung it up
BASE_KP = 1.0e5  # 1/s^2: the base follows its anchor like a stiff joint
BASE_KD = 1.0e3
ISAAC_APERTURE_MM = 20.955  # Isaac Lab's default horizontal aperture
# offset frame (x forward, z up) -> MuJoCo camera frame (looks along -z, y up)
_CAM_FROM_WORLD_CONVENTION = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class CameraSpec:
    name: str
    height: int
    width: int
    focal_mm: float
    aperture_mm: float = ISAAC_APERTURE_MM
    body: str | None = None  # mounted on this body, else free (posed from eye and target)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz, x forward z up
    near: float = 0.05
    far: float = 40.0

    @property
    def fovy_deg(self) -> float:
        hfov = 2 * math.atan(self.aperture_mm / (2 * self.focal_mm))
        return math.degrees(2 * math.atan(math.tan(hfov / 2) * self.height / self.width))


class World:
    def __init__(self, stage, dt: float = 1 / 120, substeps: int = SUBSTEPS, options: usd_mjcf.Options | None = None):
        self.stage = stage
        self.dt = dt
        self.substeps = substeps
        self.options = options or usd_mjcf.Options()
        self.cameras: dict[str, CameraSpec] = {}
        self.free_views: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._trace: _Trace | None = None
        self._recorder: _Recorder | None = None
        self._locks: _GraspLocks | None = None
        self._lock_pairs: list = []

    # ---------------------------------------------------------- build
    def add_camera(self, cam: CameraSpec) -> None:
        if self.model is not None:
            raise RuntimeError("cameras must be added before the world is built (sim.reset())")
        self.cameras[cam.name] = cam

    def build(self) -> None:
        conv = usd_mjcf.convert(self.stage, self.options)
        self.conv = conv
        spec = conv.spec
        for cam in self.cameras.values():
            if cam.body is None:
                continue
            body = spec.body(cam.body)
            if body is None:
                raise ValueError(f"camera {cam.name}: no body {cam.body}")
            c = body.add_camera()
            c.name = cam.name
            c.pos = cam.pos
            r = usd_mjcf._quat_mat(np.asarray(cam.rot, float)) @ _CAM_FROM_WORLD_CONVENTION
            c.quat = usd_mjcf._quat(r)
            c.fovy = cam.fovy_deg
        self._lights(spec)
        self._lock_pairs = _GraspLocks.declare(spec, conv) if _knob("ZENOBENCH_GRASP_LOCK", 1) else []
        drives = [j for j in conv.joints.values() if j.drive is not None]
        planar = [n for p in conv.planar.values() for n in p.joints]
        for name in [j.name for j in drives] + planar:
            a = spec.add_actuator()
            a.name, a.target = name, name
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
            a.gainprm[0], a.biasprm[1] = 1.0, -1.0  # set from the joint inertia once compiled
        w = max([c.width for c in self.cameras.values()] + [640])
        h = max([c.height for c in self.cameras.values()] + [480])
        spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
        spec.visual.quality.shadowsize = 4096
        spec.visual.map.znear = 0.002  # times the model extent (about 10 m): 2 cm, for the wrist cameras
        m = spec.compile()
        m.opt.timestep = self.dt / self.substeps
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = _knob("ZENOBENCH_IMPRATIO", IMPRATIO)
        # pinch grasps: a no-slip pass, so a held object does not creep out of the soft finger contacts
        # while it is carried (MuJoCo 3.15 already gives convex pairs several contact points)
        m.opt.noslip_iterations = int(_knob("ZENOBENCH_NOSLIP", NOSLIP_ITERATIONS))
        if not _knob("ZENOBENCH_MULTICCD", 1):
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
        if not _knob("ZENOBENCH_NATIVECCD", 1):
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
        m.opt.ccd_iterations = int(_knob("ZENOBENCH_CCD_ITERATIONS", m.opt.ccd_iterations))
        margin = _knob("ZENOBENCH_MARGIN", 0.0)
        if margin:
            m.geom_margin[:] = margin
        if _knob("ZENOBENCH_SLEEP", 1):  # resting objects sleep until touched: 4x faster in the kitchen
            m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
        m.opt.iterations = 60
        m.opt.ls_iterations = 30
        # stiff contacts (PhysX contacts are rigid); a time constant needs at least two steps
        m.geom_solref[:] = (max(_knob("ZENOBENCH_SOLREF", CONTACT_TIMECONST), 2.1 * m.opt.timestep), 1.0)
        m.geom_solimp[:, 0] = _knob("ZENOBENCH_SOLIMP0", 0.95)
        m.geom_solimp[:, 1] = _knob("ZENOBENCH_SOLIMP1", 0.99)
        self.model = m
        self.data = mujoco.MjData(m)
        self._index(conv)
        mujoco.mj_forward(m, self.data)
        self._trace = _Trace.from_env(self)
        self._recorder = _Recorder.from_env(self)
        self._locks = _GraspLocks(self, self._lock_pairs) if self._lock_pairs else None
        log.info("world: %d bodies, %d joints, %d geoms (%s)", m.nbody, m.njnt, m.ngeom, conv.stats)

    def _lights(self, spec) -> None:
        """Indoor light without a ceiling lamp model: a bright sky (the house has an open roof, as in
        Isaac's dome-lit scenes), soft ambient light, one shadowed sun and fill lights from the sides."""
        spec.visual.headlight.ambient = (0.42, 0.42, 0.42)
        spec.visual.headlight.diffuse = (0.38, 0.38, 0.38)
        spec.visual.headlight.specular = (0.03, 0.03, 0.03)
        sky = spec.add_texture()
        sky.name, sky.type = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1, sky.rgb2, sky.width, sky.height = (0.86, 0.9, 0.95), (0.62, 0.66, 0.7), 256, 1536
        sun = ((0.35, 0.45, -1.0), 0.45, True)
        fills = [((dx, dy, -0.7), 0.16, False) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        for d, diffuse, shadow in [sun, *fills]:
            light = spec.worldbody.add_light()
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            light.dir = d
            light.pos = (0, 0, 6)
            light.diffuse = (diffuse,) * 3
            light.specular = (0.05, 0.05, 0.05)
            light.castshadow = shadow

    def _index(self, conv: usd_mjcf.Conversion) -> None:
        m = self.model
        self.joint_id = {name: m.joint(name).id for name in (j.name for j in conv.joints.values())}
        drives = [j for j in conv.joints.values() if j.drive is not None]
        self._drive_index = {j.name: i for i, j in enumerate(drives)}
        self._act = np.array([m.actuator(j.name).id for j in drives], int)
        self.targets = np.array([j.drive.target for j in drives], float)
        for j in drives:
            self._set_gains(j.name, j.drive.stiffness, j.drive.damping, j.drive.max_force, j.drive.acceleration)
        self._planar = []
        for p in conv.planar.values():
            jid = [m.joint(n).id for n in p.joints]
            act = np.array([m.actuator(n).id for n in p.joints], int)
            for n in p.joints:
                self._set_gains(n, BASE_KP, BASE_KD, math.inf, True)
            self._planar.append((p, m.jnt_qposadr[jid], act, self.stage.GetPrimAtPath(p.anchor)))
        # PhysX clamps joint speed (physxJoint:maxJointVelocity) inside its solver; Zeno Malo's fingers may
        # move 0.1 m/s. Without it a contact impulse flung a 48 g finger 13 mm open in one step and the
        # pinched object fell out.
        limited = [j for j in conv.joints.values() if math.isfinite(j.max_velocity) and _knob("ZENOBENCH_VMAX", 1)]
        self._vdof = np.array([m.jnt_dofadr[self.joint_id[j.name]] for j in limited], int)
        self._vmax = np.array([j.max_velocity for j in limited], float)
        self._planar_prev = []
        for i, item in enumerate(self._planar):
            q = self._anchor_q(i)
            self.data.qpos[item[1]] = q
            self._planar_prev.append(q)

    def _set_gains(self, name: str, stiffness: float, damping: float, max_force: float, acceleration: bool) -> None:
        m = self.model
        a = m.actuator(name).id
        inertia = float(m.dof_M0[m.jnt_dofadr[m.joint(name).id]]) if acceleration else 1.0
        kp, kv = stiffness * inertia, damping * inertia
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biasprm[a, :3] = (0.0, -kp, -kv)
        if math.isfinite(max_force):
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-max_force, max_force)

    # ---------------------------------------------------------- joints and bodies
    def joint_q(self, name: str) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self.joint_id[name]]])

    def joint_v(self, name: str) -> float:
        return float(self.data.qvel[self.model.jnt_dofadr[self.joint_id[name]]])

    def set_target(self, name: str, value: float) -> None:
        i = self._drive_index.get(name)
        if i is None:
            raise KeyError(f"joint {name} has no drive")
        self.targets[i] = float(value)

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.model.body(name).id
        return self.data.xpos[b].copy(), self.data.xquat[b].copy()

    def body_velocity(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.model.body(name).id
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
        return res[3:].copy(), res[:3].copy()

    def has_body(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0

    # ---------------------------------------------------------- stepping
    def _anchor_q(self, i: int) -> np.ndarray:
        p, qadr, _, prim = self._planar[i]
        pos = prim.GetAttribute("physics:localPos0").Get()
        rot = prim.GetAttribute("physics:localRot0").Get()
        yaw = 2 * math.atan2(rot.GetImaginary()[2], rot.GetReal()) if rot is not None else 0.0
        q = np.array([pos[0] - p.origin[0], pos[1] - p.origin[1], yaw - p.origin[2]])
        cur = self.data.qpos[qadr[2]]
        q[2] = cur + (q[2] - cur + math.pi) % (2 * math.pi) - math.pi  # nearest turn: no spin at +-pi
        return q

    def step(self, n: int = 1) -> None:
        m, d = self.model, self.data
        for _ in range(n):
            if self._trace is not None:
                self._trace.record()
            if self._recorder is not None:
                self._recorder.record()
            if self._locks is not None:
                self._locks.update()
            if len(self._act):
                d.ctrl[self._act] = self.targets
            for i, (_, _, act, _) in enumerate(self._planar):
                q = self._anchor_q(i)
                vel = (q - self._planar_prev[i]) / self.dt
                self._planar_prev[i] = q
                d.ctrl[act] = q + vel * (BASE_KD / BASE_KP)  # velocity feed-forward: no lag while driving
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
                if len(self._vdof):
                    v = d.qvel[self._vdof]
                    d.qvel[self._vdof] = np.clip(v, -self._vmax, self._vmax)

    def contacts_of(self, body: str) -> list[tuple[str, float, float]]:
        """(other geom or body, distance, normal force) of every contact touching a body."""
        m, d = self.model, self.data
        a = m.body(body).id
        out = []
        for i, c in enumerate(d.contact[: d.ncon]):
            g1, g2 = c.geom1, c.geom2
            if m.geom_bodyid[g1] == a:
                other = g2
            elif m.geom_bodyid[g2] == a:
                other = g1
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            label = m.geom(other).name or m.body(m.geom_bodyid[other]).name
            out.append((label, float(c.dist), float(force[0])))
        return out

    # ---------------------------------------------------------- rendering
    def _renderer(self, height: int, width: int) -> mujoco.Renderer:
        key = (height, width)
        if key not in self._renderers:
            r = mujoco.Renderer(self.model, height, width)
            opt = r.scene_option if hasattr(r, "scene_option") else None
            self._renderers[key] = r
            if opt is not None:
                opt.geomgroup[:] = 0
                opt.geomgroup[usd_mjcf.VISUAL_GROUP] = 1
        return self._renderers[key]

    def set_view(self, name: str, eye, target) -> None:
        self.free_views[name] = (np.asarray(eye, float).reshape(3), np.asarray(target, float).reshape(3))

    def render(self, name: str) -> np.ndarray:
        """RGB uint8 image (height, width, 3) of a camera."""
        cam = self.cameras[name]
        r = self._renderer(cam.height, cam.width)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[usd_mjcf.VISUAL_GROUP] = 1
        if cam.body is not None:
            r.update_scene(self.data, camera=name, scene_option=option)
        else:
            eye, target = self.free_views.get(name, (np.array([0.0, -3.0, 2.0]), np.zeros(3)))
            view = mujoco.MjvCamera()
            view.type = mujoco.mjtCamera.mjCAMERA_FREE
            f = target - eye
            dist = float(np.linalg.norm(f)) or 1.0
            f = f / dist
            view.lookat[:] = target
            view.distance = dist
            view.azimuth = math.degrees(math.atan2(f[1], f[0]))
            view.elevation = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
            self.model.vis.global_.fovy = cam.fovy_deg
            r.update_scene(self.data, camera=view, scene_option=option)
        return r.render()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


def _knob(name: str, default: float) -> float:
    """A physics setting that an experiment can override through the environment."""
    import os

    value = os.environ.get(name)
    return float(value) if value else default


class _Trace:
    """Debug trace: `ZENOBENCH_TRACE=<body>[,<body>...]` writes JSON lines (to `ZENOBENCH_TRACE_OUT`,
    default zenobench_trace.jsonl) every `ZENOBENCH_TRACE_EVERY` steps: each body's pose and its contacts
    with the robot, plus the finger joints."""

    def __init__(self, world: World, bodies: list[str], path: str, every: int):
        self.world, self.bodies, self.every, self.tick = world, bodies, every, 0
        self.file = open(path, "w", encoding="utf-8")  # noqa: SIM115 - lives as long as the world
        self.fingers = [n for n in world.joint_id if "finger" in n]

    @classmethod
    def from_env(cls, world: World) -> _Trace | None:
        import os

        names = [b for b in os.environ.get("ZENOBENCH_TRACE", "").split(",") if b]
        names = [b for b in names if world.has_body(b)]
        if not names:
            return None
        path = os.environ.get("ZENOBENCH_TRACE_OUT", "zenobench_trace.jsonl")
        return cls(world, names, path, int(os.environ.get("ZENOBENCH_TRACE_EVERY", "4")))

    def record(self) -> None:
        import json

        self.tick += 1
        if self.tick % self.every:
            return
        w = self.world
        row = {"tick": self.tick, "fingers": {n.rsplit("/", 1)[-1]: round(w.joint_q(n), 4) for n in self.fingers}}
        if w._locks is not None:  # the weld rows of every active grasp lock: force, torque (world frame rows)
            d = w.data
            for palm, (obj, eq) in w._locks.locked.items():
                rows = np.flatnonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq))
                names = {"palm": w._locks._name(palm), "obj": w._locks._name(obj)}
                row.setdefault("locks", []).append({**names, "efc": np.round(d.efc_force[rows], 2).tolist()})
        for b in self.bodies:
            p, q = w.body_pose(b)
            row[b.rsplit("/", 1)[-1]] = {
                "pos": np.round(p, 4).tolist(),
                "quat": np.round(q, 4).tolist(),
                "contacts": [(o[-60:], round(dist, 4), round(f, 2)) for o, dist, f in w.contacts_of(b)],
            }
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()


class _Recorder:
    """Debug snapshot for offline replays: `ZENOBENCH_RECORD_AT=<tick>` saves the compiled model (.mjb) and
    the full state at that tick, then the controls of the next `ZENOBENCH_RECORD_TICKS` ticks (default
    1200), to `ZENOBENCH_RECORD_OUT` (a path stem). Replaying them with other settings tests a physics
    change on the exact moment a task went wrong, in seconds instead of a whole episode."""

    def __init__(self, world: World, at: int, ticks: int, stem: str):
        self.world, self.at, self.ticks, self.stem = world, at, ticks, stem
        self.tick, self.ctrl, self.state = 0, [], None

    @classmethod
    def from_env(cls, world: World) -> _Recorder | None:
        import os

        at = int(os.environ.get("ZENOBENCH_RECORD_AT", "0"))
        if at <= 0:
            return None
        stem = os.environ.get("ZENOBENCH_RECORD_OUT", "zenobench_record")
        return cls(world, at, int(os.environ.get("ZENOBENCH_RECORD_TICKS", "1200")), stem)

    def record(self) -> None:
        self.tick += 1
        w = self.world
        m, d = w.model, w.data
        if self.tick == self.at:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            self.state = np.zeros(mujoco.mj_stateSize(m, spec))
            mujoco.mj_getState(m, d, self.state, spec)
            mujoco.mj_saveModel(m, self.stem + ".mjb", None)
        if self.state is None or self.tick < self.at:
            return
        if len(w._act):
            d.ctrl[w._act] = w.targets
        for i, (_, _, act, _) in enumerate(w._planar):
            q = w._anchor_q(i)
            d.ctrl[act] = q + (q - w._planar_prev[i]) / w.dt * (BASE_KD / BASE_KP)
        self.ctrl.append(d.ctrl.copy())
        if len(self.ctrl) == self.ticks:
            np.savez(
                self.stem + ".npz",
                state=self.state,
                ctrl=np.array(self.ctrl),
                substeps=w.substeps,
                vdof=w._vdof,
                vmax=w._vmax,
            )
            self.state = None


LOCK_MIN_FORCE = 2.0  # N on each finger before a pinch counts as a grasp
LOCK_MAX_SLIP = (0.05, 0.6)  # m/s and rad/s of the object relative to the palm, above which it is not locked
LOCK_PATCH_RADIUS = 0.02  # m: the pads' contact patch, for the friction torque they can hold
LOCK_COOLDOWN = 30  # ticks after a slip before the object may lock again
LOCK_SMOOTHING = 1 / 12  # per tick: the grasp's load is averaged over about 0.1 s
LOCK_RELEASE_TICKS = 6  # a finger must press too lightly this many ticks in a row before the lock lets go


class _GraspLocks:
    """Static friction for pinch grasps.

    PhysX friction is static: below the friction limit a pinched object does not move in the fingers.
    MuJoCo's contacts are soft, so a squeezed object creeps, and a slightly tapered one (a bottle) is
    pressed out of the pinch even with noslip. So while both fingers of a gripper press on the same object
    and it rests in the hand, a weld (declared at build time, inactive) holds it to the palm at its current
    pose. The weld lets go when either finger stops pressing (the hand opens, the object is put down) and
    when it carries more than friction could (mu times the squeeze for force, times the pad patch radius
    for torque): the object then slips as it would in PhysX, and may lock again once it rests.
    """

    def __init__(self, world: World, pairs: list[tuple[str, tuple[str, ...], str, str]]):
        m = world.model
        self.world = world
        self.grippers: dict[int, tuple[int, ...]] = {}
        self.eq: dict[tuple[int, int], int] = {}
        for palm, fingers, obj, eq_name in pairs:
            p = m.body(palm).id
            self.grippers[p] = tuple(m.body(f).id for f in fingers)
            self.eq[(p, m.body(obj).id)] = m.equality(eq_name).id
        self.finger_of = {f: p for p, fs in self.grippers.items() for f in fs}
        self.objects = {o for _, o in self.eq}
        self.locked: dict[int, tuple[int, int]] = {}  # palm -> (object, equality id)
        self.load: dict[int, np.ndarray] = {}  # palm -> smoothed (force, torque) the grasp must supply
        self.weak: dict[int, int] = {}  # palm -> ticks a finger has pressed too lightly in a row
        self.nudged: set[int] = set()  # objects given a wake-up push this tick
        self.free_dof = {
            o: int(m.jnt_dofadr[m.body_jntadr[o]])
            for o in self.objects
            if m.body_jntnum[o] == 1 and m.jnt_type[m.body_jntadr[o]] == mujoco.mjtJoint.mjJNT_FREE
        }
        self.cooldown: dict[tuple[int, int], int] = {}

    @staticmethod
    def declare(spec, conv: usd_mjcf.Conversion) -> list[tuple[str, tuple[str, ...], str, str]]:
        """One inactive weld per (gripper palm, free body); grippers are bodies with two or more sliding
        children (the fingers)."""
        children: dict[str, list[str]] = {}
        for j in conv.joints.values():
            if j.kind == "slide":
                parent = spec.body(j.child).parent.name
                children.setdefault(parent, []).append(j.child)
        grippers = {palm: tuple(f) for palm, f in children.items() if len(f) >= 2}
        free = [b.name for b in spec.bodies if any(jt.type == mujoco.mjtJoint.mjJNT_FREE for jt in b.joints)]
        pairs = []
        for palm, fingers in grippers.items():
            for obj in free:
                e = spec.add_equality()
                e.name = f"grasp:{palm}|{obj}"
                e.type, e.objtype = mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
                e.name1, e.name2, e.active = palm, obj, False
                e.solref = (0.004, 1.0)
                pairs.append((palm, fingers, obj, e.name))
        return pairs

    def _squeeze(self) -> dict[tuple[int, int], list[float]]:
        """Per (finger, object): normal force and the largest friction coefficient of their contacts."""
        m, d = self.world.model, self.world.data
        out: dict[tuple[int, int], list[float]] = {}
        if d.ncon == 0:
            return out
        geoms = d.contact.geom[: d.ncon]
        b1, b2 = m.geom_bodyid[geoms[:, 0]], m.geom_bodyid[geoms[:, 1]]
        force = np.zeros(6)
        for i in np.flatnonzero(np.isin(b1, list(self.finger_of)) | np.isin(b2, list(self.finger_of))):
            finger, obj = (b1[i], b2[i]) if b1[i] in self.finger_of else (b2[i], b1[i])
            if obj not in self.objects:
                continue
            mujoco.mj_contactForce(m, d, int(i), force)
            entry = out.setdefault((int(finger), int(obj)), [0.0, 0.0])
            entry[0] += float(force[0])
            entry[1] = max(entry[1], float(d.contact.friction[i][0]))
        return out

    def _pinched(self, palm: int, squeeze, threshold: float = LOCK_MIN_FORCE) -> list[tuple[float, int, float]]:
        fingers = self.grippers[palm]
        found = []
        for obj in self.objects:
            sides = [squeeze.get((f, obj)) for f in fingers]
            if all(s is not None and s[0] > threshold for s in sides):
                found.append((sum(s[0] for s in sides), obj, max(s[1] for s in sides)))
        return sorted(found, reverse=True)

    def update(self) -> None:
        d = self.world.data
        for obj in self.nudged:
            d.xfrc_applied[obj] = 0.0
        self.nudged.clear()
        squeeze = self._squeeze()
        for key in list(self.cooldown):
            self.cooldown[key] -= 1
            if self.cooldown[key] <= 0:
                del self.cooldown[key]
        for palm in self.grippers:
            pinched = self._pinched(palm, squeeze)
            if palm in self.locked:
                obj, eq = self.locked[palm]
                # a contact may flicker for a tick; a hand that opens stays open
                held = next((p for p in self._pinched(palm, squeeze, LOCK_MIN_FORCE / 2) if p[1] == obj), None)
                self.weak[palm] = 0 if held is not None else self.weak.get(palm, 0) + 1
                if held is None and self.weak[palm] < LOCK_RELEASE_TICKS:
                    continue
                load = None if held is None else self._overload(palm, obj, held[0], held[2])
                if held is None or load is not None:
                    d.eq_active[eq] = 0
                    del self.locked[palm]
                    # a weld switching off does not wake a sleeping object: one tick of a negligible push does,
                    # so a released object falls instead of hanging where the hand left it
                    d.xfrc_applied[obj, 2] = -1e-9
                    self.nudged.add(obj)
                    reason = "fingers let go" if held is None else load
                    log.info("grasp lock: %s releases %s (%s)", self._name(palm), self._name(obj), reason)
                    if held is not None:
                        self.cooldown[(palm, obj)] = LOCK_COOLDOWN
                continue
            for _, obj, _ in pinched:
                if (palm, obj) in self.cooldown or any(o == obj for o, _ in self.locked.values()):
                    continue
                if self._resting(palm, obj):
                    self._lock(palm, obj)
                break

    def _resting(self, palm: int, obj: int) -> bool:
        m, d = self.world.model, self.world.data
        a, b = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, palm, a, 0)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, obj, b, 0)
        # the palm's velocity carried to the object's position, against the object's own
        lever = d.xpos[obj] - d.xpos[palm]
        palm_there = a[3:] + np.cross(a[:3], lever)
        return bool(
            np.linalg.norm(b[3:] - palm_there) < LOCK_MAX_SLIP[0] and np.linalg.norm(b[:3] - a[:3]) < LOCK_MAX_SLIP[1]
        )

    def _lock(self, palm: int, obj: int) -> None:
        m, d = self.world.model, self.world.data
        eq = self.eq[(palm, obj)]
        rot = d.xmat[palm].reshape(3, 3)
        inverse, relative = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, d.xquat[palm])
        mujoco.mju_mulQuat(relative, inverse, d.xquat[obj])
        m.eq_data[eq, 0:3] = 0.0
        m.eq_data[eq, 3:6] = rot.T @ (d.xpos[obj] - d.xpos[palm])
        m.eq_data[eq, 6:10] = relative
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self.locked[palm] = (obj, eq)
        self.load.pop(palm, None)
        self.weak.pop(palm, None)
        log.info("grasp lock: %s holds %s", self._name(palm), self._name(obj))

    def _overload(self, palm: int, obj: int, squeeze: float, mu: float) -> str | None:
        """Why friction could not hold the object, or None while it can.

        The demand is what the grasp must supply: every constraint force on the object (the weld and the
        pads together, so their internal squeeze cancels) minus its other contacts (a table it rests on).
        It is smoothed over about 0.1 s, since the palm's drive steps jolt it with brief spikes that a real
        pinch rides out, while a sustained load (a heavy or levered object) still makes it slip."""
        dof = self.free_dof.get(obj)
        if dof is None:
            return None
        demand = self.world.data.qfrc_constraint[dof : dof + 6] - self._other_contacts(palm, obj)
        now = np.array([np.linalg.norm(demand[:3]), np.linalg.norm(demand[3:6])])
        load = self.load[palm] = self.load.get(palm, now) + LOCK_SMOOTHING * (now - self.load.get(palm, now))
        force, torque = float(load[0]), float(load[1])
        if force > mu * squeeze:
            return f"slip: {force:.1f} N > {mu * squeeze:.1f} N"
        if torque > mu * squeeze * LOCK_PATCH_RADIUS:
            return f"twist: {torque:.2f} N m > {mu * squeeze * LOCK_PATCH_RADIUS:.2f} N m"
        return None

    def _other_contacts(self, palm: int, obj: int) -> np.ndarray:
        """The object's contacts with anything but this hand's fingers, in its free joint's coordinates
        (force in the world frame, torque about the body origin in the body frame)."""
        m, d = self.world.model, self.world.data
        out = np.zeros(6)
        if d.ncon == 0:
            return out
        geoms = d.contact.geom[: d.ncon]
        b1, b2 = m.geom_bodyid[geoms[:, 0]], m.geom_bodyid[geoms[:, 1]]
        other = np.where(b1 == obj, b2, b1)
        mine = ((b1 == obj) | (b2 == obj)) & ~np.isin(other, self.grippers[palm])
        origin, rot = d.xpos[obj], d.xmat[obj].reshape(3, 3)
        f6 = np.zeros(6)
        for i in np.flatnonzero(mine):
            c = d.contact[int(i)]
            mujoco.mj_contactForce(m, d, int(i), f6)
            frame = c.frame.reshape(3, 3)
            force = frame.T @ f6[:3]  # on geom2, pushed away from geom1
            torque = frame.T @ f6[3:6]
            sign = 1.0 if b2[i] == obj else -1.0
            out[:3] += sign * force
            out[3:6] += rot.T @ (sign * (np.cross(c.pos - origin, force) + torque))
        return out

    def _name(self, body: int) -> str:
        return self.world.model.body(body).name.rsplit("/", 1)[-1]
