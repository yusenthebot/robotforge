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
# An anchor moving faster than this is a placement, not a drive: ZenoBench drives at most 0.35 m/s and
# 0.8 rad/s, and puts the robot at an episode's start by dragging the anchor at 2.4 m/s straight through walls.
PLACE_SPEED = 1.5  # m/s
PLACE_TURN = 3.0  # rad/s
ISAAC_APERTURE_MM = 20.955  # Isaac Lab's default horizontal aperture
SUN_HEIGHT = 2.5  # m above the robot's base: the shadowed sun's position, which centres its shadow map
SUN_REACH = 5.0  # m around it that the shadow map covers (4096 texels: about 2.4 mm each)
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
    def __init__(
        self,
        stage,
        dt: float = 1 / 120,
        substeps: int = SUBSTEPS,
        options: usd_mjcf.Options | None = None,
        spec_hooks=(),
    ):
        self.stage = stage
        self.spec_hooks = tuple(spec_hooks)  # (spec, conversion) -> None, run just before the spec compiles
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
        self._film: _Film | None = None
        self._sun_tracks = False
        self._started = False
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
        self._lights(spec, conv)
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
        for hook in self.spec_hooks:
            hook(spec, conv)
        m = spec.compile()
        if self._sun_tracks:  # the shadow map spans SUN_REACH around the robot instead of the whole model
            m.vis.map.shadowclip = min(1.0, SUN_REACH / max(float(m.stat.extent), 1e-6))
        m.opt.timestep = self.dt / self.substeps
        if _knob("ZENOBENCH_DAMPING", 1):
            self._damp(m, conv)
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
        self._film = _Film.from_env(self)
        self._locks = _GraspLocks(self, self._lock_pairs) if self._lock_pairs else None
        log.info("world: %d bodies, %d joints, %d geoms (%s)", m.nbody, m.njnt, m.ngeom, conv.stats)

    @staticmethod
    def _damp(m: mujoco.MjModel, conv: usd_mjcf.Conversion) -> None:
        """PhysX damps a rigid body's velocity at a rate (1/s) whatever its mass; a MuJoCo joint damps by a
        force, so each free dof gets the rate times its mass or inertia. A cherry tomato that ZenoBench
        gives angular damping 3 stops rolling within a second, as it does in Isaac."""
        for body, (linear, angular) in conv.damping.items():
            j = m.body_jntadr[m.body(body).id]
            dof = m.jnt_dofadr[j]
            m.dof_damping[dof : dof + 3] = linear * m.dof_M0[dof : dof + 3]
            m.dof_damping[dof + 3 : dof + 6] = angular * m.dof_M0[dof + 3 : dof + 6]

    def _lights(self, spec, conv: usd_mjcf.Conversion) -> None:
        """Indoor light without a ceiling lamp model: a bright sky (the house has an open roof, as in
        Isaac's dome-lit scenes), soft ambient light, one shadowed sun and fill lights from the sides.

        One shadow map over the whole house gives the robot's shadow stair-stepped, blotchy edges, so the
        sun rides along with the robot's base (its direction stays fixed in the world) and its shadow map
        covers only the few metres around the robot, in detail."""
        spec.visual.headlight.ambient = (0.42, 0.42, 0.42)
        spec.visual.headlight.diffuse = (0.38, 0.38, 0.38)
        spec.visual.headlight.specular = (0.03, 0.03, 0.03)
        sky = spec.add_texture()
        sky.name, sky.type = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1, sky.rgb2, sky.width, sky.height = (0.86, 0.9, 0.95), (0.62, 0.66, 0.7), 256, 1536
        sun = ((0.35, 0.45, -1.0), 0.45, True)
        fills = [((dx, dy, -0.7), 0.16, False) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        base = next((p.body for p in conv.planar.values()), None)
        self._sun_tracks = False
        for d, diffuse, shadow in [sun, *fills]:
            if shadow and base is not None:
                # a tracking light keeps its world offset and direction: give both in the base's frame
                rot = _world_rotation(spec.body(base))
                light = spec.body(base).add_light()
                light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
                light.pos, light.dir = rot.T @ (0.0, 0.0, SUN_HEIGHT), rot.T @ np.asarray(d)
                self._sun_tracks = True
            else:
                light = spec.worldbody.add_light()
                light.pos, light.dir = (0, 0, 6), d
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
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
        # the robot's geoms, by base: switched off while the base is being placed
        self._robot_geoms = []
        for p, *_ in self._planar:
            root = m.body_rootid[m.body(p.body).id]
            g = np.flatnonzero(m.body_rootid[m.geom_bodyid] == root)
            self._robot_geoms.append((g, m.geom_contype[g].copy(), m.geom_conaffinity[g].copy()))
        self._placing = [False] * len(self._planar)

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
    def _place(self, i: int, q: np.ndarray) -> None:
        """Put the base where its anchor is, passing through whatever is in between.

        In PhysX the anchor is a rigid joint that wins over contacts, so a placement drag ends with the robot
        at the start even through a wall; here the wall would win and stop the robot short of it (inside
        it, every later motion refused). So the base follows the anchor exactly, and the robot collides with
        nothing until the anchor slows to a drive again: no arm flung by a wall, nothing knocked over."""
        m, d = self.model, self.data
        planar, qadr, act, _ = self._planar[i]
        if not self._placing[i]:
            self._placing[i] = True
            g = self._robot_geoms[i][0]
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
            log.info("base placement: the robot passes through obstacles until it arrives")
        d.qpos[qadr] = q
        d.qvel[[m.jnt_dofadr[m.joint(n).id] for n in planar.joints]] = 0.0
        d.ctrl[act] = q

    def _end_place(self, i: int) -> None:
        g, contype, conaffinity = self._robot_geoms[i]
        self.model.geom_contype[g] = contype
        self.model.geom_conaffinity[g] = conaffinity
        self._placing[i] = False
        x, y, yaw = self._planar_prev[i] + np.asarray(self._planar[i][0].origin)
        log.info("base placement: arrived at (%.2f, %.2f, %.0f deg); contacts back on", x, y, math.degrees(yaw))

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
        if not self._started:
            self._started = True
            self._jitter_start()
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
                jump = q - self._planar_prev[i]
                self._planar_prev[i] = q
                if math.hypot(jump[0], jump[1]) > PLACE_SPEED * self.dt or abs(jump[2]) > PLACE_TURN * self.dt:
                    self._place(i, q)
                    continue
                if self._placing[i]:
                    self._end_place(i)
                d.ctrl[act] = q + jump / self.dt * (BASE_KD / BASE_KP)  # velocity feed-forward: no lag
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
                if len(self._vdof):
                    v = d.qvel[self._vdof]
                    d.qvel[self._vdof] = np.clip(v, -self._vmax, self._vmax)
            if self._film is not None:
                self._film.record()

    def _jitter_start(self) -> None:
        """`ZENOBENCH_SEED=<n>` (n > 0) nudges every free object before the first step: up to
        `ZENOBENCH_JITTER` (default "1,0.5": mm along the floor, degrees about the vertical) from where the
        scene puts it, the same for the same seed. One exact start shows one outcome of a task whose
        outcome may hinge on a degree; a sweep over seeds shows how often it succeeds."""
        import os

        seed = int(os.environ.get("ZENOBENCH_SEED", "0") or 0)
        if seed <= 0:
            return
        parts = [float(v) for v in os.environ.get("ZENOBENCH_JITTER", "1,0.5").split(",")]
        if len(parts) != 2:
            raise ValueError(f"ZENOBENCH_JITTER is 'mm,deg', not {os.environ['ZENOBENCH_JITTER']!r}")
        mm, deg = parts
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        turned = np.zeros(4)
        count = 0
        for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            a = m.jnt_qposadr[j]
            d.qpos[a : a + 2] += rng.uniform(-mm, mm, 2) / 1000.0
            half = math.radians(rng.uniform(-deg, deg)) / 2
            mujoco.mju_mulQuat(turned, np.array([math.cos(half), 0.0, 0.0, math.sin(half)]), d.qpos[a + 3 : a + 7])
            d.qpos[a + 3 : a + 7] = turned
            count += 1
        mujoco.mj_forward(m, d)
        log.info("start jitter: seed %d moved %d free objects (up to %g mm, %g deg)", seed, count, mm, deg)

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
        if self._film is not None:
            self._film.close()
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


def _turn(a: float, b: float) -> float:
    """The shortest turn in degrees from angle a to angle b."""
    return (b - a + 180.0) % 360.0 - 180.0


def _world_rotation(body) -> np.ndarray:
    """A spec body's orientation in the world at the start pose (joints at zero)."""
    rot = np.eye(3)
    while body is not None and body.name != "world":
        rot = usd_mjcf._quat_mat(np.asarray(body.quat, float)) @ rot
        body = body.parent
    return rot


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


class _Film:
    """Director's view: `ZENOBENCH_FILM=<out.mp4>` films the run at 30 frames per simulated second from a
    camera that follows a body, or the middle of several (`ZENOBENCH_FILM_TARGET`, prim paths separated
    by commas; default the robot's base), from a fixed angle in the robot's frame (`ZENOBENCH_FILM_VIEW`:
    azimuth and elevation in degrees, distance in m; azimuth 0 faces the robot's front, 90 its left).
    ZenoBench's own cameras ride behind the robot, whose body then hides what its hands are doing.
    Walls and fixed furniture never block the view: when they would, the camera swings round to the
    nearest clear angle, and only moves in closer when there is none (a corridor).
    `ZENOBENCH_FILM_SIZE` is width x height (default 960x540)."""

    EVERY = 4  # ticks per frame: 30 frames per second at 120 ticks per second
    SMOOTHING = 0.06  # per frame: the camera trails its target by about half a second
    FOVY = 45.0
    SWINGS = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)  # degrees off the chosen angle
    CLEAR = 0.8  # of the chosen distance free of walls: an angle that counts as a clear view
    CLEARANCE = 0.15  # m kept between the camera and a wall
    NEAR = 0.5  # m around the target where fixed things are what the robot works at (a table), not walls

    def __init__(self, world: World, path: str, targets: list[int], heading: int | None, view, size):
        import imageio.v2 as imageio

        self.world, self.targets, self.heading = world, targets, heading
        self.azimuth, self.elevation, self.distance = view
        self.width, self.height = size
        m = world.model
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), self.width)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), self.height)
        self.writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8, macro_block_size=2)
        self.tick, self.lookat, self.yaw = 0, None, None
        self.swing, self.reach = 0.0, self.distance
        self.path = path
        self._groups = np.zeros(6, np.uint8)
        self._groups[usd_mjcf.COLLISION_GROUP] = 1
        self._hit = np.zeros(1, np.int32)

    @classmethod
    def from_env(cls, world: World) -> _Film | None:
        import atexit
        import os

        path = os.environ.get("ZENOBENCH_FILM", "")
        if not path:
            return None
        base = next((p.body for p, *_ in world._planar), None)
        names = [n for n in os.environ.get("ZENOBENCH_FILM_TARGET", "").split(",") if n] or ([base] if base else [])
        missing = [n for n in names if not world.has_body(n)]
        if not names or missing:
            raise ValueError(f"ZENOBENCH_FILM_TARGET: no body {missing or '(none given and no robot base)'}")
        view = [float(v) for v in os.environ.get("ZENOBENCH_FILM_VIEW", "35,25,2.2").split(",")]
        size = [int(v) for v in os.environ.get("ZENOBENCH_FILM_SIZE", "960x540").lower().split("x")]
        if len(view) != 3 or len(size) != 2:
            raise ValueError("ZENOBENCH_FILM_VIEW is azimuth,elevation,distance; ZENOBENCH_FILM_SIZE is WxH")
        m = world.model
        heading = m.body(base).id if base else None
        film = cls(world, path, [m.body(n).id for n in names], heading, view, size)
        atexit.register(film.close)
        log.info("film: %s, following %s", path, ", ".join(n.rsplit("/", 1)[-1] for n in names))
        return film

    def record(self) -> None:
        self.tick += 1
        if self.writer is None or self.tick % self.EVERY:
            return
        d = self.world.data
        point = np.mean([d.xpos[b] for b in self.targets], axis=0)
        yaw = 0.0
        if self.heading is not None:
            x = d.xmat[self.heading].reshape(3, 3)[:, 0]
            yaw = math.degrees(math.atan2(x[1], x[0]))
        if self.lookat is None:
            self.lookat, self.yaw = point.copy(), yaw
        else:
            self.lookat += self.SMOOTHING * (point - self.lookat)
            self.yaw += self.SMOOTHING * _turn(self.yaw, yaw)
        # a free camera's azimuth is the direction it looks in: back at the robot from the chosen side
        looking = self.yaw + self.azimuth + 180.0
        clear = self.CLEAR * self.distance
        if self._free(looking) >= clear:
            target = 0.0  # the chosen angle is clear (again)
        elif self._free(looking + self.swing) >= clear:
            target = self.swing  # stay at the clear angle found before
        else:
            reach = {s: self._free(looking + s) for s in self.SWINGS}
            target = next((s for s in self.SWINGS if reach[s] >= clear), max(self.SWINGS, key=reach.get))
        self.swing += self.SMOOTHING * _turn(self.swing, target)
        free = self._free(looking + self.swing)
        # pull in at once when a wall comes between (never film its back), drift back out slowly
        self.reach = free if free < self.reach else self.reach + self.SMOOTHING * (free - self.reach)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        cam.azimuth = looking + self.swing
        cam.elevation = -self.elevation
        cam.distance = max(self.reach, 0.2)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[usd_mjcf.VISUAL_GROUP] = 1
        w = self.world
        w.model.vis.global_.fovy = self.FOVY
        r = w._renderer(self.height, self.width)
        r.update_scene(d, camera=cam, scene_option=option)
        self.writer.append_data(r.render())

    def _free(self, azimuth: float) -> float:
        """How far the camera can back away from its target, looking along `azimuth`, before a wall or
        fixed furniture (moving bodies, the robot and what it holds, never count, nor does the furniture
        right at the target: the target may sit under the tabletop the robot works at)."""
        m, d = self.world.model, self.world.data
        a, e = math.radians(azimuth), math.radians(-self.elevation)
        back = -np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        gone = 0.0
        for _ in range(16):
            x = mujoco.mj_ray(m, d, self.lookat + gone * back, back, self._groups, 1, -1, self._hit)
            if x < 0 or gone + x >= self.distance:
                return self.distance
            gone += x
            if gone > self.NEAR and m.body_weldid[m.geom_bodyid[self._hit[0]]] == 0:  # welded to the world
                return max(0.0, gone - self.CLEARANCE)
            gone += 1e-3
        return self.distance

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            log.info("film: wrote %s", self.path)


LOCK_MIN_FORCE = 2.0  # N on each finger before a pinch counts as a grasp
LOCK_OPPOSED = 0.5  # the two fingers' pushes must point at least 120 deg apart (cosine below -0.5)
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

    def _squeeze(self) -> dict[tuple[int, int], list]:
        """Per (finger, object): normal force, the largest friction coefficient of their contacts, and the
        summed normal push on the object (world frame)."""
        m, d = self.world.model, self.world.data
        out: dict[tuple[int, int], list] = {}
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
            entry = out.setdefault((int(finger), int(obj)), [0.0, 0.0, np.zeros(3)])
            entry[0] += float(force[0])
            entry[1] = max(entry[1], float(d.contact.friction[i][0]))
            # the contact normal points from geom1 to geom2
            entry[2] = entry[2] + (1.0 if b2[i] == obj else -1.0) * float(force[0]) * d.contact.frame[i][:3]
        return out

    def _pinched(self, palm: int, squeeze, threshold: float = LOCK_MIN_FORCE) -> list[tuple[float, int, float]]:
        """(squeeze, object, mu) of the objects both fingers press from opposite sides, firmest first.

        Closed fingertips pushing down on an object (a push or a roll) press it too, but from the same side:
        that is no pinch, and locking it would drag the object along with the hand."""
        fingers = self.grippers[palm]
        found = []
        for obj in self.objects:
            sides = [squeeze.get((f, obj)) for f in fingers]
            if not all(s is not None and s[0] > threshold for s in sides):
                continue
            pushes = [s[2] / max(float(np.linalg.norm(s[2])), 1e-9) for s in sides]
            if float(pushes[0] @ pushes[1]) > -LOCK_OPPOSED:
                continue
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
