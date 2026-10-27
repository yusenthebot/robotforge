"""The Isaac Sim, Isaac Lab and Omniverse modules ZenoBench imports, backed by `world.World`.

`install()` registers them in `sys.modules` under their real names (`isaacsim.core.prims`,
`isaaclab.sim`, `isaaclab.sensors`, `isaaclab.app`, `omni.usd`, `pxr.PhysxSchema`) so ZenoBench runs
unchanged. Only the calls ZenoBench makes are provided; anything else raises a clear error.

The flow mirrors Isaac: `omni.usd.get_context().open_stage()` opens the stage with usd-core, edits go
to that stage, cameras and prims are declared, and `SimulationContext.reset()` builds the MuJoCo world
from the edited stage. Prims and cameras bind to the world lazily, so their order does not matter.

Without PyTorch installed, a small NumPy stand-in serves the handful of `torch` calls ZenoBench makes
(tensors for joint indices, targets and images).
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

from robotforge.zenobench import usd_mjcf
from robotforge.zenobench.world import SUBSTEPS, CameraSpec, World

log = logging.getLogger(__name__)


class _State:
    stage = None
    world: World | None = None
    options = usd_mjcf.Options()
    substeps = SUBSTEPS


STATE = _State()


# ------------------------------------------------------------------ torch stand-in
class Tensor(np.ndarray):
    """NumPy array with the torch methods ZenoBench calls."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def long(self):
        return np.asarray(self, np.int64).view(Tensor)

    def float(self):
        return np.asarray(self, np.float32).view(Tensor)

    def clone(self):
        return self.copy()

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self


def _t(data, dtype=None) -> Tensor:
    return np.array(np.asarray(data) if not isinstance(data, list) else data, dtype=dtype).view(Tensor)


def _torch_module() -> types.ModuleType:
    t = types.ModuleType("torch")
    t.__doc__ = "robotforge.zenobench NumPy stand-in for the torch calls ZenoBench makes"
    t.Tensor = Tensor
    t.float32, t.float64, t.float = np.float32, np.float64, np.float32
    t.int32, t.int64, t.long, t.bool = np.int32, np.int64, np.int64, np.bool_
    t.tensor = lambda data, dtype=None, device=None, **_: _t(data, dtype)
    t.as_tensor = t.tensor
    t.from_numpy = lambda a: np.asarray(a).view(Tensor)
    t.full = lambda shape, value, dtype=None, device=None, **_: np.full(shape, value, dtype or np.float32).view(Tensor)
    t.zeros = lambda *shape, dtype=None, device=None, **_: np.zeros(_shape(shape), dtype or np.float32).view(Tensor)
    t.ones = lambda *shape, dtype=None, device=None, **_: np.ones(_shape(shape), dtype or np.float32).view(Tensor)

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, fn):
            return fn

    t.no_grad = _NoGrad
    t.inference_mode = _NoGrad
    t.cuda = types.SimpleNamespace(is_available=lambda: False)
    return t


def _shape(shape):
    return tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], tuple | list) else shape


def _tensor(a, dtype=np.float32):
    """A torch tensor when real PyTorch is loaded, else the NumPy stand-in."""
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "Tensor", None) is not Tensor:
        return torch.as_tensor(np.asarray(a, dtype=dtype))
    return np.asarray(a, dtype=dtype).view(Tensor)


def _array(x) -> np.ndarray:
    if x is None:
        return None
    if hasattr(x, "detach") and not isinstance(x, np.ndarray):  # a real torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ------------------------------------------------------------------ omni.usd / isaaclab.app
class _UsdContext:
    def open_stage(self, path, *args, **kwargs) -> bool:
        from pxr import Usd

        STATE.stage = Usd.Stage.Open(str(path))
        if STATE.stage is None:
            raise RuntimeError(f"cannot open stage {path}")
        STATE.world = None
        return True

    def get_stage(self):
        return STATE.stage

    def close_stage(self, *args, **kwargs) -> bool:
        STATE.stage, STATE.world = None, None
        return True


_CONTEXT = _UsdContext()


class _App:
    def update(self) -> None:
        pass

    def close(self) -> None:
        if STATE.world is not None:
            STATE.world.close()

    def is_running(self) -> bool:
        return True


class AppLauncher:
    def __init__(self, launcher_args=None, **kwargs):
        self.app = _App()


# ------------------------------------------------------------------ isaaclab.sim
class SimulationCfg:
    def __init__(self, dt=1 / 60, device="cpu", physics_prim_path="/physicsScene", render_interval=1, **kwargs):
        self.dt, self.device, self.render_interval = dt, device, render_interval
        self.physics_prim_path = physics_prim_path
        self.extra = kwargs


class PinholeCameraCfg:
    def __init__(self, focal_length=24.0, clipping_range=(0.01, 1.0e6), horizontal_aperture=20.955, **kwargs):
        self.focal_length, self.clipping_range = focal_length, clipping_range
        self.horizontal_aperture = horizontal_aperture


class SimulationContext:
    def __init__(self, cfg: SimulationCfg | None = None):
        if STATE.stage is None:
            raise RuntimeError("open a stage first (omni.usd.get_context().open_stage)")
        self.cfg = cfg or SimulationCfg()
        STATE.world = World(STATE.stage, dt=self.cfg.dt, substeps=STATE.substeps, options=STATE.options)
        self.world = STATE.world
        for cam in _PENDING_CAMERAS:
            cam._register(self.world)
        _PENDING_CAMERAS.clear()

    def reset(self, soft: bool = False) -> None:
        if self.world.model is None:
            self.world.build()

    def step(self, render: bool = True) -> None:
        self.world.step(1)

    def render(self) -> None:
        pass

    def get_physics_dt(self) -> float:
        return self.cfg.dt

    def is_playing(self) -> bool:
        return True

    def is_stopped(self) -> bool:
        return False

    def play(self) -> None:
        pass

    def pause(self) -> None:
        pass


# ------------------------------------------------------------------ isaaclab.sensors
class _OffsetCfg:
    def __init__(self, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"):
        self.pos, self.rot, self.convention = tuple(pos), tuple(rot), convention


class CameraCfg:
    OffsetCfg = _OffsetCfg

    def __init__(
        self,
        prim_path,
        height=480,
        width=640,
        data_types=("rgb",),
        spawn=None,
        offset=None,
        update_period=0.0,
        **kwargs,
    ):
        self.prim_path, self.height, self.width = prim_path, int(height), int(width)
        self.data_types, self.spawn = list(data_types), spawn or PinholeCameraCfg()
        self.offset, self.update_period = offset or _OffsetCfg(), update_period


_PENDING_CAMERAS: list[Camera] = []


class _CameraData:
    def __init__(self):
        self.output: dict[str, object] = {}


class Camera:
    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self.data = _CameraData()
        self.name = cfg.prim_path
        self._world = None
        if STATE.world is not None and STATE.world.model is None:
            self._register(STATE.world)
        else:
            _PENDING_CAMERAS.append(self)

    def _register(self, world: World) -> None:
        cfg = self.cfg
        parent = cfg.prim_path.rsplit("/", 1)[0]
        body = parent if parent in _bodies(world) else None
        if body is not None and cfg.offset.convention != "world":
            log.warning("camera %s: offset convention %r is read as 'world'", cfg.prim_path, cfg.offset.convention)
        near, far = (tuple(cfg.spawn.clipping_range) + (40.0,))[:2]
        world.add_camera(
            CameraSpec(
                name=self.name,
                height=cfg.height,
                width=cfg.width,
                focal_mm=float(cfg.spawn.focal_length),
                aperture_mm=float(getattr(cfg.spawn, "horizontal_aperture", 20.955)),
                body=body,
                pos=cfg.offset.pos,
                rot=cfg.offset.rot,
                near=near,
                far=far,
            )
        )
        self._world = world

    def set_world_poses_from_view(self, eyes, targets, env_ids=None) -> None:
        self._world.set_view(self.name, _array(eyes)[0], _array(targets)[0])

    def update(self, dt: float = 0.0, force_recompute: bool = False) -> None:
        img = self._world.render(self.name)
        self.data.output["rgb"] = _tensor(img[None], np.uint8)

    def reset(self, env_ids=None) -> None:
        pass


def _bodies(world: World) -> set[str]:
    # before build() only the stage is known: a rigid body prim is a camera mount
    from pxr import Usd, UsdPhysics

    found = set()
    for prim in Usd.PrimRange.Stage(world.stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found.add(str(prim.GetPath()))
    return found


# ------------------------------------------------------------------ isaacsim.core
class ArticulationAction:
    def __init__(self, joint_positions=None, joint_velocities=None, joint_efforts=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_efforts = joint_efforts
        self.joint_indices = joint_indices


def _world() -> World:
    if STATE.world is None or STATE.world.model is None:
        raise RuntimeError("the simulation is not built yet: call SimulationContext.reset() first")
    return STATE.world


class SingleArticulation:
    """The joints under one prim (a robot, or a cabinet with its doors and drawers)."""

    def __init__(self, prim_path: str, name: str = "articulation", **kwargs):
        self.prim_path = prim_path.rstrip("/")
        self.name = name
        self._joints: list[usd_mjcf.JointInfo] | None = None

    def initialize(self, physics_sim_view=None) -> None:
        w = _world()
        prefix = self.prim_path + "/"
        joints = [j for j in w.conv.joints.values() if j.prim.startswith(prefix) or j.child.startswith(prefix)]
        if not joints:
            raise ValueError(f"no joints under {self.prim_path}")
        self._joints = sorted(joints, key=lambda j: j.order)
        planar = {p.body for p in w.conv.planar.values() if p.body.startswith(prefix)}
        bodies = sorted({j.child for j in joints} | planar, key=lambda b: (b.count("/"), b))
        roots = [b for b in w.conv.articulations if b.startswith(prefix)]
        self._root = roots[0] if roots else bodies[0]

    def _ensure(self) -> list[usd_mjcf.JointInfo]:
        if self._joints is None:
            self.initialize()
        return self._joints

    @property
    def dof_names(self) -> list[str]:
        return [j.prim.rsplit("/", 1)[-1] for j in self._ensure()]

    @property
    def num_dof(self) -> int:
        return len(self._ensure())

    def get_dof_index(self, dof_name: str) -> int:
        try:
            return self.dof_names.index(dof_name)
        except ValueError:
            raise KeyError(f"{self.prim_path} has no joint {dof_name}") from None

    def get_joint_positions(self, joint_indices=None):
        w = _world()
        q = np.array([w.joint_q(j.name) for j in self._ensure()])
        return _tensor(q if joint_indices is None else q[_array(joint_indices).astype(int)])

    def get_joint_velocities(self, joint_indices=None):
        w = _world()
        v = np.array([w.joint_v(j.name) for j in self._ensure()])
        return _tensor(v if joint_indices is None else v[_array(joint_indices).astype(int)])

    def apply_action(self, control_actions: ArticulationAction) -> None:
        w = _world()
        joints = self._ensure()
        idx = range(len(joints)) if control_actions.joint_indices is None else _array(control_actions.joint_indices)
        idx = [int(i) for i in np.atleast_1d(np.asarray(list(idx) if isinstance(idx, range) else idx))]
        if control_actions.joint_positions is not None:
            vals = np.atleast_1d(_array(control_actions.joint_positions).astype(float))
            for i, v in zip(idx, vals, strict=True):
                w.set_target(joints[i].name, float(v))
        if control_actions.joint_efforts is not None or control_actions.joint_velocities is not None:
            raise NotImplementedError("only position targets are supported")

    def get_world_pose(self):
        p, q = _world().body_pose(self._root)
        return _tensor(p), _tensor(q)


class SingleRigidPrim:
    def __init__(self, prim_path: str, name: str = "rigid_prim", **kwargs):
        self.prim_path = prim_path
        self.name = name

    def initialize(self, physics_sim_view=None) -> None:
        w = _world()
        if not w.has_body(self.prim_path):
            log.debug("%s is not a rigid body; its pose is read from the stage", self.prim_path)

    def get_world_pose(self):
        w = _world()
        if w.has_body(self.prim_path):
            p, q = w.body_pose(self.prim_path)
        else:
            m = usd_mjcf._mat(_xform_cache().GetLocalToWorldTransform(STATE.stage.GetPrimAtPath(self.prim_path)))
            r, p, _ = usd_mjcf._rigid(m)
            q = usd_mjcf._quat(r)
        return _tensor(p), _tensor(q)

    def get_linear_velocity(self):
        return _tensor(_world().body_velocity(self.prim_path)[0])

    def get_angular_velocity(self):
        return _tensor(_world().body_velocity(self.prim_path)[1])


def _xform_cache():
    from pxr import UsdGeom

    return UsdGeom.XformCache()


# ------------------------------------------------------------------ pxr.PhysxSchema
def _physx_schema() -> types.ModuleType:
    """Applied PhysX schemas without NVIDIA's plugin: the attributes keep their PhysX names."""
    from pxr import Sdf, Usd

    mod = types.ModuleType("pxr.PhysxSchema")

    class _Attr:
        def __init__(self, prim, name):
            self.prim, self.name = prim, name

        def Set(self, value, time=None):  # noqa: N802 - USD naming
            a = self.prim.GetAttribute(self.name)
            if not a:
                a = self.prim.CreateAttribute(self.name, _value_type(value, Sdf))
            return a.Set(value)

        def Get(self, time=None):  # noqa: N802
            a = self.prim.GetAttribute(self.name)
            return a.Get() if a else None

    class _API:
        schema = ""
        prefix = ""

        def __init__(self, prim=None):
            self.prim = prim

        @classmethod
        def Apply(cls, prim):  # noqa: N802
            prim.AddAppliedSchema(cls.schema)
            return cls(prim)

        @classmethod
        def Get(cls, stage, path):  # noqa: N802
            return cls(stage.GetPrimAtPath(path))

        def GetPrim(self):  # noqa: N802
            return self.prim

        def __bool__(self):
            return bool(self.prim) and self.schema in _applied(self.prim)

        def __getattr__(self, attr):
            for head in ("Create", "Get"):
                if attr.startswith(head) and attr.endswith("Attr"):
                    stem = attr[len(head) : -4]
                    name = f"{self.prefix}:{stem[0].lower()}{stem[1:]}"

                    def accessor(default=None, *args, _name=name):
                        a = _Attr(self.prim, _name)
                        if default is not None:
                            a.Set(default)
                        return a

                    return accessor
            raise AttributeError(f"PhysxSchema.{type(self).__name__} has no {attr} in the robotforge stand-in")

    def make(name):
        stem = name[len("Physx") : -len("API")]
        return type(name, (_API,), {"schema": name, "prefix": "physx" + stem})

    def __getattr__(name):
        if name.startswith("Physx") and name.endswith("API"):
            cls = make(name)
            setattr(mod, name, cls)
            return cls
        raise AttributeError(name)

    mod.__getattr__ = __getattr__
    mod._API = _API

    orig_remove, orig_has = Usd.Prim.RemoveAPI, Usd.Prim.HasAPI

    def remove_api(self, schema, *args):
        if isinstance(schema, type) and issubclass(schema, _API):
            return self.RemoveAppliedSchema(schema.schema)
        return orig_remove(self, schema, *args)

    def has_api(self, schema, *args):
        if isinstance(schema, type) and issubclass(schema, _API):
            return schema.schema in _applied(self)
        return orig_has(self, schema, *args)

    Usd.Prim.RemoveAPI = remove_api
    Usd.Prim.HasAPI = has_api
    return mod


def _applied(prim) -> list[str]:
    """Applied schema names as authored, unregistered ones (PhysX without its plugin) included."""
    op = prim.GetMetadata("apiSchemas")
    return list(op.ApplyOperations([])) if op is not None else []


def _value_type(value, Sdf):  # noqa: N803 - the pxr module
    if isinstance(value, bool):
        return Sdf.ValueTypeNames.Bool
    if isinstance(value, int):
        return Sdf.ValueTypeNames.Int
    if isinstance(value, float):
        return Sdf.ValueTypeNames.Float
    if isinstance(value, str):
        return Sdf.ValueTypeNames.Token
    if hasattr(value, "__len__") and len(value) == 3:
        return Sdf.ValueTypeNames.Float3
    raise TypeError(f"no USD type for {value!r}")


# ------------------------------------------------------------------ NumPy 1.x behaviour
def _numpy_compat() -> None:
    """ZenoBench was written against Isaac Sim's NumPy 1.26, where `np.cross` still accepted 2D vectors
    (a scalar z for two of them, a z of 0 for one). NumPy 2 rejects them; restore the old result."""
    if getattr(np.cross, "__robotforge_compat__", False):
        return
    original = np.cross

    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        if axis is None and axisa == axisb == axisc == -1:
            a_, b_ = np.asarray(a), np.asarray(b)
            if a_.shape[-1:] == (2,) and b_.shape[-1:] == (2,):
                return a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
            if a_.shape[-1:] == (2,) or b_.shape[-1:] == (2,):
                a_, b_ = _with_z(a_), _with_z(b_)
                return original(a_, b_)
        return original(a, b, axisa, axisb, axisc, axis)

    cross.__robotforge_compat__ = True
    cross.__doc__ = original.__doc__
    np.cross = cross


def _with_z(v: np.ndarray) -> np.ndarray:
    return np.concatenate([v, np.zeros(v.shape[:-1] + (1,), v.dtype)], -1) if v.shape[-1] == 2 else v


# ------------------------------------------------------------------ install
def _module(name: str, **attrs) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None or not getattr(mod, "__robotforge_shim__", False):
        mod = types.ModuleType(name)
        mod.__robotforge_shim__ = True
        sys.modules[name] = mod
    for k, v in attrs.items():
        setattr(mod, k, v)
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, mod)
    return mod


def install(options: usd_mjcf.Options | None = None, substeps: int = SUBSTEPS) -> None:
    """Register the stand-in modules. Real Isaac modules, if present, are left alone."""
    if options is not None:
        STATE.options = options
    STATE.substeps = substeps
    _numpy_compat()
    for real in ("isaacsim", "isaaclab", "omni"):
        if real in sys.modules and not getattr(sys.modules[real], "__robotforge_shim__", False):
            raise RuntimeError(f"the real {real} is already imported; the MuJoCo stand-in cannot replace it")
    import pxr

    if "torch" not in sys.modules:
        try:
            import importlib.util

            have_torch = importlib.util.find_spec("torch") is not None
        except (ImportError, ValueError):
            have_torch = False
        if not have_torch:
            sys.modules["torch"] = _torch_module()
    if not hasattr(pxr, "PhysxSchema") or not hasattr(pxr.PhysxSchema, "_API"):
        mod = _physx_schema()
        sys.modules["pxr.PhysxSchema"] = mod
        pxr.PhysxSchema = mod
    _module("omni")
    _module("omni.usd", get_context=lambda: _CONTEXT)
    _module("isaaclab")
    _module("isaaclab.app", AppLauncher=AppLauncher)
    _module(
        "isaaclab.sim",
        SimulationContext=SimulationContext,
        SimulationCfg=SimulationCfg,
        PinholeCameraCfg=PinholeCameraCfg,
    )
    _module("isaaclab.sensors", Camera=Camera, CameraCfg=CameraCfg)
    _module("isaacsim")
    _module("isaacsim.core")
    _module("isaacsim.core.prims", SingleArticulation=SingleArticulation, SingleRigidPrim=SingleRigidPrim)
    _module("isaacsim.core.utils")
    _module("isaacsim.core.utils.types", ArticulationAction=ArticulationAction)


def zeno_options() -> usd_mjcf.Options:
    """Conversion settings for ZenoBench scenes (Zeno Malo's base anchor, wheels dragged, not rolling)."""
    import os

    return usd_mjcf.Options(
        planar_anchors=("/World/ZenoMalo/base_anchor",),
        no_collision_bodies=("_wheel_link", "_steering_link"),
        max_hull_vertices=int(os.environ.get("ZENOBENCH_MAXHULLVERT", "0")),
        capsule_pads=("_finger_link",) if os.environ.get("ZENOBENCH_CAPSULE_PADS", "0") == "1" else (),
    )
