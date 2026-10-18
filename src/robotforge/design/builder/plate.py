"""A template or library robot carried on a plate, so parts and sensors can go on a robot that is not built from parts.

Parts and sensors are added only to robots built from parts. Told to rebuild a Go2 from parts to give it a lidar, an
agent would have replaced the dog with a 1.1 kg puck; the gallery's go2_arm shows the way that keeps it: a thin plate
is the robot's base, the Go2 hangs under it as a mounted part and walks on its own gait, and new parts go on the
plate. This works out that build for a robot already in the world, from where its body's top is, so the plate sits
on its back (or, for a robot fixed to the floor, under it).
"""

from __future__ import annotations

import itertools
from typing import Any

import mujoco
import numpy as np

THICK = 0.012  # m: the plate's thickness
GAP = 0.0015  # m between the plate and the robot's top
SHARE = 0.6  # of the robot's body length and width the plate covers
SIDE = (0.06, 0.3)  # m: the plate's length and width stay within these
MASS = (0.02, 0.4)  # kg: about 3% of the robot, within these
FIXED_PLATE = [0.2, 0.2, THICK]  # m: the plate under a robot fixed to the floor
PROBES = 7  # rays a side, down onto the robot over the plate's area
MAX_LAYERS = 8  # shapes one of them passes through on its way down


def mount_of(entry: Any) -> dict[str, Any] | None:
    """The mount that carries this world entry's robot as a part, or None for a robot from a file (only templates
    and library robots can be mounted)."""
    if entry.template is not None:
        return {"template": entry.template, **({"params": dict(entry.params)} if entry.params else {})}
    if entry.library is not None:
        return {"library": entry.library, **({"params": dict(entry.params)} if entry.params else {})}
    return None


def mounts(build: dict[str, Any], mount: dict[str, Any]) -> bool:
    """Whether `build` carries the robot `mount` names (the same template or library robot) on one of its parts."""
    key = "template" if "template" in mount else "library"
    parts = build.get("parts", []) if isinstance(build, dict) else []
    return any(
        isinstance(p, dict) and isinstance(p.get("robot"), dict) and p["robot"].get(key) == mount[key] for p in parts
    )


def carried(model: mujoco.MjModel, robot: str, mount: dict[str, Any], build: dict[str, Any] | None = None) -> dict:
    """The build that keeps `robot` (compiled in `model`, its bodies named <robot>/...) as it is, hung under a plate,
    with `build`'s robot-wide settings, its base's sensors on the plate and its parts on the plate too."""
    build = build or {}
    root = _root(model, robot)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[key] if key >= 0 else model.qpos0
    mujoco.mj_kinematics(model, data)
    corners = _corners(model, data, root)
    mass = round(float(np.clip(0.03 * model.body_subtreemass[root], *MASS)), 3)
    base: dict[str, Any] = {"mass": mass}  # a box, the base's own shape
    name = _free_name(mount.get("template") or mount.get("library") or "body", build)
    if _floating(model, root):
        low, high = corners.min(axis=0), corners.max(axis=0)
        size = [round(float(np.clip(SHARE * (high[i] - low[i]), *SIDE)), 2) for i in (0, 1)]
        base["size"] = [*size, THICK]
        top = _top_under(model, data, root, size[0] / 2, size[1] / 2, float(high[2]))
        at: Any = [0, 0, -round(top + THICK / 2 + GAP, 4)]
    else:
        base.update(size=FIXED_PLATE, fixed=True)
        at = "top"
    sensors = (build.get("base") or {}).get("sensors") if isinstance(build.get("base"), dict) else None
    if sensors:
        base["sensors"] = sensors
    out = {k: v for k, v in build.items() if k not in ("base", "parts")}
    out["base"] = base
    out["parts"] = [{"name": name, "at": at, "robot": mount}, *build.get("parts", [])]
    return out


def _root(model: mujoco.MjModel, robot: str) -> int:
    for b in range(1, model.nbody):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{robot}/"):
            return int(model.body_rootid[b])
    raise ValueError(f"no robot named {robot!r} in the compiled world")


def _floating(model: mujoco.MjModel, root: int) -> bool:
    first = int(model.body_jntadr[root])
    return first >= 0 and int(model.jnt_type[first]) == mujoco.mjtJoint.mjJNT_FREE


def _corners(model: mujoco.MjModel, data: mujoco.MjData, root: int) -> np.ndarray:
    """The corners of the boxes round the root body's own shapes (the trunk, not the legs), in the root's frame."""
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    frame, origin = data.xmat[root].reshape(3, 3), data.xpos[root]
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != root:
            continue
        rot, box = data.geom_xmat[g].reshape(3, 3), model.geom_aabb[g]
        world = data.geom_xpos[g] + (box[:3] + signs * box[3:]) @ rot.T
        out.append((world - origin) @ frame)
    if not out:
        return np.zeros((1, 3))
    return np.concatenate(out)


def _top_under(
    model: mujoco.MjModel, data: mujoco.MjData, root: int, half_x: float, half_y: float, high: float
) -> float:
    """How high the root body reaches under a plate of these half sizes, in its own frame: rays straight down onto
    it over the plate's area. The boxes round its shapes say too much for a mesh turned in its frame: on the rover
    template a dressing mesh's box reached 7.1 cm where its lid is 3.7 cm, and the plate floated above the lid.
    Anything else the rays meet first (a leg, a shelf above) is passed through; `high` (the boxes' top) when no ray
    meets the root body."""
    frame, origin = data.xmat[root].reshape(3, 3), data.xpos[root]
    down, hit, best = -frame[:, 2], np.zeros(1, np.int32), None
    for fx, fy in itertools.product(np.linspace(-1.0, 1.0, PROBES), repeat=2):
        start = origin + frame @ np.array([fx * half_x, fy * half_y, high + 0.05])
        for _ in range(MAX_LAYERS):
            far = mujoco.mj_ray(model, data, start, down, None, 1, -1, hit)
            if far < 0:
                break
            point = start + down * far
            if int(model.geom_bodyid[int(hit[0])]) == root:
                z = float((point - origin) @ frame[:, 2])
                best = z if best is None else max(best, z)
                break
            start = point + down * 1e-4
    return high if best is None else best


def _free_name(wanted: str, build: dict[str, Any]) -> str:
    taken = {p.get("name") for p in build.get("parts", []) if isinstance(p, dict)}
    name = "".join(c if c.isalnum() else "_" for c in wanted.lower()) or "body"
    if not name[0].isalpha():
        name = f"robot_{name}"
    return next(n for n in (name, *(f"{name}_{i}" for i in range(2, 99))) if n not in taken)
