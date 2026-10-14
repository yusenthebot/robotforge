"""Placed parts -> MjSpec: bodies, joints with real actuators, shapes with mass from material, sensors, mounts.

Names inside the robot are the part names: a part's body, joint and actuator are all
called after it, so `left_wheel` is the body, the hinge and the motor at once. Sensors
are `<part>_<type>` (the base's IMU is plain `imu`, like the templates').
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.design import components
from robotforge.design.builder.expand import Placed, link_end, shape_box
from robotforge.design.builder.schema import (
    BaseSpec,
    BuildSpec,
    PartSpec,
    SensorSpec,
    actuator_for,
    axis_vector,
    control_of,
)
from robotforge.design.templates.base import DARK, GEOM, SHELL, new_spec, servo_joint

LINEAR_SATURATION = 0.005  # m of position error at which a linear actuator pushes with full force
PASSIVE_DAMPING = 0.01  # N*m*s/rad on unpowered joints, so they settle instead of swinging forever
WHEEL_DAMPING = 0.002
INVISIBLE = [0.0, 0.0, 0.0, 0.0]
MOTOR_YELLOW = [0.95, 0.8, 0.2, 1.0]
FOOT_FRICTION = [1.2, 0.02, 0.002]
CASTER_FRICTION = [0.02, 0.001, 0.0001]
MIN_MASS = 0.005  # kg for a bare joint with no shape: MuJoCo needs mass on every moving body


@dataclass
class Assembly:
    spec: mujoco.MjSpec
    home_qpos: dict[str, float] = field(default_factory=dict)
    home_ctrl: dict[str, float] = field(default_factory=dict)
    floating: bool = True
    mounted: dict[str, str] = field(default_factory=dict)  # part -> "arm template" etc.
    notes: list[str] = field(default_factory=list)


def assemble(build: BuildSpec, placed: list[Placed]) -> Assembly:
    spec = new_spec()
    accent = components.color(build.color)
    out = Assembly(spec=spec, floating=not build.base.fixed)
    base = spec.worldbody.add_body(name="base")
    if not build.base.fixed:
        base.add_freejoint(name="root")
    _shape(base, build.base, build, np.eye(3), accent, "base")
    _sensors(spec, base, "base", build.base, np.eye(3))
    bodies: dict[str, mujoco.MjsBody] = {"base": base}
    for part in placed:
        parent = bodies[part.parent]
        body = parent.add_body(name=part.name, pos=part.pos.tolist(), quat=_quat(part.rot))
        bodies[part.name] = body
        _joint(spec, parent, body, part, build, out)
        _shape(body, part.spec, build, part.local, accent, part.name)
        if part.spec.foot:
            _foot(body, part)
        _sensors(spec, body, part.name, part.spec, part.local)
        if part.spec.robot is not None:
            _mount(spec, body, part, out)
    return out


# ---- joints and actuators ---------------------------------------------------------------------------


def _joint(
    spec: mujoco.MjSpec, parent: mujoco.MjsBody, body: mujoco.MjsBody, part: Placed, build: BuildSpec, out: Assembly
) -> None:
    j = part.spec.joint
    if j.type == "fixed":
        if part.spec.shape == "none" and part.spec.robot is None:
            _min_mass(body)
        return
    axis = part.axis(axis_vector(j.axis)).tolist()
    hinge = j.type == "hinge"
    scale = math.radians(1.0) if hinge else 1.0
    if j.continuous:
        joint_range = None
    elif j.range is not None:
        joint_range = (j.range[0] * scale, j.range[1] * scale)
    else:
        joint_range = (-math.pi / 2, math.pi / 2)
    home = j.home * scale if not j.continuous else 0.0
    act_spec = actuator_for(build, part.spec)
    name = part.name

    if act_spec is None:  # passive: a hinge or slide that only physics moves
        joint = body.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_HINGE if hinge else mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            damping=j.damping if j.damping is not None else PASSIVE_DAMPING,
            stiffness=j.stiffness or 0.0,
            springref=home,
        )
        _limit(joint, joint_range)
        if part.spec.shape == "none":
            _min_mass(body)
        out.home_qpos[name] = home
        return

    actuator = _geared(components.actuator(act_spec.model), act_spec.gear)
    control = control_of(act_spec)
    extra_damping = j.damping or 0.0
    if hinge and control == "position":
        joint = servo_joint(spec, body, name, axis, joint_range, actuator)
        # MuJoCo 3.x keeps damping and stiffness as [linear, 0, 0] coefficient triples.
        joint.damping = [float(joint.damping[0]) + extra_damping, 0.0, 0.0]
        joint.stiffness = [j.stiffness or 0.0, 0.0, 0.0]
        joint.springref = home
        out.home_ctrl[name] = home
    else:
        joint = body.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_HINGE if hinge else mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            armature=actuator.armature,
            stiffness=j.stiffness or 0.0,
            springref=home,
        )
        _limit(joint, joint_range)
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        if control == "position":  # a linear actuator: lead screws barely back-drive, hence the damping
            joint.damping = [actuator.damping + extra_damping, 0.0, 0.0]
            act.set_to_position(kp=actuator.stall_torque / LINEAR_SATURATION)
            act.ctrlrange = list(joint_range)
            out.home_ctrl[name] = home
        elif control == "velocity":
            # Stiff speed loop: full torque at ~5 % speed error, like a motor driver with feedback.
            joint.damping = [WHEEL_DAMPING + extra_damping, 0.0, 0.0]
            act.set_to_velocity(kv=actuator.stall_torque / (0.05 * actuator.no_load_speed))
            act.ctrlrange = [-actuator.no_load_speed, actuator.no_load_speed]
            out.home_ctrl[name] = 0.0
        else:  # torque: the control is the torque itself (for RL and hand-written controllers)
            joint.damping = [WHEEL_DAMPING + extra_damping, 0.0, 0.0]
            act.set_to_motor()
            act.ctrlrange = [-actuator.stall_torque, actuator.stall_torque]
            out.home_ctrl[name] = 0.0
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-actuator.stall_torque, actuator.stall_torque]
    out.home_qpos[name] = home
    _actuator_housing(parent, body, part, actuator, np.asarray(axis), build.servo_caps)


def _geared(part: components.Actuator, gear: float) -> components.Actuator:
    if gear == 1.0:
        return part
    return dataclasses.replace(
        part,
        stall_torque=part.stall_torque * gear,
        no_load_speed=part.no_load_speed / gear,
        armature=part.armature * gear * gear,
    )


def _limit(joint: mujoco.MjsJoint, joint_range: tuple[float, float] | None) -> None:
    if joint_range is None:
        joint.limited = mujoco.mjtLimited.mjLIMITED_FALSE
        return
    joint.range = list(joint_range)
    joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE


def _actuator_housing(
    parent: mujoco.MjsBody,
    body: mujoco.MjsBody,
    part: Placed,
    actuator: components.Actuator,
    axis: np.ndarray,
    visible: bool,
) -> None:
    """The servo or motor itself: drawn where a builder would put it and carrying its mass, never colliding."""
    rgba = DARK if visible else INVISIBLE
    sx, sy, sz = actuator.size
    if part.spec.shape == "wheel":
        # A gear motor sits inboard of the wheel, on the chassis, so it does not spin with the wheel.
        axis_parent = part.rot @ axis
        inboard = -axis_parent if float(axis_parent @ part.pos) > 0 else axis_parent
        half_w = part.spec.size[1] / 2
        length = min(sx, 0.06)
        center = part.pos + inboard * (half_w + length / 2 + 0.003)
        parent.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[max(sy, sz) / 2, length / 2, 0],
            pos=center.tolist(),
            quat=_z_to(axis_parent),
            rgba=MOTOR_YELLOW if visible else INVISIBLE,
            mass=actuator.mass,
            contype=0,
            conaffinity=0,
        )
        return
    if actuator.kind == "linear":
        # The actuator body stays on the parent and pushes the part along the axis.
        axis_parent = part.rot @ axis
        parent.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[sy / 2, sz / 2, sx / 2],
            pos=part.pos.tolist(),
            quat=_z_to(axis_parent),
            rgba=rgba,
            mass=actuator.mass,
            contype=0,
            conaffinity=0,
        )
        return
    # A rotary servo as a knuckle around its output axis at the joint.
    body.add_geom(
        type=GEOM.mjGEOM_CYLINDER,
        size=[min(sx, sy) / 2 + 0.002, sz / 2, 0],
        quat=_z_to(axis),
        rgba=rgba,
        mass=actuator.mass,
        contype=0,
        conaffinity=0,
    )


def _min_mass(body: mujoco.MjsBody) -> None:
    body.add_geom(type=GEOM.mjGEOM_SPHERE, size=[0.005, 0, 0], mass=MIN_MASS, rgba=INVISIBLE, contype=0, conaffinity=0)


# ---- shapes -----------------------------------------------------------------------------------------


def _shape(
    body: mujoco.MjsBody,
    spec: BaseSpec | PartSpec,
    build: BuildSpec,
    local: np.ndarray,
    accent: list[float],
    owner: str,
) -> None:
    shape = spec.shape
    if shape == "none":
        return
    material = spec.material or ("rubber" if shape == "caster" else build.material)
    is_base = isinstance(spec, BaseSpec)
    default_rgba = accent if is_base else (DARK if shape in ("wheel", "caster") else SHELL)
    rgba = components.color(spec.color) if spec.color is not None else default_rgba
    kwargs: dict = {"rgba": rgba}
    if spec.mass is not None:
        kwargs["mass"] = spec.mass
    else:
        kwargs["density"] = components.MATERIALS[material]
    if spec.friction is not None:
        kwargs["friction"] = [spec.friction, 0.02, 0.002]
    elif shape == "wheel":
        kwargs["friction"] = FOOT_FRICTION
    elif shape == "caster":
        kwargs["friction"] = CASTER_FRICTION
    if not getattr(spec, "collide", True):
        kwargs["contype"] = 0
        kwargs["conaffinity"] = 0

    low, high = shape_box(spec)
    center = local @ ((low + high) / 2)
    size = spec.size
    if shape == "box":
        body.add_geom(type=GEOM.mjGEOM_BOX, size=[s / 2 for s in size], pos=center.tolist(), **kwargs)
    elif shape in ("sphere", "caster"):
        body.add_geom(type=GEOM.mjGEOM_SPHERE, size=[size[0] / 2, 0, 0], pos=center.tolist(), **kwargs)
    elif shape == "wheel":
        axis = local @ np.array([0.0, 1.0, 0.0])
        r, half_w = size[0] / 2, size[1] / 2
        body.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[r, half_w, 0], pos=center.tolist(), quat=_z_to(axis), **kwargs)
        # Hub cap: a light disc on the outside face so you can see the wheel turn.
        body.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[r * 0.5, half_w + 0.0008, 0],
            pos=center.tolist(),
            quat=_z_to(axis),
            rgba=SHELL,
            mass=0.0005,
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[r * 0.42, half_w + 0.0012, r * 0.07],
            pos=center.tolist(),
            quat=_z_to(axis),
            rgba=accent,
            mass=0.0002,
            contype=0,
            conaffinity=0,
        )
    else:  # capsule or cylinder from end to end
        start, end = _segment(spec)
        gtype = GEOM.mjGEOM_CAPSULE if shape == "capsule" else GEOM.mjGEOM_CYLINDER
        body.add_geom(type=gtype, size=[size[0] / 2, 0, 0], fromto=[*(local @ start), *(local @ end)], **kwargs)


def _segment(spec: BaseSpec | PartSpec) -> tuple[np.ndarray, np.ndarray]:
    offset = np.asarray(spec.offset, float)
    length = spec.size[1]
    direction = getattr(spec, "direction", None)
    if direction is not None:
        d = np.asarray(axis_vector(direction), float)
        return offset, offset + d * length
    u = np.asarray(axis_vector(getattr(spec, "axis", "z")), float)
    return offset - u * length / 2, offset + u * length / 2


def link_radius(spec: BaseSpec | PartSpec) -> float:
    if spec.shape in ("capsule", "cylinder", "sphere", "caster", "wheel"):
        return spec.size[0] / 2
    if spec.shape == "box":
        return min(spec.size) / 2
    return 0.01


def _foot(body: mujoco.MjsBody, part: Placed) -> None:
    radius = max(0.01, 1.25 * link_radius(part.spec))
    body.add_geom(
        type=GEOM.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=part.vec(link_end(part.spec)).tolist(),
        rgba=DARK,
        friction=FOOT_FRICTION,
        density=components.MATERIALS["rubber"],
    )


# ---- sensors ----------------------------------------------------------------------------------------


def _sensors(
    spec: mujoco.MjSpec, body: mujoco.MjsBody, owner: str, part: BaseSpec | PartSpec, local: np.ndarray
) -> None:
    from robotforge.design.builder.expand import anchor

    used: dict[str, int] = {}
    for sensor in part.sensors:
        where = local @ anchor(part, sensor.at, "the base" if owner == "base" else f"part {owner!r}")
        look = local @ np.asarray(axis_vector(sensor.direction), float)
        name = sensor.name or _sensor_name(owner, sensor, used)
        _add_sensor(spec, body, name, sensor, where, look, part)


def _sensor_name(owner: str, sensor: SensorSpec, used: dict[str, int]) -> str:
    kind = {"camera": "cam", "rangefinder": "range"}.get(sensor.type, sensor.type)
    base = kind if owner == "base" and sensor.type == "imu" else f"{owner}_{kind}"
    used[base] = used.get(base, 0) + 1
    return base if used[base] == 1 else f"{base}_{used[base]}"


def _add_sensor(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    name: str,
    sensor: SensorSpec,
    where: np.ndarray,
    look: np.ndarray,
    part: BaseSpec | PartSpec,
) -> None:
    S, OBJ = mujoco.mjtSensor, mujoco.mjtObj
    if sensor.type == "imu":
        # Same names as the templates' IMU: <prefix>imu (site), accel, gyro, orientation.
        prefix = name[: -len("imu")] if name.endswith("imu") else f"{name}_"
        site = f"{prefix}imu"
        body.add_site(name=site, pos=where.tolist(), size=[0.005, 0, 0], rgba=INVISIBLE)
        for kind, stype in (
            ("accel", S.mjSENS_ACCELEROMETER),
            ("gyro", S.mjSENS_GYRO),
            ("orientation", S.mjSENS_FRAMEQUAT),
        ):
            spec.add_sensor(name=f"{prefix}{kind}", type=stype, objtype=OBJ.mjOBJ_SITE, objname=site)
        return
    if sensor.type == "touch":
        radius = max(0.012, 1.4 * link_radius(part)) if isinstance(part, PartSpec) and part.foot else None
        radius = radius or max(0.01, 1.3 * link_radius(part))
        body.add_site(name=name, pos=where.tolist(), size=[radius, 0, 0], rgba=INVISIBLE)
        spec.add_sensor(name=name, type=S.mjSENS_TOUCH, objtype=OBJ.mjOBJ_SITE, objname=name)
        return
    if sensor.type == "camera":
        body.add_camera(name=name, pos=where.tolist(), xyaxes=_camera_axes(look), fovy=sensor.fov)
        # A small lens so you can see where the camera sits and which way it looks.
        body.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[0.009, 0.006, 0],
            pos=(where - look * 0.004).tolist(),
            quat=_z_to(look),
            rgba=[0.08, 0.09, 0.1, 1],
            mass=0.003,
            contype=0,
            conaffinity=0,
        )
        return
    if sensor.type == "rangefinder":
        body.add_site(name=name, pos=where.tolist(), quat=_z_to(look), size=[0.004, 0, 0], rgba=[1, 0.3, 0.2, 0.8])
        spec.add_sensor(name=name, type=S.mjSENS_RANGEFINDER, objtype=OBJ.mjOBJ_SITE, objname=name, cutoff=sensor.range)
        return
    # lidar: a spinning puck, modelled as a ring of rangefinders in the owner's horizontal plane
    up = np.array([0.0, 0.0, 1.0])
    body.add_geom(
        type=GEOM.mjGEOM_CYLINDER,
        size=[0.022, 0.012, 0],
        pos=(where + up * 0.012).tolist(),
        rgba=[0.1, 0.11, 0.13, 1],
        mass=0.06,
        contype=0,
        conaffinity=0,
    )
    for i in range(sensor.rays):
        angle = 2 * math.pi * i / sensor.rays
        ray = np.array([math.cos(angle), math.sin(angle), 0.0])
        site = f"{name}_{i}"
        body.add_site(name=site, pos=(where + up * 0.012).tolist(), quat=_z_to(ray), size=[0.002, 0, 0], rgba=INVISIBLE)
        spec.add_sensor(name=site, type=S.mjSENS_RANGEFINDER, objtype=OBJ.mjOBJ_SITE, objname=site, cutoff=sensor.range)


def _camera_axes(look: np.ndarray) -> list[float]:
    """MuJoCo cameras look along their -z with +y up; keep the image upright where possible."""
    z = -look / np.linalg.norm(look)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    y = up - z * (up @ z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return [*x.tolist(), *y.tolist()]


# ---- mounted robots ---------------------------------------------------------------------------------


def _mount(spec: mujoco.MjSpec, body: mujoco.MjsBody, part: Placed, out: Assembly) -> None:
    from robotforge.design import library, templates

    mount = part.spec.robot
    try:
        if mount.template is not None:
            child = templates.build(mount.template, mount.params)
            what = f"{mount.template} template"
        else:
            child = library.build(mount.library, mount.params)
            what = library.get(mount.library).title
    except ValueError as exc:
        raise ValueError(f"part {part.name!r}: cannot build the mounted robot: {exc}") from exc
    root = child.spec.worldbody.first_body()
    for joint in list(root.joints) if root is not None else []:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:  # it rides on this part now instead of moving freely
            child.spec.delete(joint)
    if child.spec.option.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = max(spec.option.impratio, child.spec.option.impratio)
    for name in ("timestep", "integrator", "cone", "impratio"):
        setattr(child.spec.option, name, getattr(spec.option, name))
    frame = body.add_frame(pos=[0, 0, 0])
    spec.attach(child.spec, prefix=f"{part.name}/", frame=frame)
    out.home_qpos.update({f"{part.name}/{k}": v for k, v in child.home_qpos.items()})
    out.home_ctrl.update({f"{part.name}/{k}": v for k, v in child.home_ctrl.items()})
    out.mounted[part.name] = what
    out.notes += [f"{part.name}: {n}" for n in child.notes]


# ---- small helpers ----------------------------------------------------------------------------------


def _quat(rot: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(rot, float).ravel())
    return q.tolist()


def _z_to(direction: np.ndarray) -> list[float]:
    """Quaternion turning +z onto `direction`."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, d)
    return q.tolist()
