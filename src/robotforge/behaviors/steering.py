"""Bounded planar independent steering, commanding only the source model's actuators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.behaviors import _wheel_radius, home_pose
from robotforge.sim.actuators import drives_joint, is_position_servo, is_velocity_servo

ALIGN_TOL = 0.08  # rad; all wheels wait until their steering servos face the requested direction


class UnsupportedSteering(ValueError):
    """A steered robot whose geometry cannot be driven by the bounded planar controller."""


@dataclass(frozen=True)
class Module:
    wheel: int
    steering: int
    qpos: int
    wheel_dof: int
    home: float
    heading: float  # rolling direction for positive wheel speed at home, in the root frame
    sign: float  # steering axis points up (+1) or down (-1)
    pivot: tuple[float, float]
    offset: float  # signed distance from steering axis along the positive wheel axle
    radius: float
    lower: float
    upper: float
    wheel_lower: float
    wheel_upper: float


def _candidates(model: mujoco.MjModel, robot: str) -> list[tuple[int, int, tuple[int, ...]]]:
    """Velocity hinges and movable ancestors, scoped by both actuator and body ownership."""
    prefix = f"{robot}/"
    found = []
    for a in range(model.nu):
        if not (model.actuator(a).name or "").startswith(prefix) or not drives_joint(model, a):
            continue
        j = int(model.actuator_trnid[a, 0])
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not is_velocity_servo(model, a):
            continue
        body = int(model.jnt_bodyid[j])
        if not (model.body(body).name or "").startswith(prefix):
            continue
        ancestors = []
        while body and (model.body(body).name or "").startswith(prefix):
            start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
            ancestors.extend(
                k for k in range(start, start + count) if k != j and model.jnt_type[k] != mujoco.mjtJoint.mjJNT_FREE
            )
            body = int(model.body_parentid[body])
        found.append((a, j, tuple(ancestors)))
    return found


def _steering_context(
    model: mujoco.MjModel,
    robot: str,
    candidates: list[tuple[int, int, tuple[int, ...]]],
    home_qpos: np.ndarray | None = None,
) -> tuple[int, mujoco.MjData] | None:
    """Find yaw-like wheel carriers at home; pitch/roll legs and sliding suspension stay legacy drives.

    A tilted yaw axis is still steering: the stricter planar binder must refuse it, rather than let
    differential drive lock it at home. Keep the entire ancestry for that binder to reject mixed chains.
    """
    hinges = {j for _, _, chain in candidates for j in chain if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    if not hinges:
        return None
    root = next(b for b in range(1, model.nbody) if (model.body(b).name or "").startswith(f"{robot}/"))
    data = mujoco.MjData(model)
    data.qpos[:] = home_pose(model) if home_qpos is None else home_qpos
    mujoco.mj_kinematics(model, data)
    up = data.xmat[root].reshape(3, 3)[:, 2]
    if not any(abs(float(data.xaxis[j] @ up)) > 1e-4 for j in hinges):
        return None
    return root, data


def has_steering(model: mujoco.MjModel, robot: str) -> bool:
    """Whether wheel ancestry changes yaw in the home root frame, including unsupported tilted axes."""
    return _steering_context(model, robot, _candidates(model, robot)) is not None


def steering_binding(
    model: mujoco.MjModel, robot: str, home_qpos: np.ndarray | None = None
) -> tuple[Module, ...] | None:
    candidates = _candidates(model, robot)
    context = _steering_context(model, robot, candidates, home_qpos)
    if context is None:
        return None
    root, data = context

    def refuse(reason: str) -> None:
        raise UnsupportedSteering(f"{robot}: independent steering is unsupported: {reason}")

    if len(candidates) < 3 or any(len(chain) != 1 for _, _, chain in candidates):
        refuse(
            "needs at least three wheels, each on one independent steering hinge; "
            "mixed or articulated drives do not fit"
        )
    frame = data.xmat[root].reshape(3, 3)
    actuators: dict[int, list[int]] = {}
    for a in range(model.nu):
        if drives_joint(model, a):
            actuators.setdefault(int(model.actuator_trnid[a, 0]), []).append(a)
    modules = []
    for wheel, joint, (steer,) in candidates:
        if model.jnt_limited[joint]:
            refuse("wheel hinges must rotate continuously")
        if model.jnt_bodyid[joint] == model.jnt_bodyid[steer]:
            refuse("wheel and steering need separate bodies")
        for e in range(model.neq):
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and (
                model.eq_obj1id[e] in (joint, steer) or model.eq_obj2id[e] in (joint, steer)
            ):
                refuse("wheel or steering joints are coupled by an equality constraint")
        powered = actuators.get(steer, [])
        if len(powered) != 1 or len(actuators[joint]) != 1 or not is_position_servo(model, powered[0]):
            refuse("each wheel needs one velocity servo and each steering hinge one position servo")
        actuator = powered[0]
        if not (model.actuator(actuator).name or "").startswith(f"{robot}/"):
            refuse("steering actuator belongs to another robot")
        if model.jnt_type[steer] != mujoco.mjtJoint.mjJNT_HINGE:
            refuse("wheel carriers must rotate about vertical hinges")
        for a in (wheel, actuator):
            if not np.allclose(model.actuator_gear[a], [1, 0, 0, 0, 0, 0], atol=1e-8):
                refuse("only unit-gear servos are supported")
            if model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE or model.actuator_biasprm[a, 0] != 0:
                refuse("filtered or biased servos are unsupported")
            if not model.actuator_ctrllimited[a] or not np.all(np.isfinite(model.actuator_ctrlrange[a])):
                refuse("every wheel and steering servo needs finite control limits")
        lower, upper = map(float, model.actuator_ctrlrange[actuator])
        if model.jnt_limited[steer]:
            lower, upper = max(lower, float(model.jnt_range[steer, 0])), min(upper, float(model.jnt_range[steer, 1]))
        if upper - lower < math.pi or upper - lower > 4 * math.pi:
            refuse("steering travel must span pi to four pi radians to cover every rolling direction")
        lo, hi = map(float, model.actuator_ctrlrange[wheel])
        if not lo < 0 < hi:
            refuse("wheel control ranges must allow both forward and reverse, including zero")
        axle, axis = frame.T @ data.xaxis[joint], frame.T @ data.xaxis[steer]
        if abs(axle[2]) > 1e-4 or abs(axis[2]) < 1 - 1e-6:
            refuse("requires horizontal wheel axles and vertical steering axes in the home root frame")
        pivot = frame.T @ (data.xanchor[steer] - data.xpos[root])
        delta = frame.T @ (data.xanchor[joint] - data.xanchor[steer])
        rolling = np.cross(axle, [0.0, 0.0, 1.0])
        if abs(float(delta @ rolling)) > 1e-4:
            refuse("wheel trail along its rolling direction is unsupported")
        body = int(model.jnt_bodyid[joint])
        if not any(
            model.geom_bodyid[g] == body and (model.geom_contype[g] or model.geom_conaffinity[g])
            for g in range(model.ngeom)
        ):
            refuse("every wheel needs collision geometry to measure its radius")
        radius = _wheel_radius(model, data, joint)
        if not math.isfinite(radius) or radius <= 1e-4:
            refuse("wheel radius must be finite and positive")
        modules.append(
            Module(
                wheel,
                actuator,
                int(model.jnt_qposadr[steer]),
                int(model.jnt_dofadr[joint]),
                float(data.qpos[model.jnt_qposadr[steer]]),
                math.atan2(rolling[1], rolling[0]),
                float(np.sign(axis[2])),
                (float(pivot[0]), float(pivot[1])),
                float(delta @ axle),
                radius,
                lower,
                upper,
                lo,
                hi,
            )
        )
    if len({m.steering for m in modules}) != len(modules):
        refuse("steering hinges are shared by multiple wheels")
    if np.linalg.matrix_rank(np.array([m.pivot for m in modules]) - np.mean([m.pivot for m in modules], axis=0)) < 2:
        refuse("steering pivots must span a two-dimensional footprint")
    return tuple(modules)


def steering_speed(modules: tuple[Module, ...]) -> float:
    """Conservative bidirectional speed, bounded by every wheel's source control limits."""
    return min(min(-m.wheel_lower, m.wheel_upper) * m.radius for m in modules)


class SteeringDrive:
    rate_hz = 50.0

    def __init__(self, model: mujoco.MjModel, robot: str, modules: tuple[Module, ...]):
        self.robot, self.modules = robot, modules
        self.max_speed = steering_speed(modules)
        self.speed = self.turn = 0.0
        self.radius = min(m.radius for m in modules)
        # Navigation's terrain limits use these to distinguish driven wheels from passive casters.
        self.left = tuple(m.wheel for m in modules if m.pivot[1] >= 0)
        self.right = tuple(m.wheel for m in modules if m.pivot[1] < 0)
        self._wanted = np.zeros(len(modules))
        self.aligning = False

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        if not math.isfinite(speed) or not math.isfinite(turn):
            raise ValueError("drive speed and turn must be finite")
        self.speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        self.turn = float(np.clip(turn, -4.0, 4.0))

    def status(self) -> dict:
        return {
            "behavior": "drive",
            "drive": "independent_steering",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
            "aligning": self.aligning,
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.speed == 0 and self.turn == 0:
            self._wanted[:] = 0
            self.aligning = False
            for m in self.modules:
                data.ctrl[m.wheel] = 0.0
                data.ctrl[m.steering] = np.clip(data.qpos[m.qpos], m.lower, m.upper)
            return
        targets, speeds, errors = [], [], []
        for m in self.modules:
            vx, vy = self.speed - self.turn * m.pivot[1], self.turn * m.pivot[0]
            current = float(data.qpos[m.qpos])
            if math.hypot(vx, vy) < 1e-10:
                target = float(np.clip(current, m.lower, m.upper))
            else:
                base = m.home + m.sign * (math.atan2(vy, vx) - m.heading)
                step = math.pi
                first, last = math.ceil((m.lower - base) / step), math.floor((m.upper - base) / step)
                target = min((base + k * step for k in range(first, last + 1)), key=lambda q: abs(q - current))
            angle = m.heading + m.sign * (target - m.home)
            # A lateral axle offset rotates with the module; yaw adds -turn * offset to rolling velocity.
            speed = (vx * math.cos(angle) + vy * math.sin(angle) - self.turn * m.offset) / m.radius
            targets.append(target)
            speeds.append(speed)
            errors.append(abs(target - current))
        # Preserve the requested body twist when any wheel reaches its own (possibly asymmetric) limit.
        scale = max(
            1.0, *(v / (m.wheel_upper if v >= 0 else m.wheel_lower) for m, v in zip(self.modules, speeds, strict=True))
        )
        self.aligning = max(errors) > ALIGN_TOL
        for i, (m, target, speed) in enumerate(zip(self.modules, targets, speeds, strict=True)):
            data.ctrl[m.steering] = target
            self._wanted[i] = 0.0 if self.aligning else speed / scale
            data.ctrl[m.wheel] = np.clip(self._wanted[i], m.wheel_lower, m.wheel_upper)

    def push_limit(self, model: mujoco.MjModel) -> float | None:
        if not all(model.actuator_forcelimited[m.wheel] for m in self.modules):
            return None
        return sum(min(abs(float(v)) for v in model.actuator_forcerange[m.wheel]) / m.radius for m in self.modules)

    def spin(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        moving = np.abs(self._wanted) > 1e-6
        if not np.any(moving):
            return 1.0
        actual = np.array([abs(data.qvel[m.wheel_dof]) for m in self.modules])
        return float(np.mean(actual[moving] / np.abs(self._wanted[moving])))
