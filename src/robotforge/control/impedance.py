"""Compliant arm control: joint impedance, Cartesian impedance and operational-space control, with touch in the loop.

All of them follow a `ToolTarget` that a task hands out every control step (where the tool point goes, how fast, how it
is turned, and, when it presses on a surface, along which normal and with how many newtons) and turn it into joint
torques, which `TorqueMap` converts to whatever the arm's actuators take (position servos included).

- "joint_impedance": a spring and a damper in every joint pull the joints toward an inverse-kinematics answer for the
  target, plus gravity. Soft joints give way to contact; what the tool pushes with is stiffness times how far the
  joints are held off their goal.
- "cartesian_impedance": a spring and a damper pull the tool point itself (and its orientation) toward the target,
  `torque = J^T F + gravity`, with a weak spring in the joints the tool does not need. The tool behaves like a mass on
  a spring, whatever the pose of the arm.
- "osc": operational-space control (Khatib). The target's error is turned into the acceleration the tool should have,
  the tool-space inertia, the arm's Coriolis and gravity torques are cancelled exactly, so the tool answers like a
  unit mass in every pose, and the leftover joints are steered in the null space of the task. Along a surface normal
  it closes a force loop on the measured contact force (hybrid force and motion control).
- "ik" and "pd" on a contact task: the same differential inverse kinematics, then the arm's own position servos, or a
  stiff joint PD with gravity compensation. They know nothing of force, which is what the comparison is for.

The force the tool feels is measured the way a wrist force-torque sensor would: the contact forces on the hand and what
it holds (`ToolForce`), low-pass filtered. Forces here are the force the tool exerts on its surroundings, in newtons,
unless a name says otherwise.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.control.actuation import TorqueMap
from robotforge.control.classic import positive
from robotforge.control.tasks import Task, TaskError
from robotforge.control.tasks.base import home_qpos
from robotforge.control.tasks.tool import TargetSource, ToolForce, ToolTarget, rotation_error

IK_DAMPING = 0.05  # damped least squares: keeps the joint answers sane near a singular pose
IK_SPEED = 2.0  # rad/s: the fastest a reference joint moves toward an inverse-kinematics answer
ROT_WEIGHT = 0.5  # m per rad: how much an orientation error counts in inverse kinematics
SENSE_HZ = 40.0  # cut-off of the low-pass filter on the measured force
FORCE_MAX = 200.0  # N: the most a force loop commands, whatever it is asked
POSTURE_OMEGA = 3.0  # rad/s: the natural frequency of the weak pull toward the rest pose in the null space


def per_axis(name: str, value: float | list[float], size: int, high: float = 1e5) -> np.ndarray:
    """A gain given as one number or one per axis, as an array of `size` positive numbers."""
    values = np.atleast_1d(np.asarray(value, float))
    if values.size not in (1, size):
        raise TaskError(f"{name} takes one number or {size}, got {values.size}")
    return (
        np.full(size, positive(name, float(values[0]), high))
        if values.size == 1
        else np.array([positive(name, float(v), high) for v in values])
    )


class _Arm:
    """Kinematics and dynamics of one arm as a controller reads them every step."""

    def __init__(self, model: mujoco.MjModel, task: Any):
        self.site = task.site
        self.q, self.d = task.qadr, task.dadr
        self.body = int(model.site_bodyid[self.site])
        self.n = len(self.d)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._jdp = np.zeros((3, model.nv))
        self._jdr = np.zeros((3, model.nv))
        self._M = np.zeros((model.nv, model.nv))
        self.limited = model.jnt_limited[task.joints].astype(bool)
        self.lo = np.where(self.limited, model.jnt_range[task.joints, 0], -np.inf)
        self.hi = np.where(self.limited, model.jnt_range[task.joints, 1], np.inf)

    def jacobian(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.site)
        return np.vstack([self._jacp[:, self.d], self._jacr[:, self.d]])

    def jacobian_rate_dq(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """J-dot times the joint speeds: the acceleration the tool has with the joints not accelerating."""
        point = data.site_xpos[self.site]
        mujoco.mj_jacDot(model, data, self._jdp, self._jdr, point, self.body)
        dq = data.qvel[self.d]
        return np.concatenate([self._jdp[:, self.d] @ dq, self._jdr[:, self.d] @ dq])

    def mass(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_fullM(model, data, self._M)
        return self._M[np.ix_(self.d, self.d)]

    def bias(self, data: mujoco.MjData) -> np.ndarray:
        """Gravity and Coriolis torques the arm needs held up, less what the model cancels itself (gravcomp)."""
        return data.qfrc_bias[self.d] - data.qfrc_gravcomp[self.d]


class DiffIK:
    """Reference joint angles that follow a moving tool target: one damped least-squares step per control step,
    started from where the arm is, with the joints left over steered to a rest pose."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, task: Any, arm: _Arm):
        self.model, self.arm = model, arm
        self.scratch = mujoco.MjData(model)
        self.rest = home_qpos(model)[arm.q]
        self.q_ref = data.qpos[arm.q].copy()
        self.dq_ref = np.zeros(arm.n)
        self.dt = model.opt.timestep

    def step(self, data: mujoco.MjData, target: ToolTarget, hold_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arm, s = self.arm, self.scratch
        s.qpos[:] = data.qpos
        s.qpos[arm.q] = self.q_ref
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        J = arm.jacobian(self.model, s)
        err = np.empty(6)
        err[:3] = target.pos - s.site_xpos[arm.site]
        rot = target.rot
        err[3:] = ROT_WEIGHT * rotation_error(s.site_xmat[arm.site].reshape(3, 3), rot) if rot is not None else 0.0
        J = J.copy()
        J[3:] *= ROT_WEIGHT
        if rot is None:
            J, err = J[:3], err[:3]
        JJt = J @ J.T + IK_DAMPING**2 * np.eye(len(err))
        pinv = J.T @ np.linalg.inv(JJt)
        dq = pinv @ err + (np.eye(arm.n) - pinv @ J) @ (POSTURE_OMEGA * 0.1 * (self.rest - self.q_ref))
        step = IK_SPEED * self.dt
        dq = np.clip(dq, -step, step)
        new = np.clip(self.q_ref + dq, arm.lo, arm.hi)
        self.dq_ref = (new - self.q_ref) / self.dt
        self.q_ref = new
        return self.q_ref, self.dq_ref


class _Controller:
    """Common to the arm controllers: the task's target source, the force sense, the torque conversion."""

    label = "arm"

    def __init__(self, task: Any, model: mujoco.MjModel, data: mujoco.MjData, source: TargetSource, **gains: Any):
        if not hasattr(task, "site") or not hasattr(task, "joints"):
            raise TaskError(f"{self.label} drives an arm with a tool point (the reach, wipe, press... tasks)")
        self.task, self.source = task, source
        self.arm = _Arm(model, task)
        self.torque = TorqueMap(model, task.actuators)
        self.rate_hz = 1.0 / model.opt.timestep
        self.dt = model.opt.timestep
        self.sense = tool_force(task, model)
        self.feel = np.zeros(3)  # the filtered force on the tool
        self._alpha = 1.0 - math.exp(-2 * math.pi * SENSE_HZ * self.dt)
        self.hold_rot = data.site_xmat[task.site].reshape(3, 3).copy()
        self.rest = home_qpos(model)[task.qadr]
        self.tripped_at: float | None = None
        self.last: ToolTarget | None = None
        self._configure(model, data, **gains)

    def _configure(self, model: mujoco.MjModel, data: mujoco.MjData, **gains: Any) -> None:
        if gains:
            raise TaskError(f"{self.label}: unknown settings {sorted(gains)}")

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.feel += self._alpha * (self.sense.read(model, data) - self.feel)
        target = self.source.target(data, self.feel)
        self.last = target
        tau = self.compute(model, data, target)
        data.ctrl[self.task.actuators] = self.torque.ctrl(data, tau)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        raise NotImplementedError

    def wanted_rot(self, target: ToolTarget) -> np.ndarray | None:
        return target.rot

    def posture(self, M: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """A weak spring and damper toward the rest pose, torque per joint scaled by the joint's inertia."""
        w = POSTURE_OMEGA
        inertia = np.diag(M)
        q, dq = data.qpos[self.arm.q], data.qvel[self.arm.d]
        return inertia * (w * w * (self.rest - q) - 2 * w * dq)

    def feedforward(self, target: ToolTarget) -> np.ndarray:
        """The force the tool applies to its surroundings at the target (world axes): pressing along -normal."""
        if target.normal is None or target.push <= 0:
            return np.zeros(3)
        return -np.asarray(target.normal, float) * min(float(target.push), FORCE_MAX)

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"controller": self.label}
        if self.last is not None and self.last.normal is not None:
            out["pressing_n"] = round(float(-self.feel @ self.last.normal), 2)
        return out


class ServoPath(_Controller):
    """ "ik" on a contact task: inverse kinematics at every step, the angles sent to the arm's own position servos."""

    label = "ik"

    def _configure(self, model: mujoco.MjModel, data: mujoco.MjData, **gains: Any) -> None:
        super()._configure(model, data, **gains)
        self.ik = DiffIK(model, data, self.task, self.arm)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.feel += self._alpha * (self.sense.read(model, data) - self.feel)
        target = self.source.target(data, self.feel)
        self.last = target
        q_ref, _ = self.ik.step(data, target, self.hold_rot)
        data.ctrl[self.task.actuators] = q_ref


class JointPDPath(_Controller):
    """ "pd" on a contact task: inverse kinematics at every step, a stiff joint PD on it with gravity compensation."""

    label = "pd"

    def _configure(self, model: mujoco.MjModel, data: mujoco.MjData, omega: float = 20.0, zeta: float = 1.0) -> None:
        self.omega, self.zeta = positive("omega", omega, 200.0), positive("zeta", zeta, 10.0)
        self.ik = DiffIK(model, data, self.task, self.arm)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        q_ref, dq_ref = self.ik.step(data, target, self.hold_rot)
        a = self.arm
        q, dq = data.qpos[a.q], data.qvel[a.d]
        inertia = np.diag(a.mass(model, data))
        w, z = self.omega, self.zeta
        return inertia * (w * w * (q_ref - q) + 2 * z * w * (dq_ref - dq)) + a.bias(data)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "omega": self.omega, "zeta": self.zeta}


class JointImpedance(_Controller):
    """ "joint_impedance": a spring and damper in every joint toward the inverse-kinematics angles, gravity held up, and
    the target's pressing force added through the Jacobian transpose."""

    label = "joint_impedance"

    def _configure(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        omega: float = 8.0,
        zeta: float = 1.0,
        stiffness: float | list[float] | None = None,
    ) -> None:
        self.omega, self.zeta = positive("omega", omega, 200.0), positive("zeta", zeta, 10.0)
        self.ik = DiffIK(model, data, self.task, self.arm)
        probe = mujoco.MjData(model)
        probe.qpos[:] = data.qpos
        mujoco.mj_forward(model, probe)
        inertia = np.diag(self.arm.mass(model, probe))  # fixed: an impedance is a constant spring, not one that
        # stiffens where the arm is heavy
        if stiffness is None:
            self.K = inertia * self.omega**2
        else:
            self.K = per_axis("stiffness", stiffness, self.arm.n)
        self.D = 2 * self.zeta * np.sqrt(self.K * inertia)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        q_ref, dq_ref = self.ik.step(data, target, self.hold_rot)
        a = self.arm
        q, dq = data.qpos[a.q], data.qvel[a.d]
        tau = self.K * (q_ref - q) + self.D * (dq_ref - dq) + a.bias(data)
        force = self.feedforward(target)
        if force.any():
            J = a.jacobian(model, data)
            tau = tau + J[:3].T @ force
        return tau

    def status(self) -> dict[str, Any]:
        return {**super().status(), "stiffness_nm_rad": [round(float(k), 1) for k in self.K]}


class CartesianImpedance(_Controller):
    """ "cartesian_impedance": the tool is a mass on a spring and damper toward the target, torque = J^T F + gravity
    plus the pressing force, with a weak spring in the joints the tool does not need."""

    label = "cartesian_impedance"

    def _configure(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stiffness: float | list[float] = 500.0,
        rot_stiffness: float = 30.0,
        zeta: float = 1.0,
    ) -> None:
        linear = per_axis("stiffness", stiffness, 3)
        self.K = np.concatenate([linear, np.full(3, positive("rot_stiffness", rot_stiffness))])
        self.zeta = positive("zeta", zeta, 10.0)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        a = self.arm
        J = a.jacobian(model, data)
        M = a.mass(model, data)
        dq = data.qvel[a.d]
        x, R = data.site_xpos[a.site], data.site_xmat[a.site].reshape(3, 3)
        v = J @ dq
        rot = self.wanted_rot(target)
        err = np.zeros(6)
        err[:3] = target.pos - x
        K = self.K.copy()
        if rot is None:
            K[3:] = 0.0
        else:
            err[3:] = rotation_error(R, rot)
        vel = np.concatenate([target.vel, np.zeros(3)])
        Minv = np.linalg.inv(M)
        lam = np.linalg.inv(J @ Minv @ J.T + 1e-4 * np.eye(6))
        D = 2 * self.zeta * np.sqrt(self.K * np.diag(lam))
        force = K * err + D * (vel - v)
        force[:3] += self.feedforward(target)
        pinv_t = np.linalg.pinv(J.T)
        null = np.eye(a.n) - J.T @ pinv_t
        return J.T @ force + null @ self.posture(M, data) + a.bias(data)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "stiffness_n_m": [round(float(k), 1) for k in self.K[:3]]}


class OperationalSpace(_Controller):
    """ "osc": operational-space control. The tool follows the target like a unit mass on a critically damped spring
    in every pose (tool-space inertia, Coriolis and gravity cancelled), the spare joints move in the null space, and
    along a surface normal the commanded force is closed on the measured one."""

    label = "osc"

    def _configure(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        omega: float = 14.0,
        zeta: float = 1.0,
        rot_omega: float = 10.0,
        force_gain: float = 0.6,
        force_integral: float = 4.0,
    ) -> None:
        self.omega, self.zeta = positive("omega", omega, 200.0), positive("zeta", zeta, 10.0)
        self.rot_omega = positive("rot_omega", rot_omega, 200.0)
        self.force_gain = positive("force_gain", force_gain, 20.0)  # N commanded per N of force error
        self.force_integral = positive("force_integral", force_integral, 200.0)  # per second
        self.wound = 0.0  # N, the force loop's integral term

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        a = self.arm
        J = a.jacobian(model, data)
        M = a.mass(model, data)
        dq = data.qvel[a.d]
        x, R = data.site_xpos[a.site], data.site_xmat[a.site].reshape(3, 3)
        v = J @ dq
        rot = self.wanted_rot(target)
        normal = None if target.normal is None else np.asarray(target.normal, float)
        w, z = self.omega, self.zeta
        err_p = target.pos - x
        vel_err = target.vel - v[:3]
        if target.pressing and normal is not None:  # the normal axis is under force control: no position on it
            err_p = err_p - normal * float(normal @ err_p)
            vel_err = vel_err - normal * float(normal @ vel_err)
        accel = np.zeros(6)
        accel[:3] = w * w * err_p + 2 * z * w * vel_err
        if rot is not None:
            wr = self.rot_omega
            accel[3:] = wr * wr * rotation_error(R, rot) - 2 * z * wr * v[3:]
        else:
            accel[3:] = -2 * self.zeta * self.rot_omega * v[3:]
        Minv = np.linalg.inv(M)
        lam = np.linalg.inv(J @ Minv @ J.T + 1e-4 * np.eye(6))
        force = lam @ (accel - a.jacobian_rate_dq(model, data))
        if target.pressing and normal is not None:
            force[:3] -= normal * float(normal @ force[:3])  # on the normal the force loop commands, not Lambda a
            force[:3] += self._press(normal, target.push)
        else:
            self.wound = 0.0
            force[:3] += self.feedforward(target)
        # Dynamically consistent null space: the spare joints cannot disturb the tool.
        jbar_t = lam @ J @ Minv
        null = np.eye(a.n) - J.T @ jbar_t
        return J.T @ force + null @ self.posture(M, data) + a.bias(data)

    def _press(self, normal: np.ndarray, wanted: float) -> np.ndarray:
        """The force on the tool along the normal: the wanted push, plus a correction from the measured push."""
        measured = max(0.0, -float(self.feel @ normal))
        error = wanted - measured
        self.wound = float(np.clip(self.wound + self.force_integral * error * self.dt, -wanted, FORCE_MAX))
        push = float(np.clip(wanted + self.force_gain * error + self.wound, 0.0, FORCE_MAX))
        return -normal * push

    def status(self) -> dict[str, Any]:
        return {**super().status(), "omega": self.omega, "force_loop": self.force_gain > 0}


# ---- wiring -----------------------------------------------------------------------------------------------


CONTROLLERS: dict[str, type[_Controller]] = {
    "joint_impedance": JointImpedance,
    "cartesian_impedance": CartesianImpedance,
    "osc": OperationalSpace,
}
PATH_CONTROLLERS: dict[str, type[_Controller]] = {"ik": ServoPath, "pd": JointPDPath}
IMPEDANCE_METHODS = tuple(CONTROLLERS)


def tool_force(task: Task, model: mujoco.MjModel) -> ToolForce:
    """The task's force sense (made once, and including bodies the task says the tool holds)."""
    sense = getattr(task, "_tool_force", None)
    if sense is None:
        sense = ToolForce.of_arm(model, task, set(getattr(task, "held_bodies", ())))
        task._tool_force = sense  # type: ignore[attr-defined]
    return sense


def make_arm_controller(
    method: str, task: Task, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any] | None = None
) -> _Controller:
    """The controller `method` for a task that hands out tool targets (`task.target_source`)."""
    table = {**PATH_CONTROLLERS, **CONTROLLERS}
    source = task.target_source(model, data)  # type: ignore[attr-defined]
    return table[method](task, model, data, source, **(params or {}))
