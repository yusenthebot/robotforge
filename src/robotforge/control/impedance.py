"""Compliant arm control: joint impedance, Cartesian impedance and operational-space control, with touch in the loop.

All of them follow a `ToolTarget` that a task hands out every control step (where the tool point goes, how fast, which
way it points, and, when it presses on a surface, along which normal and with how many newtons) and turn it into joint
torques, which `TorqueMap` converts to whatever the arm's actuators take (position servos included).

- "joint_impedance": a spring and a damper in every joint pull the joints toward an inverse-kinematics answer for the
  target, plus gravity. Soft joints give way to contact; what the tool pushes with is stiffness times how far the
  joints are held off their goal.
- "cartesian_impedance": a spring and a damper pull the tool point itself (and the way it points) toward the target,
  `torque = J^T F + gravity`, with a weak spring in the joints the tool does not need. The tool behaves like a mass on
  a spring, whatever the pose of the arm.
- "osc": operational-space control (Khatib). The target's error is turned into the acceleration the tool should have,
  the tool-space inertia, the arm's Coriolis and gravity torques are cancelled exactly, so the tool answers like a
  unit mass in every pose, and the leftover joints are steered in the null space of the task. Along a surface normal
  it does not use the tool-space inertia: a soft spring keeps the tool at the surface, a damper (the most one control
  step can carry) calms the contact, and a PI loop closes the commanded push on the measured contact force (hybrid
  force and motion control).
- "ik" and "pd" on a contact task: the same differential inverse kinematics, then the arm's own position servos, or a
  stiff joint PD with gravity compensation. They know nothing of force, which is what the comparison is for.

`PathPush` is the force-limited Cartesian impedance behind `operate(..., force=...)`.

The force the tool feels is measured the way a wrist force-torque sensor would: the contact forces on the hand and what
it holds (`ToolForce`), low-pass filtered. Forces here are the force the tool exerts on its surroundings, in newtons,
unless a name says otherwise.
"""

from __future__ import annotations

import math
import types
from typing import Any

import mujoco
import numpy as np

from robotforge.control.actuation import TorqueMap
from robotforge.control.classic import positive
from robotforge.control.tasks import Task, TaskError
from robotforge.control.tasks.base import home_qpos
from robotforge.control.tasks.tool import TargetSource, ToolForce, ToolTarget, attitude, rotation_error

IK_DAMPING = 0.05  # damped least squares: keeps the joint answers sane near a singular pose
IK_SPEED = 2.0  # rad/s: the fastest a reference joint moves toward an inverse-kinematics answer
ROT_WEIGHT = 0.5  # m per rad: how much an orientation error counts in inverse kinematics
SENSE_HZ = 40.0  # cut-off of the low-pass filter on the measured force
FORCE_MAX = 200.0  # N: the most a force loop commands, whatever it is asked
POSTURE_OMEGA = 3.0  # rad/s: the natural frequency of the weak pull toward the rest pose in the null space
TOOL_REGULARIZE = 1e-4  # added to the tool-space inertia before it is inverted
DAMPER_MAX = 3000.0  # N s/m: the most damping the normal axis is given while pressing


def per_axis(name: str, value: float | list[float], size: int, high: float = 1e5) -> np.ndarray:
    """A gain given as one number or one per axis, as an array of `size` positive numbers."""
    values = np.atleast_1d(np.asarray(value, float))
    if values.size not in (1, size):
        raise TaskError(f"{name} takes one number or {size}, got {values.size}")
    if values.size == 1:
        return np.full(size, positive(name, float(values[0]), high))
    return np.array([positive(name, float(v), high) for v in values])


class _Arm:
    """Kinematics and dynamics of one arm as a controller reads them every step."""

    def __init__(self, model: mujoco.MjModel, task: Any):
        self.site = task.site
        self.q, self.d = task.qadr, task.dadr
        self.body = int(model.site_bodyid[self.site])
        self.n = len(self.d)
        nv = model.nv
        self._jacp, self._jacr = np.zeros((3, nv)), np.zeros((3, nv))
        self._jdp, self._jdr = np.zeros((3, nv)), np.zeros((3, nv))
        self._M = np.zeros((nv, nv))
        limited = model.jnt_limited[task.joints].astype(bool)
        self.lo = np.where(limited, model.jnt_range[task.joints, 0], -np.inf)
        self.hi = np.where(limited, model.jnt_range[task.joints, 1], np.inf)

    def jacobian(self, model: mujoco.MjModel, data: mujoco.MjData, rows: np.ndarray) -> np.ndarray:
        """The tool's Jacobian: 3 rows for the point, then the rotation Jacobian rows that `rows` (k x 3) picks."""
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.site)
        return np.vstack([self._jacp[:, self.d], rows @ self._jacr[:, self.d]])

    def accel_drift(self, model: mujoco.MjModel, data: mujoco.MjData, rows: np.ndarray) -> np.ndarray:
        """J-dot times the joint speeds: the acceleration the tool has with the joints not accelerating."""
        mujoco.mj_jacDot(model, data, self._jdp, self._jdr, data.site_xpos[self.site], self.body)
        dq = data.qvel[self.d]
        return np.concatenate([self._jdp[:, self.d] @ dq, rows @ (self._jdr[:, self.d] @ dq)])

    def mass(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_fullM(model, data, self._M)
        return self._M[np.ix_(self.d, self.d)]

    def bias(self, data: mujoco.MjData) -> np.ndarray:
        """Gravity and Coriolis torques the arm needs held up, less what the model cancels itself (gravcomp)."""
        return data.qfrc_bias[self.d] - data.qfrc_gravcomp[self.d]


class DiffIK:
    """Reference joint angles that follow a moving tool target: one damped least-squares step per control step,
    started from where the arm is, with the joints left over steered to a rest pose."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, arm: _Arm):
        self.model, self.arm = model, arm
        self.scratch = mujoco.MjData(model)
        self.rest = home_qpos(model)[arm.q]
        self.q_ref = data.qpos[arm.q].copy()
        self.dq_ref = np.zeros(arm.n)
        self.dt = model.opt.timestep

    def step(self, data: mujoco.MjData, target: ToolTarget) -> tuple[np.ndarray, np.ndarray]:
        arm, s = self.arm, self.scratch
        s.qpos[:] = data.qpos
        s.qpos[arm.q] = self.q_ref
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        rows, turn = attitude(s.site_xmat[arm.site].reshape(3, 3), target)
        J = arm.jacobian(self.model, s, rows)
        J[3:] *= ROT_WEIGHT
        err = np.concatenate([target.pos - s.site_xpos[arm.site], ROT_WEIGHT * turn])
        pinv = J.T @ np.linalg.inv(J @ J.T + IK_DAMPING**2 * np.eye(len(err)))
        pull = POSTURE_OMEGA * self.dt * (self.rest - self.q_ref)  # rad per step toward the rest pose
        dq = pinv @ err + (np.eye(arm.n) - pinv @ J) @ pull
        step = IK_SPEED * self.dt
        new = np.clip(self.q_ref + np.clip(dq, -step, step), arm.lo, arm.hi)
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
        self.feel = np.zeros(3)  # the filtered force the surroundings exert on the tool
        self._alpha = 1.0 - math.exp(-2 * math.pi * SENSE_HZ * self.dt)
        self.rest = home_qpos(model)[task.qadr]
        self.last: ToolTarget | None = None
        self.normal_damping = 0.0
        self._configure(model, data, **gains)

    def _configure(self, model: mujoco.MjModel, data: mujoco.MjData, **gains: Any) -> None:
        if gains:
            raise TaskError(f"{self.label}: unknown settings {sorted(gains)}")

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.feel += self._alpha * (self.sense.read(model, data) - self.feel)
        target = self.source.target(data, self.feel)
        self.last = target
        data.ctrl[self.task.actuators] = self.command(model, data, target)

    def command(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        return self.torque.ctrl(data, self.compute(model, data, target))

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        raise NotImplementedError

    def posture(self, M: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """A weak spring and damper toward the rest pose, torque per joint scaled by the joint's inertia."""
        w = POSTURE_OMEGA
        q, dq = data.qpos[self.arm.q], data.qvel[self.arm.d]
        return np.diag(M) * (w * w * (self.rest - q) - 2 * w * dq)

    def normal_damper(self, J: np.ndarray, Minv: np.ndarray, normal: np.ndarray) -> float:
        """N s/m of damping along a surface normal while pressing: `normal_damping` times the most one control step can
        carry (a damper D on a mass m is stable while D * dt / m is under 1). A stiff contact and a sliding tool with
        friction make the arm hop on the surface; without a strong damper along the normal, the force flickers between
        zero and twice its value."""
        mass = 1.0 / max(float(normal @ (J[:3] @ Minv @ J[:3].T) @ normal), 1e-6)
        return min(self.normal_damping * mass / self.dt, DAMPER_MAX)

    @staticmethod
    def feedforward(target: ToolTarget) -> np.ndarray:
        """The force the tool applies to its surroundings at the target (world axes): pressing along -normal."""
        if target.normal is None or target.push <= 0:
            return np.zeros(3)
        return -np.asarray(target.normal, float) * min(float(target.push), FORCE_MAX)

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"controller": self.label}
        if self.last is not None and self.last.normal is not None:
            out["pressing_n"] = round(float(self.feel @ np.asarray(self.last.normal)), 2)
        return out


class ServoPath(_Controller):
    """ "ik" on a contact task: inverse kinematics at every step, the angles sent to the arm's own position servos."""

    label = "ik"

    def _configure(self, model: mujoco.MjModel, data: mujoco.MjData, **gains: Any) -> None:
        super()._configure(model, data, **gains)
        self.ik = DiffIK(model, data, self.arm)

    def command(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        return self.ik.step(data, target)[0]


class JointPDPath(_Controller):
    """ "pd" on a contact task: inverse kinematics at every step, a stiff joint PD on it with gravity compensation."""

    label = "pd"

    def _configure(self, model: mujoco.MjModel, data: mujoco.MjData, omega: float = 20.0, zeta: float = 1.0) -> None:
        self.omega, self.zeta = positive("omega", omega, 200.0), positive("zeta", zeta, 10.0)
        self.ik = DiffIK(model, data, self.arm)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        q_ref, dq_ref = self.ik.step(data, target)
        a = self.arm
        inertia = np.diag(a.mass(model, data))
        w, z = self.omega, self.zeta
        return inertia * (w * w * (q_ref - data.qpos[a.q]) + 2 * z * w * (dq_ref - data.qvel[a.d])) + a.bias(data)

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
        self.ik = DiffIK(model, data, self.arm)
        probe = mujoco.MjData(model)
        probe.qpos[:] = data.qpos
        mujoco.mj_forward(model, probe)
        # Fixed at the pose it starts in: an impedance is a constant spring, it does not stiffen where the arm is heavy.
        inertia = np.diag(self.arm.mass(model, probe))
        self.K = inertia * self.omega**2 if stiffness is None else per_axis("stiffness", stiffness, self.arm.n)
        self.D = 2 * self.zeta * np.sqrt(self.K * inertia)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        q_ref, dq_ref = self.ik.step(data, target)
        a = self.arm
        tau = self.K * (q_ref - data.qpos[a.q]) + self.D * (dq_ref - data.qvel[a.d]) + a.bias(data)
        force = self.feedforward(target)
        if force.any():
            tau = tau + a.jacobian(model, data, np.zeros((0, 3)))[:3].T @ force
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
        soft_stiffness: float | list[float] | None = None,
        rot_stiffness: float = 30.0,
        zeta: float = 1.0,
        normal_damping: float = 0.6,
    ) -> None:
        self.normal_damping = positive("normal_damping", normal_damping, 0.95)
        self.K = per_axis("stiffness", stiffness, 3)
        self.K_soft = self.K if soft_stiffness is None else per_axis("soft_stiffness", soft_stiffness, 3)
        self.K_rot = positive("rot_stiffness", rot_stiffness)
        self.zeta = positive("zeta", zeta, 10.0)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        a = self.arm
        R = data.site_xmat[a.site].reshape(3, 3)
        rows, turn = attitude(R, target)
        J = a.jacobian(model, data, rows)
        M = a.mass(model, data)
        linear = (1 - target.compliance) * self.K + target.compliance * self.K_soft
        K = np.concatenate([linear, np.full(len(turn), self.K_rot)])
        vel = np.concatenate([target.vel, np.zeros(len(turn))])
        err = np.concatenate([target.pos - data.site_xpos[a.site], turn])
        Minv = np.linalg.inv(M)
        lam = np.linalg.inv(J @ Minv @ J.T + TOOL_REGULARIZE * np.eye(len(err)))
        D = 2 * self.zeta * np.sqrt(K * np.diag(lam))
        v = J @ data.qvel[a.d]
        force = K * err + D * (vel - v)
        force[:3] += self.feedforward(target)
        if target.pressing and target.normal is not None:
            n = np.asarray(target.normal, float)
            force[:3] -= n * self.normal_damper(J, Minv, n) * float(n @ v[:3])
        null = np.eye(a.n) - J.T @ np.linalg.pinv(J.T)
        return J.T @ force + null @ self.posture(M, data) + a.bias(data)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "stiffness_n_m": [round(float(k), 1) for k in self.K]}


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
        soft_omega: float | None = None,
        rot_omega: float = 10.0,
        normal_omega: float = 30.0,
        force_gain: float = 0.6,
        force_integral: float = 4.0,
        normal_damping: float = 0.6,
    ) -> None:
        self.omega, self.zeta = positive("omega", omega, 200.0), positive("zeta", zeta, 10.0)
        self.soft_omega = self.omega if soft_omega is None else positive("soft_omega", soft_omega, 200.0)
        self.rot_omega = positive("rot_omega", rot_omega, 200.0)
        self.normal_omega = positive("normal_omega", normal_omega, 200.0)  # the soft spring that keeps the pad down
        self.force_gain = positive("force_gain", force_gain, 20.0)  # N commanded per N of force error
        self.force_integral = positive("force_integral", force_integral, 200.0)  # per second
        self.normal_damping = positive("normal_damping", normal_damping, 0.95)
        self.wound = 0.0  # N, the force loop's integral term

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, target: ToolTarget) -> np.ndarray:
        a = self.arm
        R = data.site_xmat[a.site].reshape(3, 3)
        rows, turn = attitude(R, target)
        J = a.jacobian(model, data, rows)
        M = a.mass(model, data)
        v = J @ data.qvel[a.d]
        w, z, wr = self.omega, self.zeta, self.rot_omega
        w = (1 - target.compliance) * w + target.compliance * self.soft_omega
        normal = None if target.normal is None else np.asarray(target.normal, float)
        err_p = target.pos - data.site_xpos[a.site]
        vel_err = target.vel - v[:3]
        pressing = target.pressing and normal is not None
        err_n = vel_n = 0.0
        if pressing:  # the error along the normal is the normal channel's, not the tool-space inertia's
            err_n, vel_n = float(normal @ err_p), float(normal @ vel_err)
            err_p, vel_err = err_p - normal * err_n, vel_err - normal * vel_n
        accel = np.concatenate([w * w * err_p + 2 * z * w * vel_err, wr * wr * turn - 2 * z * wr * v[3:]])
        Minv = np.linalg.inv(M)
        lam = np.linalg.inv(J @ Minv @ J.T + TOOL_REGULARIZE * np.eye(len(accel)))
        force = lam @ (accel - a.accel_drift(model, data, rows))
        if pressing:
            # Along the normal only the force loop and a soft spring act: the push the tool-space inertia would add for
            # the sideways errors (it couples the axes) is taken out, or the force would swing with every sideways step.
            force[:3] -= normal * float(normal @ force[:3])
            mass = 1.0 / max(float(normal @ (J[:3] @ Minv @ J[:3].T) @ normal), 1e-6)
            wn = self.normal_omega
            at_surface = err_n + target.depth  # the target's depth is a position controller's push
            hold = wn * wn * at_surface if target.anchored else 0.0
            spring = mass * (hold + 2 * z * wn * vel_n)
            force[:3] += normal * spring + self._press(normal, target.push)
            force[:3] -= normal * self.normal_damper(J, Minv, normal) * float(normal @ v[:3])
        else:
            self.wound = 0.0
            force[:3] += self.feedforward(target)
        null = np.eye(a.n) - J.T @ (lam @ J @ Minv)  # dynamically consistent: the spare joints cannot move the tool
        return J.T @ force + null @ self.posture(M, data) + a.bias(data)

    def _press(self, normal: np.ndarray, wanted: float) -> np.ndarray:
        """The force on the tool along the normal: the wanted push, plus a correction from the measured push."""
        error = wanted - max(0.0, float(self.feel @ normal))
        self.wound = float(np.clip(self.wound + self.force_integral * error * self.dt, -wanted, FORCE_MAX))
        push = float(np.clip(wanted + self.force_gain * error + self.wound, 0.0, FORCE_MAX))
        return -normal * push

    def status(self) -> dict[str, Any]:
        return {**super().status(), "omega": self.omega, "force_gain": self.force_gain}


class PathPush:
    """Cartesian impedance that follows a moving pose (the joint values of a path, worked out beforehand) and pushes
    with at most `limit` newtons: operate's way of turning a valve or tipping a lever by force. The tool is a mass on a
    spring and damper toward where the path says it should be, the spring's pull is cut off at the limit, so a handle
    that holds back more than that stops the hand, which then stands pushing at the limit."""

    LEVER = 0.12  # m: the limit on the turning torque is the limit on the force at this arm

    def __init__(self, model: mujoco.MjModel, arm: Any, limit: float, stiffness: float = 3000.0, zeta: float = 1.0):
        owner = arm.robot.split("/", 1)[0] + "/"
        self.task = types.SimpleNamespace(
            site=arm.site,
            joints=list(arm.joints),
            qadr=np.asarray(arm.chain.qpos_adr),
            dadr=np.asarray(arm.chain.dof_adr),
            actuators=np.asarray(arm.actuators),
            robot=types.SimpleNamespace(prefix=owner, name=arm.robot),
        )
        self.arm = _Arm(model, self.task)
        self.torque = TorqueMap(model, self.task.actuators)
        self.limit = positive("force", limit, 1000.0)
        self.K, self.K_rot, self.zeta = positive("stiffness", stiffness), 60.0, zeta
        self.scratch = mujoco.MjData(model)
        self.model = model
        self.sense = ToolForce.of_arm(model, self.task)
        self.rest = home_qpos(model)[self.task.qadr]
        self.error = 0.0  # m: how far the tool is from where the path puts it
        self.peak_commanded = 0.0  # N: the most the spring asked for, after the limit
        self.peak_felt = 0.0  # N: the most the hand's contacts pushed back with
        self.held_at_limit = 0.0  # s spent pushing at the limit

    def command(self, data: mujoco.MjData, q_path: np.ndarray, dt: float) -> np.ndarray:
        """The actuators' commands that push the hand toward the pose the path has at joint values `q_path`."""
        model, a = self.model, self.arm
        s = self.scratch
        s.qpos[:] = data.qpos
        s.qpos[self.task.qadr] = q_path
        mujoco.mj_kinematics(model, s)
        x_ref, R_ref = s.site_xpos[a.site].copy(), s.site_xmat[a.site].reshape(3, 3).copy()
        J = a.jacobian(model, data, np.eye(3))
        M = a.mass(model, data)
        v = J @ data.qvel[a.d]
        x, R = data.site_xpos[a.site], data.site_xmat[a.site].reshape(3, 3)
        err = np.concatenate([x_ref - x, rotation_error(R, R_ref)])
        lam = np.linalg.inv(J @ np.linalg.inv(M) @ J.T + TOOL_REGULARIZE * np.eye(6))
        K = np.concatenate([np.full(3, self.K), np.full(3, self.K_rot)])
        force = K * err - 2 * self.zeta * np.sqrt(K * np.diag(lam)) * v
        push = float(np.linalg.norm(force[:3]))
        if push > self.limit:
            force[:3] *= self.limit / push
            self.held_at_limit += dt
        turn = float(np.linalg.norm(force[3:]))
        if turn > self.limit * self.LEVER:
            force[3:] *= self.limit * self.LEVER / turn
        self.error = float(np.linalg.norm(err[:3]))
        self.peak_commanded = max(self.peak_commanded, min(push, self.limit))
        self.peak_felt = max(self.peak_felt, float(np.linalg.norm(self.sense.read(model, data))))
        null = np.eye(a.n) - J.T @ np.linalg.pinv(J.T)
        tau = J.T @ force + null @ _posture(M, data, a, self.rest) + a.bias(data)
        return self.torque.ctrl(data, tau)


def _posture(M: np.ndarray, data: mujoco.MjData, arm: _Arm, rest: np.ndarray) -> np.ndarray:
    w = POSTURE_OMEGA
    return np.diag(M) * (w * w * (rest - data.qpos[arm.q]) - 2 * w * data.qvel[arm.d])


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
    gains = {**getattr(task, "gains", {}).get(method, {}), **(params or {})}  # what suits the task, unless told
    return table[method](task, model, data, source, **gains)
