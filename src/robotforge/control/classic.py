"""Classic controllers: hold a pose, servo IK, joint PD with gravity compensation, Cartesian
(operational-space) PD, LQR, and a go-to-goal driver.

Every class follows the session's Controller protocol (`rate_hz`, `update`, `status`),
so it can drive the live world and the viewer exactly like the built-in behaviors.
"""

from __future__ import annotations

import copy
import math
import types
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.drive import DiffDrive
from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.control.actuation import ActionScale, TorqueMap
from robotforge.control.tasks import Task, TaskError
from robotforge.control.tasks.base import home_qpos


def min_jerk(s: float) -> tuple[float, float]:
    """Smooth 0 -> 1 profile (position, and its rate per unit s) with zero speed at both ends."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


def physics_rate(model: mujoco.MjModel) -> float:
    return 1.0 / model.opt.timestep


class Hold:
    """Send fixed commands: for servo robots that means "hold this pose" (a stiff position loop)."""

    def __init__(self, task: Task, ctrl: np.ndarray, label: str = "hold"):
        self.task = task
        self.ctrl = np.asarray(ctrl, float).copy()
        self.label = label
        self.rate_hz = 50.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.ctrl[self.task.actuators] = self.ctrl

    def status(self) -> dict[str, Any]:
        return {"controller": self.label, "robot": self.task.robot.name}


class GaitBase:
    """The built-in trot gait, able to say what it will command next, so MPC can correct it."""

    def __init__(self, task: Task, gait):
        self.task = task
        self.gait = gait
        self.rate_hz = gait.rate_hz

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.gait.update(model, data)

    def status(self) -> dict[str, Any]:
        return self.gait.status()

    def preview(self, model: mujoco.MjModel, data: mujoco.MjData, times: np.ndarray) -> np.ndarray:
        # Run a throwaway copy of the gait forward in time; it only reads the clock and the trunk
        # attitude (held at its current value) and writes ctrl.
        ghost = copy.copy(self.gait)
        ghost.legs = [copy.copy(leg) for leg in self.gait.legs]
        scratch = types.SimpleNamespace(time=0.0, xquat=data.xquat, ctrl=np.array(data.ctrl))
        out = np.empty((len(times), len(self.task.actuators)))
        for i, dt in enumerate(times):
            scratch.time = float(data.time) + float(dt)
            ghost.update(model, scratch)
            out[i] = scratch.ctrl[self.task.actuators]
        return out


# ---- arms -----------------------------------------------------------------------------------


class _ReachPlan:
    """Shared by the arm controllers: an IK goal and a smooth min-jerk path to it."""

    SPEED = 0.25  # m/s average tool speed along the path

    def __init__(self, task: Task, model: mujoco.MjModel, data: mujoco.MjData):
        if not hasattr(task, "site"):
            raise TaskError(f"{type(self).__name__} works with the reach task")
        self.task = task
        self.qadr, self.dadr = task.qadr, task.dadr
        body = int(model.site_bodyid[task.site])
        self.chain = Chain.from_joints(model, task.joints, body, model.site_pos[task.site].copy())
        self.ik = IKSolver(model, data.qpos.copy())
        self.q_start = data.qpos[self.qadr].copy()
        self.q_goal, self.ik_error = self.ik.solve(self.chain, task.target, self.q_start, iterations=300, damping=1e-4)
        self.x_start = data.site_xpos[task.site].copy()
        dist = float(np.linalg.norm(task.target - self.x_start))
        self.duration = float(np.clip(dist / self.SPEED, 0.6, 2.5))
        self.t0 = float(data.time)

    def joint_ref(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        s, ds = min_jerk((t - self.t0) / self.duration)
        dq = self.q_goal - self.q_start
        return self.q_start + s * dq, ds * dq / self.duration

    def tool_ref(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        s, ds = min_jerk((t - self.t0) / self.duration)
        dx = self.task.target - self.x_start
        return self.x_start + s * dx, ds * dx / self.duration


class ServoIK:
    """ "ik": solve inverse kinematics once and let each joint's own position servo follow a smooth path.

    The simplest arm control, and what most hobby arms do. Servos are springs, so gravity
    makes the arm sag a little below the goal; the next two controllers fix that."""

    def __init__(self, task: Task, model: mujoco.MjModel, data: mujoco.MjData):
        self.task = task
        self.plan = _ReachPlan(task, model, data)
        self.rate_hz = 100.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        q, _ = self.plan.joint_ref(data.time)
        data.ctrl[self.task.actuators] = q

    def status(self) -> dict[str, Any]:
        return {"controller": "ik", "ik_error_m": round(self.plan.ik_error, 4)}


class JointPD:
    """ "pd": joint-space PD on the IK path plus gravity compensation.

    torque = M (w^2 (q_ref - q) + 2 z w (dq_ref - dq)) + gravity and Coriolis. Gains are scaled by
    each joint's inertia so one natural frequency `omega` fits small and large arms."""

    def __init__(self, task: Task, model: mujoco.MjModel, data: mujoco.MjData, omega: float = 20.0, zeta: float = 1.0):
        self.task = task
        self.plan = _ReachPlan(task, model, data)
        self.torque = TorqueMap(model, task.actuators)
        self.omega, self.zeta = omega, zeta
        self.rate_hz = physics_rate(model)
        self._M = np.zeros((model.nv, model.nv))

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        q_ref, dq_ref = self.plan.joint_ref(data.time)
        q, dq = data.qpos[self.plan.qadr], data.qvel[self.plan.dadr]
        mujoco.mj_fullM(model, data, self._M)
        inertia = np.diag(self._M)[self.plan.dadr]
        w, z = self.omega, self.zeta
        tau = inertia * (w * w * (q_ref - q) + 2 * z * w * (dq_ref - dq)) + data.qfrc_bias[self.plan.dadr]
        data.ctrl[self.task.actuators] = self.torque.ctrl(data, tau)

    def status(self) -> dict[str, Any]:
        return {"controller": "pd", "omega": self.omega, "zeta": self.zeta}


class CartesianPD:
    """ "cartesian": operational-space control. A virtual spring pulls the tool straight toward the
    target in space (no IK), torque = J^T F + gravity; a weak posture spring uses the leftover joints."""

    def __init__(self, task: Task, model: mujoco.MjModel, data: mujoco.MjData, omega: float = 15.0, zeta: float = 1.0):
        self.task = task
        self.plan = _ReachPlan(task, model, data)
        self.torque = TorqueMap(model, task.actuators)
        self.omega, self.zeta = omega, zeta
        self.rate_hz = physics_rate(model)
        self.q_rest = home_qpos(model)[self.plan.qadr]
        self._jacp = np.zeros((3, model.nv))
        self._M = np.zeros((model.nv, model.nv))

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        d = self.plan.dadr
        x_ref, v_ref = self.plan.tool_ref(data.time)
        mujoco.mj_jacSite(model, data, self._jacp, None, self.task.site)
        J = self._jacp[:, d]
        mujoco.mj_fullM(model, data, self._M)
        M = self._M[np.ix_(d, d)]
        Minv = np.linalg.inv(M)
        lam = np.linalg.inv(J @ Minv @ J.T + 1e-4 * np.eye(3))  # task-space inertia, damped near singularities
        x, v = data.site_xpos[self.task.site], J @ data.qvel[d]
        w, z = self.omega, self.zeta
        force = lam @ (w * w * (x_ref - x) + 2 * z * w * (v_ref - v))
        q, dq = data.qpos[self.plan.qadr], data.qvel[d]
        posture = np.diag(M) * (4.0 * (self.q_rest - q) - 4.0 * dq)
        null = np.eye(len(d)) - J.T @ (lam @ J @ Minv)
        tau = J.T @ force + null @ posture + data.qfrc_bias[d]
        data.ctrl[self.task.actuators] = self.torque.ctrl(data, tau)

    def status(self) -> dict[str, Any]:
        return {"controller": "cartesian", "omega": self.omega}


# ---- LQR ------------------------------------------------------------------------------------


def solve_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Discrete algebraic Riccati equation by the structure-preserving doubling algorithm.

    Converges quadratically, so it handles a 2 ms physics step (eigenvalues very close to 1)
    where the plain Riccati iteration would need tens of thousands of steps."""
    n = A.shape[0]
    Ak, Gk, Hk = A.copy(), B @ np.linalg.solve(R, B.T), Q.copy()
    eye = np.eye(n)
    for _ in range(iterations):
        W = np.linalg.solve(eye + Gk @ Hk, eye)
        A_next = Ak @ W @ Ak
        G_next = Gk + Ak @ W @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ W @ Ak
        done = np.linalg.norm(H_next - Hk) <= 1e-10 * max(1.0, np.linalg.norm(H_next))
        Ak, Gk, Hk = A_next, (G_next + G_next.T) / 2, (H_next + H_next.T) / 2
        if done:
            break
    return Hk


class LQR:
    """ "lqr": the optimal linear controller around a balance point.

    MuJoCo linearizes one physics step around the robot's home pose (mjd_transitionFD gives
    x' = A x + B u), the Riccati equation turns the weights Q (how much each error matters) and
    R (how expensive pushing is) into a gain matrix K, and then u = u_home - K (x - x_home)."""

    def __init__(
        self,
        task: Task,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_weights: list[float] | None = None,
        r_weight: float = 0.1,
    ):
        self.task = task
        self.dofs = task.robot.dofs()
        free_or_ball = (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL))
        if any(model.jnt_type[model.dof_jntid[d]] in free_or_ball for d in self.dofs):
            raise TaskError("this LQR handles robots on hinge and slide joints (no floating base)")
        nd, acts = len(self.dofs), task.actuators
        self.qpos_eq = data.qpos.copy()
        home = home_qpos(model)
        joints = sorted({int(model.dof_jntid[d]) for d in self.dofs})
        for j in joints:
            self.qpos_eq[model.jnt_qposadr[j]] = home[model.jnt_qposadr[j]]
        self.ctrl_eq = np.zeros(len(acts))
        lin = mujoco.MjData(model)
        lin.qpos[:] = self.qpos_eq
        lin.qvel[:] = 0
        lin.ctrl[:] = data.ctrl
        lin.ctrl[acts] = self.ctrl_eq
        mujoco.mj_forward(model, lin)
        nx = 2 * model.nv + model.na
        A, B = np.zeros((nx, nx)), np.zeros((nx, model.nu))
        mujoco.mjd_transitionFD(model, lin, 1e-6, True, A, B, None, None)
        idx = np.concatenate([self.dofs, model.nv + self.dofs])
        self.A, self.B = A[np.ix_(idx, idx)], B[np.ix_(idx, acts)]
        weights = q_weights if q_weights is not None else [1.0] * (2 * nd)
        if len(weights) != 2 * nd:
            raise TaskError(f"q_weights needs {2 * nd} numbers: {nd} positions then {nd} speeds")
        self.Q = np.diag(np.asarray(weights, float))
        self.R = r_weight * np.eye(len(acts))
        P = solve_dare(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R + self.B.T @ P @ self.B, self.B.T @ P @ self.A)
        # Unlimited hinges (a pole that may spin all the way round) measure angle errors modulo a turn.
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        dof_joints = model.dof_jntid[self.dofs]
        self.wrap = (model.jnt_type[dof_joints] == hinge) & ~model.jnt_limited[dof_joints].astype(bool)
        self.scale = ActionScale(model, acts)
        self.rate_hz = physics_rate(model)
        self._dq = np.zeros(model.nv)
        closed = np.abs(np.linalg.eigvals(self.A - self.B @ self.K))
        self.stable = bool(closed.max() < 1.0)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_differentiatePos(model, self._dq, 1.0, self.qpos_eq, data.qpos)
        dq = self._dq[self.dofs]
        dq = np.where(self.wrap, (dq + math.pi) % (2 * math.pi) - math.pi, dq)
        x = np.concatenate([dq, data.qvel[self.dofs]])
        data.ctrl[self.task.actuators] = self.scale.clip(self.ctrl_eq - self.K @ x)

    def status(self) -> dict[str, Any]:
        return {"controller": "lqr", "gains": np.round(self.K, 3).tolist(), "stable": self.stable}


class StateFeedback:
    """ "pd" for balance: a hand-tuned rule, push = k_angle * angle + k_spin * spin + small cart terms.

    It is what you would write by intuition ("push toward where the pole leans"); LQR computes
    the same kind of rule but with gains chosen optimally from the physics."""

    def __init__(self, task: Task, gains: list[float] | None = None):
        if not hasattr(task, "hinge"):
            raise TaskError("this rule is written for the cart-pole balance task")
        self.task = task
        self.gains = np.asarray(gains if gains is not None else [0.3, 4.0, 0.5, 0.6], float)
        if self.gains.shape != (4,):
            raise TaskError("gains are [cart position, pole angle, cart speed, pole spin]")
        self.rate_hz = 100.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        x = np.array(self.task.state(data))
        data.ctrl[self.task.actuators[0]] = float(np.clip(self.gains @ x, -1.0, 1.0))

    def status(self) -> dict[str, Any]:
        return {"controller": "pd", "gains": self.gains.tolist()}


# ---- wheels ---------------------------------------------------------------------------------


class GoToGoal:
    """ "pd" for drive_to: turn toward the goal, drive forward as long as it is roughly ahead,
    slow down when close. Speed and turn rate are proportional to the errors."""

    def __init__(self, task: Task, model: mujoco.MjModel, k_turn: float = 3.0, k_speed: float = 1.5):
        if not hasattr(task, "goal"):
            raise TaskError("go-to-goal driving works with the drive_to task")
        self.task = task
        try:
            self.drive = DiffDrive(model, task.robot.name)
        except (ValueError, KeyError, StopIteration) as exc:
            raise TaskError(str(exc)) from exc
        self.k_turn, self.k_speed = k_turn, k_speed
        self.rate_hz = 50.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pos = self.task.position(data)
        to_goal = self.task.goal - pos
        dist = float(np.linalg.norm(to_goal))
        heading = float(self.task._heading(data.qpos))
        err = (math.atan2(to_goal[1], to_goal[0]) - heading + math.pi) % (2 * math.pi) - math.pi
        if dist < 0.5 * self.task.params.tolerance:
            speed = turn = 0.0
        else:
            turn = self.k_turn * err
            speed = min(self.drive.max_speed, self.k_speed * dist) * max(0.0, math.cos(err)) ** 2
        self.drive.command(speed, turn)
        self.drive.update(model, data)

    def status(self) -> dict[str, Any]:
        return {"controller": "pd", **self.drive.status()}
