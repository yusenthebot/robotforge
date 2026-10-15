"""Two-wheeled self-balancing robots: an LQR balancer on a model MuJoCo linearizes for us.

A robot standing on two wheels on one axle falls over unless it keeps driving its
wheels under its center of mass, like a Segway. The controller looks at four numbers:
how far the body leans (pitch) and how fast, where the axle is and how fast it rolls.
MuJoCo predicts how a small change in each of them, and in the wheel command, plays
out over one control period (finite differences on the real model, contacts and motor
limits included); the Riccati equation turns that linear model into the best gains.
Driving is the same balancing with a moving reference: lean, roll, keep up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.behaviors import find_wheels
from robotforge.sim.geometry import collides, lowest_point

CONTROL_HZ = 100.0
FALL_ANGLE = 0.8  # rad from balance: past this the robot is lying down
SUPPORT_TOLERANCE = 0.003  # m: geoms this close to the lowest point carry the robot
# Weights for the LQR: errors in [pitch rad, pitch rate rad/s, position m, speed m/s], and the wheel command.
Q_WEIGHTS = (60.0, 1.0, 4.0, 2.0)
R_WEIGHT = 0.05
SPEED_SMOOTHING = 0.6  # s: commands ramp in, a sudden full-speed command would tip it over
MAX_LEAN_SPEED = 0.6  # share of the wheels' top speed offered for driving; the rest is for balancing
POSITION_SLACK = 0.4  # m: after a shove, stop chasing the old spot beyond this


@dataclass(frozen=True)
class Balancer:
    """Where a two-wheeled robot's parts are in the model."""

    robot: str
    root: int  # body with the free joint
    qadr: int  # free joint qpos address
    dadr: int  # free joint dof address
    wheels: tuple[int, int]  # left, right actuator ids
    signs: tuple[float, float]  # +1 when a positive command rolls that wheel forward
    wheel_bodies: tuple[int, int]
    wheel_dofs: tuple[int, int]
    radius: float
    track: float
    top_speed: float  # wheel rad/s


def find_balancer(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray | None = None) -> Balancer | None:
    """A floating robot whose only ground contacts at its home pose are two speed-driven coaxial wheels."""
    wheels = find_wheels(model, robot)
    if wheels is None or len(wheels.left) != 1 or len(wheels.right) != 1:
        return None
    left, right = wheels.left[0], wheels.right[0]
    joints = [int(model.actuator_trnid[a, 0]) for a in (left, right)]
    bodies = [int(model.jnt_bodyid[j]) for j in joints]
    root = int(model.body_rootid[bodies[0]])
    free = [
        j
        for j in range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if not free:
        return None
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 if home_qpos is None else home_qpos
    mujoco.mj_kinematics(model, data)
    own = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root and collides(model, g)]
    if not own:
        return None
    low = {g: float(lowest_point(model, data, g)[2]) for g in own}
    bottom = min(low.values())
    support = {int(model.geom_bodyid[g]) for g, z in low.items() if z < bottom + SUPPORT_TOLERANCE}
    if not support <= set(bodies):
        return None  # a caster, a foot or the body itself touches the floor too: it stands on its own
    return Balancer(
        robot=robot,
        root=root,
        qadr=int(model.jnt_qposadr[free[0]]),
        dadr=int(model.jnt_dofadr[free[0]]),
        wheels=(left, right),
        signs=(wheels.left_signs[0], wheels.right_signs[0]),
        wheel_bodies=(bodies[0], bodies[1]),
        wheel_dofs=(int(model.jnt_dofadr[joints[0]]), int(model.jnt_dofadr[joints[1]])),
        radius=wheels.radius,
        track=wheels.track,
        top_speed=float(min(model.actuator_ctrlrange[left][1], model.actuator_ctrlrange[right][1])),
    )


# ---- the planar state: lean, lean rate, axle position, axle speed --------------------------------------


def heading_frame(data: mujoco.MjData, b: Balancer) -> tuple[np.ndarray, np.ndarray]:
    """(forward, left) directions on the floor, from the body's heading."""
    x = data.xmat[b.root].reshape(3, 3)[:, 0]
    fwd = np.array([x[0], x[1], 0.0])
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return fwd, np.array([-fwd[1], fwd[0], 0.0])


def axle_point(data: mujoco.MjData, b: Balancer) -> np.ndarray:
    return (data.xpos[b.wheel_bodies[0]] + data.xpos[b.wheel_bodies[1]]) / 2


def _omega(data: mujoco.MjData, b: Balancer) -> np.ndarray:
    """Body angular velocity in world coordinates (free joints store it in the body frame)."""
    return data.xmat[b.root].reshape(3, 3) @ data.qvel[b.dadr + 3 : b.dadr + 6]


def measure(data: mujoco.MjData, b: Balancer, fwd: np.ndarray, left: np.ndarray, origin: np.ndarray) -> np.ndarray:
    z_axis = data.xmat[b.root].reshape(3, 3)[:, 2]
    pitch = math.atan2(float(z_axis @ fwd), float(z_axis[2]))
    omega = _omega(data, b)
    axle = axle_point(data, b)
    v_axle = data.qvel[b.dadr : b.dadr + 3] + np.cross(omega, axle - data.xpos[b.root])
    return np.array([pitch, float(omega @ left), float((axle - origin) @ fwd), float(v_axle @ fwd)])


# ---- model-based design ------------------------------------------------------------------------------


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    return q


def set_lean(model, data, b: Balancer, axle: np.ndarray, left: np.ndarray, angle: float) -> None:
    """Tilt the body about the axle so the wheels stay where they are."""
    q = _rotation(left, angle)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, q)
    pos = data.qpos[b.qadr : b.qadr + 3]
    data.qpos[b.qadr : b.qadr + 3] = axle + rot.reshape(3, 3) @ (pos - axle)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, data.qpos[b.qadr + 3 : b.qadr + 7].copy())
    data.qpos[b.qadr + 3 : b.qadr + 7] = out


def equilibrium(model: mujoco.MjModel, b: Balancer, home_qpos: np.ndarray) -> tuple[np.ndarray, float]:
    """Full physics state of the robot resting balanced (center of mass over the axle), and its lean."""
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)
    fwd, left = heading_frame(data, b)
    axle = axle_point(data, b)
    com = data.subtree_com[b.root]
    lean = math.atan2(float((com - axle) @ fwd), float(com[2] - axle[2]))
    set_lean(model, data, b, axle, left, -lean)
    # Let the tyres settle into the floor with the lean and the spot held, so the linearization starts at rest.
    held = data.qpos[b.qadr : b.qadr + 7].copy()
    for _ in range(int(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        z = data.qpos[b.qadr + 2]
        data.qpos[b.qadr : b.qadr + 7] = held
        data.qpos[b.qadr + 2] = z
        vz = data.qvel[b.dadr + 2]
        data.qvel[:] = 0.0
        data.qvel[b.dadr + 2] = vz
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state = np.empty(size)
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    z_axis = data.xmat[b.root].reshape(3, 3)[:, 2]
    return state, math.atan2(float(z_axis @ fwd), float(z_axis[2]))


def linearize(model: mujoco.MjModel, b: Balancer, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x' = A x + B u over one control period, for x = [lean, lean rate, position, speed] and u = wheel speed."""
    data = mujoco.MjData(model)
    full = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mujoco.mj_setState(model, data, state, full)
    mujoco.mj_forward(model, data)
    fwd, left = heading_frame(data, b)
    origin = axle_point(data, b)
    steps = max(1, round(1.0 / (CONTROL_HZ * model.opt.timestep)))
    eps = np.array([1e-3, 1e-2, 1e-3, 1e-2, 1e-2])

    def roll_out(dx: np.ndarray, du: float) -> np.ndarray:
        mujoco.mj_setState(model, data, state, full)
        mujoco.mj_forward(model, data)
        axle = axle_point(data, b)
        set_lean(model, data, b, axle, left, dx[0])
        spin = dx[1] * left  # lean rate about the axle: wheels keep still on the floor
        rot = data.xmat[b.root].reshape(3, 3)
        data.qvel[b.dadr + 3 : b.dadr + 6] += rot.T @ spin
        data.qvel[b.dadr : b.dadr + 3] += np.cross(spin, data.qpos[b.qadr : b.qadr + 3] - axle)
        data.qpos[b.qadr : b.qadr + 3] += dx[2] * fwd
        data.qvel[b.dadr : b.dadr + 3] += dx[3] * fwd
        for dof, sign in zip(b.wheel_dofs, b.signs, strict=True):
            data.qvel[dof] += sign * (dx[3] / b.radius - dx[1])
        for act, sign in zip(b.wheels, b.signs, strict=True):
            data.ctrl[act] = sign * du
        for _ in range(steps):
            mujoco.mj_step(model, data)
        return measure(data, b, fwd, left, origin)

    A = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps[i]
        A[:, i] = (roll_out(dx, 0.0) - roll_out(-dx, 0.0)) / (2 * eps[i])
    B = ((roll_out(np.zeros(4), eps[4]) - roll_out(np.zeros(4), -eps[4])) / (2 * eps[4])).reshape(4, 1)
    return A, B


def lqr_gains(A: np.ndarray, B: np.ndarray, q: tuple[float, ...] = Q_WEIGHTS, r: float = R_WEIGHT) -> np.ndarray:
    from robotforge.control.classic import solve_dare  # the control layer's Riccati solver

    Q, R = np.diag(q), np.array([[r]])
    P = solve_dare(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    if not np.all(np.isfinite(K)) or np.abs(np.linalg.eigvals(A - B @ K)).max() >= 1.0:
        raise ValueError(
            "it cannot be balanced by its wheels: no gains keep its linear model upright "
            "(are the wheel motors strong enough, and is the axle under the body?)"
        )
    return K


# ---- the controller ------------------------------------------------------------------------------------


class WheelBalancer:
    """Keeps a two-wheeled robot upright and drives it: move(speed, turn) like any rover."""

    rate_hz = CONTROL_HZ

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        home_qpos: np.ndarray,
        balancer: Balancer | None = None,
        q_weights: tuple[float, ...] = Q_WEIGHTS,
        r_weight: float = R_WEIGHT,
    ):
        b = balancer or find_balancer(model, robot, home_qpos)
        if b is None:
            raise ValueError(f"robot {robot!r} is not a two-wheeled balancing robot")
        self.robot = robot
        self.b = b
        state, self.lean = equilibrium(model, b, home_qpos)
        self.A, self.B = linearize(model, b, state)
        try:
            self.K = lqr_gains(self.A, self.B, q_weights, r_weight)
        except ValueError as exc:
            raise ValueError(f"robot {robot!r}: {exc}") from None
        self.max_speed = MAX_LEAN_SPEED * b.top_speed * b.radius
        self.command_speed = self.command_turn = 0.0
        self.speed = self.turn = 0.0
        self._ref: np.ndarray | None = None  # where the axle should be (it moves along with speed commands)
        self._last_t: float | None = None
        self.fallen = False

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        self.command_speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        self.command_turn = float(np.clip(turn, -3.0, 3.0))

    def status(self) -> dict:
        return {
            "behavior": "balance",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
            "upright": not self.fallen,
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        b = self.b
        t = float(data.time)
        dt = 0.0 if self._last_t is None or t < self._last_t else t - self._last_t
        self._last_t = t
        alpha = 1.0 - math.exp(-dt / SPEED_SMOOTHING)
        self.speed += alpha * (self.command_speed - self.speed)
        self.turn += alpha * (self.command_turn - self.turn)
        fwd, left = heading_frame(data, b)
        axle = axle_point(data, b)
        if self._ref is None or dt == 0.0:
            self._ref = axle.copy()
        self._ref = self._ref + fwd * self.speed * dt
        x = measure(data, b, fwd, left, self._ref)
        if abs(x[0] - self.lean) > FALL_ANGLE:
            self.fallen = True
        if self.fallen:
            data.ctrl[list(b.wheels)] = 0.0
            return
        if abs(x[2]) > POSITION_SLACK:  # pushed far away: balance near where it is now, not race back
            x[2] = float(np.clip(x[2], -POSITION_SLACK, POSITION_SLACK))
            self._ref = axle - fwd * x[2]
        error = x - np.array([self.lean, 0.0, 0.0, self.speed])
        roll = self.speed / b.radius - float((self.K @ error)[0])
        spin = self.turn * b.track / (2 * b.radius)
        for act, sign, side in zip(b.wheels, b.signs, (-1.0, 1.0), strict=True):
            data.ctrl[act] = sign * float(np.clip(roll + side * spin, -b.top_speed, b.top_speed))
