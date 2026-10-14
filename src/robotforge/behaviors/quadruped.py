"""Trot gait for four-legged robots: diagonal legs move together, feet follow simple arcs.

Open loop with a light attitude correction from the trunk orientation. Foot
targets are expressed in the trunk frame and turned into joint targets with
numerical IK, so the gait only needs to know which joints form each leg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain, IKSolver

# Template leg names and the trot phase offset of each leg (diagonal pairs in phase).
TEMPLATE_LEGS = {"front_left": 0.0, "rear_right": 0.0, "front_right": 0.5, "rear_left": 0.5}


@dataclass
class Leg:
    name: str
    chain: Chain
    actuators: np.ndarray  # actuator ids, same order as chain joints
    phase_offset: float
    hip: np.ndarray  # hip position in the trunk frame
    home_foot: np.ndarray  # foot position in the trunk frame at the home pose
    q: np.ndarray  # last IK solution (warm start)


class QuadrupedGait:
    rate_hz = 100.0

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        legs: list[tuple[str, list[int], int, np.ndarray, float]],
        root_joint: int,
        home_qpos: np.ndarray,
        period: float = 0.36,
        lift: float | None = None,
        attitude_gain: float = 0.6,
    ):
        """legs: (name, [hip, thigh, knee] joint ids, foot body id, foot point in body frame, phase offset)."""
        self.robot = robot
        self.period = period
        self.duty = 0.5
        self.attitude_gain = attitude_gain
        self.root_qpos = model.jnt_qposadr[root_joint]
        self.trunk = model.jnt_bodyid[root_joint]
        base = home_qpos.copy()
        base[self.root_qpos : self.root_qpos + 7] = [0, 0, 0, 1, 0, 0, 0]  # trunk frame == world frame
        self.ik = IKSolver(model, base)
        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}

        self.legs: list[Leg] = []
        for name, joints, body, point, offset in legs:
            chain = Chain.from_joints(model, joints, body, point)
            q0 = base[chain.qpos_adr].copy()
            foot = self.ik.point(chain, q0)
            self.ik.point(chain, q0)
            hip = self.ik.data.xpos[model.jnt_bodyid[joints[0]]].copy()
            acts = np.array([joint_to_act[j] for j in joints])
            self.legs.append(Leg(name, chain, acts, offset, hip, foot, q0))

        leg_len = float(np.mean([np.linalg.norm(leg.home_foot - leg.hip) for leg in self.legs]))
        self.lift = lift if lift is not None else 0.25 * leg_len
        # Stride is limited to what the legs can reach comfortably.
        self.max_stride = 0.55 * leg_len
        self.max_speed = self.max_stride / (self.period * self.duty)
        self.command_speed = 0.0
        self.command_turn = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self._last_t: float | None = None
        self._phase = 0.0
        self._amplitude = 0.0  # 0 = standing still, 1 = full stepping

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        """speed in m/s (negative = backwards), turn in rad/s (positive = left)."""
        self.command_speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        self.command_turn = float(np.clip(turn, -2.0, 2.0))

    def status(self) -> dict:
        return {
            "behavior": "walk",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = data.time
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        # Smooth command changes so the robot does not stumble on a sudden key press.
        alpha = 1.0 - math.exp(-dt / 0.25)
        self.speed += alpha * (self.command_speed - self.speed)
        self.turn += alpha * (self.command_turn - self.turn)
        moving = abs(self.command_speed) > 1e-3 or abs(self.command_turn) > 1e-3
        self._amplitude += alpha * ((1.0 if moving else 0.0) - self._amplitude)
        self._phase = (self._phase + dt / self.period) % 1.0

        quat = data.xquat[self.trunk]
        roll, pitch = _roll_pitch(quat)
        half_cycle = self.period * self.duty
        for leg in self.legs:
            # Each foot sweeps backwards during stance at the body's commanded velocity.
            v = np.array([self.speed - self.turn * leg.hip[1], self.turn * leg.hip[0]])
            stride = v * half_cycle
            norm = np.linalg.norm(stride)
            if norm > self.max_stride:
                stride *= self.max_stride / norm
            phase = (self._phase + leg.phase_offset) % 1.0
            target = leg.home_foot.copy()
            if phase < self.duty:
                u = phase / self.duty
                target[:2] += stride * (0.5 - u)
            else:
                u = (phase - self.duty) / (1.0 - self.duty)
                s = 0.5 - 0.5 * math.cos(math.pi * u)  # ease in and out
                target[:2] += stride * (s - 0.5)
                target[2] += self._amplitude * self.lift * math.sin(math.pi * u)
            # Level the trunk: extend legs on the low side.
            target[2] += self.attitude_gain * (-pitch * leg.hip[0] + roll * leg.hip[1])
            leg.q, _ = self.ik.solve(leg.chain, target, leg.q, iterations=4)
            data.ctrl[leg.actuators] = leg.q


def _roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def for_template(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> QuadrupedGait:
    """Build a gait for a robot made from the quadruped template."""

    def jid(name: str) -> int:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}/{name}")
        if j < 0:
            raise ValueError(f"robot {robot!r} has no joint {name!r}; is it a quadruped?")
        return j

    legs = []
    for leg, offset in TEMPLATE_LEGS.items():
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/{leg}_foot")
        joints = [jid(f"{leg}_hip"), jid(f"{leg}_thigh"), jid(f"{leg}_knee")]
        legs.append((leg, joints, int(model.site_bodyid[site]), model.site_pos[site].copy(), offset))
    return QuadrupedGait(model, robot, legs, jid("root"), home_qpos)
