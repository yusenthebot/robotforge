"""Humanoid behaviors: stand with balance, walk, and strike poses.

Walking follows the classic recipe for small humanoids: plan footsteps, let a
linear inverted pendulum decide how the body must sway so the zero moment point
stays inside the supporting foot (biped.py), then place each foot relative to
the torso with leg IK. Feedback from the torso IMU on ankles and hips corrects
what the plan gets wrong, the way a person shifts their weight on their feet.
Standing is the same controller with no steps planned, so a push is resisted
by the same balance feedback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.behaviors.biped import LEFT, RIGHT, Foot, GaitParams, GaitPlan, PreviewControl
from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.sim.geometry import collides, lowest_point

SIDES = ("left", "right")
SIGN = (1.0, -1.0)
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
UPPER = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "neck_yaw",
    "neck_pitch",
)
DT = 0.01  # control period, s
PREVIEW = 0.8  # s of future ZMP the body sway looks at
REPLAN_LEAD = 0.05  # s; plan pieces starting sooner than this are kept when the command changes
REPLAN_NOTICE = 0.2  # s of warning (at 0.2 m legs) before a settle and a step may be swapped; see GaitPlan.rewind
FALLEN_UP = 0.5  # torso up-axis z below this means the robot is lying down
MAX_TURN_PER_STEP = 0.4  # rad
TALL_LEG = 0.4  # m


@dataclass(frozen=True)
class Keyframe:
    """One moment of a pose: arm and head joint targets, how deep to sink and how far to lean."""

    at: float  # seconds from the start of the pose
    joints: dict[str, float] = field(default_factory=dict)  # joint -> rad; others go to their home value
    drop: float = 0.0  # hips sink by this fraction of the leg length
    lean: float = 0.0  # torso pitched forward, rad


def _arms(roll: float = 0.12, pitch: float = 0.1, elbow: float = 0.35, side: str = "both") -> dict[str, float]:
    sides = SIDES if side == "both" else (side,)
    out: dict[str, float] = {}
    for s in sides:
        out.update({f"{s}_shoulder_roll": roll, f"{s}_shoulder_pitch": pitch, f"{s}_elbow": elbow})
    return out


POSES: dict[str, tuple[str, list[Keyframe]]] = {
    "stand": ("return to the neutral standing pose", [Keyframe(0.0), Keyframe(1.0)]),
    "wave": (
        "raise the right arm and wave hello",
        [
            Keyframe(0.0),
            Keyframe(0.9, _arms(2.5, 0.15, 0.4, "right")),
            *[Keyframe(1.3 + 0.4 * i, _arms(2.1 if i % 2 == 0 else 2.7, 0.15, 0.6, "right")) for i in range(5)],
            Keyframe(3.6, _arms(2.4, 0.15, 0.4, "right")),
            Keyframe(4.5),
        ],
    ),
    "squat": (
        "bend the knees deeply with arms forward, then stand back up",
        [
            Keyframe(0.0),
            Keyframe(1.4, {**_arms(0.1, 1.3, 0.2)}, drop=0.3),
            Keyframe(2.4, {**_arms(0.1, 1.3, 0.2)}, drop=0.3),
            Keyframe(3.8),
        ],
    ),
    "arms_up": (
        "raise both arms high in a cheer",
        [
            Keyframe(0.0),
            Keyframe(0.9, {**_arms(2.7, 0.2, 0.15), "neck_pitch": -0.25}),
            Keyframe(2.0, {**_arms(2.7, 0.2, 0.15), "neck_pitch": -0.25}),
            Keyframe(2.9),
        ],
    ),
    "bow": (
        "bow politely from the hips",
        [
            Keyframe(0.0),
            Keyframe(1.2, {**_arms(0.1, 0.3, 0.3), "neck_pitch": 0.35}, lean=0.45),
            Keyframe(2.0, {**_arms(0.1, 0.3, 0.3), "neck_pitch": 0.35}, lean=0.45),
            Keyframe(3.2),
        ],
    ),
    "look_around": (
        "turn the head left, then right, then back",
        [
            Keyframe(0.0),
            Keyframe(1.0, {"neck_yaw": 1.0}),
            Keyframe(1.6, {"neck_yaw": 1.0}),
            Keyframe(2.8, {"neck_yaw": -1.0}),
            Keyframe(3.4, {"neck_yaw": -1.0}),
            Keyframe(4.3),
        ],
    ),
}


@dataclass(frozen=True)
class BalanceGains:
    """IMU feedback, as joint offset (rad) per rad of torso tilt and per rad/s of tilt rate."""

    ankle_pitch: float = 0.5
    ankle_pitch_rate: float = 0.03
    ankle_roll: float = 1.0
    ankle_roll_rate: float = 0.1
    hip_roll: float = 0.3
    hip_roll_rate: float = 0.02


@dataclass(frozen=True)
class LegRoles:
    """The joints of one leg that balance feedback works through: (index in the leg's chain, sign). The sign makes
    a positive offset turn the body about the trunk's +x (rolls) or +y (pitch), whichever way the joint's axis
    points. Our template's legs read (1, +-1), (4, -1), (5, +-1); a leg without an ankle roll (Unitree H1) has none."""

    hip_roll: tuple[int, float]
    ankle_pitch: tuple[int, float]
    ankle_roll: tuple[int, float] | None


TEMPLATE_ROLES = tuple(LegRoles((1, sign), (4, -1.0), (5, sign)) for sign in SIGN)


def turn_per_step(scale: float, leg_length: float) -> float:
    """The most a humanoid turns in one step (rad): 0.2 for small ones, more for tall ones, whose steps take longer."""
    return min(0.2 * scale, MAX_TURN_PER_STEP) if leg_length >= TALL_LEG else 0.2


class HumanoidController:
    """Keeps a two-legged robot balanced while it stands, walks or plays a pose."""

    rate_hz = 1.0 / DT

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        home_qpos: np.ndarray,
        legs: list[list[int]],
        feet: list[tuple[int, np.ndarray]],
        upper: dict[str, int],
        root_joint: int,
        imu: tuple[int, int] | None,
        gait: GaitParams | None = None,
        gains: BalanceGains | None = None,
        roles: tuple[LegRoles, LegRoles] = TEMPLATE_ROLES,
        foot_rotations: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """legs: [left, right] joint ids from hip to ankle (LEG order on our template); feet: [(foot body, sole
        point in it)] per side; upper: arm and head joint name -> id; imu: (orientation sensor adr, gyro sensor adr),
        or None to read the torso body itself; roles: the balance joints of each leg; foot_rotations: each foot
        body's orientation in the torso frame when it stands flat at home (identity on our template)."""
        self.robot = robot
        self.torso = int(model.jnt_bodyid[root_joint])
        self.roles = roles
        self.foot_rot = foot_rotations or (np.eye(3), np.eye(3))
        self.root_dof = int(model.jnt_dofadr[root_joint])
        root_adr = model.jnt_qposadr[root_joint]
        base = home_qpos.copy()
        base[root_adr : root_adr + 7] = [0, 0, 0, 1, 0, 0, 0]  # torso frame == IK base frame
        self.ik = IKSolver(model, base)
        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        self.chains = [Chain.from_joints(model, legs[s], feet[s][0], feet[s][1]) for s in (LEFT, RIGHT)]
        self.leg_acts = [np.array([joint_to_act[j] for j in legs[s]]) for s in (LEFT, RIGHT)]
        self.leg_q = [base[c.qpos_adr].copy() for c in self.chains]
        self.upper_adr = {n: int(model.jnt_qposadr[j]) for n, j in upper.items()}
        self.upper_act = {n: joint_to_act[j] for n, j in upper.items() if j in joint_to_act}
        self.upper_home = {n: float(base[a]) for n, a in self.upper_adr.items()}
        self.upper_range = {n: tuple(model.jnt_range[j]) for n, j in upper.items()}
        self.imu = imu

        # Standing geometry from the home pose.
        soles = [self.ik.point(c, q) for c, q in zip(self.chains, self.leg_q, strict=True)]
        self.stand_z = -float(np.mean([s[2] for s in soles]))  # torso above the soles
        hip = self.ik.data.xpos[model.jnt_bodyid[legs[LEFT][0]]]
        self.leg_length = float(hip[2] + self.stand_z)
        self.half_width = float(np.mean([abs(s[1]) for s in soles]))
        self.com_local = self._com_local()
        com_height = self.stand_z + float(self.com_local[2])
        scale = math.sqrt(self.leg_length / 0.2)  # pendulum timing scales with the square root of size
        self.replan_notice = round(REPLAN_NOTICE * scale / DT)
        self.gait = gait or GaitParams(
            single=0.36 * scale,
            double=0.12 * scale,
            start=0.4 * scale,
            settle=0.4 * scale,
            lift=0.12 * self.leg_length,
            half_width=self.half_width,
            min_width=1.6 * self.half_width,
            max_stride=0.25 * self.leg_length,
            # A tall robot takes longer steps in time, so at a fixed turn per step it turned slower: a Booster T1
            # (0.55 m legs, 0.8 s a step) managed 10 degrees a second, and A and D hardly turned it. Scaled like the
            # timing, it turns 24 degrees a second. Small ones keep 0.2 rad: an OP3 turning more while walking fell.
            max_turn=turn_per_step(scale, self.leg_length),
            zmp_inset=0.35 * self.half_width,  # less sway: wide steps rocked tall robots off their feet
            zmp_roll=0.5,
        )
        self.gains = gains or BalanceGains()
        self.preview = PreviewControl(DT, com_height, round(PREVIEW / DT))
        self.max_speed = self.gait.max_stride / self.gait.step_time
        self.max_turn = self.gait.max_turn / self.gait.step_time

        self.command_speed = 0.0
        self.command_turn = 0.0
        self.fallen = False
        self.plan: GaitPlan | None = None
        self.ground = 0.0
        self._com = np.zeros((3, 2))
        self._feet = np.zeros((2, 4))
        self._support = 0
        self._last_t: float | None = None
        self._pose: tuple[str, list[Keyframe]] | None = None
        self._pose_t = 0.0
        self._pending: str | None = None
        self._custom: dict[str, list[Keyframe]] = {}  # poses made on the fly, e.g. a crouch
        self._planned_for = (0.0, 0.0)  # the command the plan's future was made for
        self._idle = True  # nothing but standing still is planned (updated every tick)
        self._upper_from: dict[str, float] = dict(self.upper_home)
        self._pose_level = (0.0, 0.0)  # (drop, lean) where the last pose ended

    # ---- commands -----------------------------------------------------------------------

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        """speed in m/s (negative = backwards), turn in rad/s (positive = left)."""
        if not (math.isfinite(speed) and math.isfinite(turn)):
            raise ValueError(f"speed and turn must be finite numbers, got speed={speed} turn={turn}")
        self.command_speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        self.command_turn = float(np.clip(turn, -self.max_turn, self.max_turn))
        if (self.command_speed or self.command_turn) and (self._pose or self._pending):
            # Walking cancels a pose: glide the arms back home while the first step gets ready.
            self._upper_from = self._upper_targets()
            self._pose_level = self._level()
            self._pose = ("stand", POSES["stand"][1])
            self._pose_t = 0.0
            self._pending = None

    def limit_speed(self, top: float) -> None:
        """Lower the fastest accepted speed to what this robot is known to walk at (m/s)."""
        self.max_speed = min(self.max_speed, top)
        self.command(self.command_speed, self.command_turn)

    def play(self, name: str) -> None:
        """Strike a named pose (see POSES); walking stops first."""
        if name not in POSES:
            raise ValueError(f"unknown pose {name!r}; choose one of: {', '.join(POSES)}")
        self.command(0.0, 0.0)
        self._pending = name

    def crouch(self, drop: float, seconds: float = 1.0) -> None:
        """Sink the hips by `drop` (a share of the leg length, 0 to 0.35) and stay there, balancing."""
        if not 0.0 <= drop <= 0.35:
            raise ValueError(f"crouch drop must be between 0 and 0.35 of the leg length, got {drop}")
        self.command(0.0, 0.0)
        self._custom["crouch"] = [Keyframe(0.0), Keyframe(seconds, drop=drop)]
        self._pending = "crouch"

    @property
    def done(self) -> bool:
        """True when no pose is playing or waiting and no steps are left to take."""
        return self.fallen or (self._pose is None and self._pending is None and self._idle)

    def status(self) -> dict:
        walking = bool(self.command_speed or self.command_turn) or not self._idle
        behavior = "pose" if self._pose or self._pending else "walk" if walking else "stand"
        if self.fallen:
            behavior = "fallen"
        out = {
            "behavior": behavior,
            "robot": self.robot,
            "speed": round(self.command_speed, 3),
            "turn": round(self.command_turn, 3),
            "max_speed": round(self.max_speed, 3),
        }
        if self._pose or self._pending:
            out["pose"] = self._pose[0] if self._pose else self._pending
        if self.fallen:
            out["fallen"] = True
        return out

    # ---- control loop -------------------------------------------------------------------

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.plan is None:
            self._begin(data)
        if self.fallen or self._up(data) < FALLEN_UP:
            if not self.fallen:  # forget what it was doing, so status says what is going on
                self.command_speed = self.command_turn = 0.0
                self._pose = self._pending = None
            self.fallen = True
            return  # keep the last targets; flailing on the floor helps nobody
        ticks = 1 if self._last_t is None else min(max(round((data.time - self._last_t) / DT), 1), 10)
        self._last_t = data.time
        for _ in range(ticks):
            self._tick()
        self._drive(data)

    def _begin(self, data: mujoco.MjData) -> None:
        feet = []
        for chain, flat in zip(self.chains, self.foot_rot, strict=True):
            rot = data.xmat[chain.body].reshape(3, 3)
            sole = data.xpos[chain.body] + rot @ chain.local_point
            heading = rot @ flat.T  # the sole's own frame: forward is +x
            feet.append((sole, math.atan2(heading[1, 0], heading[0, 0])))
        self.ground = float(np.mean([f[0][2] for f in feet]))
        self.plan = GaitPlan(DT, *(Foot(float(s[0]), float(s[1]), yaw) for s, yaw in feet), self.gait)
        self._com[:] = 0.0
        self._com[0] = data.subtree_com[self.torso][:2]
        for s, chain in enumerate(self.chains):
            self.leg_q[s] = data.qpos[chain.qpos_adr].copy()
        self._upper_from = {n: float(data.qpos[a]) for n, a in self.upper_adr.items()}
        self._pose = ("stand", POSES["stand"][1])  # glide from wherever the arms are to home
        self._pose_t = 0.0
        self.fallen = self._up(data) < FALLEN_UP

    def _tick(self) -> None:
        plan = self.plan
        assert plan is not None
        command = (self.command_speed, self.command_turn)
        if command != self._planned_for:
            plan.rewind(round(REPLAN_LEAD / DT), self.replan_notice, moving=any(command))
            self._planned_for = command
        horizon = self.preview.horizon
        while len(plan) <= horizon:
            plan.extend(self.command_speed, self.command_turn)
        _, self._feet, self._support = plan.pop()
        # Snapshot for done/status, which other threads read while the simulation thread plans.
        self._idle = plan.idle()
        self._com = self.preview.step(self._com, plan.zmp_ahead(horizon))
        self._advance_pose(plan)

    def _advance_pose(self, plan: GaitPlan) -> None:
        if self._pending is not None and self._idle:
            self._upper_from = self._upper_targets()
            self._pose_level = self._level()
            frames = self._custom[self._pending] if self._pending in self._custom else POSES[self._pending][1]
            self._pose = (self._pending, frames)
            self._pose_t = 0.0
            self._pending = None
        elif self._pose is not None:
            self._pose_t += DT
            if self._pose_t > self._pose[1][-1].at:
                self._upper_from = self._upper_targets()
                self._pose_level = self._level()
                self._pose = None

    def _drive(self, data: mujoco.MjData) -> None:
        drop, lean = self._level()
        upper = self._upper_targets()
        upper = self._swing_arms(upper)
        for name, value in upper.items():
            self.ik.base_qpos[self.upper_adr[name]] = value

        left, right = self._feet
        yaw = left[3] + _wrap(right[3] - left[3]) / 2
        roll_m, pitch_m = _roll_pitch(self._imu_quat(data))
        # Feet are placed relative to the torso as planned, not as measured: re-leveling the swing
        # foot from the IMU fed the torso's own wobble back into the step and made it worse.
        planned = _rz(yaw) @ _ry(lean)
        com = self._com[0]
        offset = planned @ self.com_local
        torso = np.array([com[0] - offset[0], com[1] - offset[1], self.ground + self.stand_z - drop * self.leg_length])

        for s in (LEFT, RIGHT):
            foot = self._feet[s]
            sole = np.array([foot[0], foot[1], self.ground + foot[2]])
            target = planned.T @ (sole - torso)
            rotation = planned.T @ _rz(foot[3]) @ self.foot_rot[s]
            self.leg_q[s], _ = self.ik.solve(
                self.chains[s], target, self.leg_q[s], iterations=6, damping=1e-6, rotation=rotation, axis_weight=1.0
            )
        self.com_local = self._com_local()

        g = self.gains
        gyro = self._imu_gyro(data)
        pitch_fb = g.ankle_pitch * (pitch_m - lean) + g.ankle_pitch_rate * gyro[1]
        roll_fb = g.ankle_roll * roll_m + g.ankle_roll_rate * gyro[0]
        hip_fb = g.hip_roll * roll_m + g.hip_roll_rate * gyro[0]
        for s in (LEFT, RIGHT):
            q = self.leg_q[s].copy()
            if self._support != 2 - s:  # feet carrying weight push the body back over them
                roles = self.roles[s]
                q[roles.ankle_pitch[0]] += roles.ankle_pitch[1] * pitch_fb
                if roles.ankle_roll is not None:
                    q[roles.ankle_roll[0]] += roles.ankle_roll[1] * roll_fb
                q[roles.hip_roll[0]] += roles.hip_roll[1] * hip_fb
            data.ctrl[self.leg_acts[s]] = q
        for name, value in upper.items():
            act = self.upper_act.get(name)
            if act is not None:
                data.ctrl[act] = value

    # ---- helpers --------------------------------------------------------------------------

    def _level(self) -> tuple[float, float]:
        """(drop, lean) right now, eased between pose keyframes."""
        if self._pose is None:
            return self._pose_level
        frames = self._pose[1]
        a, b, u = _bracket(frames, self._pose_t)
        start = self._pose_level if a is frames[0] else (a.drop, a.lean)
        return start[0] + u * (b.drop - start[0]), start[1] + u * (b.lean - start[1])

    def _upper_targets(self) -> dict[str, float]:
        if self._pose is None:
            return dict(self._upper_from)
        frames = self._pose[1]
        a, b, u = _bracket(frames, self._pose_t)
        out = {}
        for name in self.upper_adr:
            start = self._upper_from[name] if a is frames[0] else a.joints.get(name, self.upper_home[name])
            end = b.joints.get(name, self.upper_home[name])
            lo, hi = self.upper_range[name]
            out[name] = float(min(max(start + u * (end - start), lo), hi))
        return out

    def _swing_arms(self, upper: dict[str, float]) -> dict[str, float]:
        """Arms swing opposite to the legs while walking, like a person's."""
        if self._pose is not None:
            return upper
        left, right = self._feet
        yaw = left[3] + _wrap(right[3] - left[3]) / 2
        ahead = math.cos(yaw) * (left[0] - right[0]) + math.sin(yaw) * (left[1] - right[1])
        swing = 2.0 * ahead  # rad per meter the left foot leads
        out = dict(upper)
        for side, sign in (("left", -1.0), ("right", 1.0)):
            key = f"{side}_shoulder_pitch"
            if key in out:
                out[key] += sign * swing
        return out

    def _com_local(self) -> np.ndarray:
        """Robot CoM in the torso frame for the current joint targets."""
        d = self.ik.data
        d.qpos[:] = self.ik.base_qpos
        for chain, q in zip(self.chains, self.leg_q, strict=True):
            d.qpos[chain.qpos_adr] = q
        mujoco.mj_kinematics(self.ik.model, d)
        mujoco.mj_comPos(self.ik.model, d)
        return d.subtree_com[self.torso].copy()

    def _up(self, data: mujoco.MjData) -> float:
        return float(data.xmat[self.torso][8])

    def _imu_quat(self, data: mujoco.MjData) -> np.ndarray:
        if self.imu is None:
            return data.xquat[self.torso]
        return data.sensordata[self.imu[0] : self.imu[0] + 4]

    def _imu_gyro(self, data: mujoco.MjData) -> np.ndarray:
        """Angular velocity in the torso frame: a free joint's rotational velocity is in its body's frame."""
        if self.imu is None:
            return data.qvel[self.root_dof + 3 : self.root_dof + 6]
        return data.sensordata[self.imu[1] : self.imu[1] + 3]


def _bracket(frames: list[Keyframe], t: float) -> tuple[Keyframe, Keyframe, float]:
    """The keyframes around time t and the eased progress between them."""
    for a, b in zip(frames, frames[1:], strict=False):
        if t <= b.at:
            u = 0.0 if b.at <= a.at else (t - a.at) / (b.at - a.at)
            u = min(max(u, 0.0), 1.0)
            return a, b, u * u * (3 - 2 * u)
    return frames[-1], frames[-1], 1.0


def _roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def for_template(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> HumanoidController:
    """Build the controller for a robot made from the humanoid template."""

    def find(obj: mujoco.mjtObj, name: str, what: str) -> int:
        i = mujoco.mj_name2id(model, obj, f"{robot}/{name}")
        if i < 0:
            raise ValueError(f"robot {robot!r} has no {what} {name!r}; is it a humanoid?")
        return i

    legs = [[find(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}", "joint") for j in LEG] for side in SIDES]
    feet = []
    for side in SIDES:
        site = find(mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot", "foot site")
        feet.append((int(model.site_bodyid[site]), model.site_pos[site].copy()))
    upper = {}
    for name in UPPER:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}/{name}")
        if j >= 0:  # arms and head are optional
            upper[name] = j
    quat = find(mujoco.mjtObj.mjOBJ_SENSOR, "orientation", "sensor")
    gyro = find(mujoco.mjtObj.mjOBJ_SENSOR, "gyro", "sensor")
    imu = (int(model.sensor_adr[quat]), int(model.sensor_adr[gyro]))
    root = find(mujoco.mjtObj.mjOBJ_JOINT, "root", "joint")
    return HumanoidController(model, robot, home_qpos, legs, feet, upper, root, imu)


AXIS_ALIGNED = 0.7  # |cos| between a joint axis and a trunk axis for the joint to count as turning about it
TEMPLATE_LEG = 0.2  # m, the leg length the balance gains and gait timing were tuned for
LEG_REACH = 0.2  # m: feet are at least this far below the trunk at home


def for_robot(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> HumanoidController:
    """The controller for any two-legged robot (a real humanoid from the library, an imported one), its legs found
    in its kinematic tree: the two branches below the floating trunk that reach the floor, with 5 or 6 hinges each.
    Its arms and head hold their home pose; poses that move them are for robots built from our template."""
    prefix = f"{robot}/"
    bodies = [b for b in range(1, model.nbody) if (model.body(b).name or "").startswith(prefix)]
    roots = [b for b in bodies if model.body_parentid[b] == 0]
    free = [
        j
        for b in roots
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(roots) != 1 or len(free) != 1:
        raise ValueError(f"robot {robot!r} has no floating trunk, so it is not a walking humanoid")
    trunk, root_joint = roots[0], free[0]
    data = _posed_at_home(model, home_qpos, root_joint)
    feet = _two_feet(model, data, set(bodies), trunk)
    if feet is None:
        raise ValueError(f"robot {robot!r}: no two legs reaching the floor found, so it is not a walking humanoid")
    legs, roles, soles, flats = [], [], [], []
    hips = _common_ancestor(model, *feet)
    for foot in feet:
        joints = _hinges_between(model, hips, foot)
        roles.append(_leg_roles(data, joints, robot))
        legs.append(joints)
        soles.append((foot, _sole(model, data, foot)))
        flats.append(data.xmat[foot].reshape(3, 3).copy())  # the trunk frame is the world frame here
    ctrl = HumanoidController(
        model, robot, home_qpos, legs, soles, {}, root_joint, None, roles=tuple(roles), foot_rotations=tuple(flats)
    )
    # A taller robot topples slower but needs more ankle per degree of tilt: Booster T1 (0.55 m legs) leaned 7 degrees
    # and wobbled on our template's gains, and stands within 2 degrees with them scaled like the gait's timing. Sideways
    # it also sways slower, so its roll is damped for longer (rate gains scaled once more): with the pitch gains alone
    # scaled, a T1 carrying an SO-100 on its head (1.2 kg) swayed further and further sideways after it stopped walking
    # and fell over within 2 s, and with every gain scaled once it still fell after walking 6 or 10 s. Scaled so, it
    # stops within 5 degrees, and a T1 on its own walks 1.26 m in 10 s at 0.12 m/s (was 1.13).
    scale = math.sqrt(ctrl.leg_length / TEMPLATE_LEG)
    base = BalanceGains()
    ctrl.gains = BalanceGains(
        ankle_pitch=base.ankle_pitch * scale,
        ankle_pitch_rate=base.ankle_pitch_rate * scale,
        ankle_roll=base.ankle_roll * scale,
        ankle_roll_rate=base.ankle_roll_rate * scale**2,
        hip_roll=base.hip_roll * scale,
        hip_roll_rate=base.hip_roll_rate * scale**2,
    )
    return ctrl


def _posed_at_home(model: mujoco.MjModel, home_qpos: np.ndarray, root_joint: int) -> mujoco.MjData:
    """The robot in its home pose with its trunk at the origin and level: trunk axes are world axes."""
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    adr = model.jnt_qposadr[root_joint]
    data.qpos[adr : adr + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)
    return data


def _two_feet(model: mujoco.MjModel, data: mujoco.MjData, bodies: set[int], trunk: int) -> tuple[int, int] | None:
    """(left foot, right foot): the lowest body on each side of the trunk, if both reach well below it."""
    lowest: dict[int, float] = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in bodies and b != trunk and collides(model, g):
            lowest[b] = min(lowest.get(b, math.inf), float(lowest_point(model, data, g)[2]))
    if not lowest:
        return None
    floor = min(lowest.values())
    near = [b for b, z in lowest.items() if z < floor + 0.02]  # both soles touch the floor at home
    left = [b for b in near if data.xpos[b][1] > 0]
    right = [b for b in near if data.xpos[b][1] < 0]
    if not left or not right or floor > -LEG_REACH:
        return None

    def foot(side: list[int]) -> int:
        # The end of the leg: an ankle's shell can reach as low as the sole beside it.
        return max(side, key=lambda b: (_depth(model, b), -lowest[b]))

    return foot(left), foot(right)


def _depth(model: mujoco.MjModel, body: int) -> int:
    depth = 0
    while body > 0:
        body = int(model.body_parentid[body])
        depth += 1
    return depth


def _common_ancestor(model: mujoco.MjModel, a: int, b: int) -> int:
    above = set()
    while a > 0:
        above.add(a)
        a = int(model.body_parentid[a])
    while b > 0 and b not in above:
        b = int(model.body_parentid[b])
    return b


def _hinges_between(model: mujoco.MjModel, top: int, body: int) -> list[int]:
    """Hinge joints from just below `top` down to `body`, in that order."""
    joints: list[int] = []
    while body > 0 and body != top:
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        joints = [j for j in range(start, start + count) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE] + joints
        body = int(model.body_parentid[body])
    return joints


def _leg_roles(data: mujoco.MjData, joints: list[int], robot: str) -> LegRoles:
    """Which joints of a leg roll the body and which pitch it, by the way their axes point at home."""
    if len(joints) not in (5, 6):
        raise ValueError(f"robot {robot!r} has legs of {len(joints)} joints; the balance controller needs 5 or 6")
    axes = [data.xaxis[j] for j in joints]
    pitch = [i for i, a in enumerate(axes) if abs(a[1]) > AXIS_ALIGNED]
    roll = [i for i, a in enumerate(axes) if abs(a[0]) > AXIS_ALIGNED]
    if len(pitch) < 2 or not roll:
        raise ValueError(f"robot {robot!r}: its legs have no hip roll and ankle pitch joints the controller knows")
    hip_roll, ankle_pitch = roll[0], pitch[-1]
    ankle_roll = next((i for i in roll if i > pitch[-2]), None)  # a roll joint below the knee
    return LegRoles(
        (hip_roll, float(np.sign(axes[hip_roll][0]))),
        (ankle_pitch, float(np.sign(axes[ankle_pitch][1]))),
        (ankle_roll, float(np.sign(axes[ankle_roll][0]))) if ankle_roll is not None else None,
    )


def _sole(model: mujoco.MjModel, data: mujoco.MjData, foot: int) -> np.ndarray:
    """The middle of a foot's underside, in the foot body's frame."""
    corners = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != foot or not collides(model, g):
            continue
        center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        for signs in np.ndindex(2, 2, 2):
            corners.append(data.geom_xpos[g] + rot @ (center + (np.array(signs) * 2 - 1) * half))
    points = np.array(corners)
    sole = np.array([*(points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2, points[:, 2].min()])
    return data.xmat[foot].reshape(3, 3).T @ (sole - data.xpos[foot])
