"""Trot gait for four-legged robots (diagonal legs move together) and tripod gait for six-legged ones.

Foot paths are planned open loop, with light feedback from the trunk: its attitude
levels the body and its measured velocity corrects the stride. Foot targets are
expressed in the trunk frame and turned into joint targets with numerical IK, so
the gait only needs to know which joints form each leg. Legs come from our
template's names or, for any other robot (library, imported, built from parts), from
its kinematic tree: four or six chains of three hinges from the floating trunk to a foot.
A tripod gait is the trot's six-legged sibling: front and rear legs of one side step
with the middle leg of the other, so three feet always hold the robot up.
A push (a shove, or a drag in the viewer) makes the robot step along with it instead of
fighting it, standing or walking; see PUSH_DRIFT.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.behaviors.footing import LIFT_FIRST, Footing
from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.sim.actuators import drives_joint, is_torque_motor
from robotforge.sim.geometry import collides, lowest_point

# Trot phase offset of each leg (diagonal pairs in phase). Also our template's leg names.
TEMPLATE_LEGS = {"front_left": 0.0, "rear_right": 0.0, "front_right": 0.5, "rear_left": 0.5}
# Which legs step together, as each leg's phase offset in the cycle: trot (diagonal pairs), pace
# (same-side pairs) and bound (front pair, then rear pair).
PATTERNS = {
    "trot": TEMPLATE_LEGS,
    "pace": {"front_left": 0.0, "rear_left": 0.0, "front_right": 0.5, "rear_right": 0.5},
    "bound": {"front_left": 0.0, "front_right": 0.0, "rear_left": 0.5, "rear_right": 0.5},
}
# Tripod: two triangles of feet take turns.
TRIPOD_LEGS = {
    "front_left": 0.0,
    "middle_right": 0.0,
    "rear_left": 0.0,
    "front_right": 0.5,
    "middle_left": 0.5,
    "rear_right": 0.5,
}
PLAN_HZ = 100.0  # foot targets and IK; torque-motor PD runs every physics step
# m: the IK's tolerance (its default). A foot target that stays this close to where the leg's joints already put it
# is not solved again: the IK would return the same joints. Standing still, the targets do not move, and solving
# them anyway halved the speed of a world with one dog in it (real-time factor 173 -> 90, six dogs 38 -> 15).
IK_TOL = 1e-4
# The plan's bookkeeping (trunk velocity, push watch, velocity correction, foot targets) is float arithmetic on two
# and three numbers. On small numpy arrays it took longer than the physics: a dog standing idle on its gait spent a
# quarter of its headless time in the gait (a plan took 35 us, 70 once it had walked and stopped). As plain floats,
# computed the same way in the same order, a plan takes 6 us, the share is a tenth and the robots move as before,
# bit for bit (scripts/bench.py --ab measures it).
TEMPLATE_PERIOD = 0.36  # s per gait cycle for our template dog
# Bigger robots step slower: about 0.36 s per cycle for a Go2-sized leg (hip to foot 0.28 m),
# scaled with leg length. Measured: Go2 walks best at 0.30-0.36 s, Spot (0.44 m legs) at 0.45-0.55 s.
REFERENCE_LEG, REFERENCE_PERIOD = 0.28, 0.36
# Open-loop foot paths are not enough on heavy robots: servos give way under load, so stance feet
# creep forward relative to the trunk. Menagerie's Go2 made 39% of a 0.3 m/s command and slid
# backwards at 10 cm/s when stepping in place. A PI correction on the trunk's measured velocity and
# yaw rate, active only while stepping, makes commands come true (Go2: 0.301 m/s for 0.3).
# Told to walk straight, a robot keeps the heading it had: a turn of HOLD_KP rad/s per rad off it, at most HOLD_MAX.
# Holding the turn rate at zero alone let slips add up: a Go2 on stairs ended up 55 degrees off and walked off their
# side.
HOLD_KP = 1.5
HOLD_MAX = 0.5  # rad/s
HOLD_BLOCKED = 0.3  # it makes less than this share of its speed: it is held up, and the heading is not held
VELOCITY_KP = 0.5
VELOCITY_KI = 2.0  # 1/s
# The fastest command the gait accepts, as a share of what the longest stride allows. The rest is
# headroom for the velocity correction. Measured top speeds at the full stride: template dog 0.39 of
# 0.52 m/s (it stumbles at full stride), Go2 0.62, Go1 0.66, A1 0.63 of 0.86 m/s. Library robots that
# reach less (Spot: about 0.37) carry their own limit.
SPEED_MARGIN = 0.7
# Over steps, stones and slopes a robot walks no faster than this share of its top speed, whatever it is told: at full
# speed a Go2 toppled on 8 cm stairs (0 of 3 in scripts/legged_bench.py), at 80 percent half of the dogs did.
ROUGH_SPEED = 0.6
# PD for torque-motor legs: full torque at this joint error (rad), lightly damped on top of the
# joints' own damping. The PD runs explicitly at physics rate, where heavy damping makes light knee
# joints chatter (critical damping sent Go2 bouncing in circles). On Menagerie's Go2 with its raw
# motors these walk 3.2 m in 10 s at 0.3 m/s, straight.
MOTOR_SATURATION = 0.15
MOTOR_DAMPING_RATIO = 0.25
# Push recovery. Standing still, the gait is a stiff table: dragged sideways by the viewer it pivoted on
# the feet of that side once the pull passed about half its weight (Go2 at 0.46 g, our dog at 0.58 g)
# and rolled onto its back. Trotting, its strides fought the drag and it tipped over the same way when
# grabbed by the back. So a pushed robot steps, and while it recovers (1) its sideways strides follow
# the trunk instead of fighting it and (2) its swinging feet aim at the ground as if the trunk were
# level: on a tilted trunk that lands them lower on the high side and further out on the side it falls
# to. Without (2) the Go2 still flipped; leveling roll alone left the dog drifting sideways for good.
# A short shove that the stance soaks up does not start any of this.
PUSH_DRIFT = 0.08  # standing: the trunk moved this many leg lengths from where it stood...
DRIFT_TAU = 2.0  # ...lately (s; slow creep is forgotten)
# Drift is motion off the usual (adapting at REST_TAU, like the roll). Standing on a slope, servo legs creep: our
# dog on a 15 degree ramp slid down at 8-11 mm/s, as it does without a gait. Counted as drift, that read as a
# push after 4 s and it stepped about the ramp for half a minute; a creep that slow is how it stands there.
# Speeds are in units of sqrt(g * leg length), so one setting fits small and large robots.
WALK_PUSH_SPEED = 0.08  # walking: sideways speed, averaged over a cycle (a straight trot stays under 0.012)
PUSH_LEAN = math.radians(4.0)  # roll away from the usual that counts as a push (a straight trot: under 2)
# Some robots roll more on their plain trot: the A1 4 to 5 degrees each step, so it "recovered" from pushes nobody
# gave it for its first 3 s of every walk, and on a 12 degree ramp until it stalled. A lean counts as a push while
# walking only past WOBBLE_SHARE times how far the trunk usually leans when nothing pushes it (its mean |lean|,
# learned over WOBBLE_TAU s, starting from a robot that leans PUSH_LEAN / WOBBLE_SHARE).
WOBBLE_SHARE = 2.5
# On rough ground (footing.py: steps, stones) a walking robot rocks and sways as a pushed one does; recovering then
# made a Go2 that had its front feet on the second stair turn sideways and fall back down. There a lean is no push,
# and sideways speed counts only past this many times the usual.
ROUGH_PUSH_SHARE = 2.0
WOBBLE_TAU = 1.0  # s
TURN_LEAN = math.radians(3.0)  # per rad/s of turn: turning leans the trunk and slips it sideways...
TURN_SLIP = 0.5  # ...by up to this times turn rate times leg length (Go2 at 2 rad/s: 0.18 m/s)
CALM_SPEED = 0.03  # recovered once the cycle-averaged speed off the command is this low...
RECOVER_CYCLES = 2.0  # ...for this many cycles
# s to blend recovery in: at once from a stand, gently while walking, where the trot soaks up much of a
# shove by itself (blended in as fast, recovery added about 2 degrees of tilt to such shoves)
RECOVER_RISE, RECOVER_RISE_WALKING = 0.05, 0.3
RECOVER_FALL = 0.5  # cycles to blend back into the plain gait
LATERAL_FOLLOW = 0.5  # share of the trunk's sideways velocity its strides follow (1 kept it drifting)
FOLLOW_TAU = 0.25  # cycles: the sideways velocity the strides follow is smoothed this much
REST_TAU = 1.0  # s: the usual roll (on a side slope, say) adapts this slowly
SETTLE_TIME = 0.5  # s after a start or reset during which a robot settles onto its feet unwatched...
# ...and longer, up to this many s, until its trunk is calm: put down on a 15 degree ramp, a Go2 slid 12 cm down
# and back for 0.9 s; watched from 0.5 s on, that read as a push and it stepped 0.9 m down the ramp.
SETTLE_MAX = 3.0
# Standing, calm enough to settle means this share of CALM_SPEED: a slower creep never adds up to PUSH_DRIFT
# (its drift peaks at half its speed times 1 s). Our dog put down on stepping stones slid off a stone's edge at
# 3 cm/s, under CALM_SPEED; it settled while sliding, the slide read as a push and it stepped for half a minute.
# A hand on the robot (a drag in the viewer) ends settling at once.
SETTLE_CALM = 0.5
# A trunk tilted this far is down, on its side or its back: nothing to step for. It settles again once it is up.
# Our dog flipped by a fast drag kicked its legs on its back for good.
FALLEN_TILT = math.radians(60.0)
# A walking robot blocked by a stair riser or a curb leans and wobbles like a pushed one, and kept on
# recovering: its strides followed the wobble sideways and it never climbed (the obstacle course fell from
# 12 m to 4.6 m). Walking recovery is cut off after a while and then held off, so it climbs on its plain gait.
# It is cut off only once nothing moves the trunk sideways as fast as a fresh push would: a hand that keeps
# dragging is still a push, and cut off regardless, a trotting dog dragged 0.8 m by its back over 6 s fought the
# hand and rolled over.
WALK_RECOVER_MAX = 3.0  # s of continuous recovery while walking...
WALK_RECOVER_REST = 3.0  # ...then this long on the plain gait before another can start
# Standing, likewise for a recovery that never calms, as on stepping stones, where stepping rocks the trunk
# itself; but never while a hand pulls the robot along (a drag in the viewer), and counted from when it lets go.
# After the push bench's shoves and drags, recovering lasted 0.8 to 2.4 s (4.1 s after a dog was dragged 1.6 m).
STAND_RECOVER_MAX = 6.0  # s
STAND_RECOVER_REST = 3.0  # s

# A leg joint whose range is a full turn or more (ANYmal C's knees: +-540 degrees) does not keep the knee bending one
# way: reaching for a foot target near full stretch, the IK swung a knee through straight to bend the other way and the
# robot spun round and fell (5 s into a 0.2 m/s walk). Such joints may turn this far either side of home while walking.
UNLIMITED = 2 * math.pi  # rad
BRANCH = math.pi / 2  # rad

# (name, [hip, thigh, knee] joint ids, foot body id, foot point in that body's frame, phase offset)
LegSpec = tuple[str, list[int], int, np.ndarray, float]


@dataclass
class Leg:
    name: str
    chain: Chain
    actuators: np.ndarray  # actuator ids, same order as chain joints
    phase_offset: float
    hip: np.ndarray  # hip position in the trunk frame
    home_foot: np.ndarray  # foot position in the trunk frame at the home pose
    q: np.ndarray  # last IK solution (warm start)
    solved_for: tuple[float, float, float]  # the foot target q was solved for...
    error: float = math.inf  # ...and how far q's foot is from it (inf: unknown, solve again)
    hip_xy: tuple[float, float] = field(init=False, repr=False)  # hip and home_foot as floats, for the plan
    home: tuple[float, float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.hip_xy = (float(self.hip[0]), float(self.hip[1]))
        self.home = (float(self.home_foot[0]), float(self.home_foot[1]), float(self.home_foot[2]))


@dataclass(frozen=True)
class MotorDrive:
    """PD for torque-motor actuators: the gait computes joint targets, this turns them into torques."""

    actuators: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    gear: np.ndarray
    lower: np.ndarray  # control limits (torque / gear)
    upper: np.ndarray


def bind_legs(
    model: mujoco.MjModel, robot: str, legs: list[LegSpec], root_joint: int, home_qpos: np.ndarray
) -> tuple[np.ndarray, list[Leg], MotorDrive | None]:
    """Leg chains and standing geometry, measured on private kinematics without constructing IK."""
    root_qpos = model.jnt_qposadr[root_joint]
    base = home_qpos.copy()
    base[root_qpos : root_qpos + 7] = [0, 0, 0, 1, 0, 0, 0]  # trunk frame == world frame
    data = mujoco.MjData(model)
    data.qpos[:] = base
    mujoco.mj_kinematics(model, data)
    joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if drives_joint(model, a)}

    bound: list[Leg] = []
    for name, joints, body, point, offset in legs:
        missing = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints if j not in joint_to_act]
        if missing:
            raise ValueError(f"robot {robot!r} has no actuator on {', '.join(missing)}, so it cannot walk")
        q0 = base[model.jnt_qposadr[joints]].copy()
        chain = _keep_branch(Chain.from_joints(model, joints, body, point), q0)
        foot = data.xpos[body] + data.xmat[body].reshape(3, 3) @ point
        hip = data.xpos[model.jnt_bodyid[joints[0]]].copy()
        acts = np.array([joint_to_act[j] for j in joints])
        bound.append(Leg(name, chain, acts, offset, hip, foot, q0, tuple(foot.tolist())))
    motors = _motor_drive(model, np.concatenate([leg.actuators for leg in bound]))
    return base, bound, motors


def gait_period(leg_length: float) -> float:
    return float(np.clip(REFERENCE_PERIOD * leg_length / REFERENCE_LEG, 0.3, 0.7))


def stride_speed(max_stride: float, period: float, duty: float = 0.5) -> float:
    return SPEED_MARGIN * (max_stride / (period * duty))


class QuadrupedGait:
    rate_hz = PLAN_HZ

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        legs: list[LegSpec],
        root_joint: int,
        home_qpos: np.ndarray,
        period: float | None = None,
        lift: float | None = None,
        attitude_gain: float = 0.6,
    ):
        self.robot = robot
        self.duty = 0.5
        self.attitude_gain = attitude_gain
        # Standing still, level the trunk as while walking. Off: hold the stance the servos would, leveling only as
        # much as it steps (see _plan); for a robot an open world stands on its gait before anyone moved it.
        self.level_standing = True
        self.root_qpos = model.jnt_qposadr[root_joint]
        self.root_dof = model.jnt_dofadr[root_joint]
        self.trunk = model.jnt_bodyid[root_joint]
        base, self.legs, self.motors = bind_legs(model, robot, legs, root_joint, home_qpos)
        self.ik = IKSolver(model, base)
        self.ik.point(self.legs[-1].chain, self.legs[-1].q)
        self._actuators = np.concatenate([leg.actuators for leg in self.legs])
        self._targets = np.concatenate([leg.q for leg in self.legs])
        self._targets_from = [leg.q for leg in self.legs]  # the solutions _targets holds (rebuilt when one changes)
        if self.motors is not None:
            # Torque motors need their PD loop at physics rate; IK still runs at PLAN_HZ.
            self.rate_hz = 1.0 / model.opt.timestep

        leg_len = float(np.mean([np.linalg.norm(leg.home_foot - leg.hip) for leg in self.legs]))
        # Feet placed on what the robot sees of the ground; None in a world with nothing but a flat floor.
        footing = Footing(model, self.trunk, [(leg.chain.body, leg.chain.local_point) for leg in self.legs], leg_len)
        self.footing: Footing | None = footing if footing.needed else None
        if period is None:
            period = gait_period(leg_len)
        self.leg_length = leg_len
        self.lift = lift if lift is not None else 0.25 * leg_len
        # Stride is limited to what the legs can reach comfortably.
        self.max_stride = 0.55 * leg_len
        self._top: float | None = None  # a known top speed for this robot, if any
        self.set_period(period)
        self.command_speed = 0.0
        self.command_turn = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self._last_t: float | None = None
        self._next_plan = -math.inf
        self._phase = 0.0
        self._amplitude = 0.0  # 0 = standing still, 1 = full stepping
        # Trunk velocity (forward, left) in its heading frame and yaw rate, averaged; and the
        # integrated error of the same. Tuples of floats, rebound and never changed in place, so a shallow copy
        # of the gait (the control layer's preview) cannot disturb the live one.
        self._v_meas = _ZERO3
        self._v_int = _ZERO3
        self._saturated = False  # the last plan had to shorten a stride
        # Push recovery (see PUSH_DRIFT), also updated by rebinding only.
        self._own_pattern = [leg.phase_offset for leg in self.legs]  # recovery is tuned for this pattern
        self._v_fast = _ZERO3  # the trunk velocity again, smoothed over a quarter cycle
        self._rest_roll = 0.0  # the roll the trunk usually holds here
        self._wobble = PUSH_LEAN / WOBBLE_SHARE  # how far it usually leans off that, walking (see WOBBLE_SHARE)
        self._started_at = -math.inf  # sim time of the start, or of the last reset under it
        self._calm_since = -math.inf  # while settling: since when the trunk has been calm
        self._settled_at = -math.inf  # pushes count from this time on (inf: not settled yet)
        self._drift = _ZERO2  # how far the trunk moved while standing still, lately (forward, left)
        self._creep = _ZERO2  # how it usually moves standing here: a slow creep down a slope, say
        self._recover_until = -math.inf
        self._walk_recover_since: float | None = None  # when the current recovery while walking started
        self._walk_rest_until = -math.inf  # no recovery while walking before this
        self._stand_recover_since: float | None = None  # the same standing, from when no hand pulled it any more
        self._stand_recovered_at = -math.inf  # the last time it recovered standing
        self._stand_rest_until = -math.inf
        self._recovery = 0.0  # 0 = the plain gait, 1 = recovering, blended in between
        self._held_yaw: float | None = None  # the heading kept while told to go straight (see HOLD_KP)
        self._hold_turn = 0.0  # rad/s of turn that brings it back to it

    def set_period(self, period: float) -> None:
        """Seconds per gait cycle; the fastest accepted speed follows from it and the longest stride."""
        self.period = period
        self._stride_speed = self.max_stride / (self.period * self.duty)  # body speed at the longest stride
        self.max_speed = stride_speed(self.max_stride, self.period, self.duty)
        if self._top is not None:
            self.max_speed = min(self.max_speed, self._top)

    def set_pattern(self, offsets: dict[str, float]) -> None:
        """Which legs step together: a phase offset per leg name (see PATTERNS)."""
        missing = [leg.name for leg in self.legs if leg.name not in offsets]
        if missing:
            raise ValueError(f"the pattern has no phase for {', '.join(missing)}")
        for leg in self.legs:
            leg.phase_offset = offsets[leg.name]

    def stand_leveled(self) -> None:
        """Level the trunk standing still from now on too (see level_standing). Its legs then change the stance under
        it, so a robot not recovering from a push settles anew, as on a gait started now: read as a push, that change
        sent a Go2 standing on a 17 to 20 degree ramp stepping down it and off."""
        if self.level_standing:
            return
        self.level_standing = True
        if self._recovery == 0.0 and self._last_t is not None:
            self._started_at = self._calm_since = self._last_t
            self._settled_at = math.inf

    def limit_slope(self, slope: float) -> None:
        """Do not walk forward onto ground steeper than this (m/m, stairs measured by their rise over run): stand
        before it, still able to turn and back off."""
        if self.footing is not None:
            self.footing.slope_limit = slope

    @property
    def blocked(self) -> bool:
        """Whether it stands before ground too steep for it (see limit_slope)."""
        return self.footing is not None and self.footing.too_steep

    def limit_speed(self, top: float) -> None:
        """Lower the fastest accepted speed to what this robot is known to reach (m/s)."""
        self._top = top
        self.max_speed = min(self.max_speed, top)
        self.command(self.command_speed, self.command_turn)

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
            **({"blocked": "the ground ahead is too steep for it"} if self.blocked else {}),
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        time_went_back = data.time < self._next_plan - 2.0 / PLAN_HZ  # e.g. the data was reset
        if data.time + 1e-9 >= self._next_plan or time_went_back:
            self._plan(model, data)
            self._next_plan = data.time + 1.0 / PLAN_HZ
        if self.motors is None:
            data.ctrl[self._actuators] = self._targets
        else:
            _apply_pd(self.motors, data, self._targets)

    def _plan(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = data.time
        if self._last_t is None or t < self._last_t:  # a new start, or the data was reset under us
            if self.footing is not None:
                self.footing.reset()
            self._started_at = self._calm_since = t
            self._settled_at = math.inf
            # Recovery times are absolute: after a rewind they would keep it stepping until the old clock
            self._recover_until = self._walk_rest_until = self._stand_rest_until = -math.inf
            self._walk_recover_since = self._stand_recover_since = None
            self._recovery = 0.0
            self._drift = _ZERO2
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        # Smooth command changes so the robot does not stumble on a sudden key press.
        alpha = 1.0 - math.exp(-dt / 0.25)
        forward = self.command_speed
        if self.blocked and forward > 0.0:
            forward = 0.0
        elif self.footing is not None and self.footing.rough_ahead:
            forward = float(np.clip(forward, -ROUGH_SPEED * self.max_speed, ROUGH_SPEED * self.max_speed))
        self.speed += alpha * (forward - self.speed)
        self.turn += alpha * (self.command_turn - self.turn)
        moving = abs(forward) > 1e-3 or abs(self.command_turn) > 1e-3

        quat = data.xquat[self.trunk]
        qw, qx, qy, qz = quat.tolist()
        roll, pitch = _roll_pitch((qw, qx, qy, qz))
        measured = self._measure(data)
        self._hold_heading(data, moving)
        self._v_meas = _toward(self._v_meas, measured, 1.0 - math.exp(-dt / self.period))
        self._v_fast = _toward(self._v_fast, measured, 1.0 - math.exp(-dt / (FOLLOW_TAU * self.period)))
        fallen = 1.0 - 2.0 * (qx**2 + qy**2) < math.cos(FALLEN_TILT)  # its z axis that far from up
        held = any(data.qfrc_applied[self.root_dof : self.root_dof + 6].tolist())  # a hand pulls it (a viewer drag)
        recovering = self._recovering(t, dt, measured, roll, moving, fallen, held)
        if recovering and not moving:  # pushed while standing: start stepping right away
            self._amplitude += (1.0 - math.exp(-dt / RECOVER_RISE)) * (1.0 - self._amplitude)
        else:
            self._amplitude += alpha * ((1.0 if moving else 0.0) - self._amplitude)
        self._phase = (self._phase + dt / self.period) % 1.0

        (vx, vy), turn = self._corrected_velocity(dt)
        w = self._recovery
        # The trunk is leveled while the robot walks. Standing, a gait without level_standing levels it only as much
        # as it steps (recovering, or stopping) and otherwise holds the stance its servos would: leveled standing on
        # a 17 to 20 degree ramp, a Go2 nobody had moved, which stood there on its servos, pushed its downhill legs
        # out, its feet slid, the pitch grew with them and it went over backwards. A robot that was moved keeps its
        # leveled stand: on a 15 degree ramp a Go2 walked up and stopped stood 10 degrees tilted, unleveled 19.
        leveling = self.attitude_gain * (1.0 if moving or self.level_standing else max(self._amplitude, w))
        still = not moving and self._amplitude == 0.0 and w == 0.0 and self.speed == 0.0 and self.turn == 0.0
        if still and leveling == 0.0:
            return  # it has not stepped since it started: every foot target is still its home (see IK_TOL)
        level = None
        if w > 1e-3:
            vi0, vi1, vi2 = self._v_int
            if moving or not held:
                # Sideways strides follow the trunk, and the velocity correction lets go of that axis.
                vy = (1.0 - w) * vy + w * LATERAL_FOLLOW * self._v_fast[1]
                self._v_int = (vi0, (1.0 - w) * vi1, vi2)
            else:
                # A hand pulling a standing robot: its strides follow forwards too. Braking against a forward
                # pull by the back pitched a never-moved dog to 61 degrees and onto its back.
                follow = w * LATERAL_FOLLOW
                vx, vy = (1.0 - w) * vx + follow * self._v_fast[0], (1.0 - w) * vy + follow * self._v_fast[1]
                self._v_int = ((1.0 - w) * vi0, (1.0 - w) * vi1, vi2)
            level = _tilt(quat)
        half_cycle = self.period * self.duty
        saturated = False
        strides = []
        for leg in self.legs:
            # Each foot sweeps backwards during stance at the body's commanded velocity.
            hx, hy = leg.hip_xy
            sx, sy = (vx - turn * hy) * half_cycle, (vy + turn * hx) * half_cycle  # the stride
            norm = math.sqrt(sx * sx + sy * sy)
            if norm > self.max_stride:
                shorten = self.max_stride / norm
                sx, sy = sx * shorten, sy * shorten
                saturated = True
            strides.append((sx, sy))
        # Where a swinging foot lands depends on where the trunk gets to meanwhile, which is somewhere between the
        # command and how fast it really goes: on stairs a Go2 made half its command, and feet aimed with the command
        # came down on the nosing ahead of them and slid off; aimed with the measured speed alone, a Spot slowed on a
        # ramp aimed short, came down early on the slope, made shorter steps and slowed to a crawl.
        ahead = (0.5 * (vx + self._v_meas[0]), 0.5 * (vy + self._v_meas[1]))
        footing = self._footing(model, data, t, dt, moving, strides, ahead)
        if footing is not None:  # the trunk follows the ground it walks on (see footing.py)
            pitch, roll = pitch - footing.pitch, roll - footing.roll
            if level is not None:  # a pushed robot aims its feet as if level as it should be here, not as on flat
                level = _attitude(footing.roll, footing.pitch).T @ level
        for i, leg in enumerate(self.legs):
            hx, hy = leg.hip_xy
            sx, sy = strides[i]
            phase = (self._phase + leg.phase_offset) % 1.0
            fx, fy, fz = leg.home
            if phase < self.duty:
                u = phase / self.duty
                fx, fy = fx + sx * (0.5 - u), fy + sy * (0.5 - u)
                if footing is not None:
                    dx, dz, _ = footing.offset(i, fx, fy, None)
                    fx, fz = fx + dx, fz + dz
            else:
                u = (phase - self.duty) / (1.0 - self.duty)
                ahead = u
                if footing is not None and footing.climbs(i):  # up a step: rise first, then move on to it
                    ahead = min(max((u - LIFT_FIRST) / (1.0 - LIFT_FIRST), 0.0), 1.0)
                s = 0.5 - 0.5 * math.cos(math.pi * ahead)  # ease in and out
                fx, fy = fx + sx * (s - 0.5), fy + sy * (s - 0.5)
                lift = self.lift
                if footing is not None:
                    dx, dz, extra = footing.offset(i, fx, fy, s, u)
                    fx, fz, lift = fx + dx, fz + dz, lift + extra
                fz += self._amplitude * lift * math.sin(math.pi * u)
            # Level the trunk: extend legs on the low side.
            fz += leveling * (-pitch * hx + roll * hy)
            if level is not None and phase >= self.duty:
                # A swinging foot aims where it would if the trunk were level (see PUSH_DRIFT).
                aim = np.array([fx, fy, fz])
                fx, fy, fz = ((1.0 - w) * aim + w * (level.T @ aim)).tolist()
            # Solved again unless the IK would return the same joints at once (see IK_TOL).
            ox, oy, oz = leg.solved_for
            if leg.error + math.sqrt((fx - ox) * (fx - ox) + (fy - oy) * (fy - oy) + (fz - oz) * (fz - oz)) >= IK_TOL:
                leg.q, error = self.ik.solve(leg.chain, np.array([fx, fy, fz]), leg.q, iterations=4, tol=IK_TOL)
                # The error is q's own only when the IK stopped on it; after its last iteration it is the step before.
                leg.solved_for, leg.error = (fx, fy, fz), error if error < IK_TOL else math.inf
        if any(leg.q is not q for leg, q in zip(self.legs, self._targets_from, strict=True)):
            self._targets_from = [leg.q for leg in self.legs]
            self._targets = np.concatenate(self._targets_from)
        self._saturated = saturated

    def _footing(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
        dt: float,
        moving: bool,
        strides: list[tuple[float, float]],
        velocity: tuple[float, float],
    ) -> Footing | None:
        """The terrain corrections of this plan, or None on flat ground or before the robot ever stepped."""
        footing = self.footing
        if footing is None or (not footing.active and not moving and self._amplitude < 0.01):
            return None
        swings: list[float | None] = []
        landings, stands = [], []
        for leg, (sx, sy) in zip(self.legs, strides, strict=True):
            phase = (self._phase + leg.phase_offset) % 1.0
            fx, fy, _ = leg.home
            swing = None if phase < self.duty else (phase - self.duty) / (1.0 - self.duty)
            swings.append(swing if self._amplitude >= 0.01 else None)  # feet on the ground: nothing swings
            landings.append((fx + 0.5 * sx, fy + 0.5 * sy))
            u = phase / self.duty if phase < self.duty else 0.0
            stands.append((fx + sx * (0.5 - u), fy + sy * (0.5 - u)))
        if not footing.active:
            footing.start(model, data, [s is not None for s in swings])
        swing_time = self.period * (1.0 - self.duty)
        footing.plan(model, data, t, dt, self.period, swings, landings, stands, swing_time, velocity)
        return footing

    def _recovering(
        self,
        t: float,
        dt: float,
        measured: tuple[float, float, float],
        roll: float,
        moving: bool,
        fallen: bool,
        held: bool,
    ) -> bool:
        """Whether the robot is recovering from a push; also eases the recovery in and out."""
        if fallen:  # down: nothing to step for, and once up again it settles anew (see FALLEN_TILT)
            self._started_at = self._calm_since = t
            self._settled_at = math.inf
            self._recover_until = -math.inf
        elif self._settled_at == math.inf:  # settled once calm for a whole cycle (see SETTLE_MAX, SETTLE_CALM)
            if not self._calm(moving, 1.0 if moving else SETTLE_CALM):
                self._calm_since = t
            since = t - self._started_at
            calm = since >= SETTLE_TIME and t - self._calm_since >= self.period
            if held or calm or since >= SETTLE_MAX:  # a hand on it: watch it from now on
                self._settled_at = t
        if t < self._settled_at:  # still settling onto its feet: nothing to recover from yet
            self._rest_roll = roll
            recovering = False
        else:
            recovering = self._watch_for_pushes(t, roll - self._rest_roll, moving, held)
        if moving or recovering or self._amplitude >= 0.05 or t < self._settled_at:
            self._drift, self._creep = _ZERO2, _ZERO2
        else:  # how it moves off its usual slow creep (see DRIFT_TAU)
            (mx, my, _), (cx, cy), (dx, dy) = measured, self._creep, self._drift
            k = 1.0 - math.exp(-dt / REST_TAU)
            cx, cy = cx + k * (mx - cx), cy + k * (my - cy)
            decay = math.exp(-dt / DRIFT_TAU)
            self._creep = (cx, cy)
            self._drift = (decay * dx + dt * (mx - cx), decay * dy + dt * (my - cy))
        if not recovering:
            if moving and t >= self._settled_at:
                self._wobble += (1.0 - math.exp(-dt / WOBBLE_TAU)) * (abs(roll - self._rest_roll) - self._wobble)
            self._rest_roll += (1.0 - math.exp(-dt / REST_TAU)) * (roll - self._rest_roll)
        rise = RECOVER_RISE_WALKING if moving else RECOVER_RISE
        tau = rise if recovering else RECOVER_FALL * self.period
        self._recovery += (1.0 - math.exp(-dt / tau)) * (float(recovering) - self._recovery)
        return recovering

    def _watch_for_pushes(self, t: float, lean: float, moving: bool, held: bool) -> bool:
        """Start (or keep up) a recovery when the trunk moves or leans in a way the gait does not."""
        if [leg.phase_offset for leg in self.legs] != self._own_pattern:
            return False  # tuned for the gait's own trot or tripod; a pace or a bound rocks by design
        scale = math.sqrt(9.81 * self.leg_length)
        slip = TURN_SLIP * abs(self.turn) * self.leg_length
        sideways_push = WALK_PUSH_SPEED * scale + slip  # how fast a push that starts a walking recovery moves it
        if t < self._recover_until:  # keep at it until the trunk is calm again
            pushed = not self._calm(moving)
        elif not moving and self._amplitude < 0.05:  # standing still
            dx, dy = self._drift
            pushed = math.sqrt(dx * dx + dy * dy) > PUSH_DRIFT * self.leg_length or abs(lean) > PUSH_LEAN
        elif moving and self.footing is not None and self.footing.active and t < self.footing.rough_until:
            # Steps and stones rock a walking trunk as a push would: only a much faster sideways shove counts there.
            pushed = abs(self._v_meas[1]) > ROUGH_PUSH_SHARE * sideways_push
        elif moving:
            pushed = abs(self._v_meas[1]) > sideways_push
            pushed = pushed or abs(lean) > max(PUSH_LEAN, WOBBLE_SHARE * self._wobble) + TURN_LEAN * abs(self.turn)
        else:  # settling into a stand after a walk
            pushed = False
        if pushed:
            self._recover_until = t + RECOVER_CYCLES * self.period
        recovering = self._limit_walking(t, t < self._recover_until, moving, sideways_push)
        return self._limit_standing(t, recovering, moving, held)

    def _calm(self, moving: bool, share: float = 1.0) -> bool:
        """The trunk moves as told, averaged over a cycle: sideways (and standing, forwards) under this share of
        CALM_SPEED."""
        off = abs(self._v_meas[1]) if moving else math.hypot(self._v_meas[0] - self.speed, self._v_meas[1])
        slip = TURN_SLIP * abs(self.turn) * self.leg_length
        return off <= share * CALM_SPEED * math.sqrt(9.81 * self.leg_length) + slip

    def _limit_walking(self, t: float, recovering: bool, moving: bool, sideways_push: float) -> bool:
        """Cut a recovery while walking off after WALK_RECOVER_MAX, then rest from it (see WALK_RECOVER_MAX)."""
        if not (recovering and moving):
            self._walk_recover_since = None
            return recovering
        if t < self._walk_rest_until:
            self._recover_until = -math.inf
            return False
        if self._walk_recover_since is None:
            self._walk_recover_since = t
        elif t - self._walk_recover_since > WALK_RECOVER_MAX and abs(self._v_meas[1]) <= sideways_push:
            self._recover_until = -math.inf
            self._walk_rest_until = t + WALK_RECOVER_REST
            self._walk_recover_since = None
            return False
        return True

    def _limit_standing(self, t: float, recovering: bool, moving: bool, held: bool) -> bool:
        """Cut recovering while standing off after STAND_RECOVER_MAX, then rest from it (see STAND_RECOVER_MAX).
        Recoveries less than a rest apart count as one: on stones one ends and the trunk's roll starts the next."""
        if moving or held:  # walking has its own limit; a hand pulling it along gets steps for as long as it pulls
            self._stand_recover_since = None
            if held:
                self._stand_rest_until = -math.inf
            return recovering
        if t < self._stand_rest_until:
            self._recover_until = -math.inf
            return False
        if not recovering:
            if self._stand_recover_since is not None and t - self._stand_recovered_at > STAND_RECOVER_REST:
                self._stand_recover_since = None
            return False
        if self._stand_recover_since is None:
            self._stand_recover_since = t
        elif t - self._stand_recover_since > STAND_RECOVER_MAX:
            self._recover_until = -math.inf
            self._stand_rest_until = t + STAND_RECOVER_REST
            self._stand_recover_since = None
            return False
        self._stand_recovered_at = t
        return True

    def _measure(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """The trunk's velocity (forward, left) in its heading frame and its yaw rate, right now."""
        r = data.xmat[self.trunk].tolist()  # row by row; MuJoCo makes it from xquat, as mju_quat2Mat would
        yaw = math.atan2(r[3], r[0])
        c, s = math.cos(yaw), math.sin(yaw)
        # Linear velocity in the world frame, angular in the body frame: its yaw rate is the rotation's last row
        # times it.
        vx, vy, _, wx, wy, wz = data.qvel[self.root_dof : self.root_dof + 6].tolist()
        yaw_rate = r[6] * wx + r[7] * wy + r[8] * wz
        return (c * vx + s * vy, -s * vx + c * vy, yaw_rate)

    def _hold_heading(self, data: mujoco.MjData, moving: bool) -> None:
        """Told to go straight, keep the heading it had when told (see HOLD_KP): a turn rate held at zero still lets
        slips and stumbles add up."""
        if not moving or abs(self.command_turn) > 1e-3 or self._recovery > 0.5:
            self._held_yaw = None
            self._hold_turn = 0.0
            return
        r = data.xmat[self.trunk]
        yaw = math.atan2(float(r[3]), float(r[0]))
        if self._held_yaw is None:
            self._held_yaw = yaw
        error = (self._held_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        self._hold_turn = min(max(HOLD_KP * error, -HOLD_MAX), HOLD_MAX)
        if abs(self._v_meas[0]) < HOLD_BLOCKED * abs(self.speed):
            # Held up (a wall, a riser it cannot climb), turning back only twists it against what holds it: a dog
            # pinned on a wall wobbled sideways and recovered as if pushed for 4.3 s at a time (2.5 without this).
            self._hold_turn = 0.0

    def _corrected_velocity(self, dt: float) -> tuple[tuple[float, float], float]:
        """The body velocity and turn rate to step for: the command plus a PI correction, each limited."""
        (w0, w1, w2), (m0, m1, m2) = (self.speed, 0.0, self.turn + self._hold_turn), self._v_meas
        top = 0.5 * self._stride_speed  # forward and sideways; the turn at 1 rad/s
        if self._amplitude < 0.05:  # standing: nothing to correct, and no windup for the next start
            self._v_int = _ZERO3
        elif not self._saturated:  # strides at their limit cannot go faster; integrating then only winds up
            gain = VELOCITY_KI * dt
            i0, i1, i2 = self._v_int
            self._v_int = (
                min(max(i0 + gain * (w0 - m0), -top), top),
                min(max(i1 + gain * (w1 - m1), -top), top),
                min(max(i2 + gain * (w2 - m2), -1.0), 1.0),
            )
        amp, (i0, i1, i2) = self._amplitude, self._v_int
        vx = w0 + min(max(amp * (VELOCITY_KP * (w0 - m0) + i0), -top), top)
        vy = w1 + min(max(amp * (VELOCITY_KP * (w1 - m1) + i1), -top), top)
        turn = w2 + min(max(amp * (VELOCITY_KP * (w2 - m2) + i2), -1.0), 1.0)
        return (vx, vy), turn


_ZERO2 = (0.0, 0.0)
_ZERO3 = (0.0, 0.0, 0.0)


def _toward(average: tuple[float, float, float], now: tuple[float, float, float], share: float) -> tuple[float, ...]:
    """An exponential average moved `share` of the way to `now`."""
    (a0, a1, a2), (n0, n1, n2) = average, now
    return (a0 + share * (n0 - a0), a1 + share * (n1 - a1), a2 + share * (n2 - a2))


def _tilt(quat: np.ndarray) -> np.ndarray:
    """The trunk's rotation without its heading: the roll and pitch part, as a 3x3 matrix."""
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot


def _attitude(roll: float, pitch: float) -> np.ndarray:
    """The rotation of a trunk with this roll and pitch (as _roll_pitch reads them) and no heading."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]])


def _roll_pitch(quat: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


# ---- actuators ------------------------------------------------------------------------------


def _motor_drive(model: mujoco.MjModel, actuators: np.ndarray) -> MotorDrive | None:
    """PD gains for torque-motor legs, scaled to each motor's torque limit; None for servo legs."""
    motors = [is_torque_motor(model, int(a)) for a in actuators]
    if not any(motors):
        return None
    if not all(motors):
        raise ValueError("this robot's legs mix position servos and torque motors; the gait drives one kind")
    if not model.actuator_ctrllimited[actuators].all():
        raise ValueError("this robot's leg motors have no torque limits (ctrlrange); the gait needs them")
    joints = model.actuator_trnid[actuators, 0]
    gear = model.actuator_gear[actuators, 0]
    lower, upper = model.actuator_ctrlrange[actuators, 0], model.actuator_ctrlrange[actuators, 1]
    torque_limit = np.maximum(np.abs(lower), np.abs(upper)) * np.abs(gear)
    kp = torque_limit / MOTOR_SATURATION
    dof = model.jnt_dofadr[joints]
    inertia = model.dof_M0[dof]  # each joint's own inertia, armature included
    # Explicit damping is stable only while kd * dt stays well below the inertia it acts on.
    kd = np.minimum(2.0 * MOTOR_DAMPING_RATIO * np.sqrt(kp * inertia), 0.5 * inertia / model.opt.timestep)
    return MotorDrive(actuators, model.jnt_qposadr[joints], dof, kp, kd, gear, lower.copy(), upper.copy())


def _keep_branch(chain: Chain, home: np.ndarray) -> Chain:
    """The chain with each joint of a full turn or more of range kept within BRANCH of its home angle."""
    wide = chain.upper - chain.lower >= UNLIMITED - 1e-9
    if not wide.any():
        return chain
    lower = np.where(wide, np.maximum(chain.lower, home - BRANCH), chain.lower)
    upper = np.where(wide, np.minimum(chain.upper, home + BRANCH), chain.upper)
    return dataclasses.replace(chain, lower=lower, upper=upper)


def _apply_pd(drive: MotorDrive, data: mujoco.MjData, targets: np.ndarray) -> None:
    torque = drive.kp * (targets - data.qpos[drive.qpos_adr]) - drive.kd * data.qvel[drive.dof_adr]
    data.ctrl[drive.actuators] = np.clip(torque / drive.gear, drive.lower, drive.upper)


# ---- finding the legs -----------------------------------------------------------------------


def template_binding(model: mujoco.MjModel, robot: str) -> tuple[int, list[LegSpec]]:
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
    return jid("root"), legs


def for_template(
    model: mujoco.MjModel,
    robot: str,
    home_qpos: np.ndarray,
    period: float = TEMPLATE_PERIOD,
    lift: float | None = None,
) -> QuadrupedGait:
    """Build a gait for a robot made from the quadruped template (period: s per cycle, lift: m)."""

    root, legs = template_binding(model, robot)
    return QuadrupedGait(model, robot, legs, root, home_qpos, period=period, lift=lift)


def robot_binding(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> tuple[int, list[LegSpec], float | None]:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/front_left_foot") >= 0:
        return (*template_binding(model, robot), TEMPLATE_PERIOD)
    found = discover_legs(model, robot, home_qpos)
    if found is None:
        raise ValueError(
            f"robot {robot!r} does not look like a legged walker "
            "(expected a floating trunk with four or six legs of three hinge joints each)"
        )
    root, legs = found
    return root, legs, None


def for_robot(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> QuadrupedGait:
    """Build a gait for any four- or six-legged robot: our template by name, others by their kinematic tree."""
    root, legs, period = robot_binding(model, robot, home_qpos)
    return QuadrupedGait(model, robot, legs, root, home_qpos, period=period)


def discover_legs(
    model: mujoco.MjModel, robot: str, home_qpos: np.ndarray | None = None
) -> tuple[int, list[LegSpec]] | None:
    """Find the four (or six) legs of a floating robot named `robot`, or None if it has no such legs.

    A leg is a chain of exactly three hinge joints from the trunk to a leaf body whose lowest
    collision geom (the foot) is at least 5 cm below the trunk origin at the home pose. Legs are
    named by where their hips are, assuming the trunk's +x points forward and +y left (the
    convention of MuJoCo Menagerie and URDF); six legs are three per side, front to rear. Robots
    with two-joint or four-joint legs, wheels on the legs, or an extra leaf with three hinges that
    reaches the floor are not recognized.
    """
    prefix = f"{robot}/"
    bodies = [b for b in range(1, model.nbody) if _name(model, mujoco.mjtObj.mjOBJ_BODY, b).startswith(prefix)]
    roots = [b for b in bodies if model.body_parentid[b] == 0]
    if len(roots) != 1 or model.body_jntnum[roots[0]] < 1:
        return None
    trunk = roots[0]
    root_joint = int(model.body_jntadr[trunk])
    if int(model.jnt_type[root_joint]) != mujoco.mjtJoint.mjJNT_FREE:
        return None

    # Pose the robot at home with the trunk at the origin, as the gait's IK will.
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 if home_qpos is None else home_qpos
    adr = model.jnt_qposadr[root_joint]
    data.qpos[adr : adr + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)

    parents = {int(model.body_parentid[b]) for b in bodies}
    candidates = []
    for leaf in (b for b in bodies if b not in parents):
        path = _path_from(model, trunk, leaf)
        joints = [j for b in path for j in _hinges(model, b)]
        if len(joints) != 3:
            continue
        foot = _foot(model, data, path[path.index(model.jnt_bodyid[joints[-1]]) :])
        if foot is None or foot[2][2] > -0.05:  # a leg reaches down below the trunk
            continue
        candidates.append((joints, foot[0], foot[1], data.xpos[model.jnt_bodyid[joints[0]]].copy()))
    if len(candidates) not in (4, 6):
        return None

    center = np.mean([hip for *_, hip in candidates], axis=0)
    sides: dict[str, list] = {"left": [], "right": []}
    for leg in candidates:
        sides["left" if leg[3][1] > center[1] else "right"].append(leg)
    per_side = len(candidates) // 2
    if any(len(side) != per_side for side in sides.values()):
        return None
    rows = ("front", "rear") if per_side == 2 else ("front", "middle", "rear")
    phases = TEMPLATE_LEGS if per_side == 2 else TRIPOD_LEGS
    legs: list[LegSpec] = []
    for side, members in sides.items():
        for row, (joints, body, point, _hip) in zip(rows, sorted(members, key=lambda c: -c[3][0]), strict=True):
            legs.append((f"{row}_{side}", joints, body, point, phases[f"{row}_{side}"]))
    return root_joint, legs


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or ""


def _path_from(model: mujoco.MjModel, top: int, body: int) -> list[int]:
    """Bodies from just below `top` down to `body`."""
    path = []
    while body not in (top, 0):
        path.append(body)
        body = int(model.body_parentid[body])
    return path[::-1]


def _hinges(model: mujoco.MjModel, body: int) -> list[int]:
    start, count = model.body_jntadr[body], model.body_jntnum[body]
    return [j for j in range(start, start + count) if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_HINGE]


def _foot(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]) -> tuple[int, np.ndarray, np.ndarray] | None:
    """(body, point in body frame, point in trunk frame) of the lowest collision geom on `bodies`."""
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies and collides(model, g)]
    if not geoms:
        return None
    lowest = min(geoms, key=lambda g: lowest_point(model, data, g)[2])
    body = int(model.geom_bodyid[lowest])
    if int(model.geom_type[lowest]) == mujoco.mjtGeom.mjGEOM_SPHERE:
        world = data.geom_xpos[lowest].copy()  # a ball foot rolls; track its center
    else:
        world = lowest_point(model, data, lowest)
    rot = data.xmat[body].reshape(3, 3)
    return body, rot.T @ (world - data.xpos[body]), world
