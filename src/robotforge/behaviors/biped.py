"""Walking plans for two-legged robots: footsteps, a reference ZMP and the body sway it needs.

A biped stays up while the zero moment point (ZMP, where the ground pushes back)
is inside the foot it stands on. GaitPlan turns speed and turn commands into
footsteps and a ZMP path that hops from foot to foot; PreviewControl finds the
center of mass (CoM) motion that produces that ZMP on a linear inverted
pendulum (Kajita et al., "Biped walking pattern generation by using preview
control of zero-moment point", ICRA 2003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81
LEFT, RIGHT = 0, 1
SIDE_SIGN = (1.0, -1.0)  # +y for the left foot, -y for the right


class PreviewControl:
    """LQ preview controller on the cart-table model: CoM jerk is the input, ZMP the output."""

    def __init__(self, dt: float, com_height: float, horizon: int, q: float = 1.0, r: float = 1e-6):
        A = np.array([[1.0, dt, dt * dt / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        B = np.array([[dt**3 / 6], [dt * dt / 2], [dt]])
        C = np.array([[1.0, 0.0, -com_height / GRAVITY]])
        P = q * C.T @ C
        for _ in range(20000):  # discrete Riccati equation by fixed-point iteration (3x3, fast)
            S = r + (B.T @ P @ B).item()
            P_next = A.T @ P @ A + q * C.T @ C - (A.T @ P @ B) @ (B.T @ P @ A) / S
            done = np.max(np.abs(P_next - P)) <= 1e-11 * np.max(np.abs(P_next))
            P = P_next
            if done:
                break
        S = r + (B.T @ P @ B).item()
        self.K = (B.T @ P @ A) / S  # (1, 3) state feedback
        Ac = A - B @ self.K
        gains, X = [], q * C.T
        for _ in range(horizon):
            gains.append((B.T @ X).item() / S)
            X = Ac.T @ X
        # A finite horizon drops the tail of the gains, which would leave the CoM short of a ZMP held
        # still; rescale so a constant reference is tracked exactly.
        self.f = np.array(gains) * (self.K[0, 0] / sum(gains))  # (horizon,) weights on the future ZMP
        self.A, self.B, self.C = A, B, C
        self.horizon = horizon

    def step(self, state: np.ndarray, zmp_ahead: np.ndarray) -> np.ndarray:
        """state (3, 2): CoM [position, velocity, acceleration] in x and y; zmp_ahead (horizon, 2)."""
        u = -self.K @ state + self.f @ zmp_ahead
        return self.A @ state + self.B @ u

    def zmp(self, state: np.ndarray) -> np.ndarray:
        return (self.C @ state)[0]


@dataclass(frozen=True)
class GaitParams:
    single: float  # s on one foot
    double: float  # s with both feet down while the weight moves across
    start: float  # s to shift the weight onto the first stance foot
    settle: float  # s to bring the weight back to the middle after the last step
    lift: float  # m swing foot clearance
    half_width: float  # m from the body's middle line to each foot
    min_width: float  # m feet are never placed closer than this (between centers)
    max_stride: float  # m body advance per step
    max_turn: float  # rad heading change per step
    zmp_inset: float  # m the ZMP sits inside the stance foot center, toward the other foot
    zmp_roll: float  # fraction of the step the ZMP rolls heel to toe while on one foot

    @property
    def step_time(self) -> float:
        return self.single + self.double


@dataclass
class Foot:
    x: float
    y: float
    yaw: float

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def local(self, dx: float, dy: float) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.x + c * dx - s * dy, self.y + s * dx + c * dy])


class GaitPlan:
    """The gait's future at control rate: reference ZMP, both sole poses and which feet carry weight.

    Feet are [x, y, z, yaw] in the plan frame (z = height above the ground). Support is 0 for
    both feet, 1 when standing on the left foot, 2 when standing on the right.
    """

    def __init__(self, dt: float, left: Foot, right: Foot, params: GaitParams):
        self.dt = dt
        self.p = params
        self.feet_end = [left, right]
        self.zmp_end = (left.xy() + right.xy()) / 2
        self.standing = True
        self.next_swing = LEFT
        self.frame = self._middle()
        self.last_stride = 0.0
        self._zmp = np.zeros((0, 2))
        self._feet = np.zeros((0, 2, 4))
        self._support = np.zeros(0, dtype=int)
        self._i = 0
        self._base = 0  # absolute tick number of buffer row 0
        # (absolute tick, planner state, kind of piece) where each planned piece begins
        self._checkpoints: list[tuple[int, tuple, str]] = []

    # ---- reading ----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._zmp) - self._i

    def pop(self) -> tuple[np.ndarray, np.ndarray, int]:
        i = self._i
        self._i += 1
        return self._zmp[i], self._feet[i], int(self._support[i])

    def zmp_ahead(self, n: int) -> np.ndarray:
        return self._zmp[self._i : self._i + n]

    def idle(self) -> bool:
        """True when nothing but standing still is planned."""
        return self.standing and bool(np.all(self._support[self._i :] == 0)) and self._still()

    def rewind(self, lead: int, notice: int, moving: bool) -> None:
        """Forget the pieces of the plan that start at least `lead` ticks from now, so a new command
        takes effect after the step in progress instead of after everything already planned.

        Swapping a settle (weight drifts back to the middle) for a step (weight jumps to one foot), or
        the other way round, needs `notice` ticks of warning: with less, the body sway cannot follow,
        the robot starts to rock sideways and a few quick taps on a key tip it over.
        """
        self._forget_past()
        now = self._base + self._i
        for k, (tick, state, kind) in enumerate(self._checkpoints):
            if tick < now + lead:
                continue
            if tick < now + notice and (kind == "settle") != (self._kind(state, moving) == "settle"):
                continue
            del self._checkpoints[k:]
            end = tick - self._base
            self._zmp, self._feet, self._support = self._zmp[:end], self._feet[:end], self._support[:end]
            feet, zmp, self.standing, self.next_swing, self.frame, self.last_stride = state
            self.feet_end, self.zmp_end = list(feet), zmp.copy()
            return

    def _forget_past(self) -> None:
        """Checkpoints behind the current tick can never be rewound to; keep the list short."""
        now = self._base + self._i
        drop = next((k for k, c in enumerate(self._checkpoints) if c[0] >= now), len(self._checkpoints))
        del self._checkpoints[:drop]

    def _still(self) -> bool:
        rest = self._feet[self._i :]
        return len(rest) == 0 or bool(np.allclose(rest, rest[-1], atol=1e-9))

    # ---- planning ---------------------------------------------------------------------

    def extend(self, speed: float, turn: float) -> None:
        """Append the next piece of the plan for the commanded speed (m/s) and turn rate (rad/s)."""
        p = self.p
        self._forget_past()
        state = (list(self.feet_end), self.zmp_end.copy(), self.standing, self.next_swing, self.frame, self.last_stride)
        moving = abs(speed) > 1e-3 or abs(turn) > 1e-3
        kind = self._kind(state, moving)
        self._checkpoints.append((self._base + len(self._zmp), state, kind))
        if kind == "start":
            self.frame = self._middle()
            self.next_swing = LEFT if turn >= 0 else RIGHT  # open toward the turn
            stance = self.next_swing ^ 1
            self._hold_shift(self._stance_zmp(stance, 0.0), p.start)
            self.standing = False
            self._step(speed, turn, double_support=False)
        elif kind == "step":
            # Walking on, slowing down (one step), reversing (two) or closing up the feet to stop.
            self._step(speed, turn, double_support=True)
        elif kind == "settle":
            middle = (self.feet_end[LEFT].xy() + self.feet_end[RIGHT].xy()) / 2
            self._hold_shift(middle, p.settle)
            self.standing = True
            self.last_stride = 0.0
        else:
            self._hold_shift(self.zmp_end, 0.1)

    def _kind(self, state: tuple, moving: bool) -> str:
        """Which piece extend() appends from a planner state: start, step, settle or hold."""
        feet, _, standing, next_swing, frame, last_stride = state
        if moving and standing:
            return "start"
        if moving or abs(last_stride) > 1e-3:
            return "step"
        if standing:
            return "hold"
        target = Foot(*self._place(frame, next_swing), frame[2])
        foot = feet[next_swing]
        apart = math.hypot(foot.x - target.x, foot.y - target.y) > 0.004 or abs(_wrap(foot.yaw - target.yaw)) > 0.03
        return "step" if apart else "settle"

    def _step(self, speed: float, turn: float, double_support: bool) -> None:
        p = self.p
        swing, stance = self.next_swing, self.next_swing ^ 1
        stride = float(np.clip(speed * p.step_time, -p.max_stride, p.max_stride))
        # Starting or stopping takes one step; reversing takes two, so the body sway can follow.
        stride = float(np.clip(stride, self.last_stride - p.max_stride, self.last_stride + p.max_stride))
        dyaw = float(np.clip(turn * p.step_time, -p.max_turn, p.max_turn))
        x, y, yaw = self.frame
        mid_yaw = yaw + dyaw / 2
        frame = (x + stride * math.cos(mid_yaw), y + stride * math.sin(mid_yaw), yaw + dyaw)
        tx, ty = self._place(frame, swing)
        target = self._keep_apart(Foot(tx, ty, frame[2]), self.feet_end[stance], swing)

        zmp_stance = self._stance_zmp(stance, -0.5 * p.zmp_roll * stride)
        if double_support:
            self._hold_shift(zmp_stance, p.double)
        self._swing(swing, target, stance, stride)
        self.frame = frame
        self.last_stride = stride
        self.next_swing = stance

    def _swing(self, swing: int, target: Foot, stance: int, stride: float) -> None:
        p = self.p
        n = max(1, round(p.single / self.dt))
        u = np.arange(1, n + 1) / n
        # Lift first, travel in the middle, set down last, so the sole never scuffs the floor.
        travel = np.clip((u - 0.15) / 0.7, 0.0, 1.0)
        s = 0.5 - 0.5 * np.cos(np.pi * travel)
        start = self.feet_end[swing]
        feet = np.repeat(self._feet_row()[None], n, axis=0)
        feet[:, swing, 0] = start.x + s * (target.x - start.x)
        feet[:, swing, 1] = start.y + s * (target.y - start.y)
        feet[:, swing, 2] = p.lift * np.sin(np.pi * u) ** 2
        feet[:, swing, 3] = start.yaw + s * _wrap(target.yaw - start.yaw)
        roll = p.zmp_roll * stride
        zmp = np.array([self._stance_zmp(stance, roll * (v - 0.5)) for v in u])
        self._append(zmp, feet, np.full(n, 1 + stance))
        self.feet_end[swing] = target
        self.zmp_end = zmp[-1]

    def _hold_shift(self, zmp_to: np.ndarray, duration: float) -> None:
        """Both feet down while the ZMP moves smoothly to `zmp_to`."""
        n = max(1, round(duration / self.dt))
        u = np.arange(1, n + 1) / n
        s = (0.5 - 0.5 * np.cos(np.pi * u))[:, None]
        zmp = self.zmp_end + s * (np.asarray(zmp_to) - self.zmp_end)
        self._append(zmp, np.repeat(self._feet_row()[None], n, axis=0), np.zeros(n, dtype=int))
        self.zmp_end = np.asarray(zmp_to, float).copy()

    def _append(self, zmp: np.ndarray, feet: np.ndarray, support: np.ndarray) -> None:
        self._zmp = np.concatenate([self._zmp[self._i :], zmp])
        self._feet = np.concatenate([self._feet[self._i :], feet])
        self._support = np.concatenate([self._support[self._i :], support])
        self._base += self._i
        self._i = 0

    # ---- geometry ---------------------------------------------------------------------

    def _feet_row(self) -> np.ndarray:
        return np.array([[f.x, f.y, 0.0, f.yaw] for f in self.feet_end])

    def _middle(self) -> tuple[float, float, float]:
        left, right = self.feet_end
        mid = (left.xy() + right.xy()) / 2
        return float(mid[0]), float(mid[1]), left.yaw + _wrap(right.yaw - left.yaw) / 2

    def _place(self, frame: tuple[float, float, float], side: int) -> tuple[float, float]:
        x, y, yaw = frame
        dy = SIDE_SIGN[side] * self.p.half_width
        return x - math.sin(yaw) * dy, y + math.cos(yaw) * dy

    def _stance_zmp(self, stance: int, forward: float) -> np.ndarray:
        inward = -SIDE_SIGN[stance] * self.p.zmp_inset
        return self.feet_end[stance].local(forward, inward)

    def _keep_apart(self, target: Foot, stance: Foot, swing: int) -> Foot:
        """Never put the swing foot on or across the stance foot (e.g. while turning)."""
        c, s = math.cos(stance.yaw), math.sin(stance.yaw)
        dx, dy = target.x - stance.x, target.y - stance.y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        side = SIDE_SIGN[swing]
        if side * ly >= self.p.min_width:
            return target
        ly = side * self.p.min_width
        return Foot(stance.x + c * lx - s * ly, stance.y + s * lx + c * ly, target.yaw)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
