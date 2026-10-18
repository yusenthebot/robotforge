"""Lesson: how a joint holds an angle (P, D, steady-state error, feedforward and I).

The rig is one arm on a motor. The learner is the control engineer: every step starts with a guess, then
a step test from hanging straight down to 120 degrees that lands in the experiment record, so runs with
different gains lie over each other. A live formula shows each term of the controller with today's numbers.
`JointLesson` holds what any one-joint PID lesson needs; the PiPER lesson (pd_piper.py) reuses it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import mujoco
import numpy as np

from robotforge.learn.engine import (
    Bench,
    Control,
    Lesson,
    LessonContext,
    Metric,
    Plot,
    Quiz,
    Series,
    Step,
    Sweep,
    Text,
)

PIVOT = (0.0, 0.0, 0.62)  # shoulder height, m
LENGTH = 0.36  # pivot to the weight, m
TIP_MASS = 0.5
LINK_MASS = 0.2
TORQUE_MAX = 4.0  # N*m, enough to lift the arm with some to spare
TEST_START = 0.0  # degrees: the test starts hanging straight down...
TEST_TARGET = 120.0  # ...and aims here
TEST_SECONDS = 3.0
SETTLE_BAND = 3.0  # degrees, for two stars
FINE_BAND = 1.0  # degrees, for three stars: needs the steady-state error gone
SAMPLE_DT = 0.01
INTEGRAL_LIMIT = 2.0  # rad*s: the integral never grows past this
REST_SPAN = 2.0  # degrees: within this over the last half second, the arm has come to rest


def _deg(rad: float) -> float:
    return math.degrees(rad)


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


def _signed(value: float, digits: int) -> str:
    """`value` with its sign, as the live formula shows it; one that rounds to zero reads +0.00, not -0.00."""
    text = f"{value:+.{digits}f}"
    return text.replace("-", "+") if float(text) == 0 else text


class JointPD:
    """u = Kp*e - Kd*omega + Ki*integral(e) + gravity feedforward, clipped to the motor's limit.

    Angles are in the lesson's frame: `sign` and `offset` map the joint's own reading onto it, so on any
    arm a bigger angle means lifting. The integral stops growing while the motor is already at its limit
    and the error would push it further (anti-windup), otherwise a long climb charges it up and it
    overshoots afterwards.
    """

    rate_hz = 500.0

    def __init__(
        self,
        model: mujoco.MjModel,
        ctx: LessonContext,
        prefix: str = "",
        values: Mapping[str, float] | None = None,
        *,
        joint: str | None = None,
        actuator: str | None = None,
        torque_max: float = TORQUE_MAX,
        sign: float = 1.0,
        offset: float = 0.0,
        gravity: Callable[[mujoco.MjData], float] | None = None,
    ) -> None:
        """`prefix` names one rig among several; `values` (default: the lesson's) gives its gains and target.
        `gravity` is the controller's own model of what gravity takes, in the lesson's frame (default: the
        simulation's exact value, a perfect model)."""
        self.ctx = ctx
        self.values = values if values is not None else ctx.values
        self.joint = model.joint(joint or f"{prefix}shoulder")
        self.qadr = int(self.joint.qposadr[0])
        self.dadr = int(self.joint.dofadr[0])
        self.act = model.actuator(actuator or f"{prefix}motor").id
        self.torque_max = torque_max
        self.sign = sign
        self.offset = offset
        self.gravity = gravity
        self.integral = 0.0
        self.last_t = None
        self.next_sample = 0.0
        self.samples: deque[dict[str, float]] = deque(maxlen=2048)
        self.terms = {"u": 0.0, "raw": 0.0, "p": 0.0, "d": 0.0, "i": 0.0, "ff": 0.0, "e": 0.0, "w": 0.0}

    def angle(self, data: mujoco.MjData) -> float:
        """The joint angle in the lesson's frame, radians."""
        return self.sign * float(data.qpos[self.qadr]) + self.offset

    def set_angle(self, data: mujoco.MjData, rad: float) -> None:
        data.qpos[self.qadr] = (rad - self.offset) / self.sign
        data.qvel[self.dadr] = 0.0

    def speed(self, data: mujoco.MjData) -> float:
        return self.sign * float(data.qvel[self.dadr])

    def needed(self, data: mujoco.MjData) -> float:
        """The torque that holds the arm still right now (gravity and any load), in the lesson's frame."""
        return self.sign * float(data.qfrc_bias[self.dadr])

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        v = self.values
        q, w = self.angle(data), self.speed(data)
        target = math.radians(v["target"])
        e = target - q
        dt = 0.0 if self.last_t is None else max(0.0, data.time - self.last_t)
        self.last_t = data.time
        p, d = v["kp"] * e, -v["kd"] * w
        ff = 0.0
        if v["gravity"]:
            ff = self.gravity(data) if self.gravity is not None else self.needed(data)
        if v["ki"] > 0:
            before = p + d + v["ki"] * self.integral + ff
            if abs(before) < self.torque_max or e * before < 0:  # anti-windup: no charging at the limit
                self.integral = float(np.clip(self.integral + e * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT))
        else:
            self.integral = 0.0
        i = v["ki"] * self.integral
        raw = p + d + i + ff
        u = float(np.clip(raw, -self.torque_max, self.torque_max))
        data.ctrl[self.act] = self.sign * u
        self.terms = {"u": u, "raw": raw, "p": p, "d": d, "i": i, "ff": ff, "e": e, "w": w}
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            self.samples.append(
                {"t": data.time, "q": _deg(q), "target": v["target"], "w": w, "u": u, "p": p, "d": d, "iff": i + ff}
            )

    def restart_integral(self) -> None:
        self.integral = 0.0

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-pd", **{k: round(x, 3) for k, x in self.terms.items()}}


def grade(
    times: list[float], angles: list[float], target: float = TEST_TARGET, start: float = TEST_START
) -> dict[str, Any]:
    """Score a step test from its curve (seconds since the step, degrees): overshoot past the target as a
    share of the step, the time it settles into the +-3 and +-1 degree bands for good, how far off it
    ends (mean of the last half second, if it has come to rest), and stars. Live tests and sweeps grade
    the same way."""
    direction = 1.0 if target >= start else -1.0
    beyond = max((direction * (a - target) for a in angles), default=0.0)
    overshoot = max(0.0, beyond) / max(abs(target - start), 1e-9) * 100

    def settled(band: float) -> float | None:
        since = None  # when it last came inside the band, if it stayed there
        for t, q in zip(times, angles, strict=True):
            inside = abs(q - target) <= band
            since = (since if since is not None else t) if inside else None
        return since

    tail = [q for t, q in zip(times, angles, strict=True) if times and t >= times[-1] - 0.5]
    resting = bool(tail) and max(tail) - min(tail) <= REST_SPAN  # a swinging arm ends nowhere in particular
    end_err = abs(target - float(np.mean(tail))) if resting else None
    settle, fine = settled(SETTLE_BAND), settled(FINE_BAND)
    stars = 0
    if settle is not None:
        stars = 1
        if settle <= 1.5 and overshoot <= 15:
            stars = 2
        if fine is not None and fine <= 1.0 and overshoot <= 5:
            stars = 3
    return {
        "settle_s": None if settle is None else round(settle, 2),
        "fine_s": None if fine is None else round(fine, 2),
        "overshoot_pct": round(overshoot, 1),
        "end_err": None if end_err is None else round(end_err, 2),
        "stars": stars,
    }


# ---- one-joint lessons in general ---------------------------------------------------------------------

GAINS = ("kp", "kd", "ki", "gravity")


def _runs(ctx: LessonContext, **want: Callable[[float], bool]) -> list[dict[str, Any]]:
    """Recorded runs whose settings pass every test in `want` (knob -> predicate)."""
    return [r for r in ctx.runs if all(test(r["params"].get(k, 0.0)) for k, test in want.items())]


class JointLesson(Lesson):
    """A lesson around one joint under `JointPD`: a step test from `test_start` to `test_target` degrees
    that is graded and recorded, sweeps of one gain on copies of the rig, and the controller's terms as a
    live formula. Subclasses build the rig and say how to make its controller."""

    test_start = TEST_START
    test_target = TEST_TARGET
    torque_max = TORQUE_MAX
    kick_speed = 5.0  # rad/s added by Push

    def __init__(self) -> None:
        self.pd: JointPD | None = None
        self.test: dict[str, Any] | None = None
        self.model: mujoco.MjModel | None = None

    def make_pd(self, model: mujoco.MjModel, ctx: LessonContext) -> JointPD:
        raise NotImplementedError

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.pd = self.make_pd(model, ctx)
        return {"lesson": self.pd}

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        assert self.pd is not None
        if control == "kick":
            data.qvel[self.pd.dadr] += self.pd.sign * self.kick_speed
        elif control == "test":
            self.pd.set_angle(data, math.radians(self.test_start))
            self.pd.restart_integral()
            ctx.values["target"] = self.test_target
            self.test = {"t0": data.time, "times": [], "angles": []}
            ctx.results["test"] = {"running": True}
        elif control in ("ki", "gravity"):
            self.pd.restart_integral()

    def drain_samples(self) -> list[dict[str, float]]:
        if self.pd is None:
            return []
        out = list(self.pd.samples)
        self.pd.samples.clear()
        self._follow_test(out)
        return out

    def _follow_test(self, samples: list[dict[str, float]]) -> None:
        """Follow a running test: collect its curve until it has run TEST_SECONDS."""
        test = self.test
        if test is None or not samples:
            return
        for s in samples:
            t = s["t"] - test["t0"]
            if t < 0:
                continue
            if t > TEST_SECONDS:
                self._finish_test()
                return
            test["times"].append(t)
            test["angles"].append(s["q"])

    def _finish_test(self) -> None:
        assert self.test is not None and self.pd is not None
        test, self.test = self.test, None
        result = grade(test["times"], test["angles"], self.test_target, self.test_start)
        ctx = self.pd.ctx
        ctx.results["test"] = {"running": False, **result}
        params = {k: ctx.values.get(k, 0.0) for k in (self.bench.params if self.bench else GAINS)}
        ctx.record_run(params, result, test["times"], test["angles"])

    def fresh_data(self, model: mujoco.MjModel) -> mujoco.MjData:
        """The rig at rest in its home pose, for a sweep's private run."""
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        return data

    def run_sweep(self, sweep: Sweep, ctx: LessonContext) -> list[dict[str, Any]]:
        """The step test once per value, on a fresh copy of the rig: the live joint keeps running."""
        assert self.model is not None
        model = self.model
        points = []
        for value in sweep.values:
            values = {**ctx.values, sweep.control: float(value), "target": self.test_target}
            pd = self.make_pd(model, LessonContext(values=values))
            data = self.fresh_data(model)
            pd.set_angle(data, math.radians(self.test_start))
            mujoco.mj_forward(model, data)
            while data.time <= TEST_SECONDS:
                pd.update(model, data)
                mujoco.mj_step(model, data)
            times = [s["t"] for s in pd.samples]
            angles = [s["q"] for s in pd.samples]
            points.append({"x": float(value), **grade(times, angles, self.test_target, self.test_start)})
        return points

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        assert self.pd is not None
        q = _deg(self.pd.angle(data))
        target = ctx.values["target"]
        if self.controller_off(ctx):
            readout = _text(f"控制器关着 · 角度 {q:.1f}°", f"Controller off · angle {q:.1f}°")
        else:
            readout = _text(
                f"角度 {q:.1f}° · 目标 {target:g}° · 差 {_signed(target - q, 1)}°",
                f"angle {q:.1f}° · target {target:g}° · off by {_signed(target - q, 1)}°",
            )
        if self.test is not None:
            readout = _text("测试中……", "Testing...")
        return {
            "angle": round(q, 2),
            "error": round(target - q, 2),
            "torque": round(self.pd.terms["u"], 3),
            "torque_max": self.torque_max,
            "gravity_torque": round(self.pd.needed(data), 3),
            "readout": readout,
            "formula": self.formula(model, data, ctx),
        }

    @staticmethod
    def controller_off(ctx: LessonContext) -> bool:
        v = ctx.values
        return v["kp"] == 0 and v["kd"] == 0 and v["ki"] == 0 and not v["gravity"]

    def formula(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> list[dict[str, Any]]:
        """The controller's sum with today's numbers, one row per term; each row appears once its step's
        `show` flag is on (f_p, f_d, f_i, f_g)."""
        assert self.pd is not None
        v, t, limit = ctx.values, self.pd.terms, self.torque_max
        saturated = abs(t["raw"]) > limit + 1e-9
        motor: dict[str, Any] = {
            "show": "f_p",
            "label": _text("电机出力", "Motor"),
            "expr": _text("上面几项加起来", "the terms added up"),
            "value": f"{_signed(t['u'], 2)} N·m",
            "color": "ink",
        }
        if saturated:
            motor["note"] = _text(
                f"算出来要 {_signed(t['raw'], 1)}，可电机最多 ±{limit:g} N·m，只能出满力",
                f"the sum asks for {_signed(t['raw'], 1)}, but the motor tops out at ±{limit:g} N·m",
            )
        ff_on = bool(v["gravity"])
        return [
            {
                "show": "f_p",
                "label": _text("P 项", "P"),
                "expr": f"Kp × e = {v['kp']:g} × {_signed(t['e'], 3)} rad",
                "value": f"{_signed(t['p'], 2)} N·m",
                "color": "blue",
            },
            {
                "show": "f_d",
                "label": _text("D 项", "D"),
                "expr": f"−Kd × ω = −{v['kd']:g} × {_signed(t['w'], 2)} rad/s",
                "value": f"{_signed(t['d'], 2)} N·m",
                "color": "violet",
            },
            {
                "show": "f_i",
                "label": _text("I 项", "I"),
                "expr": f"Ki × ∫e = {v['ki']:g} × {_signed(self.pd.integral, 3)}",
                "value": f"{_signed(t['i'], 2)} N·m",
                "color": "green",
            },
            {
                "show": "f_i",
                "label": _text("前馈", "Feedforward"),
                "expr": _text("模型算出的重力", "gravity from the model") if ff_on else _text("关着", "off"),
                "value": f"{_signed(t['ff'], 2)} N·m",
                "color": "green",
            },
            motor,
            {
                "show": "f_g",
                "label": _text("重力要", "Gravity needs"),
                "expr": _text("把手臂托住要的力矩", "to hold the arm up"),
                "value": f"{_signed(self.pd.needed(data), 2)} N·m",
                "color": "muted",
            },
        ]


# ---- PID I: the weighted arm ---------------------------------------------------------------------------


def _swung_by_hand(ctx: LessonContext) -> bool:
    return "dragged" in ctx.events and any(abs(s["q"]) > 40 for s in ctx.samples)


def _p_compared(ctx: LessonContext) -> bool:
    """Two P-only tests whose Kp differ at least twofold."""
    kps = [r["params"]["kp"] for r in _runs(ctx, kd=lambda x: x <= 0.1, ki=lambda x: x == 0, gravity=lambda x: x == 0)]
    kps = [k for k in kps if k > 0]
    return len(kps) >= 2 and max(kps) >= 2 * min(kps)


def _damped(ctx: LessonContext) -> bool:
    """Swept Kd, then ran a test that settles quickly without overshooting."""
    good = [
        r
        for r in _runs(ctx, kp=lambda x: x >= 20)
        if r["metrics"]["overshoot_pct"] <= 5 and r["metrics"]["settle_s"] is not None
        if r["metrics"]["settle_s"] <= 0.7
    ]
    return "swept:kd" in ctx.events and bool(good)


def _stiffer(ctx: LessonContext) -> bool:
    """A test with Kp 70 or more that settled, P and D only."""
    runs = _runs(ctx, kp=lambda x: x >= 70, ki=lambda x: x == 0, gravity=lambda x: x == 0)
    return any(r["metrics"]["settle_s"] is not None for r in runs)


def _ff_and_i(ctx: LessonContext) -> bool:
    """One test with the feedforward alone and one with the integral alone."""
    ff = _runs(ctx, gravity=lambda x: x > 0, ki=lambda x: x == 0, kp=lambda x: x > 0)
    i = _runs(ctx, gravity=lambda x: x == 0, ki=lambda x: x > 0, kp=lambda x: x > 0)
    return bool(ff) and bool(i)


def _challenge(ctx: LessonContext) -> bool:
    return ctx.results.get("test", {}).get("stars", 0) >= 2


STEPS = (
    Step(
        id="feel",
        title=_text("认识这个关节", "Meet the joint"),
        body=_text(
            "这是一根装在电机上的手臂，末端有一个配重。现在控制器关着，电机不出力，重力让它垂下来。\n"
            "用鼠标把手臂拖起来再松手，看看它怎么动。",
            "This is an arm on a motor with a weight at its end. "
            "The controller is off, so the motor does nothing and gravity lets the arm hang.\n"
            "Drag the arm up with the mouse and let go. Watch how it moves.",
        ),
        done=_text(
            "没有控制时，手臂只受重力和一点点摩擦，所以会来回摆很久。控制器的工作，就是决定电机每一刻该出多大的力。",
            "Without control the arm only feels gravity and a little friction, so it swings for a long time. "
            "A controller decides, moment by moment, how hard the motor should push.",
        ),
        check=_swung_by_hand,
        show=("scale",),
    ),
    Step(
        id="p",
        title=_text("比例控制 P：像一根弹簧", "Proportional control: a spring"),
        body=_text(
            "控制器每时每刻算一次误差 e = 目标 − 当前角度，P 控制让电机出力 u = Kp × e："
            "离目标越远推得越用力，像一根把手臂拉向目标的弹簧。下面的公式会实时填上数字。\n"
            "把 Kp 设成 10，点「试一次」：手臂从垂直向下出发，去 120°。结果会留在右边的实验记录里。\n"
            "然后先猜一猜，再把 Kp 调到 40 试一次。",
            "Every moment the controller works out the error e = target - angle, and P control makes the motor "
            "push with u = Kp * e: the further off, the harder it pushes, like a spring pulling the arm to the "
            "target. The formula below fills in the numbers live.\n"
            "Set Kp to 10 and press Run test: the arm starts hanging straight down and heads for 120 degrees. "
            "The result stays in the experiment record on the right.\n"
            "Then make a guess and run it again with Kp 40.",
        ),
        done=_text(
            "两条曲线几乎重合，都在 9° 到 170° 之间来回摆。原因有两个：弹簧只会往目标拉，不会刹车；"
            "而且离目标远时 Kp × e 早就超过了电机的 4 N·m 上限（公式里会提示），电机已经在出满力，"
            "弹簧再硬也推不出更多。真实的电机都有这个上限。",
            "The two curves almost coincide, both swinging between 9 and 170 degrees. Two reasons: a spring "
            "only pulls toward the target and never brakes, and far from the target Kp * e is way past the "
            "motor's 4 N*m limit (the formula says so), so the motor is already pushing flat out and a stiffer "
            "spring cannot push any harder. Every real motor has such a limit.",
        ),
        quiz=Quiz(
            question=_text("Kp 从 10 加到 40，会怎样？", "Kp goes from 10 to 40. What happens?"),
            options=(
                _text("稳稳停在 120°", "It stops neatly at 120 degrees"),
                _text("还是来回摆，曲线和 Kp 10 差不多", "It still swings, much like Kp 10"),
                _text("摆得更猛，甩过头更多", "It swings harder and flies further past"),
            ),
            answer=1,
            explain=_text(
                "P 只管把手臂拉向目标，没有东西把摆动的能量吃掉。",
                "P only pulls toward the target; nothing soaks up the swing's energy.",
            ),
            predict=True,
        ),
        check=_p_compared,
        unlock=("kp", "test"),
        show=("scale", "target", "error", "torque", "f_p"),
    ),
    Step(
        id="d",
        title=_text("微分控制 D：加一个阻尼器", "Derivative control: a damper"),
        body=_text(
            "D 项看的是转速 ω：u = Kp × e − Kd × ω。转得越快，往回拉得越狠，像门上的阻尼器。\n"
            "Kp 留在 40，先猜一猜，再点「扫一遍 Kd」：它会把 Kd 从 0 到 6 各测一次，画出超调和稳定时间。"
            "最后挑一个你觉得最好的 Kd，自己试一次。",
            "The D term looks at the turning speed omega: u = Kp * e - Kd * omega. The faster it turns, the "
            "harder it is held back, like a door damper.\n"
            "Keep Kp at 40, make a guess, then press Sweep Kd: it runs the test for Kd from 0 to 6 and plots "
            "overshoot and settling time. Finally pick the Kd you like best and run it yourself.",
        ),
        done=_text(
            "超调随 Kd 一路降到 0，稳定时间却先降后升：Kd 3 左右最快（0.4 秒），Kd 6 一点不晃，却要 0.95 秒才爬到位。"
            "太小会晃，太大会迟钝，调参就是找中间那一点。",
            "Overshoot falls to zero as Kd grows, but settling time first drops, then rises: about Kd 3 is "
            "quickest (0.4 s), while Kd 6 never wobbles yet needs 0.95 s to creep in. Too little wobbles, too "
            "much is sluggish; tuning is finding the spot between.",
        ),
        quiz=Quiz(
            question=_text("如果 Kd 加到很大（6），会怎样？", "What if Kd is very large (6)?"),
            options=(
                _text("越大越好，最快稳住", "The bigger the better: fastest to settle"),
                _text("一点不晃，但慢吞吞地爬到位", "No wobble at all, but it creeps in slowly"),
                _text("晃得更厉害", "It wobbles even more"),
            ),
            answer=1,
            explain=_text(
                "阻尼太大，手臂每动一点都被拽住，所以不晃却很慢。",
                "With too much damping every movement is held back, so it never wobbles but is slow.",
            ),
            predict=True,
        ),
        check=_damped,
        unlock=("kd", "kick"),
        show=("f_d",),
    ),
    Step(
        id="why_low",
        title=_text("为什么停在目标下面？", "Why does it stop short?"),
        body=_text(
            "看公式：手臂停住后，D 项是 0（不转了），电机全靠 P 项把手臂托住，所以 Kp × e 必须等于重力要的力矩。"
            "这就是说，手臂必须留着一点误差 e。\n"
            "先猜，再把 Kp 调到 80、Kd 调到 5 试一次（Kd 跟着加，免得晃）。",
            "Look at the formula: once the arm is still, D is zero (nothing turns), and the motor holds the arm "
            "with the P term alone, so Kp * e must equal what gravity needs. In other words the arm has to keep "
            "a little error e.\n"
            "Guess first, then run a test with Kp 80 and Kd 5 (more Kd too, so it does not wobble).",
        ),
        done=_text(
            "Kp 翻倍，误差减半：2.7° 变成 1.3°（看实验记录里的“最后差”）。e = 重力 ÷ Kp，Kp 越大误差越小，"
            "但只要还要出力，误差就到不了 0。这叫稳态误差。",
            "Double Kp, half the error: 2.7 degrees became 1.3 (see 'ended off by' in the record). e = gravity / "
            "Kp: a bigger Kp shrinks it, but as long as the motor must push, the error never reaches zero. This "
            "is steady-state error.",
        ),
        quiz=Quiz(
            question=_text(
                "Kp 40 时最后差 2.7°。Kp 80 时会差多少？", "With Kp 40 it ends 2.7 degrees short. With Kp 80?"
            ),
            options=(
                _text("大约 1.3°，少一半", "About 1.3 degrees, half as much"),
                _text("还是 2.7°", "Still 2.7 degrees"),
                _text("0°，正好到位", "0 degrees, spot on"),
            ),
            answer=0,
            explain=_text(
                "Kp × e 要等于同样的重力，Kp 翻倍，e 就减半。",
                "Kp * e must match the same gravity, so twice the Kp means half the e.",
            ),
            predict=True,
        ),
        check=_stiffer,
        show=("gravity", "f_g"),
    ),
    Step(
        id="ff_i",
        title=_text("消掉误差：前馈和积分", "Removing the error: feedforward and integral"),
        body=_text(
            "两种办法：\n"
            "① 重力补偿（前馈）：既然算得出重力要多少，就提前把它加进电机输出。\n"
            "② 积分 I：把误差一点点累加起来，只要还有误差，这一项就越来越大，直到误差消失。\n"
            "先猜，再各试一次：一次只开重力补偿，一次只开积分（Ki 设成 10）。两条曲线会叠在一起。",
            "Two ways:\n"
            "1. Gravity compensation (feedforward): we can work out what gravity needs, so add it to the motor "
            "output up front.\n"
            "2. Integral I: keep adding up the error; while any is left this term keeps growing, until it is "
            "gone.\n"
            "Guess, then run one test with only gravity compensation and one with only the integral (Ki 10). "
            "The two curves will lie over each other.",
        ),
        done=_text(
            "前馈一上来就把重力那份力补上，0.4 秒就差 0°；积分得先攒误差，Ki 10 测满 3 秒还差 1.1°"
            "（我们测的：Ki 40 才追上前馈，Ki 60 开始冲过头）。"
            "前馈快，但得有准确的模型；积分不需要模型，却慢。真实的机器人两个一起用：PID 加前馈。",
            "Feedforward supplies gravity's share from the start: 0 degrees off within 0.4 s. The integral has "
            "to collect error first: with Ki 10 it is still 1.1 degrees off after the full 3 s (in our runs Ki 40 "
            "catches up with feedforward and Ki 60 starts to overshoot). Feedforward is fast but needs a good "
            "model; the integral needs no model but is slow. Real robots use both: PID plus feedforward.",
        ),
        quiz=Quiz(
            question=_text("哪个能更快把误差消掉？", "Which removes the error faster?"),
            options=(
                _text("重力补偿（前馈）", "Gravity compensation (feedforward)"),
                _text("积分 I", "The integral"),
                _text("一样快", "Both the same"),
            ),
            answer=0,
            explain=_text(
                "前馈不用等误差出现，积分要等误差攒起来。",
                "Feedforward does not wait for an error; the integral must let one build up first.",
            ),
            predict=True,
        ),
        check=_ff_and_i,
        unlock=("gravity", "ki"),
        show=("f_i",),
    ),
    Step(
        id="challenge",
        title=_text("挑战：一秒内稳稳到位", "Challenge: on target within a second"),
        body=_text(
            "点「试一次」，目标还是从垂直向下到 120°。\n"
            f"两颗星：1.5 秒内稳定在 ±{SETTLE_BAND:g}° 以内，超调不超过 15%。"
            f"三颗星：1 秒内稳定在 ±{FINE_BAND:g}° 以内，超调不超过 5%。\n"
            "所有旋钮都可以调，目标角度也解锁了，可以随便玩。",
            "Press Run test; it is the same climb from hanging down to 120 degrees.\n"
            f"Two stars: within ±{SETTLE_BAND:g} degrees in 1.5 s, overshooting by at most 15%. "
            f"Three stars: within ±{FINE_BAND:g} degree in 1 s, overshooting by at most 5%.\n"
            "Every knob is yours now, and the target slider is unlocked to play with.",
        ),
        done=_text(
            "你刚刚亲手整定了一个 PID 控制器。机械臂、无人机、四足机器人的每一个关节里，跑的都是这一套。"
            "下一课换成真机 PiPER 的肩关节，看看这套增益搬过去还灵不灵。",
            "You just tuned a PID controller by hand. Every joint of robot arms, drones and legged robots runs "
            "this loop. The next lesson moves to the real PiPER's shoulder: do these gains still work there?",
        ),
        check=_challenge,
        unlock=("target",),
    ),
)


def rig_spec(name: str) -> mujoco.MjSpec:
    """An empty stage for joint rigs: floor, light and the materials the viewer styles."""
    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.option.timestep = 0.002
    w = spec.worldbody
    w.add_light(pos=[0.5, -1.0, 2.5], dir=[-0.2, 0.4, -1.0], diffuse=[0.8, 0.8, 0.8])
    floor = spec.add_material(name="rf_floor", rgba=[0.62, 0.66, 0.7, 1])
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material=floor.name)
    for material in ("rf_dark", "rf_metal", "rf_shell", "rf_rubber"):
        spec.add_material(name=material)
    return spec


def add_rig(spec: mujoco.MjSpec, prefix: str = "", x: float = 0.0, weight_rgba: list[float] | None = None) -> None:
    """One motor joint swinging a weighted arm, named with `prefix` and standing at `x` (several fit side by side)."""
    G = mujoco.mjtGeom
    w = spec.worldbody
    # A weighted pedestal and a pillar up to the shoulder; the motor sits on top, facing the camera.
    w.add_geom(
        name=f"{prefix}foot",
        type=G.mjGEOM_CYLINDER,
        size=[0.16, 0.02, 0],
        pos=[x, 0, 0.02],
        rgba=[0.16, 0.18, 0.21, 1],
        material="rf_dark",
    )
    w.add_geom(
        name=f"{prefix}pillar",
        type=G.mjGEOM_BOX,
        size=[0.035, 0.035, PIVOT[2] / 2],
        pos=[x, 0.05, PIVOT[2] / 2],
        rgba=[0.78, 0.8, 0.83, 1],
        material="rf_metal",
    )
    w.add_geom(
        name=f"{prefix}motor",
        type=G.mjGEOM_CYLINDER,
        size=[0.05, 0.035, 0],
        pos=[x, 0.0, PIVOT[2]],
        quat=[0.7071068, 0.7071068, 0, 0],
        rgba=[0.16, 0.18, 0.21, 1],
        material="rf_dark",
    )
    arm = w.add_body(name=f"{prefix}arm", pos=[PIVOT[0] + x, PIVOT[1], PIVOT[2]])  # its parts touch nothing
    # Axis -y: positive angles swing the arm toward +x, to the right as the camera sees it.
    arm.add_joint(
        name=f"{prefix}shoulder",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, -1, 0],
        damping=0.004,
        frictionloss=0.002,
        armature=0.002,
    )
    arm.add_geom(
        name=f"{prefix}hub",
        contype=0,
        conaffinity=0,
        type=G.mjGEOM_CYLINDER,
        size=[0.03, 0.02, 0],
        pos=[0, -0.05, 0],
        quat=[0.7071068, 0.7071068, 0, 0],
        rgba=[0.85, 0.87, 0.9, 1],
        material="rf_metal",
        mass=0.01,
    )
    arm.add_geom(
        name=f"{prefix}link",
        contype=0,
        conaffinity=0,
        type=G.mjGEOM_CAPSULE,
        fromto=[0, -0.06, 0, 0, -0.06, -LENGTH],
        size=[0.018, 0, 0],
        rgba=[0.95, 0.96, 0.97, 1],
        material="rf_shell",
        mass=LINK_MASS,
    )
    arm.add_geom(
        name=f"{prefix}weight",
        contype=0,
        conaffinity=0,
        type=G.mjGEOM_SPHERE,
        size=[0.045, 0, 0],
        pos=[0, -0.06, -LENGTH],
        rgba=weight_rgba or [0.95, 0.66, 0.0, 1],
        material="rf_shell",
        mass=TIP_MASS,
    )
    arm.add_site(name=f"{prefix}tip", pos=[0, -0.06, -LENGTH], size=[0.01, 0, 0])
    motor = spec.add_actuator(name=f"{prefix}motor", target=f"{prefix}shoulder", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    motor.ctrllimited = True
    motor.ctrlrange = [-TORQUE_MAX, TORQUE_MAX]
    motor.gear = [1, 0, 0, 0, 0, 0]


GAIN_CONTROLS = (
    Control("kp", "slider", _text("Kp 比例", "Kp proportional"), 0, 80, 0.5, 0),
    Control("kd", "slider", _text("Kd 微分", "Kd derivative"), 0, 6, 0.05, 0),
    Control("ki", "slider", _text("Ki 积分", "Ki integral"), 0, 60, 0.5, 0),
    Control("gravity", "toggle", _text("重力补偿（前馈）", "Gravity compensation (feedforward)")),
)

ANGLE_PLOTS = (
    Plot(
        "angle",
        _text("角度", "Angle"),
        "°",
        (
            Series("target", _text("目标", "target"), "amber", dashed=True),
            Series("q", _text("实际", "actual"), "ink"),
        ),
    ),
    Plot(
        "torque",
        _text("电机力矩", "Motor torque"),
        "N·m",
        (
            Series("u", _text("总输出", "total"), "ink"),
            Series("p", _text("P 项", "P"), "blue"),
            Series("d", _text("D 项", "D"), "violet"),
            Series("iff", _text("I + 前馈", "I + feedforward"), "green"),
        ),
    ),
)


def step_bench(sweeps: tuple[Sweep, ...], reference: float, extra: tuple[tuple[str, Text], ...] = ()) -> Bench:
    """The experiment record of a step test: settings, overshoot, settling, the final error. `extra` adds
    more settings to each run, as (control, short name)."""
    params = (*GAINS, *(key for key, _ in extra))
    short = (_text("Kp", "Kp"), _text("Kd", "Kd"), _text("Ki", "Ki"), _text("前馈", "FF"))
    short += tuple(name for _, name in extra)
    return Bench(
        title=_text("实验记录：阶跃响应", "Experiment record: step responses"),
        x=_text("测试开始后", "since the step"),
        x_unit="s",
        y=_text("角度", "angle"),
        y_unit="°",
        params=params,
        short=short,
        metrics=(
            Metric("overshoot_pct", _text("超调", "overshoot"), "%"),
            Metric("settle_s", _text("稳到 ±3°", "settled to ±3°"), "s"),
            Metric("end_err", _text("最后差", "ended off by"), "°"),
        ),
        reference=reference,
        sweeps=sweeps,
        zoom=6.0,  # the last few degrees are where feedforward, integral and Kp differ
    )


class PDLesson(JointLesson):
    id = "pd_control"
    chapter = _text("控制原理", "Control")
    title = _text("PID（一）：让关节停在目标角度", "PID I: holding a joint on target")
    summary = _text(
        "先猜再试：弹簧、阻尼、稳态误差、前馈和积分，每一次测试都留在记录里叠着比。",
        "Guess, then try: springs, damping, steady-state error, feedforward and integral, every test kept "
        "in a record to lay over the others.",
    )
    minutes = 15
    controls = (
        Control("target", "slider", _text("目标角度", "Target angle"), 0, 170, 1, TEST_TARGET, "°"),
        *GAIN_CONTROLS,
        Control("kick", "button", _text("推一下", "Push")),
        Control("test", "button", _text("试一次", "Run test")),
    )
    plots = ANGLE_PLOTS
    steps = STEPS
    bench = step_bench(
        (
            Sweep("kp", (2, 5, 10, 15, 20, 30, 40, 60, 80), ("overshoot_pct", "settle_s")),
            Sweep("kd", (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6), ("overshoot_pct", "settle_s")),
        ),
        TEST_TARGET,
    )

    def build(self) -> mujoco.MjModel:
        spec = rig_spec("pd_lesson")
        add_rig(spec)
        self.model = spec.compile()
        return self.model

    def make_pd(self, model: mujoco.MjModel, ctx: LessonContext) -> JointPD:
        return JointPD(model, ctx)

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "joint",
            "pivot": [PIVOT[0], PIVOT[1] - 0.06, PIVOT[2]],
            "axis": [0, -1, 0],
            "zero": [0, 0, -1],  # where the arm points at 0 degrees
            "swing": [1, 0, 0],  # where it points at +90 degrees
            "length": LENGTH,
            "body": int(model.body("arm").id),
            "weight": [0, -0.06, -LENGTH],  # in the arm's frame
        }

    def camera(self) -> dict[str, list[float]]:
        # A little right of the pivot, where the arm swings up, but not so far that the dial's -90° ran off a phone.
        return {"eye": [0.3, -2.3, 0.95], "target": [0.06, 0.0, 0.6]}
