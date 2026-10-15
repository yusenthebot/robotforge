"""Lesson: how a joint holds an angle (P, D, steady-state error, feedforward and I).

The rig is one arm on a motor. The learner is the control engineer: they turn Kp, Kd, Ki and gravity
compensation, watch the angle and torque plots, and finish by tuning it to settle fast without overshoot.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import mujoco
import numpy as np

from robotforge.learn.engine import Bench, Control, Lesson, LessonContext, Metric, Plot, Quiz, Series, Step, Sweep

PIVOT = (0.0, 0.0, 0.62)  # shoulder height, m
LENGTH = 0.36  # pivot to the weight, m
TIP_MASS = 0.5
LINK_MASS = 0.2
TORQUE_MAX = 4.0  # N*m, enough to lift the arm with some to spare
TEST_TARGET = 120.0  # degrees, the challenge
TEST_SECONDS = 3.0
SETTLE_BAND = 3.0  # degrees, for two stars
FINE_BAND = 1.0  # degrees, for three stars: needs the steady-state error gone
SAMPLE_DT = 0.01


def _deg(rad: float) -> float:
    return math.degrees(rad)


class JointPD:
    """u = Kp*e - Kd*omega + Ki*integral(e) + gravity feedforward, clipped to the motor's limit."""

    rate_hz = 500.0

    def __init__(self, model: mujoco.MjModel, ctx: LessonContext) -> None:
        self.ctx = ctx
        self.joint = model.joint("shoulder")
        self.qadr = int(self.joint.qposadr[0])
        self.dadr = int(self.joint.dofadr[0])
        self.act = model.actuator("motor").id
        self.integral = 0.0
        self.last_t = None
        self.next_sample = 0.0
        self.samples: deque[dict[str, float]] = deque(maxlen=2048)
        self.terms = {"u": 0.0, "p": 0.0, "d": 0.0, "i": 0.0, "ff": 0.0}

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        v = self.ctx.values
        q, w = float(data.qpos[self.qadr]), float(data.qvel[self.dadr])
        target = math.radians(v["target"])
        e = target - q
        dt = 0.0 if self.last_t is None else max(0.0, data.time - self.last_t)
        self.last_t = data.time
        if v["ki"] > 0:
            self.integral = float(np.clip(self.integral + e * dt, -2.0, 2.0))  # bounded: no wind-up forever
        else:
            self.integral = 0.0
        p, d, i = v["kp"] * e, -v["kd"] * w, v["ki"] * self.integral
        ff = float(data.qfrc_bias[self.dadr]) if v["gravity"] else 0.0  # what gravity pulls with right now
        u = float(np.clip(p + d + i + ff, -TORQUE_MAX, TORQUE_MAX))
        data.ctrl[self.act] = u
        self.terms = {"u": u, "p": p, "d": d, "i": i, "ff": ff}
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            self.samples.append(
                {"t": data.time, "q": _deg(q), "target": v["target"], "w": w, "u": u, "p": p, "d": d, "iff": i + ff}
            )

    def restart_integral(self) -> None:
        self.integral = 0.0

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-pd", **{k: round(x, 3) for k, x in self.terms.items()}}


def grade(times: list[float], angles: list[float]) -> dict[str, Any]:
    """Score a step test from its curve (seconds since the step, degrees): overshoot, the time it settles
    into the +-3 and +-1 degree bands for good, and stars. Live tests and sweeps grade the same way."""
    peak = max(angles, default=0.0)
    overshoot = max(0.0, (peak - TEST_TARGET) / TEST_TARGET * 100)

    def settled(band: float) -> float | None:
        since = None  # when it last came inside the band, if it stayed there
        for t, q in zip(times, angles, strict=True):
            inside = abs(q - TEST_TARGET) <= band
            since = (since if since is not None else t) if inside else None
        return since

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
        "stars": stars,
    }


def _oscillating(ctx: LessonContext) -> bool:
    """Swinging back and forth: the speed changed sign a few times with a real amplitude."""
    window = ctx.recent(3.0)
    if len(window) < 100:
        return False
    flips = sum(1 for a, b in zip(window, window[1:], strict=False) if a["w"] * b["w"] < 0)
    span = max(s["q"] for s in window) - min(s["q"] for s in window)
    return flips >= 3 and span > 6.0


def _settled(ctx: LessonContext, seconds: float = 1.0, speed: float = 0.05) -> bool:
    window = ctx.recent(seconds)
    return len(window) >= seconds / SAMPLE_DT * 0.8 and all(abs(s["w"]) < speed for s in window)


def _steady_error(ctx: LessonContext) -> float:
    window = ctx.recent(0.5)
    return abs(np.mean([s["target"] - s["q"] for s in window])) if window else 99.0


def _swung_by_hand(ctx: LessonContext) -> bool:
    return "dragged" in ctx.events and any(abs(s["q"]) > 40 for s in ctx.samples)


def _p_only(ctx: LessonContext) -> bool:
    v = ctx.values
    return v["kp"] >= 10 and v["kd"] < 0.2 and abs(v["target"] - 90) <= 20 and _oscillating(ctx)


def _damped(ctx: LessonContext) -> bool:
    v = ctx.values
    return v["kp"] >= 10 and v["kd"] >= 0.3 and _settled(ctx) and abs(v["target"] - 90) <= 30


def _no_steady_error(ctx: LessonContext) -> bool:
    v = ctx.values
    return (v["gravity"] or v["ki"] > 0) and v["kp"] > 0 and _settled(ctx, 0.8) and _steady_error(ctx) < 1.0


def _challenge(ctx: LessonContext) -> bool:
    return ctx.results.get("test", {}).get("stars", 0) >= 2


STEPS = (
    Step(
        id="feel",
        title={"zh": "认识这个关节", "en": "Meet the joint"},
        body={
            "zh": "这是一根装在电机上的手臂，末端有一个配重。"
            "现在控制器关着，电机不出力，重力让它垂下来。\n"
            "用鼠标把手臂拖起来再松手，看看它怎么动。",
            "en": "This is an arm on a motor with a weight at its end. "
            "The controller is off, so the motor does nothing and gravity lets the arm hang.\n"
            "Drag the arm up with the mouse and let go. Watch how it moves.",
        },
        done={
            "zh": "没有控制时，手臂只受重力和一点点摩擦，所以会来回摆很久。"
            "控制器的工作，就是决定电机每一刻该出多大的力。",
            "en": "Without control the arm only feels gravity and a little friction, so it swings for a long time. "
            "A controller decides, moment by moment, how hard the motor should push.",
        },
        check=_swung_by_hand,
        show=("scale",),
    ),
    Step(
        id="p",
        title={"zh": "比例控制 P：像一根弹簧", "en": "Proportional control: a spring"},
        body={
            "zh": "控制器每时每刻算一次误差：e = 目标角度 − 当前角度。\n"
            "P 控制让电机输出 u = Kp × e：离目标越远，推得越用力，就像一根把手臂拉向目标的弹簧。\n"
            "把目标设为 90°（水平），再把 Kp 调到 10 以上，Kd 保持 0。",
            "en": "Every moment the controller works out the error: e = target angle - current angle.\n"
            "P control makes the motor push with u = Kp * e: the further from the target, the harder it pushes, "
            "like a spring pulling the arm toward the target.\n"
            "Set the target to 90 degrees (level), then raise Kp above 10 and keep Kd at 0.",
        },
        done={
            "zh": "只有弹簧、没有阻尼，所以它一直来回晃。还注意到了吗？它晃的中心比 90° 低一点，这个后面会讲。",
            "en": "A spring with nothing to calm it keeps swinging. Did you notice it swings around a point a bit "
            "below 90 degrees? We will come back to that.",
        },
        check=_p_only,
        unlock=("kp", "target"),
        show=("scale", "target", "error", "torque"),
    ),
    Step(
        id="d",
        title={"zh": "微分控制 D：加一个阻尼器", "en": "Derivative control: a damper"},
        body={
            "zh": "D 项看的是转速 ω：u = Kp × e − Kd × ω。\n"
            "转得越快，反方向的力越大，就像门上的阻尼器，把来回的晃动吃掉。\n"
            "慢慢加大 Kd，直到手臂稳稳停住。也可以点「推一下」，看它怎么恢复。",
            "en": "The D term looks at the turning speed omega: u = Kp * e - Kd * omega.\n"
            "The faster it turns, the harder it is held back, like a door damper soaking up the swinging.\n"
            "Raise Kd slowly until the arm comes to rest. Press Push to see how it recovers.",
        },
        done={
            "zh": "阻尼把摆动的能量消耗掉了。Kd 太小会晃，太大会变得迟钝。调参就是在两者之间找平衡。",
            "en": "The damper drains the swing's energy. Too little Kd and it wobbles, too much and it turns sluggish. "
            "Tuning is finding the balance.",
        },
        check=_damped,
        unlock=("kd", "kick"),
    ),
    Step(
        id="why_low",
        title={"zh": "为什么停在目标下面？", "en": "Why does it stop short?"},
        body={
            "zh": "看角度曲线：手臂停住了，但停在目标下方几度。这是为什么？",
            "en": "Look at the angle plot: the arm has stopped, but a few degrees below the target. Why?",
        },
        done={
            "zh": "这叫稳态误差。重力一直往下拉，需要电机持续出一个力矩；P 控制只有在有误差时才出力，"
            "所以手臂只能“留着一点误差”来产生这个力矩。Kp 越大误差越小，但永远不会是零。",
            "en": "This is steady-state error. Gravity pulls all the time, so the motor must keep pushing; "
            "P control only pushes when there is an error, so the arm has to keep a little error to hold itself up. "
            "A bigger Kp shrinks it but never makes it zero.",
        },
        quiz=Quiz(
            question={"zh": "选一个最好的解释", "en": "Pick the best explanation"},
            options=(
                {"zh": "电机的力气不够", "en": "The motor is too weak"},
                {
                    "zh": "重力需要一个持续的力矩，而 P 项只有在有误差时才出力",
                    "en": "Gravity needs a constant torque, and the P term only pushes when there is an error",
                },
                {"zh": "Kd 太大，把手臂拖住了", "en": "Kd is too big and holds the arm back"},
            ),
            answer=1,
            explain={
                "zh": "对：要抵消重力就得一直出力，P 控制只能靠误差来出力。",
                "en": "Right: holding against gravity takes a constant push, "
                "and P control can only push through an error.",
            },
        ),
        show=("gravity",),
    ),
    Step(
        id="ff_i",
        title={"zh": "消除误差：前馈或积分", "en": "Remove the error: feedforward or integral"},
        body={
            "zh": "两种办法，选一个试试：\n"
            "① 重力补偿（前馈）：既然知道重力有多大，就提前把它加进电机输出里。\n"
            "② 积分 I：把误差一点点累加起来，只要还有误差，力矩就继续变大，直到误差消失。\n"
            "打开其中一个，让误差变成 0。",
            "en": "Two ways; try one:\n"
            "1. Gravity compensation (feedforward): we know how hard gravity pulls, "
            "so add that to the motor output up front.\n"
            "2. Integral I: keep adding up the error; while any error is left "
            "the push keeps growing, until it is gone.\n"
            "Switch one on and bring the error to zero.",
        },
        done={
            "zh": "前馈靠模型，反应快但模型要准；积分不需要模型，但慢，还可能冲过头。"
            "真实机器人通常两个一起用，这就是 PID。",
            "en": "Feedforward relies on a model: fast, but the model must be right. "
            "The integral needs no model but is slow and can overshoot. Real robots usually use both: that is PID.",
        },
        check=_no_steady_error,
        unlock=("gravity", "ki"),
    ),
    Step(
        id="challenge",
        title={"zh": "挑战：一秒内稳稳到位", "en": "Challenge: on target within a second"},
        body={
            "zh": "点「开始测试」：手臂会从垂直向下出发，目标 120°。\n"
            f"两颗星：1.5 秒内稳定在 ±{SETTLE_BAND:g}° 以内，超调不超过 15%。"
            f"三颗星：1 秒内稳定在 ±{FINE_BAND:g}° 以内，超调不超过 5%（提示：先把稳态误差消掉）。\n"
            "Kp、Kd、Ki 和重力补偿都可以随便调。",
            "en": "Press Run test: the arm starts hanging straight down and aims for 120 degrees.\n"
            f"Two stars: within +-{SETTLE_BAND:g} degrees in 1.5 s, overshooting by at most 15%. "
            f"Three stars: within +-{FINE_BAND:g} degree in 1 s, overshooting by at most 5% "
            "(hint: get rid of the steady-state error first).\n"
            "Change Kp, Kd, Ki and gravity compensation as you like.",
        },
        done={
            "zh": "你刚刚亲手整定了一个 PID 控制器。机械臂、无人机、四足机器人的每一个关节里，跑的都是这一套。",
            "en": "You just tuned a PID controller by hand. "
            "Every joint of robot arms, drones and legged robots runs this loop.",
        },
        check=_challenge,
        unlock=("test",),
    ),
)


class PDLesson(Lesson):
    id = "pd_control"
    chapter = {"zh": "控制原理", "en": "Control"}
    title = {"zh": "让关节停在目标角度：PID 控制", "en": "Holding a joint on target: PID control"}
    summary = {
        "zh": "亲手调 Kp、Kd、Ki，看懂弹簧、阻尼、稳态误差和前馈。",
        "en": "Turn Kp, Kd and Ki yourself and see springs, damping, steady-state error and feedforward.",
    }
    minutes = 12
    controls = (
        Control("target", "slider", {"zh": "目标角度", "en": "Target angle"}, 0, 170, 1, 90, "°"),
        Control("kp", "slider", {"zh": "Kp 比例", "en": "Kp proportional"}, 0, 80, 0.5, 0),
        Control("kd", "slider", {"zh": "Kd 微分", "en": "Kd derivative"}, 0, 6, 0.05, 0),
        Control("ki", "slider", {"zh": "Ki 积分", "en": "Ki integral"}, 0, 60, 0.5, 0),
        Control("gravity", "toggle", {"zh": "重力补偿（前馈）", "en": "Gravity compensation (feedforward)"}),
        Control("kick", "button", {"zh": "推一下", "en": "Push"}),
        Control("test", "button", {"zh": "开始测试", "en": "Run test"}),
    )
    plots = (
        Plot(
            "angle",
            {"zh": "角度", "en": "Angle"},
            "°",
            (
                Series("target", {"zh": "目标", "en": "target"}, "amber", dashed=True),
                Series("q", {"zh": "实际", "en": "actual"}, "ink"),
            ),
        ),
        Plot(
            "torque",
            {"zh": "电机力矩", "en": "Motor torque"},
            "N·m",
            (
                Series("u", {"zh": "总输出", "en": "total"}, "ink"),
                Series("p", {"zh": "P 项", "en": "P"}, "blue"),
                Series("d", {"zh": "D 项", "en": "D"}, "violet"),
                Series("iff", {"zh": "I + 前馈", "en": "I + feedforward"}, "green"),
            ),
        ),
    )
    steps = STEPS
    bench = Bench(
        title={"zh": "实验记录：阶跃响应", "en": "Experiment record: step responses"},
        x={"zh": "测试开始后", "en": "since the step"},
        x_unit="s",
        y={"zh": "角度", "en": "angle"},
        y_unit="°",
        params=("kp", "kd", "ki", "gravity"),
        short=(
            {"zh": "Kp", "en": "Kp"},
            {"zh": "Kd", "en": "Kd"},
            {"zh": "Ki", "en": "Ki"},
            {"zh": "前馈", "en": "FF"},
        ),
        metrics=(
            Metric("overshoot_pct", {"zh": "超调", "en": "overshoot"}, "%"),
            Metric("settle_s", {"zh": "稳到 ±3°", "en": "settled to ±3°"}, "s"),
            Metric("fine_s", {"zh": "稳到 ±1°", "en": "settled to ±1°"}, "s"),
        ),
        reference=TEST_TARGET,
        sweeps=(
            Sweep("kp", (2, 5, 10, 15, 20, 30, 40, 60, 80), ("overshoot_pct", "settle_s")),
            Sweep("kd", (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6), ("overshoot_pct", "settle_s")),
        ),
    )

    def __init__(self) -> None:
        self.pd: JointPD | None = None
        self.test: dict[str, Any] | None = None
        self.model: mujoco.MjModel | None = None

    def build(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.modelname = "pd_lesson"
        spec.option.timestep = 0.002
        G = mujoco.mjtGeom
        w = spec.worldbody
        w.add_light(pos=[0.5, -1.0, 2.5], dir=[-0.2, 0.4, -1.0], diffuse=[0.8, 0.8, 0.8])
        floor = spec.add_material(name="rf_floor", rgba=[0.62, 0.66, 0.7, 1])
        w.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[3, 3, 0.05], material=floor.name)
        for name in ("rf_dark", "rf_metal", "rf_shell", "rf_rubber"):
            spec.add_material(name=name)
        # A weighted pedestal and a pillar up to the shoulder; the motor sits on top, facing the camera.
        w.add_geom(
            name="foot",
            type=G.mjGEOM_CYLINDER,
            size=[0.16, 0.02, 0],
            pos=[0, 0, 0.02],
            rgba=[0.16, 0.18, 0.21, 1],
            material="rf_dark",
        )
        w.add_geom(
            name="pillar",
            type=G.mjGEOM_BOX,
            size=[0.035, 0.035, PIVOT[2] / 2],
            pos=[0, 0.05, PIVOT[2] / 2],
            rgba=[0.78, 0.8, 0.83, 1],
            material="rf_metal",
        )
        w.add_geom(
            name="motor",
            type=G.mjGEOM_CYLINDER,
            size=[0.05, 0.035, 0],
            pos=[0, 0.0, PIVOT[2]],
            quat=[0.7071068, 0.7071068, 0, 0],
            rgba=[0.16, 0.18, 0.21, 1],
            material="rf_dark",
        )
        arm = w.add_body(name="arm", pos=list(PIVOT))  # swings in free air: its parts touch nothing
        # Axis -y: positive angles swing the arm toward +x, to the right as the camera sees it.
        arm.add_joint(
            name="shoulder",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, -1, 0],
            damping=0.004,
            frictionloss=0.002,
            armature=0.002,
        )
        arm.add_geom(
            name="hub",
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
            name="link",
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
            name="weight",
            contype=0,
            conaffinity=0,
            type=G.mjGEOM_SPHERE,
            size=[0.045, 0, 0],
            pos=[0, -0.06, -LENGTH],
            rgba=[0.95, 0.66, 0.0, 1],
            material="rf_shell",
            mass=TIP_MASS,
        )
        arm.add_site(name="tip", pos=[0, -0.06, -LENGTH], size=[0.01, 0, 0])
        motor = spec.add_actuator(name="motor", target="shoulder", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        motor.ctrllimited = True
        motor.ctrlrange = [-TORQUE_MAX, TORQUE_MAX]
        motor.gear = [1, 0, 0, 0, 0, 0]
        self.model = spec.compile()
        return self.model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.pd = JointPD(model, ctx)
        return {"lesson": self.pd}

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        assert self.pd is not None
        if control == "kick":
            data.qvel[self.pd.dadr] += 5.0
        elif control == "test":
            data.qpos[self.pd.qadr] = 0.0
            data.qvel[self.pd.dadr] = 0.0
            self.pd.restart_integral()
            ctx.values["target"] = TEST_TARGET
            self.test = {"t0": data.time, "times": [], "angles": []}
            ctx.results["test"] = {"running": True}
        elif control in ("ki", "gravity"):
            self.pd.restart_integral()

    def drain_samples(self) -> list[dict[str, float]]:
        if self.pd is None:
            return []
        out = list(self.pd.samples)
        self.pd.samples.clear()
        self._grade(out)
        return out

    def _grade(self, samples: list[dict[str, float]]) -> None:
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
        result = grade(test["times"], test["angles"])
        ctx = self.pd.ctx
        ctx.results["test"] = {"running": False, **result}
        params = {k: ctx.values[k] for k in ("kp", "kd", "ki", "gravity")}
        ctx.record_run(params, result, test["times"], test["angles"])

    def run_sweep(self, sweep: Sweep, ctx: LessonContext) -> list[dict[str, Any]]:
        """The step test once per value, on a fresh copy of the rig: the live joint keeps running."""
        assert self.model is not None
        model = self.model
        points = []
        for value in sweep.values:
            values = {**ctx.values, sweep.control: float(value), "target": TEST_TARGET}
            pd = JointPD(model, LessonContext(values=values))
            data = mujoco.MjData(model)
            times, angles = [], []
            while data.time <= TEST_SECONDS:
                pd.update(model, data)
                mujoco.mj_step(model, data)
            for sample in pd.samples:
                times.append(sample["t"])
                angles.append(sample["q"])
            points.append({"x": float(value), **grade(times, angles)})
        return points

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        assert self.pd is not None
        q = float(data.qpos[self.pd.qadr])
        return {
            "angle": round(_deg(q), 2),
            "error": round(ctx.values["target"] - _deg(q), 2),
            "torque": round(self.pd.terms["u"], 3),
            "torque_max": TORQUE_MAX,
            "gravity_torque": round(float(data.qfrc_bias[self.pd.dadr]), 3),
        }

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
        return {"eye": [0.3, -2.3, 0.95], "target": [0.12, 0.0, 0.6]}
