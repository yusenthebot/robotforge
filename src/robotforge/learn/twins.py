"""Twins: copies of one robot, each with its own settings, put through the same test side by side.

Changing a knob and waiting for the next run hides what changed. With twins the difference plays out at
the same moment, in the same plot, from the same push: the beginner sees cause and effect at once.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

from robotforge.control.methods import make_controller
from robotforge.control.tasks import make_task
from robotforge.design.world import World, compile_world
from robotforge.learn import pd
from robotforge.learn.engine import Control, Lesson, LessonContext, Plot, Quiz, Series, Step, Text

SAMPLE_DT = 0.01  # s between plot samples


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


@dataclass(frozen=True)
class Twin:
    name: str  # prefix of its robot ("a" -> "a/...") and suffix of its knobs ("kp_a")
    label: Text
    color: str  # plot colour key, also used for its heading and name tag
    values: dict[str, float] = field(default_factory=dict)  # its knob settings to start with


@dataclass(frozen=True)
class Knob:
    """A setting every twin has its own copy of."""

    id: str
    kind: Literal["slider", "choice"]
    label: Text
    min: float = 0.0
    max: float = 1.0
    step: float = 0.1
    unit: str = ""
    options: tuple[Text, ...] = ()


class TwinValues(Mapping[str, float]):
    """One twin's view of the lesson's settings: its own knobs under their plain names, shared ones as is."""

    def __init__(self, values: dict[str, float], twin: str, fixed: Mapping[str, float] | None = None) -> None:
        self.values = values
        self.twin = twin
        self.fixed = fixed or {}  # settings the bench does not offer, at the value they stay at

    def __getitem__(self, key: str) -> float:
        own = f"{key}_{self.twin}"
        if own in self.values:
            return self.values[own]
        return self.values[key] if key in self.values else self.fixed[key]

    def __iter__(self) -> Iterator[str]:
        suffix = f"_{self.twin}"
        return iter({k.removesuffix(suffix) for k in self.values} | set(self.fixed))

    def __len__(self) -> int:
        return sum(1 for _ in self)


class Watch:
    """A controller that only observes, every physics step: the lesson's measure() (peaks are brief: a
    controller's answer to a push can be over in a few milliseconds) and, every SAMPLE_DT, its sample()."""

    def __init__(self, lesson: TwinsLesson, model: mujoco.MjModel) -> None:
        self.lesson = lesson
        self.rate_hz = 1.0 / model.opt.timestep
        self.next_sample = -1.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.lesson.measure(model, data)
        if data.time >= self.next_sample:
            self.next_sample = float(data.time) + SAMPLE_DT
            self.lesson.samples.append({"t": float(data.time), **self.lesson.sample(model, data)})

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-watch"}


class TwinsLesson(Lesson):
    kind = "bench"
    chapter = _text("调参台", "Tuning bench")
    twins: tuple[Twin, ...] = ()
    knobs: tuple[Knob, ...] = ()
    shared: tuple[Control, ...] = ()  # settings and buttons all twins share: the test, the target
    fixed: dict[str, float] = {}  # settings the controllers read that this bench keeps out of reach

    def __init__(self) -> None:
        self.playground = None
        self.samples: deque[dict[str, float]] = deque(maxlen=4096)
        self.ctx: LessonContext | None = None

    @property
    def controls(self) -> tuple[Control, ...]:  # type: ignore[override]
        own = tuple(
            Control(
                f"{k.id}_{t.name}",
                k.kind,
                k.label,
                k.min,
                k.max,
                k.step,
                float(t.values[k.id]),
                k.unit,
                k.options,
                group=t.name,
            )
            for t in self.twins
            for k in self.knobs
        )
        return self.shared + own

    @property
    def groups(self) -> dict[str, dict[str, Any]]:  # type: ignore[override]
        return {t.name: {"label": t.label, "color": t.color} for t in self.twins}

    def view(self, twin: Twin) -> TwinValues:
        assert self.ctx is not None
        return TwinValues(self.ctx.values, twin.name, self.fixed)

    def knob_ids(self, twin: Twin) -> list[str]:
        return [f"{k.id}_{twin.name}" for k in self.knobs]

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        return {}

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called every physics step while the lesson runs, e.g. to track the peaks of a test."""

    def drain_samples(self) -> list[dict[str, float]]:
        with self.playground.session.lock:  # the Watch appends from the physics thread
            out = list(self.samples)
            self.samples.clear()
        return out

    def columns(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "label": t.label, "color": t.color, "real": False, "sub": None} for t in self.twins]


def _row(label: Text, cells: list[Any], values: list[float | None] | None = None, high: bool = False) -> dict:
    """A table row; `values` marks the best twin (lowest, or highest with high=True) unless they tie."""
    best = None
    if values is not None:
        known = [v for v in values if v is not None]
        if len(known) == len(values) and len(set(known)) > 1:
            target = max(known) if high else min(known)
            best = values.index(target)
    return {"label": label, "cells": cells, "best": best}


def _num(value: float | None, unit: str = "", digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}{(' ' + unit) if unit else ''}"


# ---- PID triplets: three joints, the same step --------------------------------------------------------

PID_TWINS = (
    Twin("a", _text("A", "A"), "amber", {"kp": 10.0, "kd": 0.3}),
    Twin("b", _text("B", "B"), "blue", {"kp": 40.0, "kd": 0.5}),
    Twin("c", _text("C", "C"), "violet", {"kp": 40.0, "kd": 3.0}),
)
WEIGHT_RGBA = {"amber": [0.95, 0.66, 0.0, 1], "blue": [0.23, 0.44, 0.85, 1], "violet": [0.54, 0.36, 0.94, 1]}
SPACING = 0.55  # m between the rigs


def _test_done(ctx: LessonContext) -> bool:
    return "triplet" in ctx.results


def _a_tuned(ctx: LessonContext) -> bool:
    return ctx.results.get("triplet", {}).get("a", {}).get("stars", 0) >= 2


def _feedforward_tested(ctx: LessonContext) -> bool:
    test = ctx.results.get("triplet", {})
    return bool(test.get("gravity")) and all(test.get(t.name, {}).get("settle_s") is not None for t in PID_TWINS)


class PIDTwins(TwinsLesson):
    id = "pid_twins"
    title = _text("PID 三胞胎：同一个阶跃，三组参数", "PID triplets: one step, three settings")
    summary = _text(
        "三个一模一样的关节，各用一组 Kp、Kd，同时跳到同一个角度。谁快、谁冲过头、谁晃，一眼就看出来。",
        "Three identical joints, each with its own Kp and Kd, jump to the same angle at once. Who is fast, who "
        "overshoots and who wobbles is plain to see.",
    )
    minutes = 6
    twins = PID_TWINS
    fixed = {"ki": 0.0}  # PD only here: the integral term is the PID lesson's subject
    knobs = (
        Knob("kp", "slider", _text("Kp 比例", "Kp proportional"), 0, 80, 0.5),
        Knob("kd", "slider", _text("Kd 微分", "Kd derivative"), 0, 6, 0.05),
    )
    shared = (
        Control("test", "button", _text("三个一起测试", "Test all three")),
        Control("target", "slider", _text("目标角度（三个一起）", "Target angle (all three)"), 0, 170, 1, 90, "°"),
        Control("gravity", "toggle", _text("重力补偿（前馈，三个一起）", "Gravity compensation (feedforward, all)")),
    )
    plots = (
        Plot(
            "angle",
            _text("角度", "Angle"),
            "°",
            (
                Series("target", _text("目标", "target"), "muted", dashed=True),
                *(Series(f"q_{t.name}", t.label, t.color) for t in PID_TWINS),
            ),
        ),
        Plot(
            "torque",
            _text("电机力矩", "Motor torque"),
            "N·m",
            tuple(Series(f"u_{t.name}", t.label, t.color) for t in PID_TWINS),
        ),
    )
    steps = (
        Step(
            id="compare",
            title=_text("同一个测试，三组参数", "One test, three settings"),
            body=_text(
                "三个关节完全一样，只有参数不同：A 软（Kp 10，Kd 0.3），B 硬但阻尼小（Kp 40，Kd 0.5），"
                "C 硬而且阻尼足（Kp 40，Kd 3）。\n先猜：谁最先到 120°？谁会冲过头？然后点「三个一起测试」。",
                "The three joints are identical except for their settings: A is soft (Kp 10, Kd 0.3), B is stiff with "
                "little damping (Kp 40, Kd 0.5), C is stiff and well damped (Kp 40, Kd 3).\nGuess first: who reaches "
                "120° first? Who overshoots? Then press Test all three.",
            ),
            done=_text(
                "Kd 像刹车：A 又软又几乎没刹车，晃过头、被重力拖着停不准；B 够硬但刹车太弱，冲得更远、晃更久；"
                "C 一样硬，刹车够，快到时提前减速，又快又稳。Kp 决定冲得多猛，Kd 决定能不能停住。",
                "Kd acts like a brake: A is soft with almost no brake, so it swings past and gravity keeps it off "
                "target; B is stiff but its brake is weak, so it flies further and swings longer; C is just as stiff "
                "with enough brake, slowing before it arrives: fast and steady. Kp sets how hard it charges, Kd "
                "whether it can stop.",
            ),
            check=_test_done,
            unlock=("test", "target", "gravity"),
            show=("labels",),
        ),
        Step(
            id="tune_a",
            title=_text("把 A 调好", "Tune A"),
            body=_text(
                "现在只调 A 的 Kp 和 Kd，让它拿到两颗星：1.5 秒内稳到 ±3° 以内，超调不超过 15%。"
                "B 和 C 留着当参照。每次改完点「三个一起测试」。",
                "Now tune only A's Kp and Kd until it earns two stars: inside +-3° within 1.5 s and no more than 15% "
                "overshoot. Leave B and C as they are, for reference. Press Test all three after each change.",
            ),
            done=_text(
                "调 PID 的老办法：先加 Kp 直到够快，晃起来了就加 Kd 压住。参照物就在旁边，改了什么一目了然。",
                "The classic way to tune PID: raise Kp until it is fast enough, then add Kd until the swinging stops. "
                "With the others right next to it, you see exactly what each change did.",
            ),
            check=_a_tuned,
            unlock=("kp_a", "kd_a"),
            show=("labels",),
        ),
        Step(
            id="feedforward",
            title=_text("三个一起加上前馈", "Feedforward for all three"),
            body=_text(
                "打开「重力补偿」再测一次：控制器先算出重力要多少力矩并直接加上，三个关节会怎样？",
                "Turn on Gravity compensation and test again: the controller works out what gravity needs and adds "
                "it up front. What happens to all three?",
            ),
            done=_text(
                "连最软的 A 都能停准了：反馈（PID）负责纠正误差，前馈负责提前算好的部分，两者配合最好。",
                "Even soft A now stops on target: feedback (PID) corrects errors, feedforward supplies what can be "
                "worked out in advance, and they work best together.",
            ),
            check=_feedforward_tested,
            unlock=("kp_b", "kd_b", "kp_c", "kd_c"),
            show=("labels",),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.model: mujoco.MjModel | None = None
        self.pds: dict[str, pd.JointPD] = {}
        self.test: dict[str, Any] | None = None

    def build(self) -> mujoco.MjModel:
        spec = pd.rig_spec("pid_twins")
        for i, twin in enumerate(self.twins):
            pd.add_rig(spec, f"{twin.name}/", (i - 1) * SPACING, WEIGHT_RGBA[twin.color])
        self.model = spec.compile()
        return self.model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.test = None
        self.pds = {t.name: pd.JointPD(model, ctx, f"{t.name}/", self.view(t)) for t in self.twins}
        return {**{f"pd_{name}": ctrl for name, ctrl in self.pds.items()}, "watch": Watch(self, model)}

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        assert self.ctx is not None
        out = {"target": float(self.ctx.values["target"])}
        for name, ctrl in self.pds.items():
            out[f"q_{name}"] = math.degrees(float(data.qpos[ctrl.qadr]))
            out[f"u_{name}"] = ctrl.terms["u"]
        return out

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control == "test":
            for ctrl in self.pds.values():
                data.qpos[ctrl.qadr] = 0.0
                data.qvel[ctrl.dadr] = 0.0
                ctrl.restart_integral()
            ctx.values["target"] = pd.TEST_TARGET
            self.test = {"t0": float(data.time), "curves": {t.name: ([], []) for t in self.twins}}
        elif control == "gravity" or control.startswith(("kp_", "kd_")):
            for ctrl in self.pds.values():
                ctrl.restart_integral()

    def drain_samples(self) -> list[dict[str, float]]:
        out = super().drain_samples()
        test = self.test
        if test is None:
            return out
        for s in out:
            t = s["t"] - test["t0"]
            if t < 0:
                continue
            if t > pd.TEST_SECONDS:
                self._finish()
                break
            for name, (times, angles) in test["curves"].items():
                times.append(t)
                angles.append(s[f"q_{name}"])
        return out

    def _finish(self) -> None:
        assert self.test is not None and self.ctx is not None
        test, self.test = self.test, None
        ctx = self.ctx
        result: dict[str, Any] = {"gravity": bool(ctx.values["gravity"])}
        for twin in self.twins:
            graded = pd.grade(*test["curves"][twin.name])
            view = self.view(twin)
            result[twin.name] = {**graded, "kp": view["kp"], "kd": view["kd"]}
        ctx.results["triplet"] = result

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        return {"table": self._table(ctx), "readout": _text("正在测试……", "Testing...") if self.test else _text("", "")}

    def _table(self, ctx: LessonContext) -> dict[str, Any]:
        rows: list[dict[str, Any]] = [
            {"heading": _text("现在的设置", "Settings now")},
            _row(_text("Kp", "Kp"), [_num(self.view(t)["kp"]) for t in self.twins]),
            _row(_text("Kd", "Kd"), [_num(self.view(t)["kd"], digits=2) for t in self.twins]),
        ]
        test = ctx.results.get("triplet")
        if test:
            got = [test[t.name] for t in self.twins]
            ff = _text("（开了前馈）", " (feedforward on)") if test["gravity"] else _text("", "")
            rows += [
                {"heading": {k: f"{v}{ff[k]}" for k, v in _text("上次测试", "Last test").items()}},
                _row(
                    _text("超调", "Overshoot"),
                    [_num(r["overshoot_pct"], "%") for r in got],
                    [r["overshoot_pct"] for r in got],
                ),
                _row(
                    _text("稳到 ±3°", "Settled to ±3°"),
                    [_num(r["settle_s"], "s", 2) for r in got],
                    [r["settle_s"] for r in got],
                ),
                _row(
                    _text("稳到 ±1°", "Settled to ±1°"),
                    [_num(r["fine_s"], "s", 2) for r in got],
                    [r["fine_s"] for r in got],
                ),
                _row(_text("星", "Stars"), ["★" * r["stars"] + "☆" * (3 - r["stars"]) for r in got]),
            ]
        return {"columns": self.columns(), "rows": rows}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "faceoff",
            "entrants": [
                {
                    **c,
                    "base": [(i - 1) * SPACING, 0.0, 0.0],
                    "body": int(model.body(f"{c['name']}/arm").id),
                    "top": pd.PIVOT[2] + 0.1,
                    "cloud": [],
                }
                for i, c in enumerate(self.columns())
            ],
            "targets": [],
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [0.0, -3.0, 1.0], "target": [0.0, 0.0, 0.5]}


# ---- LQR twins: two cart-poles, the same push ---------------------------------------------------------

Q_ANGLE = (1.0, 10.0, 50.0, 200.0, 1000.0)  # how much a lean costs
R_EFFORT = (0.02, 0.2, 2.0, 20.0)  # how much pushing costs
PUSH = 0.6  # rad/s kick to each pole
WATCH_SECONDS = 4.0
SETTLED_DEG = 0.5

CART_COLORS = {"amber": "orange", "blue": "blue", "violet": "purple"}
LQR_TWINS = (
    Twin("a", _text("A", "A"), "amber", {"q": 2, "r": 1}),
    Twin("b", _text("B", "B"), "blue", {"q": 2, "r": 3}),
)


def _pushed(ctx: LessonContext) -> bool:
    return "push" in ctx.results


def _compared_q(ctx: LessonContext) -> bool:
    push = ctx.results.get("push", {})
    return len({push.get(t.name, {}).get("q") for t in LQR_TWINS}) > 1


class LQRTwins(TwinsLesson):
    id = "lqr_twins"
    title = _text("LQR 双胞胎：Q 和 R 怎么选", "LQR twins: choosing Q and R")
    summary = _text(
        "两台倒立摆用 LQR 平衡，各有一组 Q（多在乎误差）和 R（多舍不得用力）。同时推一下，看误差和力气怎么交换。",
        "Two cart-poles balance with LQR, each with its own Q (how much errors matter) and R (how costly "
        "pushing is). Push both at once and watch error trade against effort.",
    )
    minutes = 6
    twins = LQR_TWINS
    knobs = (
        Knob(
            "q",
            "choice",
            _text("角度权重 Q", "Angle weight Q"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in Q_ANGLE),
        ),
        Knob(
            "r",
            "choice",
            _text("力气成本 R", "Effort cost R"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in R_EFFORT),
        ),
    )
    shared = (Control("push", "button", _text("两个一起推", "Push both")),)
    plots = (
        Plot(
            "lean",
            _text("杆的倾角", "Pole lean"),
            "°",
            tuple(Series(f"lean_{t.name}", t.label, t.color) for t in LQR_TWINS),
        ),
        Plot(
            "effort",
            _text("电机出力（满力 = 100%）", "Motor effort (full = 100%)"),
            "%",
            tuple(Series(f"effort_{t.name}", t.label, t.color) for t in LQR_TWINS),
        ),
    )
    steps = (
        Step(
            id="trade",
            title=_text("误差和力气的交换", "Trading error for effort"),
            body=_text(
                "LQR 不是让你直接调增益，而是告诉它“什么最重要”：Q 是误差的代价，R 是用力的代价，"
                "它自己算出最划算的增益。\nA 的 R 是 0.2（用力很便宜），B 的 R 是 20（用力很贵），Q 一样。"
                "先猜：被推一下，谁晃得多？谁用力多？然后点「两个一起推」。",
                "With LQR you do not set gains directly; you say what matters: Q is the price of an error, R the price "
                "of pushing, and it works out the gains that cost least.\nA has R = 0.2 (pushing is cheap), B has "
                "R = 20 (pushing is expensive), the same Q. Guess first: after a push, who leans more? Who pushes "
                "harder? Then press Push both.",
            ),
            done=_text(
                "B 用的力不到 A 的一半，代价是晃得稍多、回正稍慢。R 大，控制器就“省着用力”：动作温和，"
                "电机和电池都轻松，但误差大一点。真实机器人常这样取舍。",
                "B uses less than half of A's effort, at the price of a little more lean and a slower recovery. A big "
                "R makes the controller frugal: gentler moves, easier on motors and batteries, a bit more error. Real "
                "robots make this trade all the time.",
            ),
            check=_pushed,
            unlock=("push",),
            show=("labels",),
        ),
        Step(
            id="weights",
            title=_text("再调 Q", "Now change Q"),
            body=_text(
                "把 B 的 R 调回和 A 一样，再把 B 的 Q 调到 1000（很在乎倾角），A 的 Q 调到 1，然后再推。",
                "Set B's R back to A's, then set B's Q to 1000 (it cares a lot about the lean) and A's Q to 1, "
                "and push again.",
            ),
            done=_text(
                "Q 大的那台倾角更小：它拼命把杆扶正，代价是小车跑得更远（我们只给倾角加了权，没给位置加）。"
                "Q 和 R 的比例决定了控制器的“性格”。",
                "The one with the big Q leans less: it fights hard to keep the pole upright, at the price of the cart "
                "running further (only the lean got more weight, not the cart's position). The ratio of Q to R sets "
                "the controller's character.",
            ),
            check=_compared_q,
            unlock=("q_a", "r_a", "q_b", "r_b"),
            show=("labels",),
        ),
        Step(
            id="quiz",
            title=_text("想一想", "Think about it"),
            body=_text("一台用电池的机器人，电机又容易发热。", "A battery-powered robot whose motors run hot."),
            done=_text("", ""),
            quiz=Quiz(
                question=_text("它的 LQR 应该怎么设？", "How should its LQR be set?"),
                options=(
                    _text("R 调小，让它更用力", "Make R small so it pushes harder"),
                    _text("R 调大，用力更省，接受误差稍大", "Make R large: frugal effort, a bit more error accepted"),
                    _text("Q 和 R 都调成 0", "Set both Q and R to 0"),
                ),
                answer=1,
                explain=_text(
                    "对。R 大，控制器就把“用力”看得很贵，出力更小更平滑，省电也少发热。"
                    "Q 和 R 不能是 0：没有代价，就没有“最划算”可言。",
                    "Right. A large R makes effort expensive, so the output is smaller and smoother: less battery, "
                    "less heat. Q and R cannot be 0: with no costs there is nothing to optimize.",
                ),
            ),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.model: mujoco.MjModel | None = None
        self.watching: dict[str, Any] | None = None

    def build(self) -> mujoco.MjModel:
        robots = [
            {
                "name": t.name,
                "template": "cartpole",
                "params": {"color": CART_COLORS[t.color]},  # the cart wears its plot colour
                "pos": [0.0, (i - (len(self.twins) - 1) / 2) * 0.9],
            }
            for i, t in enumerate(self.twins)
        ]
        self.model = compile_world(World.model_validate({"name": "lqr_twins", "robots": robots})).model
        return self.model

    def _controller(self, model: mujoco.MjModel, data: mujoco.MjData, twin: Twin) -> Any:
        view = self.view(twin)
        params = {"q_weights": [5.0, Q_ANGLE[int(view["q"])], 0.5, 1.0], "r_weight": R_EFFORT[int(view["r"])]}
        return make_controller("lqr", make_task("balance", model, twin.name), model, data, params)

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.watching = None
        return {**{t.name: self._controller(model, data, t) for t in self.twins}, "watch": Watch(self, model)}

    def _hinge(self, model: mujoco.MjModel, twin: Twin) -> Any:
        return model.joint(f"{twin.name}/hinge")

    def _actuator(self, model: mujoco.MjModel, twin: Twin) -> int:
        return next(a for a in range(model.nu) if (model.actuator(a).name or "").startswith(f"{twin.name}/"))

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        out = {}
        for twin in self.twins:
            out[f"lean_{twin.name}"] = math.degrees(float(data.qpos[self._hinge(model, twin).qposadr[0]]))
            a = self._actuator(model, twin)
            out[f"effort_{twin.name}"] = 100 * float(data.ctrl[a]) / float(model.actuator_ctrlrange[a][1])
        return out

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        session = self.playground.session
        model = session.model
        twin = next((t for t in self.twins if control in self.knob_ids(t)), None)
        if twin is not None:  # new weights: LQR works out new gains for that twin
            session.set_controller(twin.name, self._controller(model, data, twin))
        elif control == "push" and self.watching is None:
            for t in self.twins:
                data.qvel[self._hinge(model, t).dofadr[0]] += PUSH
            self.watching = {
                "t0": float(data.time),
                "settings": {t.name: {k: self.view(t)[k] for k in ("q", "r")} for t in self.twins},
                **{t.name: {"lean": 0.0, "effort": 0.0, "travel": 0.0, "settle": None} for t in self.twins},
            }

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        w = self.watching
        if w is None:
            return
        t = float(data.time) - w["t0"]
        for twin in self.twins:
            lean = abs(math.degrees(float(data.qpos[self._hinge(model, twin).qposadr[0]])))
            a = self._actuator(model, twin)
            cart = model.jnt_qposadr[model.body_jntadr[model.body(f"{twin.name}/cart").id]]
            m = w[twin.name]
            m["lean"] = max(m["lean"], lean)
            m["effort"] = max(m["effort"], abs(float(data.ctrl[a])) / float(model.actuator_ctrlrange[a][1]))
            m["travel"] = max(m["travel"], abs(float(data.qpos[cart])))
            m["settle"] = None if lean > SETTLED_DEG else (m["settle"] if m["settle"] is not None else t)

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        w = self.watching
        if w is not None:
            t = float(data.time) - w["t0"]
            if t >= WATCH_SECONDS:
                ctx.results["push"] = {
                    twin.name: {
                        **w["settings"][twin.name],
                        "lean": round(w[twin.name]["lean"], 2),
                        "effort": round(100 * w[twin.name]["effort"]),
                        "travel": round(100 * w[twin.name]["travel"], 1),
                        "settle": None if w[twin.name]["settle"] is None else round(w[twin.name]["settle"], 2),
                    }
                    for twin in self.twins
                }
                self.watching = None
        return {
            "table": self._table(model, ctx),
            "readout": _text("推了，看着……", "Pushed, watching...") if w else _text("", ""),
        }

    def _table(self, model: mujoco.MjModel, ctx: LessonContext) -> dict[str, Any]:
        controllers = [self.playground.session.controllers.get(t.name) for t in self.twins]
        gains = [np.round(getattr(c, "K", np.zeros((1, 4)))[0], 1) for c in controllers]
        rows: list[dict[str, Any]] = [
            {"heading": _text("现在的设置", "Settings now")},
            _row(_text("Q（角度）", "Q (angle)"), [f"{Q_ANGLE[int(self.view(t)['q'])]:g}" for t in self.twins]),
            _row(_text("R（用力）", "R (effort)"), [f"{R_EFFORT[int(self.view(t)['r'])]:g}" for t in self.twins]),
            _row(_text("算出的角度增益", "Angle gain it worked out"), [f"{abs(g[1]):.1f}" for g in gains]),
        ]
        push = ctx.results.get("push")
        if push:
            got = [push[t.name] for t in self.twins]
            rows += [
                {"heading": _text("上次推完", "After the last push")},
                _row(
                    _text("最大倾角", "Largest lean"), [_num(r["lean"], "°", 2) for r in got], [r["lean"] for r in got]
                ),
                _row(_text("最大出力", "Peak effort"), [f"{r['effort']}%" for r in got], [r["effort"] for r in got]),
                _row(
                    _text("回正（±0.5°）", "Back upright (±0.5°)"),
                    [_num(r["settle"], "s", 2) for r in got],
                    [r["settle"] for r in got],
                ),
                _row(
                    _text("小车跑了", "Cart travelled"),
                    [_num(r["travel"], "cm") for r in got],
                    [r["travel"] for r in got],
                ),
            ]
        return {"columns": self.columns(), "rows": rows}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "faceoff",
            "entrants": [
                {
                    **c,
                    "base": [0.0, (i - (len(self.twins) - 1) / 2) * 0.9, 0.0],
                    "body": int(model.body(f"{c['name']}/cart").id),
                    "top": 1.0,  # just above the pole's tip (0.9 m)
                    "cloud": [],
                }
                for i, c in enumerate(self.columns())
            ],
            "targets": [],
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [1.6, -2.6, 1.5], "target": [0.0, 0.0, 0.45]}
