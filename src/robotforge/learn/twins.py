"""Twins: copies of one robot, each with its own settings, put through the same test side by side.

Changing a knob and waiting for the next run hides what changed. With twins the difference plays out at
the same moment, in the same plot, from the same push: the beginner sees cause and effect at once.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

from robotforge.behaviors import quadruped
from robotforge.control.methods import make_controller
from robotforge.control.tasks import make_task
from robotforge.design.world import World, compile_world
from robotforge.learn import pd
from robotforge.learn.engine import Control, Lesson, LessonContext, Plot, Quiz, Series, Step, Text

SAMPLE_DT = 0.01  # s between plot samples
GRAVITY = 9.81


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


# ---- dogs: shared helpers ----------------------------------------------------------------------------

DOG_LANES = 1.2  # m between the two dogs
TRUNK_DOWN = 0.5  # cosine of the trunk's tilt below which a dog is down


def _dog_world(name: str, twins: tuple[Twin, ...], start_x: float, terrain: list[dict] | None = None) -> World:
    robots = [
        {"name": t.name, "template": "quadruped", "params": {"color": CART_COLORS[t.color]}, "pos": [start_x, y]}
        for t, y in zip(twins, _lanes(len(twins)), strict=True)
    ]
    return World.model_validate(
        {"name": name, "floor": {"style": "concrete"}, "terrain": terrain or [], "robots": robots}
    )


def _lanes(n: int) -> list[float]:
    return [(i - (n - 1) / 2) * DOG_LANES for i in range(n)]


class DogTwins(TwinsLesson):
    """Twin dogs that can be put back on their start line, with controllers rebuilt from their settings."""

    start_x = 0.0

    def __init__(self) -> None:
        super().__init__()
        self.model: mujoco.MjModel | None = None
        self.start: np.ndarray | None = None
        self.trunks: dict[str, int] = {}

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.start = data.qpos.copy()
        self.trunks = {t.name: int(model.body(f"{t.name}/trunk").id) for t in self.twins}
        return {**{t.name: self.controller(model, data, t) for t in self.twins}, "watch": Watch(self, model)}

    def controller(self, model: mujoco.MjModel, data: mujoco.MjData, twin: Twin) -> Any:
        raise NotImplementedError

    def restart(self, data: mujoco.MjData) -> None:
        """Every dog back on its start line, still, with a fresh controller: each test starts the same."""
        assert self.start is not None
        session = self.playground.session
        model = session.model
        data.qpos[:] = self.start
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for twin in self.twins:
            session.set_controller(twin.name, self.controller(model, data, twin))

    def tilt(self, data: mujoco.MjData, twin: Twin) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[self.trunks[twin.name]][8])))))

    def down(self, data: mujoco.MjData, twin: Twin) -> bool:
        return float(data.xmat[self.trunks[twin.name]][8]) < TRUNK_DOWN

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "faceoff",
            "entrants": [
                {
                    **c,
                    "base": [self.start_x, y, 0.0],
                    "body": int(model.body(f"{c['name']}/trunk").id),
                    "top": 0.55,
                    "cloud": [],
                }
                for c, y in zip(self.columns(), _lanes(len(self.twins)), strict=True)
            ],
            "targets": [],
        }


# ---- MPC twins: two dogs standing, the same shoves ----------------------------------------------------

HORIZONS = (0.1, 0.2, 0.35, 0.6)  # s of future each plan looks at
SAMPLES = (4, 16, 32, 64)  # random futures tried per plan
NOISES = (0.02, 0.05, 0.1, 0.3)  # how bold those tries are, as a fraction of each motor's range
SHOVE_EVERY = 2.5  # s between shoves
SHOVE_STEP = 0.2  # m/s of sideways kick added per level
SHOVE_LEVELS = 10

MPC_TWINS = (
    Twin("a", _text("A", "A"), "amber", {"horizon": 2, "samples": 2, "noise": 1}),
    Twin("b", _text("B", "B"), "blue", {"horizon": 2, "samples": 2, "noise": 3}),
)


def _shoved(ctx: LessonContext) -> bool:
    return "shoves" in ctx.results


def _compared_horizons(ctx: LessonContext) -> bool:
    shoves = ctx.results.get("shoves", {})
    return len({shoves.get(t.name, {}).get("horizon") for t in MPC_TWINS}) > 1


class MPCTwins(DogTwins):
    id = "mpc_twins"
    title = _text("MPC 双胞胎：预测多远、试得多大胆", "MPC twins: how far ahead, how bold")
    summary = _text(
        "两只机器狗都用 MPC 站着，各有一组预测时长、样本数和探索噪声。推得一次比一次狠，看哪组设置扛得住。",
        "Two robot dogs stand with MPC, each with its own horizon, sample count and exploration noise. The "
        "shoves get harder each time: see which settings hold.",
    )
    minutes = 7
    twins = MPC_TWINS
    knobs = (
        Knob(
            "horizon",
            "choice",
            _text("预测时长（秒）", "Horizon (s)"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in HORIZONS),
        ),
        Knob(
            "samples",
            "choice",
            _text("每次试几种未来", "Futures tried per plan"),
            options=tuple(_text(str(v), str(v)) for v in SAMPLES),
        ),
        Knob(
            "noise",
            "choice",
            _text("探索噪声", "Exploration noise"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in NOISES),
        ),
    )
    shared = (Control("shove", "button", _text("开始推（两只一起）", "Start shoving (both)")),)
    plots = (
        Plot(
            "tilt",
            _text("机身倾斜", "Body tilt"),
            "°",
            tuple(Series(f"tilt_{t.name}", t.label, t.color) for t in MPC_TWINS),
        ),
    )
    steps = (
        Step(
            id="bold",
            title=_text("MPC 在想什么", "What MPC is thinking"),
            body=_text(
                "MPC（模型预测控制）每 20 毫秒做一次计划：在脑子里把未来 0.35 秒模拟几十遍，每遍随机改一点电机指令，"
                "挑一个最不容易摔的去执行。\n"
                "A 和 B 只差一个设置：探索噪声，也就是每次随机改多大。A 是 0.05（只敢微调），"
                "B 是 0.3（敢大改）。先猜谁扛得多，再点「开始推」。",
                "Every 20 ms, MPC (model predictive control) makes a plan: it simulates the next 0.35 s dozens of "
                "times "
                "in its head, each time changing the motor commands at random, and carries out the one least likely "
                "to fall.\nA and B differ in one setting only, the exploration noise: how big those random changes "
                "are. A uses 0.05 (small tweaks), B uses 0.3 (bold changes). Guess who lasts longer, then press "
                "Start shoving.",
            ),
            done=_text(
                "B 通常扛得更多：我们各测了 6 次，噪声 0.05 扛 5 到 6 级，噪声 0.3 扛 6 到 10 级。"
                "试的动作太小，被猛推时就想不到“大步迈出去”这种救法。每次结果有起伏（尝试是随机的），多推几次看看。"
                "站着不动时 B 并不更抖：每次只执行试出来最好的那个，不好的尝试都被扔掉了。",
                "B usually lasts longer: in six runs each, noise 0.05 survived 5 to 6 levels and noise 0.3 6 to 10. "
                "With tiny tries, a hard shove never leads it to a bold rescue like a big step. Results vary from "
                "run to run (the tries are random), so shove a few times. And B is no shakier when standing still: "
                "only the best try is carried out, the bad ones are thrown away.",
            ),
            check=_shoved,
            unlock=("shove",),
            show=("labels",),
        ),
        Step(
            id="horizon",
            title=_text("预测越远越好吗？", "Is looking further always better?"),
            body=_text(
                "把两只的噪声都设成 0.3，A 的预测时长调到 0.1 秒，B 留在 0.35 秒，再推一次。然后可以试试 0.6 秒。",
                "Set both noises to 0.3, A's horizon to 0.1 s and leave B at 0.35 s, then shove again. Afterwards "
                "try 0.6 s.",
            ),
            done=_text(
                "0.1 秒太短，摔倒的苗头还没出现在预测里，它就来不及了。"
                "可 0.6 秒也不更好（我们各测 6 次，平均扛过：0.1 秒 6 级，0.35 秒 8.5 级，0.6 秒 6.8 级）："
                "随机的未来越长越乱，同样几十个样本里更难挑出好计划，计算量也翻倍。预测时长要恰到好处。",
                "0.1 s is too short: the coming fall is not in the prediction yet when it should act. But 0.6 s is "
                "not better either (six runs each, average levels: 0.1 s 6, 0.35 s 8.5, 0.6 s 6.8): longer random "
                "futures are messier, the same few dozen samples rarely contain a good plan, and the computing "
                "doubles. The horizon has a sweet spot.",
            ),
            check=_compared_horizons,
            unlock=("horizon_a", "samples_a", "noise_a", "horizon_b", "samples_b", "noise_b"),
            show=("labels",),
        ),
        Step(
            id="cost",
            title=_text("算力账", "Counting the cost"),
            body=_text(
                "表里的“每次计划模拟几步”= 样本数 × 预测时长 ÷ 物理步长，而计划每 20 毫秒就要做一次。",
                "'Physics steps per plan' in the table is samples x horizon / physics step, and a new plan is due "
                "every 20 ms.",
            ),
            done=_text("", ""),
            quiz=Quiz(
                question=_text(
                    "为什么真机器狗很少用这么“暴力”的 MPC？", "Why do real robot dogs rarely use MPC this brute-force?"
                ),
                options=(
                    _text("真机没有电机", "Real robots have no motors"),
                    _text(
                        "机载电脑每 20 毫秒算不完上万步仿真，所以用简化模型，或者用学出来的策略",
                        "Its onboard computer cannot simulate tens of thousands of steps every 20 ms, so it uses "
                        "simplified models or a learned policy",
                    ),
                    _text("预测未来是不可能的", "Predicting the future is impossible"),
                ),
                answer=1,
                explain=_text(
                    "对。真机常用简化的模型（比如把身体当成一个质点）做 MPC，"
                    "或者像 G1 那样，把“试错”提前在仿真里做完，学成一个神经网络。",
                    "Right. Real robots often run MPC on a simplified model (such as treating the body as a single "
                    "mass), or, like the G1, do the trial and error ahead of time in simulation and learn a neural "
                    "network.",
                ),
            ),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.level = 0
        self.next_shove: float | None = None
        self.runs: dict[str, dict[str, Any]] = {}
        self.trial = 0  # each shoving gets fresh random tries, so trying again can turn out differently

    def build(self) -> mujoco.MjModel:
        self.model = compile_world(_dog_world("mpc_twins", self.twins, self.start_x)).model
        return self.model

    def settings(self, twin: Twin) -> dict[str, float]:
        v = self.view(twin)
        return {
            "horizon": HORIZONS[int(v["horizon"])],
            "samples": SAMPLES[int(v["samples"])],
            "noise": NOISES[int(v["noise"])],
        }

    def controller(self, model: mujoco.MjModel, data: mujoco.MjData, twin: Twin) -> Any:
        snapshot = copy.copy(data)
        task = make_task("stand", model, twin.name)
        task.begin(snapshot)
        seed = 2 * self.trial + self.twins.index(twin)
        return make_controller("mpc", task, model, snapshot, {**self.settings(twin), "seed": seed})

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        return {f"tilt_{t.name}": self.tilt(data, t) for t in self.twins}

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        twin = next((t for t in self.twins if control in self.knob_ids(t)), None)
        if twin is not None:
            self.playground.session.set_controller(
                twin.name, self.controller(self.playground.session.model, data, twin)
            )
        elif control == "shove" and self.next_shove is None:
            self.trial += 1
            self.restart(data)
            self.level = 0
            self.next_shove = float(data.time) + 1.0
            self.runs = {t.name: {**self.settings(t), "survived": None} for t in self.twins}

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.next_shove is None:
            return
        for twin in self.twins:
            run = self.runs[twin.name]
            if run["survived"] is None and self.down(data, twin):
                run["survived"] = max(0, self.level - 1)
        if all(r["survived"] is not None for r in self.runs.values()):
            self._finish()
        elif data.time >= self.next_shove:
            if self.level >= SHOVE_LEVELS:
                for run in self.runs.values():
                    run["survived"] = SHOVE_LEVELS if run["survived"] is None else run["survived"]
                self._finish()
                return
            self.level += 1
            side = 1.0 if self.level % 2 else -1.0
            for twin in self.twins:
                if self.runs[twin.name]["survived"] is None:
                    dof = int(model.jnt_dofadr[model.body_jntadr[self.trunks[twin.name]]])
                    data.qvel[dof + 1] += side * SHOVE_STEP * self.level  # the same kick for both
            self.next_shove = float(data.time) + SHOVE_EVERY

    def _finish(self) -> None:
        assert self.ctx is not None
        self.ctx.results["shoves"] = {name: dict(run) for name, run in self.runs.items()}
        self.next_shove = None

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        readout = _text(f"第 {self.level} 级", f"level {self.level}") if self.next_shove is not None else _text("", "")
        return {"table": self._table(model, ctx), "readout": readout}

    def _table(self, model: mujoco.MjModel, ctx: LessonContext) -> dict[str, Any]:
        settings = [self.settings(t) for t in self.twins]
        rows: list[dict[str, Any]] = [
            {"heading": _text("现在的设置", "Settings now")},
            _row(_text("预测时长", "Horizon"), [f"{s['horizon']:g} s" for s in settings]),
            _row(_text("每次试几种未来", "Futures per plan"), [str(s["samples"]) for s in settings]),
            _row(_text("探索噪声", "Exploration noise"), [f"{s['noise']:g}" for s in settings]),
            _row(
                _text("每次计划模拟几步", "Physics steps per plan"),
                [f"{round(s['samples'] * s['horizon'] / model.opt.timestep):,}" for s in settings],
            ),
        ]
        shoves = ctx.results.get("shoves")
        if shoves:
            got = [shoves[t.name] for t in self.twins]
            rows += [
                {"heading": _text("上次推的结果", "Last shoving")},
                _row(
                    _text("当时的设置", "Ran with"),
                    [f"{r['horizon']:g} s · {r['samples']} · {r['noise']:g}" for r in got],
                ),
                _row(
                    _text("扛过的级数", "Levels survived"),
                    [str(r["survived"]) for r in got],
                    [r["survived"] for r in got],
                    high=True,
                ),
            ]
        return {"columns": self.columns(), "rows": rows}

    def camera(self) -> dict[str, list[float]]:
        # From in front of the dogs: they stand side by side and the sideways shoves read as left and right.
        return {"eye": [2.6, -0.5, 1.1], "target": [0.0, 0.0, 0.2]}


# ---- gait twins: two dogs walking, the same speed ------------------------------------------------------

PERIODS = (0.25, 0.36, 0.5, 0.7)  # s per gait cycle
LIFTS = (0.5, 1.0, 1.6)  # times the usual foot lift
CURB_X = 1.0  # m: where the curb crosses both lanes
CURB_HEIGHT = 0.03
FLAT_SECONDS = 5.0
CURB_SECONDS = 12.0
SETTLE = 1.5  # s of walking before the measuring starts

GAIT_TWINS = (
    Twin("a", _text("A", "A"), "amber", {"period": 0, "lift": 1}),
    Twin("b", _text("B", "B"), "blue", {"period": 2, "lift": 1}),
)


def _walked_flat(ctx: LessonContext) -> bool:
    return "flat" in ctx.results


def _tried_curb(ctx: LessonContext) -> bool:
    curb = ctx.results.get("curb", {})
    return len({curb.get(t.name, {}).get("lift") for t in GAIT_TWINS}) > 1


class GaitTwins(DogTwins):
    id = "gait_twins"
    title = _text("步态双胞胎：步子的节奏和抬腿高度", "Gait twins: step rhythm and foot lift")
    summary = _text(
        "两只机器狗用同样的速度走，各有自己的步子周期和抬腿高度。看谁省力、谁稳、谁能跨过路沿。",
        "Two robot dogs walk at the same speed, each with its own step period and foot lift. See who saves "
        "energy, who is steady, and who gets over the curb.",
    )
    minutes = 7
    twins = GAIT_TWINS
    start_x = -0.5
    knobs = (
        Knob(
            "period",
            "choice",
            _text("步子周期（秒）", "Step period (s)"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in PERIODS),
        ),
        Knob(
            "lift", "choice", _text("抬腿高度", "Foot lift"), options=tuple(_text(f"×{v:g}", f"×{v:g}") for v in LIFTS)
        ),
    )
    shared = (
        Control("flat", "button", _text("平地走 5 秒", "Walk 5 s on the flat")),
        Control("curb", "button", _text("走过路沿", "Walk over the curb")),
        Control("speed", "slider", _text("速度（两只一起）", "Speed (both)"), 0.1, 0.3, 0.05, 0.2, "m/s"),
    )
    plots = (
        Plot(
            "speed",
            _text("前进速度", "Forward speed"),
            "m/s",
            tuple(Series(f"v_{t.name}", t.label, t.color) for t in GAIT_TWINS),
        ),
        Plot(
            "sway",
            _text("机身晃动", "Body sway"),
            "°/s",
            tuple(Series(f"sway_{t.name}", t.label, t.color) for t in GAIT_TWINS),
        ),
    )
    steps = (
        Step(
            id="rhythm",
            title=_text("步子的节奏", "The rhythm of the steps"),
            body=_text(
                "两只用同样的速度走。A 的步子周期是 0.25 秒（小碎步），B 是 0.5 秒（慢而大步）。"
                "点「平地走 5 秒」，看表里的“最快能走”“晃动”和“每米能耗”。",
                "Both walk at the same speed. A's step period is 0.25 s (quick little steps), B's is 0.5 s (slow, long "
                "steps). Press Walk 5 s on the flat and compare 'top speed', 'sway' and 'energy per metre'.",
            ),
            done=_text(
                "小碎步能走得更快（腿迈得勤），但费力；慢大步更省力，可最高速度低。"
                "周期太长（0.7 秒）连 0.2 米/秒都走不到，"
                "还晃得厉害。动物也一样：跑得越快，步频越高。",
                "Quick steps can go faster (the legs swing more often) but cost more; slow long steps save energy but "
                "top out lower. Too long a period (0.7 s) cannot even reach 0.2 m/s and sways badly. Animals do the "
                "same: the faster they go, the quicker their steps.",
            ),
            check=_walked_flat,
            unlock=("flat", "speed"),
            show=("labels",),
        ),
        Step(
            id="lift",
            title=_text("抬腿要多高？", "How high to lift the feet?"),
            body=_text(
                "把两只的周期都设成 0.36 秒，A 的抬腿调到 ×0.5，B 调到 ×1.6。先「平地走 5 秒」比能耗，"
                "再点「走过路沿」：前面有一道 3 厘米高的坎。",
                "Set both periods to 0.36 s, A's lift to x0.5 and B's to x1.6. First Walk 5 s on the flat and compare "
                "energy, then press Walk over the curb: a 3 cm step lies ahead.",
            ),
            done=_text(
                "平地上抬腿低最省力（我们测到能耗只有一半不到），可脚一碰到 3 厘米的坎就被卡住。抬得高费力，但过得去。"
                "所以崎岖地形上的机器狗都抬腿高，平地上可以抬低省电。",
                "On the flat a low lift saves the most (less than half the energy in our tests), but the foot catches "
                "on the 3 cm curb and the dog is stuck. A high lift costs more but gets across. That is why dogs on "
                "rough ground lift high, and on the flat can lift low to save battery.",
            ),
            check=_tried_curb,
            unlock=("curb", "period_a", "lift_a", "period_b", "lift_b"),
            show=("labels",),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.test: str | None = None
        self.t0 = 0.0
        self.runs: dict[str, dict[str, Any]] = {}
        self.base_lift: float | None = None

    def build(self) -> mujoco.MjModel:
        curb = {"type": "curb", "pos": [CURB_X, 0.0], "size": [0.12, 3.0], "height": CURB_HEIGHT}
        self.model = compile_world(_dog_world("gait_twins", self.twins, self.start_x, [curb])).model
        return self.model

    def settings(self, twin: Twin) -> dict[str, float]:
        v = self.view(twin)
        return {"period": PERIODS[int(v["period"])], "lift": LIFTS[int(v["lift"])]}

    def controller(self, model: mujoco.MjModel, data: mujoco.MjData, twin: Twin) -> Any:
        home = self.playground.behaviors.home_qpos()
        if self.base_lift is None:
            self.base_lift = quadruped.for_template(model, twin.name, home).lift
        s = self.settings(twin)
        return quadruped.for_template(model, twin.name, home, period=s["period"], lift=self.base_lift * s["lift"])

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        out = {}
        for twin in self.twins:
            trunk = self.trunks[twin.name]
            dof = int(model.jnt_dofadr[model.body_jntadr[trunk]])
            out[f"v_{twin.name}"] = float(data.qvel[dof])
            out[f"sway_{twin.name}"] = math.degrees(float(np.linalg.norm(data.qvel[dof + 3 : dof + 5])))
        return out

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        session = self.playground.session
        twin = next((t for t in self.twins if control in self.knob_ids(t)), None)
        if twin is not None and self.test is None:
            session.set_controller(twin.name, self.controller(session.model, data, twin))
        elif control in ("flat", "curb") and self.test is None:
            self.restart(data)
            self.test, self.t0 = control, float(data.time)
            for t in self.twins:
                session.controllers[t.name].command(float(ctx.values["speed"]), 0.0)
            self.runs = {
                t.name: {
                    **self.settings(t),
                    "top": round(session.controllers[t.name].max_speed, 2),
                    "energy": 0.0,
                    "sway": [],
                    "x0": None,
                    "fell": False,
                }
                for t in self.twins
            }

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.test is None:
            return
        t = float(data.time) - self.t0
        for twin in self.twins:
            run = self.runs[twin.name]
            trunk = self.trunks[twin.name]
            run["fell"] = run["fell"] or self.down(data, twin)
            if t < SETTLE:
                continue
            if run["x0"] is None:
                run["x0"] = float(data.xpos[trunk][0])
            acts = [a for a in range(model.nu) if (model.actuator(a).name or "").startswith(f"{twin.name}/")]
            run["energy"] += (
                float(np.sum(np.abs(data.actuator_force[acts] * data.actuator_velocity[acts]))) * model.opt.timestep
            )
            dof = int(model.jnt_dofadr[model.body_jntadr[trunk]])
            run["sway"].append(float(np.linalg.norm(data.qvel[dof + 3 : dof + 5])))
        if t >= (FLAT_SECONDS if self.test == "flat" else CURB_SECONDS):
            self._finish(model, data)

    def _finish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        assert self.ctx is not None
        results = {}
        for twin in self.twins:
            run = self.runs[twin.name]
            trunk = self.trunks[twin.name]
            x = float(data.xpos[trunk][0])
            distance = max(0.0, x - (run["x0"] if run["x0"] is not None else x))
            seconds = (FLAT_SECONDS if self.test == "flat" else CURB_SECONDS) - SETTLE
            mass = float(model.body_subtreemass[trunk])
            results[twin.name] = {
                "period": run["period"],
                "lift": run["lift"],
                "top": run["top"],
                "speed": round(distance / seconds, 3),
                "sway": round(math.degrees(float(np.mean(run["sway"]))) if run["sway"] else 0.0, 1),
                "cot": round(run["energy"] / (mass * GRAVITY * distance), 1) if distance > 0.05 else None,
                "crossed": x > CURB_X + 0.25,
                "fell": run["fell"],
            }
            self.playground.session.controllers[twin.name].command(0.0, 0.0)
        self.ctx.results[self.test] = results
        self.test = None

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        readout = _text("正在走……", "Walking...") if self.test else _text("", "")
        return {"table": self._table(ctx), "readout": readout}

    def _table(self, ctx: LessonContext) -> dict[str, Any]:
        settings = [self.settings(t) for t in self.twins]
        rows: list[dict[str, Any]] = [
            {"heading": _text("现在的设置", "Settings now")},
            _row(_text("步子周期", "Step period"), [f"{s['period']:g} s" for s in settings]),
            _row(_text("抬腿高度", "Foot lift"), [f"×{s['lift']:g}" for s in settings]),
        ]
        for key, heading in (
            ("flat", _text("平地 5 秒", "5 s on the flat")),
            ("curb", _text("过路沿", "Over the curb")),
        ):
            got_all = ctx.results.get(key)
            if not got_all:
                continue
            got = [got_all[t.name] for t in self.twins]
            rows += [
                {"heading": heading},
                _row(_text("当时的设置", "Ran with"), [f"{r['period']:g} s · ×{r['lift']:g}" for r in got]),
                _row(
                    _text("最快能走", "Top speed"),
                    [f"{r['top']:.2f} m/s" for r in got],
                    [r["top"] for r in got],
                    high=True,
                ),
                _row(
                    _text("实际速度", "Speed reached"),
                    [f"{r['speed']:.2f} m/s" for r in got],
                    [r["speed"] for r in got],
                    high=True,
                ),
                _row(_text("晃动", "Sway"), [f"{r['sway']:.0f} °/s" for r in got], [r["sway"] for r in got]),
                _row(_text("每米能耗", "Energy per metre"), [_num(r["cot"]) for r in got], [r["cot"] for r in got]),
            ]
            if key == "curb":
                rows.append(
                    _row(
                        _text("跨过路沿", "Got over"),
                        [_text("过去了", "yes") if r["crossed"] else _text("卡住了", "stuck") for r in got],
                    )
                )
        return {"columns": self.columns(), "rows": rows}

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [0.8, -3.7, 1.5], "target": [0.8, 0.0, 0.15]}  # the start, the curb and beyond it
