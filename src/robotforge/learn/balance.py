"""Lesson: how a humanoid keeps its balance. Centre of mass, the support patch and the capture point.

Our template humanoid (ZMP walking, ankle feedback) and the Unitree G1 (a learned policy) stand in two
lanes. Every push puts both back on their marks, lets them settle (ours crouches first if asked), then
gives the chosen one a kick of a set speed. The floor shows the patch between the feet that touch the
ground, the centre of mass dropped onto it, and the capture point: where the centre of mass would come
to rest over if the robot let its body swing like an inverted pendulum, x_cp = x + v * sqrt(z / g). Inside
the patch, ankles can stop the swing; outside it, only a step can.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import humanoid
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Bench, Control, Lesson, LessonContext, Metric, Plot, Quiz, Series, Step, Text
from robotforge.learn.gait import _hull, _trunk

ROBOTS = ("ours", "g1")
# Side by side along the push direction, seen side on: the capture point runs ahead of the robot across
# the screen. The G1 stands ahead of ours, so stepping under a push takes it further away, not into ours.
MARKS = {"ours": (-0.6, 0.0), "g1": (0.8, 0.0)}
PUSH_AT = 2.2  # s after the restart: standing (and crouched) by then
WATCH = 3.0  # s after the push
SAMPLE_DT = 0.01
UPRIGHT = 0.5  # cosine of the torso's tilt below which a humanoid is down
GRAVITY = 9.81


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


def _signed(value: float, digits: int) -> str:
    """`value` with its sign, as the live formula shows it; one that rounds to zero reads +0.0, not -0.0."""
    text = f"{value:+.{digits}f}"
    return text.replace("-", "+") if float(text) == 0 else text


def _runs(ctx: LessonContext, **want: Any) -> list[dict[str, Any]]:
    return [r for r in ctx.runs if all(test(r["params"].get(k, 0.0)) for k, test in want.items())]


def _stood(run: dict[str, Any]) -> bool:
    return run["metrics"]["stood"] >= 1


def _pushed_gently(ctx: LessonContext) -> bool:
    return any(_stood(r) for r in _runs(ctx, robot=lambda v: v == 0))


def _found_the_edge(ctx: LessonContext) -> bool:
    forward = _runs(ctx, robot=lambda v: v == 0, direction=lambda v: v == 0, crouch=lambda v: v < 0.05)
    return any(_stood(r) for r in forward) and any(not _stood(r) for r in forward)


def _sideways(ctx: LessonContext) -> bool:
    side = _runs(ctx, robot=lambda v: v == 0, direction=lambda v: v == 1, strength=lambda v: v >= 0.5)
    return any(_stood(r) for r in side)


def _crouched(ctx: LessonContext) -> bool:
    low = _runs(ctx, robot=lambda v: v == 0, direction=lambda v: v == 0, crouch=lambda v: v >= 0.25)
    return any(_stood(r) and r["params"]["strength"] >= 0.4 for r in low)


def _g1_stepped(ctx: LessonContext) -> bool:
    big = _runs(ctx, robot=lambda v: v == 1, direction=lambda v: v == 0, strength=lambda v: v >= 0.6)
    return any(_stood(r) for r in big)


STEPS = (
    Step(
        id="stand",
        title=_text("站着的时候，重心在哪", "Standing: where is the centre of mass"),
        body=_text(
            "左边是我们的小人形（0.45 米），右边是 Unitree G1（1.3 米）。先看我们的。\n"
            "地上紫色的那块是两只脚踩住的地方，叫支撑面；黄点是重心投到地上的位置。"
            "空心圈是“捕获点”，后面会讲。\n先猜，再用 0.1 m/s 轻轻推一下。",
            "On the left is our small humanoid (0.45 m), on the right the Unitree G1 (1.3 m). Ours first.\n"
            "The violet patch on the floor is where the two feet press down: the support. The amber dot is the "
            "centre of mass dropped onto the floor. The hollow ring is the capture point; more on it soon.\n"
            "Guess, then give it a gentle 0.1 m/s push.",
        ),
        done=_text(
            "站着时黄点在紫色那块里面，推一下它晃了晃又回来了。只要重心的投影落在支撑面里，脚踝就能把身体推回来。",
            "Standing, the amber dot sits inside the violet patch; after the push the body swayed and came "
            "back. As long as the centre of mass is over the support, the ankles can push the body back.",
        ),
        quiz=Quiz(
            question=_text(
                "站着不动时，重心的投影在哪里？", "Standing still, where is the centre of mass over the floor?"
            ),
            options=(
                _text("两只脚踩住的那块里面", "Inside the patch between the feet"),
                _text("脚的前面", "In front of the feet"),
                _text("哪都可以", "Anywhere"),
            ),
            answer=0,
            explain=_text("出了那块，就没有东西把它撑住了。", "Outside it, nothing holds it up."),
            predict=True,
        ),
        check=_pushed_gently,
        unlock=("push", "strength"),
        show=("labels", "support", "f_cp"),
    ),
    Step(
        id="edge",
        title=_text("推到倒：捕获点", "Push until it falls: the capture point"),
        body=_text(
            "被推以后身体往前冲，像倒立摆一样。捕获点 = 重心 + 速度 × √(重心高度 ÷ g)：身体带着这个速度，"
            "重心要冲到这里才停得下来。公式里会实时算出它在脚尖前面还是后面。\n"
            "先猜，再分别用 0.3 和 0.4 m/s 往前推。",
            "A push sends the body forward like an inverted pendulum. The capture point = centre of mass + "
            "speed * sqrt(height / g): with that speed the centre of mass only comes to a stop out there. The "
            "formula works out live whether that is in front of the toes or behind them.\n"
            "Guess, then push forward with 0.3 and with 0.4 m/s.",
        ),
        done=_text(
            "0.3 m/s 时捕获点在重心前 4.6 cm，脚尖在 5.1 cm，还在脚里，站住了；0.4 m/s 时是 6.1 cm，"
            "出了脚尖，倒了。一个公式就预言了倒不倒。",
            "At 0.3 m/s the capture point is 4.6 cm ahead of the centre of mass and the toes 5.1 cm: still "
            "under the feet, it stands. At 0.4 m/s it is 6.1 cm, past the toes, and it falls. One formula "
            "predicted it.",
        ),
        quiz=Quiz(
            question=_text(
                "往前推 0.4 m/s，我们的小人形会倒吗？", "A 0.4 m/s push forward: does our small humanoid fall?"
            ),
            options=(_text("不会", "No"), _text("会", "Yes")),
            answer=1,
            explain=_text("捕获点 6.1 cm，脚尖只有 5.1 cm。", "Capture point 6.1 cm, toes only 5.1 cm."),
            predict=True,
        ),
        check=_found_the_edge,
        show=("capture", "f_edge"),
    ),
    Step(
        id="side",
        title=_text("侧着推", "A push from the side"),
        body=_text(
            "「方向」选「往左」。两只脚左右分开站，侧面的支撑更宽。先猜，再用 0.5 m/s 侧推。",
            "Set Direction to the left. The feet stand apart, so the support is wider sideways. Guess, then push "
            "sideways with 0.5 m/s.",
        ),
        done=_text(
            "侧面脚边离重心 7.3 cm，比脚尖的 5.1 cm 远，所以 0.5 m/s 也站得住（我们测到 0.6 才倒）。"
            "人被侧推时会把脚分得更开，就是这个道理。",
            "Sideways the edge of the feet is 7.3 cm from the centre of mass, more than the toes' 5.1 cm, so "
            "0.5 m/s is still fine (in our runs it fell at 0.6). That is why people widen their stance when "
            "shoved from the side.",
        ),
        quiz=Quiz(
            question=_text("侧推 0.5 m/s 呢？", "And 0.5 m/s from the side?"),
            options=(_text("站得住", "It stands"), _text("会倒", "It falls")),
            answer=0,
            explain=_text("侧面的支撑更宽。", "The support is wider sideways."),
            predict=True,
        ),
        check=_sideways,
        unlock=("direction",),
    ),
    Step(
        id="crouch",
        title=_text("蹲低一点", "Crouch a little"),
        body=_text(
            "方向换回「往前」，把「蹲低」调到 0.3（髋往下沉腿长的三成）。先猜，再用 0.4 m/s 往前推。",
            "Back to forward, and set Crouch to 0.3 (the hips sink by 30% of the leg length). Guess, then push "
            "forward with 0.4 m/s.",
        ),
        done=_text(
            "站住了。重心从 22.8 cm 降到 17.6 cm，√(高度 ÷ g) 跟着变小，同样 0.4 m/s 的捕获点从 6.1 缩到 5.4 cm，"
            "刚好在脚尖附近，脚踝还救得回来。所以摔跤手和滑板手都压低重心。",
            "It stands. The centre of mass drops from 22.8 to 17.6 cm, sqrt(height / g) shrinks with it, and "
            "the same 0.4 m/s puts the capture point at 5.4 instead of 6.1 cm, right at the toes, where the "
            "ankles can still save it. That is why wrestlers and skateboarders keep low.",
        ),
        quiz=Quiz(
            question=_text("蹲低以后，0.4 m/s 往前推会怎样？", "Crouched, what does a 0.4 m/s push forward do?"),
            options=(
                _text("还是倒", "It still falls"),
                _text("站得住", "It stands"),
                _text("倒得更快", "It falls faster"),
            ),
            answer=1,
            explain=_text("重心低了，捕获点离得更近。", "A lower centre of mass keeps the capture point closer."),
            predict=True,
        ),
        check=_crouched,
        unlock=("crouch",),
    ),
    Step(
        id="g1",
        title=_text("换成真机：G1 会迈步", "Now the G1: it steps"),
        body=_text(
            "「哪个机器人」选 G1，「蹲低」回到 0。G1 用的是强化学习练出来的策略。"
            "先猜，再用 0.8 m/s 往前推（比把我们的小人形推倒的力度大一倍）。",
            "Set Which robot to the G1 and Crouch back to 0. The G1 runs a policy learned with reinforcement "
            "learning. Guess, then push it forward with 0.8 m/s (twice what knocked ours over).",
        ),
        done=_text(
            "捕获点冲到脚尖前十几厘米，G1 却没倒：它往前迈了几步，让脚追上捕获点（0.6 m/s 时迈了 40 cm，"
            "1.0 m/s 时 1.2 米）。这叫迈步策略。我们的控制器只会用脚踝，不会为了站稳迈步。"
            "G1 是在仿真里被推着练出来的。",
            "The capture point lands well past its toes, yet the G1 stays up: it steps forward to bring its "
            "feet under the capture point (40 cm for 0.6 m/s, 1.2 m for 1.0 m/s in our runs). That is the "
            "stepping strategy. Our controller only uses its ankles and never steps to stay up; the G1 learned "
            "it in simulation.",
        ),
        quiz=Quiz(
            question=_text("0.8 m/s 往前推，G1 会怎样？", "A 0.8 m/s push forward: what does the G1 do?"),
            options=(
                _text("原地站住", "It stays on the spot"),
                _text("往前迈几步，站住", "It steps forward a few times and stays up"),
                _text("倒了", "It falls"),
            ),
            answer=1,
            explain=_text("捕获点出了脚，只有迈步才救得回来。", "Past the feet, only a step can save it."),
            predict=True,
        ),
        check=_g1_stepped,
        unlock=("robot",),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text("在公交车上，车猛地一刹，你会怎么做？", "On a bus that brakes hard, what do you do?"),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("为什么那样做能站住？", "Why does that keep you up?"),
            options=(
                _text("往前迈一步：把脚放到捕获点那里", "Step forward: put a foot where the capture point is"),
                _text("站直不动：更稳", "Stand stiff and still: steadier"),
                _text("踮起脚尖：离地更高", "Go up on tiptoe: higher up"),
            ),
            answer=0,
            explain=_text(
                "对。迈步把支撑面挪到捕获点下面，这是人和会走路的机器人最有用的一招。",
                "Right. A step moves the support under the capture point, the most useful trick people and "
                "walking robots have.",
            ),
        ),
    ),
)


class Watch:
    """Follows the push test every physics step and samples the plots at 100 Hz."""

    rate_hz = 500.0

    def __init__(self, lesson: BalanceLesson) -> None:
        self.lesson = lesson
        self.next_sample = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.lesson.measure(model, data)
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            self.lesson.samples.append(self.lesson.sample(model, data))

    def status(self) -> dict[str, Any]:
        return {"controller": "balance-watch"}


class BalanceLesson(Lesson):
    id = "balance"
    chapter = _text("走路", "Walking")
    title = _text("人形平衡：重心、支撑面和捕获点", "Humanoid balance: centre of mass, support, capture point")
    summary = _text(
        "推一推两台人形：一个公式预言倒不倒，蹲低为什么更稳，G1 为什么会迈步。",
        "Push two humanoids: one formula predicts the fall, why crouching helps, and why the G1 steps.",
    )
    minutes = 12
    controls = (
        Control("push", "button", _text("推一下", "Push")),
        Control("strength", "slider", _text("推的力度（速度）", "Push (speed)"), 0.1, 1.0, 0.1, 0.1, "m/s"),
        Control(
            "direction",
            "choice",
            _text("方向", "Direction"),
            options=(_text("往前", "Forward"), _text("往左", "To the left")),
        ),
        Control("crouch", "slider", _text("蹲低（我们的）", "Crouch (ours)"), 0.0, 0.3, 0.05, 0.0),
        Control(
            "robot",
            "choice",
            _text("哪个机器人", "Which robot"),
            options=(_text("我们的人形", "Our humanoid"), _text("Unitree G1", "Unitree G1")),
        ),
    )
    plots = (
        Plot(
            "margin",
            _text("捕获点离脚边（正 = 还在脚里）", "Capture point inside the feet (positive = inside)"),
            "cm",
            (Series("margin", _text("余量", "margin"), "violet"),),
            window=5.0,
        ),
        Plot("tilt", _text("身体倾斜", "Body tilt"), "°", (Series("tilt", _text("倾斜", "tilt"), "ink"),), window=5.0),
    )
    steps = STEPS
    bench = Bench(
        title=_text("实验记录：推完以后身体倾斜", "Experiment record: body tilt after the push"),
        x=_text("推了以后", "since the push"),
        x_unit="s",
        y=_text("倾斜", "tilt"),
        y_unit="°",
        params=("robot", "strength", "direction", "crouch"),
        short=(_text("机器人", "robot"), _text("力度", "push"), _text("方向", "dir"), _text("蹲", "crouch")),
        metrics=(
            Metric("cp_cm", _text("捕获点", "capture pt"), "cm", ranked=False),
            Metric("edge_cm", _text("脚边", "foot edge"), "cm", ranked=False),
            Metric("stood", _text("站住", "stood"), "", lower_is_better=False, ranked=False),
        ),
    )

    def __init__(self) -> None:
        self.playground = None
        self.ctx: LessonContext | None = None
        self.start: np.ndarray | None = None
        self.controllers: dict[str, Any] = {}
        self.roots: dict[str, int] = {}
        self.bodies: dict[str, set[int]] = {}
        self.floor = -1
        self.samples: list[dict[str, float]] = []
        self.test: dict[str, Any] | None = None

    def build(self) -> mujoco.MjModel:
        world = World.model_validate(
            {
                "name": "balance",
                "floor": {"style": "concrete"},
                "robots": [
                    {"name": "ours", "template": "humanoid", "pos": list(MARKS["ours"])},
                    {"name": "g1", "library": "g1", "pos": list(MARKS["g1"])},
                ],
            }
        )
        return compile_world(world).model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.start = data.qpos.copy()
        self.floor = int(model.geom("floor").id)
        self.test = None
        self.samples = []
        self.roots = {name: _trunk(model, name) for name in ROBOTS}
        self.bodies = {
            name: {b for b in range(model.nbody) if model.body(b).name.startswith(f"{name}/")} for name in ROBOTS
        }
        self.controllers = {name: self._controller(model, name) for name in ROBOTS}
        return {**self.controllers, "watch": Watch(self)}

    def _controller(self, model: mujoco.MjModel, name: str) -> Any:
        if name == "ours":
            return humanoid.for_template(model, name, model.key("home").qpos)
        from robotforge.bridge.onnx_policy import walker_for  # the shipped G1 policy runs on numpy

        walker = walker_for(model, name, "g1")
        walker.command(0.0, 0.0)
        return walker

    @property
    def active(self) -> str:
        assert self.ctx is not None
        return ROBOTS[int(self.ctx.values["robot"])]

    # ---- the push test ---------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control != "push":
            return
        assert self.start is not None and self.playground is not None
        session = self.playground.session
        model = session.model
        data.qpos[:] = self.start  # both back on their marks, still
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for name in ROBOTS:
            self.controllers[name] = self._controller(model, name)
            session.set_controller(name, self.controllers[name])
        crouch = float(ctx.values["crouch"])
        if crouch > 0:
            self.controllers["ours"].crouch(crouch)
        self.test = {
            "robot": self.active,
            "t0": float(data.time),
            "pushed": None,  # time of the push, once given
            "times": [],
            "tilts": [],
            "fell": False,
            "settings": {k: float(ctx.values[k]) for k in ("robot", "strength", "direction", "crouch")},
        }

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        test = self.test
        if test is None:
            return
        t = float(data.time) - test["t0"]
        root = self.roots[test["robot"]]
        axis = int(test["settings"]["direction"])
        if test["pushed"] is None and t >= PUSH_AT:
            mujoco.mj_subtreeVel(model, data)
            dof = int(model.jnt_dofadr[model.body_jntadr[root]])
            data.qvel[dof + axis] += test["settings"]["strength"]  # a kick to the whole robot, through its root
            mujoco.mj_forward(model, data)
            mujoco.mj_subtreeVel(model, data)
            com, cp, edge = self._balance(model, data, test["robot"], axis)
            test.update(pushed=float(data.time), cp=cp, edge=edge, height=float(com[2]))
        if test["pushed"] is None:
            return
        if float(data.xmat[root][8]) < UPRIGHT:
            test["fell"] = True
        if float(data.time) - test["pushed"] >= WATCH or test["fell"]:
            self._finish()

    def _finish(self) -> None:
        test, self.test = self.test, None
        assert test is not None and self.ctx is not None
        stood = not test["fell"]
        result = {
            "robot": test["robot"],
            "stood": stood,
            "cp_cm": round(100 * test["cp"], 1),
            "edge_cm": round(100 * test["edge"], 1),
            "height_cm": round(100 * test["height"], 1),
        }
        self.ctx.results["push"] = result
        verdict = _text("站住了", "stayed up") if stood else _text("倒了", "fell")
        self.ctx.results["summary"] = {
            "zh": f"捕获点 {result['cp_cm']:g} cm，脚边 {result['edge_cm']:g} cm：{verdict['zh']}",
            "en": f"Capture point {result['cp_cm']:g} cm, foot edge {result['edge_cm']:g} cm: {verdict['en']}",
        }
        metrics = {"cp_cm": result["cp_cm"], "edge_cm": result["edge_cm"], "stood": float(stood)}
        self.ctx.record_run(test["settings"], metrics, test["times"], test["tilts"])

    # ---- what the robot is doing -----------------------------------------------------------------------

    def _support(self, model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
        """Where the robot touches the floor (x, y), or an empty array."""
        bodies = self.bodies[name]
        points = []
        for contact in data.contact[: data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.floor not in (g1, g2):
                continue
            if int(model.geom_bodyid[g2 if g1 == self.floor else g1]) in bodies:
                points.append(contact.pos[:2].copy())
        return np.array(points).reshape(-1, 2)

    def _balance(
        self, model: mujoco.MjModel, data: mujoco.MjData, name: str, axis: int
    ) -> tuple[np.ndarray, float, float]:
        """(centre of mass, capture point ahead of it, foot edge ahead of it) along `axis`, in m.
        Needs mj_subtreeVel to have run."""
        root = self.roots[name]
        com = data.subtree_com[root]
        speed = data.subtree_linvel[root][axis]
        cp = float(speed * math.sqrt(max(com[2], 1e-3) / GRAVITY))
        support = self._support(model, data, name)
        edge = float(support[:, axis].max() - com[axis]) if len(support) else 0.0
        return com.copy(), cp, edge

    def _tilt(self, data: mujoco.MjData, name: str) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[self.roots[name]][8])))))

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        name = self.active if self.test is None else self.test["robot"]
        axis = int(self.ctx.values["direction"]) if self.test is None else int(self.test["settings"]["direction"])
        mujoco.mj_subtreeVel(model, data)
        _, cp, edge = self._balance(model, data, name, axis)
        tilt = self._tilt(data, name)
        test = self.test
        if test is not None and test["pushed"] is not None:
            test["times"].append(round(float(data.time) - test["pushed"], 3))
            test["tilts"].append(round(tilt, 2))
        return {"t": float(data.time), "margin": 100 * (edge - cp), "tilt": tilt}

    def drain_samples(self) -> list[dict[str, float]]:
        out, self.samples = self.samples, []
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        name = self.active if self.test is None else self.test["robot"]
        axis = int(ctx.values["direction"])
        mujoco.mj_subtreeVel(model, data)
        com, cp, edge = self._balance(model, data, name, axis)
        root = self.roots[name]
        velocity = data.subtree_linvel[root][:2]
        capture = com[:2] + velocity * math.sqrt(max(com[2], 1e-3) / GRAVITY)
        support = self._support(model, data, name)
        hull = _hull(support) if len(support) > 2 else support
        readout = _text("", "")
        if self.test is not None:
            if self.test["pushed"] is None:
                readout = _text("站稳了就推……", "Settling, then the push...")
            else:
                readout = _text("推了，看它站不站得住", "Pushed: will it stay up?")
        return {
            "view": self.view(name),
            "follow": root,
            "support": hull.round(4).tolist(),
            "com": com[:2].round(4).tolist(),
            "capture": capture.round(4).tolist(),
            "formula": self._formula(com, cp, edge, axis, ctx),
            "readout": readout,
        }

    def _formula(self, com: np.ndarray, cp: float, edge: float, axis: int, ctx: LessonContext) -> list[dict[str, Any]]:
        where = _text("往前", "forward") if axis == 0 else _text("往左", "to the left")
        z = float(com[2])
        speed = cp / math.sqrt(max(z, 1e-3) / GRAVITY)
        inside = edge - cp
        margin_row: dict[str, Any] = {
            "show": "f_edge",
            "label": _text("还剩多少", "Margin"),
            "expr": _text("脚边 − 捕获点", "foot edge - capture point"),
            "value": f"{_signed(100 * inside, 1)} cm",
            "color": "violet",
        }
        if inside < 0:
            margin_row["note"] = _text("捕获点出了脚：不迈步就要倒", "The capture point is past the feet: step or fall")
        return [
            {
                "show": "f_cp",
                "label": _text("捕获点", "Capture point"),
                "expr": f"v × √(z ÷ g) = {_signed(speed, 2)} × √({z:.3f} ÷ 9.81)",
                "value": f"{_signed(100 * cp, 1)} cm",
                "color": "amber",
            },
            {
                "show": "f_edge",
                "label": _text("脚边", "Foot edge"),
                "expr": _text(f"重心{where['zh']}到支撑面的边", f"centre of mass to the support's edge, {where['en']}"),
                "value": f"{100 * edge:.1f} cm",
                "color": "ink",
            },
            margin_row,
        ]

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "gait",
            "entrants": [
                {
                    "name": name,
                    "label": label,
                    "color": color,
                    "real": name == "g1",
                    "base": [*MARKS[name], 0.0],
                    "body": _trunk(model, name),
                    "top": top,
                    "cloud": [],
                }
                for name, label, color, top in (
                    ("ours", _text("我们的人形", "Our humanoid"), "amber", 0.5),
                    ("g1", _text("Unitree G1", "Unitree G1"), "blue", 1.35),
                )
            ],
            "targets": [],
        }

    @staticmethod
    def view(name: str) -> dict[str, list[float]]:
        """Side on, close enough to see the feet: near for our 0.45 m humanoid, further back for the G1.
        The camera then follows the robot being pushed."""
        x, y = MARKS[name]
        if name == "ours":
            return {"eye": [x + 0.15, y - 1.9, 0.75], "target": [x + 0.15, y, 0.25]}
        return {"eye": [x + 0.3, y - 4.0, 1.25], "target": [x + 0.3, y, 0.6]}

    def camera(self) -> dict[str, list[float]]:
        return self.view("ours")
