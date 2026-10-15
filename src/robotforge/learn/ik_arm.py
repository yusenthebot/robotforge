"""Inverse kinematics on six-joint arms: our template arm first, then the AgileX PiPER.

The planar lesson shows the idea with two links and a formula. Real arms have six joints in 3D, where
no formula is handy: the hand is steered by the Jacobian, one damped least-squares step per control tick.
The lesson makes that visible: the target to drag, the manipulability ellipsoid that shows which ways the
hand moves easily, what the damping does at the edge of reach, and the same steps on a real arm.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.arm import DOWN, ArmMotion
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Control, Lesson, LessonContext, Plot, Quiz, Series, Step, Text

ARMS = ("ours", "piper")
LANE = 0.5  # m from the middle to each arm's base
DAMPINGS = (0.0001, 0.001, 0.01, 0.05, 0.2)  # lambda: what DLS adds to J J^T
REACHED = 0.008  # m
MOVED = 0.08  # m a target must move to count as a new one
FLAIL = 30.0  # deg per tick: a joint jumping this far is flailing
CALM = 5.0  # deg per tick
EDGE = 30.0  # % of the home pose's manipulability: near a singular pose
# Flailing with a tiny λ can throw the joints onto their limits, the arm folded away from the target; the
# clipped steps then point out of range and it never moves again (CI caught it: a 1e-9 difference in the
# aimed joints jammed 7 runs in 20). Stalled that far off with a joint on a limit, it starts over from home.
JAMMED = 0.05  # m off the target...
STALLED = 0.05  # ...with no joint turning more than this many degrees a tick...
JAM_TICKS = 25  # ...for this many ticks (0.5 s)
JAM_NOTE = 4.0  # s the note about starting over stays up
ELLIPSOID_SCALE = 0.25  # drawn size per m/rad of hand speed
START = (0.25, 0.0, 0.25)  # target relative to the active arm's base


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


class IKFollow:
    """Resolved-rate IK: every tick one damped least-squares step from where the joints are aimed now."""

    rate_hz = 50.0

    def __init__(self, lesson: IKArmLesson, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> None:
        self.lesson = lesson
        self.ctx = ctx
        home = lesson.playground.behaviors.home_qpos()
        self.arms = {name: ArmMotion(model, name, home) for name in ARMS}
        self.q = {name: arm.q.copy() for name, arm in self.arms.items()}
        self.home = {name: q.copy() for name, q in self.q.items()}
        self.stalled = 0  # ticks the active arm has been jammed
        self.note_until = -1.0  # sim time the starting-over note comes down
        self.w_home = {name: max(self.manipulability(name)[0], 1e-9) for name in ARMS}
        self.step_deg = 0.0
        self.error = 0.0
        self.last_reached: np.ndarray | None = None
        self.samples: deque[dict[str, float]] = deque(maxlen=1024)

    def active(self) -> str:
        return ARMS[int(self.ctx.values["arm"])]

    def target(self, name: str | None = None) -> np.ndarray:
        arm = self.arms[name or self.active()]
        v = self.ctx.values
        return np.array([arm.base_xy[0] + v["tx"], arm.base_xy[1] + v["ty"], v["tz"]])

    def manipulability(self, name: str) -> tuple[float, np.ndarray, np.ndarray]:
        """w = sqrt(det(J J^T)) of the hand's position Jacobian, and its SVD (directions, hand speeds)."""
        arm = self.arms[name]
        point = arm.ik.point(arm.chain, self.q[name])
        model, data = arm.ik.model, arm.ik.data
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, point, arm.chain.body)
        J = jacp[:, arm.chain.dof_adr]
        U, S, _ = np.linalg.svd(J)
        return float(np.prod(S)), U, S

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        name = self.active()
        arm = self.arms[name]
        v = self.ctx.values
        target = self.target(name)
        axis = (arm.tool_axis, DOWN) if v["down"] else None
        q, _ = arm.ik.solve(
            arm.chain, target, self.q[name], iterations=1, damping=DAMPINGS[int(v["damping"])], axis=axis
        )
        self.step_deg = math.degrees(float(np.max(np.abs(q - self.q[name]))))
        self.q[name] = q
        self._unjam(name, target, float(data.time))
        for other, a in self.arms.items():
            data.ctrl[a.actuators] = self.q[other]
        hand = arm.ik.point(arm.chain, q)
        self.error = float(np.linalg.norm(hand - target))
        w = 100 * self.manipulability(name)[0] / self.w_home[name]
        self.lesson.note(name, target, self.error, w, self.step_deg, bool(v["down"]), DAMPINGS[int(v["damping"])])
        self.samples.append(
            {"t": float(data.time), "err": 1000 * self.error, "w": min(w, 400.0), "step": min(self.step_deg, 180.0)}
        )

    def _unjam(self, name: str, target: np.ndarray, now: float) -> None:
        """Start the active arm over from home when it is stuck on its joint limits far from the target."""
        if 0 <= self.note_until < now:
            self.ctx.results.pop("summary", None)
            self.note_until = -1.0
        arm = self.arms[name]
        q = self.q[name]
        on_limit = bool(np.any((q <= arm.chain.lower + 1e-6) | (q >= arm.chain.upper - 1e-6)))
        far = float(np.linalg.norm(arm.ik.point(arm.chain, q) - target)) > JAMMED
        self.stalled = self.stalled + 1 if on_limit and far and self.step_deg < STALLED else 0
        if self.stalled < JAM_TICKS:
            return
        self.stalled = 0
        self.q[name] = self.home[name].copy()
        self.ctx.results["summary"] = _text(
            "臂甩到关节极限上卡住了，一步也挪不动：从初始姿态重新解。真正的 IK 求解器也常靠换个起点重来。",
            "The arm was thrown onto its joint limits and stuck there, unable to move a step: it starts over "
            "from its home pose. Real IK solvers restart from another pose too.",
        )
        self.note_until = now + JAM_NOTE

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-ik", "arm": self.active(), "error_mm": round(1000 * self.error, 1)}


def _reached_targets(ctx: LessonContext) -> bool:
    return ctx.results.get("ik", {}).get("reached", 0) >= 2


def _near_singular(ctx: LessonContext) -> bool:
    return "edge" in ctx.events


def _tried_damping(ctx: LessonContext) -> bool:
    return {"flail", "calm_edge"} <= ctx.events


def _down_reached(ctx: LessonContext) -> bool:
    return "down_reached" in ctx.events


def _piper_far(ctx: LessonContext) -> bool:
    return "piper_far" in ctx.events


STEPS = (
    Step(
        id="drag",
        title=_text("三维里够目标", "Reaching in 3D"),
        body=_text(
            "这是我们的六轴机械臂。拖动琥珀色小球（或用下面的 x、y、z 滑条），手会跟着走。\n"
            "二维时有公式能直接算出关节角；六个关节在三维里就没有现成的公式了。它用的办法是：每 20 毫秒，"
            "用雅可比矩阵算出“每个关节转一点，手会往哪儿挪”，朝目标走一小步。把目标换 2 个地方试试。",
            "This is our six-joint arm. Drag the amber ball (or use the x, y, z sliders below) and the hand "
            "follows.\nIn 2D a formula gives the joint angles directly; with six joints in 3D there is no handy "
            "formula. Instead, every 20 ms it uses the Jacobian, 'turn each joint a little and the hand moves this "
            "way', to take a small step toward the target. Put the target in two different places.",
        ),
        done=_text(
            "这就是迭代逆运动学：不求一步到位，每个控制周期走一小步，几百毫秒就到了。目标一直在动也没关系，"
            "它每一步都重新算。",
            "That is iterative inverse kinematics: no attempt to land in one go, just a small step every control "
            "tick, there in a few hundred milliseconds. A moving target is no problem; every step is worked out "
            "afresh.",
        ),
        check=_reached_targets,
        unlock=("tx", "ty", "tz"),
        show=("target", "error", "labels"),
    ),
    Step(
        id="ellipsoid",
        title=_text("椭球：手往哪边好动", "The ellipsoid: which ways the hand moves easily"),
        body=_text(
            "手上的半透明椭球叫可操作度椭球：关节都转同样一点点，手能往椭球长的方向挪得多，往扁的方向挪得少。\n"
            "把目标往外拉，让臂越来越直，看椭球和右边的“可操作度”曲线怎么变。",
            "The see-through ellipsoid on the hand is the manipulability ellipsoid: for the same small turn of the "
            "joints, the hand moves a lot along its long axes and little along its short ones.\nPull the target "
            "outward so the arm straightens, and watch the ellipsoid and the manipulability curve.",
        ),
        done=_text(
            "臂快伸直时，椭球被压扁了：手几乎没法再往外挪，因为所有关节都只能让它往侧面或上下动。"
            "这种姿态叫奇异位形，可操作度掉到接近 0。",
            "As the arm straightens the ellipsoid flattens: the hand can hardly move further out, because every "
            "joint now only moves it sideways or up and down. Such a pose is called a singularity, and the "
            "manipulability drops toward 0.",
        ),
        check=_near_singular,
        show=("target", "error", "ellipsoid", "labels"),
    ),
    Step(
        id="damping",
        title=_text("阻尼 λ：在奇异点附近别乱甩", "Damping λ: no flailing near a singularity"),
        body=_text(
            "雅可比在奇异点附近几乎不可逆：想让手往“动不了”的方向挪一点，算出来的关节角会大得离谱。阻尼最小二乘"
            "（DLS）在里面加了一个 λ 来压住它。\n先把 λ 调到 0.0001，把目标拉到臂够不到的远处，看“关节一步转多少”；"
            "再把 λ 调到 0.05 或 0.2，再拉一次。",
            "Near a singularity the Jacobian can barely be inverted: asking the hand to move a little in the 'stuck' "
            "direction gives absurdly large joint angles. Damped least squares (DLS) adds a λ to tame that.\n"
            "First set λ to 0.0001 and pull the target beyond the arm's reach, and watch 'joint step per tick'; "
            "then set λ to 0.05 or 0.2 and pull again.",
        ),
        done=_text(
            "λ 太小，在边上关节会一步甩出几十上百度，整条臂乱抽，甩到关节极限上还可能卡死（这时它从初始姿态重新解）；"
            "λ 大，动作平稳，代价是离目标远一点、跟得慢一点。"
            "真实机器人都会加阻尼或限速，安全比那几毫秒的精度要紧。",
            "With too small a λ, at the edge a joint jumps tens or hundreds of degrees in one step and the whole "
            "arm thrashes, and may jam on its joint limits (then it starts over from home); with a large λ the "
            "motion stays smooth, at the price of ending a little further from the target and following more "
            "slowly. Real robots always add damping or speed limits: safety matters more than those few millimetres.",
        ),
        check=_tried_damping,
        unlock=("damping",),
        show=("target", "error", "ellipsoid", "labels"),
    ),
    Step(
        id="down",
        title=_text("手爪朝下", "Gripper pointing down"),
        body=_text(
            "抓东西时手爪要垂直朝下，这又是一个要求。打开「手爪朝下」，把目标放回臂前方再够一次。",
            "To pick things up the gripper must point straight down: one more requirement. Turn on Gripper "
            "pointing down, put the target back in front of the arm and reach it again.",
        ),
        done=_text(
            "多了朝向要求，同一个臂能到的地方就变少了：位置用 3 个自由度，朝向又占了几个。所以“臂展”和“能朝下抓的范围”"
            "是两回事，真机对比那课里 PiPER 就是这样。",
            "With a direction to hold, the same arm reaches fewer places: position uses 3 degrees of freedom and "
            "the direction takes more. That is why 'reach' and 'how far it can pick straight down' differ, as the "
            "PiPER showed in the real-robot face-off.",
        ),
        check=_down_reached,
        unlock=("down",),
        show=("target", "error", "ellipsoid", "labels"),
    ),
    Step(
        id="piper",
        title=_text("换成真机：AgileX PiPER", "Now a real arm: the AgileX PiPER"),
        body=_text(
            "把「跟着目标的臂」换成 PiPER（右边那台）。关掉「手爪朝下」，"
            "把目标放到前方 0.55 米：我们的臂够不到的地方。",
            "Switch 'Arm following the target' to the PiPER (the one on the right). Turn off Gripper pointing down "
            "and put the target 0.55 m out in front, where our arm cannot reach.",
        ),
        done=_text(
            "一模一样的算法，换了一台真机照样能用：雅可比 IK 只需要知道关节怎么连，不在乎是谁造的。"
            "PiPER 的臂更长，所以能到我们够不着的地方，它的椭球在同样的距离上也更“圆”。",
            "Exactly the same algorithm works on a real arm: Jacobian IK only needs to know how the joints connect, "
            "not who built them. The PiPER's arm is longer, so it reaches where ours cannot, and at the same "
            "distance its ellipsoid is rounder.",
        ),
        check=_piper_far,
        unlock=("arm",),
        show=("target", "error", "ellipsoid", "labels"),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text("可操作度曲线掉到接近 0 的时候……", "When the manipulability curve drops near 0..."),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("可操作度接近 0 说明什么？", "What does a manipulability near 0 mean?"),
            options=(
                _text("电机没电了", "The motors are out of power"),
                _text(
                    "手在某个方向上几乎动不了，关节要转很大角度才能挪一点（奇异位形）",
                    "The hand can barely move in some direction; the joints must turn a lot to move it a little "
                    "(a singularity)",
                ),
                _text("机械臂到达了目标", "The arm has reached the target"),
            ),
            answer=1,
            explain=_text(
                "对。规划路径时要绕开奇异位形，控制时要加阻尼，这就是工业机器人程序员每天在处理的事。",
                "Right. Paths are planned around singularities and controllers add damping: that is everyday work "
                "for industrial robot programmers.",
            ),
        ),
    ),
)


class IKArmLesson(Lesson):
    id = "ik_arm"
    chapter = _text("运动学", "Kinematics")
    title = _text(
        "逆运动学（二）：六轴机械臂，从模板到 PiPER", "Inverse kinematics II: six-joint arms, template to PiPER"
    )
    summary = _text(
        "在三维里拖目标，看雅可比怎么一步步把手带过去；可操作度椭球、奇异点、阻尼 λ，最后换到真机 PiPER 上。",
        "Drag a target in 3D and watch the Jacobian steer the hand step by step: the manipulability ellipsoid, "
        "singularities, damping λ, and finally the real PiPER.",
    )
    minutes = 10
    controls = (
        Control("tx", "slider", _text("目标：往前", "Target: forward"), 0.05, 0.85, 0.01, START[0], " m"),
        Control("ty", "slider", _text("目标：左右", "Target: sideways"), -0.45, 0.45, 0.01, START[1], " m"),
        Control("tz", "slider", _text("目标：高度", "Target: height"), 0.02, 0.75, 0.01, START[2], " m"),
        Control(
            "damping",
            "choice",
            _text("阻尼 λ", "Damping λ"),
            value=3,  # 0.05: calm at the edge of reach; the lesson has the learner try 0.0001
            options=tuple(_text(f"{d:g}", f"{d:g}") for d in DAMPINGS),
        ),
        Control("down", "toggle", _text("手爪朝下", "Gripper pointing down")),
        Control(
            "arm",
            "choice",
            _text("跟着目标的臂", "Arm following the target"),
            options=(_text("我们的臂", "Our arm"), _text("PiPER", "PiPER")),
        ),
    )
    plots = (
        Plot(
            "err",
            _text("离目标多远", "Distance to the target"),
            "mm",
            (Series("err", _text("误差", "error"), "amber"),),
        ),
        Plot(
            "w",
            _text("可操作度（起始姿态 = 100）", "Manipulability (home pose = 100)"),
            "",
            (Series("w", _text("可操作度", "manipulability"), "blue"),),
        ),
        Plot(
            "step",
            _text("关节一步转多少", "Joint step per tick"),
            "°",
            (Series("step", _text("最大的那个关节", "largest joint"), "violet"),),
        ),
    )
    steps = STEPS

    def __init__(self) -> None:
        self.playground = None
        self.follow: IKFollow | None = None
        self.ctx: LessonContext | None = None
        self.bases: dict[str, list[float]] = {}

    def build(self) -> mujoco.MjModel:
        world = World.model_validate(
            {
                "name": "ik_arm",
                "floor": {"style": "concrete"},
                "robots": [
                    {"name": "ours", "template": "arm", "params": {"dof": 6, "color": "orange"}, "pos": [0, -LANE]},
                    {"name": "piper", "library": "piper", "pos": [0, LANE]},
                ],
            }
        )
        return compile_world(world).model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.follow = IKFollow(self, model, data, ctx)
        self.bases = {name: [*map(float, arm.base_xy), 0.0] for name, arm in self.follow.arms.items()}
        ctx.results.setdefault("ik", {"reached": 0})
        return {"lesson": self.follow}

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control in ("arm", "down", "damping") and self.follow is not None:
            self.follow.last_reached = None  # the same spot counts again under the new setting

    def note(
        self, arm: str, target: np.ndarray, error: float, w: float, step: float, down: bool, damping: float
    ) -> None:
        """What the controller just did, for the step checks (called every control tick)."""
        ctx, follow = self.ctx, self.follow
        assert ctx is not None and follow is not None
        if error < REACHED and (follow.last_reached is None or np.linalg.norm(target - follow.last_reached) > MOVED):
            follow.last_reached = target.copy()
            ctx.results["ik"] = {"reached": ctx.results.get("ik", {}).get("reached", 0) + 1}
            if down:
                ctx.events.add("down_reached")
            radius = float(np.linalg.norm(target[:2] - np.asarray(self.bases[arm][:2])))
            if arm == "piper" and radius > 0.5:
                ctx.events.add("piper_far")
        if w < EDGE:
            ctx.events.add("edge")
            if step > FLAIL and damping <= 0.001:
                ctx.events.add("flail")
            if damping >= 0.05 and step < CALM and error > 0.03:  # pulled past reach, yet calm
                ctx.events.add("calm_edge")

    def drain_samples(self) -> list[dict[str, float]]:
        if self.follow is None:
            return []
        with self.playground.session.lock:
            out = list(self.follow.samples)
            self.follow.samples.clear()
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        follow = self.follow
        assert follow is not None
        name = follow.active()
        arm = follow.arms[name]
        w, U, S = follow.manipulability(name)
        hand = data.site_xpos[arm.site]
        w_rel = 100 * w / follow.w_home[name]
        return {
            "arm": name,
            "hand": [round(float(x), 4) for x in hand],
            "target": [round(float(x), 4) for x in follow.target(name)],
            "ellipsoid": {"axes": np.round(U, 4).tolist(), "radii": [round(float(s) * ELLIPSOID_SCALE, 4) for s in S]},
            "readout": _text(
                f"误差 {1000 * follow.error:.0f} mm  可操作度 {w_rel:.0f}  关节一步 {follow.step_deg:.1f}°",
                f"error {1000 * follow.error:.0f} mm  manipulability {w_rel:.0f}  joint step {follow.step_deg:.1f}°",
            ),
        }

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        tags = {
            "ours": (_text("我们的臂", "Our arm"), "amber"),
            "piper": (_text("AgileX PiPER", "AgileX PiPER"), "blue"),
        }
        lanes = {"ours": -LANE, "piper": LANE}
        return {
            "kind": "arm3d",
            "bases": {name: [0.0, lanes[name], 0.0] for name in ARMS},
            "entrants": [
                {
                    "name": name,
                    "label": tags[name][0],
                    "color": tags[name][1],
                    "real": name == "piper",
                    "base": [0.0, lanes[name], 0.0],
                    "top": 0.75,
                    "cloud": [],
                }
                for name in ARMS
            ],
            "targets": [],
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [1.8, -1.9, 1.2], "target": [0.2, -0.15, 0.3]}  # our arm, with the PiPER beside it
