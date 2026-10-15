"""Lesson: inverse kinematics on a two-link arm.

Forward kinematics first (joint angles in, hand position out), then the analytic solution with the law
of cosines, the workspace, the two elbow solutions, and finally iterative IK with the Jacobian, the method
that works for any arm.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import mujoco
import numpy as np

from robotforge.learn.engine import Control, Lesson, LessonContext, Plot, Quiz, Series, Step

SHOULDER = (0.0, -0.05, 0.5)  # world position of the shoulder joint, m
L1, L2 = 0.30, 0.24
REACH = L1 + L2
INNER = abs(L1 - L2)
START_TARGET = (0.30, 0.20)  # in the arm's plane, relative to the shoulder: x right, z up
NEAR = 0.02  # m: "the hand is at the target"
SAMPLE_DT = 0.02


def forward(q1: float, q2: float) -> tuple[np.ndarray, np.ndarray]:
    """Elbow and hand positions in the arm's plane, relative to the shoulder."""
    elbow = np.array([L1 * math.cos(q1), L1 * math.sin(q1)])
    hand = elbow + np.array([L2 * math.cos(q1 + q2), L2 * math.sin(q1 + q2)])
    return elbow, hand


def analytic(x: float, z: float, elbow_up: bool) -> tuple[float, float]:
    """Law of cosines. Out of reach, the arm points straight at the target."""
    d = min(max(math.hypot(x, z), INNER + 1e-3), REACH - 1e-4)
    c2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    q2 = math.acos(max(-1.0, min(1.0, c2)))
    if elbow_up:
        q2 = -q2
    q1 = math.atan2(z, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return q1, q2


def jacobian(q1: float, q2: float) -> np.ndarray:
    s1, c1, s12, c12 = math.sin(q1), math.cos(q1), math.sin(q1 + q2), math.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def dls_step(q: np.ndarray, target: np.ndarray, gain: float = 0.6, damping: float = 0.05) -> np.ndarray:
    """One damped least-squares step: dq = J^T (J J^T + lambda^2 I)^-1 e, scaled by `gain`."""
    _, hand = forward(*q)
    e = target - hand
    J = jacobian(*q)
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(2), e)
    return q + gain * dq


class PlanarIK:
    rate_hz = 100.0

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> None:
        self.ctx = ctx
        self.qadr = [int(model.joint(n).qposadr[0]) for n in ("shoulder", "elbow")]
        self.acts = [model.actuator(n).id for n in ("shoulder", "elbow")]
        self.q_cmd = np.array([data.qpos[a] for a in self.qadr])
        self.method = -1
        self.pending = 0  # iterative steps asked for and not done yet
        self.iterations = 0
        self.next_sample = 0.0
        self.samples: deque[dict[str, float]] = deque(maxlen=1024)

    def _fresh_target(self) -> None:
        """Iterative IK needs an error to close: swing the target 100 degrees around the shoulder."""
        _, hand = forward(*self.q_cmd)
        angle = math.atan2(hand[1], hand[0]) + math.radians(100)
        radius = min(max(float(np.linalg.norm(hand)), 0.22), 0.45)
        tx, tz = radius * math.cos(angle), max(radius * math.sin(angle), -0.3)
        self.ctx.values["tx"], self.ctx.values["tz"] = round(tx, 3), round(tz, 3)

    def target(self) -> np.ndarray:
        return np.array([self.ctx.values["tx"], self.ctx.values["tz"]])

    def actual(self, data: mujoco.MjData) -> np.ndarray:
        return np.array([float(data.qpos[a]) for a in self.qadr])

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        v = self.ctx.values
        method = int(v["method"])
        if method != self.method:  # switching methods starts from where the arm is now
            self.method = method
            self.q_cmd = self.actual(data)
            self.iterations = 0
            self.pending = 0
            if method == 2:
                self._fresh_target()
        if method == 0:
            self.q_cmd = np.radians([v["theta1"], v["theta2"]])
        elif method == 1:
            self.q_cmd = np.array(analytic(v["tx"], v["tz"], bool(v["elbow_up"])))
        elif self.pending > 0:
            self.pending -= 1
            self.iterations += 1
            self.q_cmd = dls_step(self.q_cmd, self.target())
        for act, q in zip(self.acts, self.q_cmd, strict=True):
            lo, hi = model.actuator_ctrlrange[act]
            data.ctrl[act] = float(np.clip(q, lo, hi))
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            q = self.actual(data)
            _, hand = forward(*q)
            self.samples.append(
                {
                    "t": data.time,
                    "err": float(np.linalg.norm(self.target() - hand) * 100),
                    "q1": math.degrees(q[0]),
                    "q2": math.degrees(q[1]),
                }
            )

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-ik", "method": self.method, "iterations": self.iterations}


def _hand_error(ctx: LessonContext) -> float:
    return ctx.samples[-1]["err"] / 100 if ctx.samples else 1.0


def _fk_reached(ctx: LessonContext) -> bool:
    return int(ctx.values["method"]) == 0 and _hand_error(ctx) < 0.03


def _followed(ctx: LessonContext) -> bool:
    return int(ctx.values["method"]) == 1 and len(ctx.results.get("visited", [])) >= 3


def _went_outside(ctx: LessonContext) -> bool:
    return "outside" in ctx.events


def _both_elbows(ctx: LessonContext) -> bool:
    return {"elbow:0", "elbow:1"} <= ctx.events


def _iterated(ctx: LessonContext) -> bool:
    it = ctx.results.get("iterate", {})
    return int(ctx.values["method"]) == 2 and it.get("n", 0) >= 3 and _hand_error(ctx) < 0.01


STEPS = (
    Step(
        id="fk",
        title={"zh": "正运动学：关节角决定手在哪", "en": "Forward kinematics: angles decide where the hand is"},
        body={
            "zh": "这只手臂有两个关节：肩 θ1 和肘 θ2。知道这两个角，就能算出手的位置：\n"
            "x = L1·cos θ1 + L2·cos(θ1+θ2)，z = L1·sin θ1 + L2·sin(θ1+θ2)\n"
            "这叫正运动学。现在只用两个角度滑块，把手移进琥珀色的目标点。是不是有点别扭？",
            "en": "This arm has two joints: the shoulder theta1 and the elbow theta2. Knowing both angles gives "
            "the hand's position:\n"
            "x = L1 cos(theta1) + L2 cos(theta1 + theta2), z = L1 sin(theta1) + L2 sin(theta1 + theta2)\n"
            "That is forward kinematics. Using only the two angle sliders, put the hand on the amber target. "
            "A bit awkward, isn't it?",
        },
        done={
            "zh": "用角度去凑位置很别扭：动一个关节，手会沿着弧线跑。我们真正想说的是“手去那里”，这就需要反过来算。",
            "en": "Steering a position with angles is clumsy: each joint swings the hand along an arc. "
            "What we want to say is 'hand, go there', which needs the calculation the other way round.",
        },
        check=_fk_reached,
        unlock=("theta1", "theta2"),
        show=("target", "angles"),
    ),
    Step(
        id="ik",
        title={"zh": "逆运动学：从目标反推关节角", "en": "Inverse kinematics: from the target back to angles"},
        body={
            "zh": "把方法切到「解析 IK」，然后拖动琥珀色目标点，手会自己跟过去。\n"
            "它是怎么算的：肩、肘、手组成一个三角形，三条边是 L1、L2 和肩到目标的距离 d。\n"
            "余弦定理给出肘角：cos θ2 = (d² − L1² − L2²) / (2·L1·L2)，再由 θ2 算出 θ1。\n"
            "把目标拖到三个不同的地方试试。",
            "en": "Switch the method to Analytic IK, then drag the amber target: the hand follows.\n"
            "How: shoulder, elbow and hand form a triangle with sides L1, L2 and d, the distance to the target.\n"
            "The law of cosines gives the elbow: cos(theta2) = (d^2 - L1^2 - L2^2) / (2 L1 L2), "
            "and theta1 follows from theta2.\n"
            "Drag the target to three different places.",
        },
        done={
            "zh": "这就是逆运动学 (IK)：给出手要去的位置，算出每个关节该转多少。机械臂抓东西、四足落脚，都靠它。",
            "en": "That is inverse kinematics (IK): from where the hand should go to how far each joint turns. "
            "Arms grasping and legs stepping all rely on it.",
        },
        check=_followed,
        unlock=("method", "tx", "tz"),
        show=("triangle",),
    ),
    Step(
        id="workspace",
        title={"zh": "工作空间：哪里够得着", "en": "The workspace: what it can reach"},
        body={
            "zh": "浅色圆环是手能到达的地方：最远 L1+L2，最近 |L1−L2|。\n把目标拖到圆环外面，看看手臂怎么办。",
            "en": "The shaded ring is everywhere the hand can reach: at most L1 + L2 away, at least |L1 - L2|.\n"
            "Drag the target outside the ring and see what the arm does.",
        },
        done={
            "zh": "目标在工作空间外就没有解，余弦定理里的 cos θ2 会超过 1。IK 只能退而求其次：把手臂伸直，指向目标。",
            "en": "Outside the workspace there is no solution: cos(theta2) would have to exceed 1. "
            "The best IK can do is stretch the arm straight toward the target.",
        },
        check=_went_outside,
        show=("workspace",),
    ),
    Step(
        id="elbow",
        title={"zh": "两个解：肘朝上还是朝下", "en": "Two answers: elbow up or elbow down"},
        body={
            "zh": "θ2 = ± arccos(…)：同一个目标点通常有两个解。虚线是另一个解。\n"
            "把目标放回圆环里，切换「肘部朝上」，看手臂换一种姿势够到同一点。",
            "en": "theta2 = +- arccos(...): the same target usually has two solutions. "
            "The dashed arm is the other one.\n"
            "Put the target back inside the ring and flip Elbow up: the arm reaches the same point another way.",
        },
        done={
            "zh": "真实机器人要从多个解里挑一个：避开障碍、离关节极限远一点、转动最少。这也是 IK 的一部分。",
            "en": "Real robots pick one of several solutions: one that avoids obstacles, stays away from joint limits, "
            "or moves the least. Choosing is part of IK too.",
        },
        check=_both_elbows,
        unlock=("elbow_up",),
        show=("other",),
    ),
    Step(
        id="iterate",
        title={"zh": "迭代 IK：用雅可比一步步逼近", "en": "Iterative IK: closing in with the Jacobian"},
        body={
            "zh": "6 个关节的机械臂没有这么简单的公式。通用办法是迭代：\n"
            "看一眼误差 e（手到目标的蓝色箭头），用雅可比矩阵 J 算出每个关节该转多少才能把误差缩小：\n"
            "Δq = Jᵀ(JJᵀ + λ²I)⁻¹·e\n"
            "把方法切到「迭代 IK」（目标会跳到一个新位置），一次次点「迭代一步」，"
            "看误差曲线怎么下降，直到小于 1 厘米。",
            "en": "A six-joint arm has no formula this simple. The general method is to iterate:\n"
            "look at the error e (the blue arrow from hand to target), and use the Jacobian J to work out how much "
            "each joint should turn to shrink it: dq = J^T (J J^T + lambda^2 I)^-1 e.\n"
            "Switch the method to Iterative IK (the target jumps somewhere new) and press Step a few times. "
            "Watch the error fall below 1 cm.",
        },
        done={
            "zh": "每一步只是“往对的方向挪一点”，几步之后就到了。"
            "λ 是阻尼，防止手臂伸直时一步迈得太大。机器人里的 IK 求解器大多是这样工作的。",
            "en": "Each step only nudges the arm the right way, and a few steps get there. Lambda damps the step so it "
            "does not jump when the arm is nearly straight. Most IK solvers in robots work like this.",
        },
        check=_iterated,
        unlock=("iterate",),
        show=("error",),
    ),
    Step(
        id="why_iterate",
        title={"zh": "想一想", "en": "Think about it"},
        body={
            "zh": "为什么工业机械臂和人形机器人一般用迭代法，而不是解析公式？",
            "en": "Why do industrial arms and humanoids usually use the iterative method rather than a formula?",
        },
        done={
            "zh": "你已经会了运动学的两个方向：正运动学算手在哪，逆运动学算关节怎么转。"
            "下一课会用它让机械臂真正去抓东西。",
            "en": "You now know kinematics both ways: forward tells where the hand is, inverse tells how to turn the "
            "joints. Next, use it to make an arm really pick things up.",
        },
        quiz=Quiz(
            question={"zh": "选一个最好的理由", "en": "Pick the best reason"},
            options=(
                {"zh": "迭代法算得更精确", "en": "Iteration is more precise"},
                {
                    "zh": "关节一多就没有通用公式，而迭代法对任何结构都能用，还能加限制条件",
                    "en": "With many joints there is no general formula, while iteration works for any arm "
                    "and can respect extra limits",
                },
                {"zh": "迭代法不需要知道连杆长度", "en": "Iteration does not need the link lengths"},
            ),
            answer=1,
            explain={
                "zh": "对：解析解只对特定结构存在；迭代法只要能算雅可比就能用，还能顺便避开关节极限。",
                "en": "Right: formulas exist only for particular arms; iteration only needs the Jacobian, "
                "and can keep joints inside their limits along the way.",
            },
        ),
    ),
)


class IKLesson(Lesson):
    id = "inverse_kinematics"
    chapter = {"zh": "运动学", "en": "Kinematics"}
    title = {"zh": "逆运动学（一）：手怎么够到目标", "en": "Inverse kinematics I: reaching a target"}
    summary = {
        "zh": "从关节角到手的位置，再反过来：余弦定理、工作空间、两个解和雅可比迭代。",
        "en": "From joint angles to hand position and back: law of cosines, workspace, two solutions, Jacobian steps.",
    }
    minutes = 12
    controls = (
        Control(
            "method",
            "choice",
            {"zh": "方法", "en": "Method"},
            options=(
                {"zh": "关节角", "en": "Joint angles"},
                {"zh": "解析 IK", "en": "Analytic IK"},
                {"zh": "迭代 IK", "en": "Iterative IK"},
            ),
        ),
        Control("theta1", "slider", {"zh": "肩 θ1", "en": "Shoulder theta1"}, -90, 180, 1, 10, "°"),
        Control("theta2", "slider", {"zh": "肘 θ2", "en": "Elbow theta2"}, -150, 150, 1, 60, "°"),
        Control("tx", "slider", {"zh": "目标 x", "en": "Target x"}, -0.6, 0.6, 0.01, START_TARGET[0], " m"),
        Control("tz", "slider", {"zh": "目标 z", "en": "Target z"}, -0.45, 0.6, 0.01, START_TARGET[1], " m"),
        Control("elbow_up", "toggle", {"zh": "肘部朝上", "en": "Elbow up"}, value=1),
        Control("iterate", "button", {"zh": "迭代一步", "en": "Step"}),
    )
    plots = (
        Plot(
            "error",
            {"zh": "手到目标的距离", "en": "Hand to target"},
            "cm",
            (Series("err", {"zh": "误差", "en": "error"}, "blue"),),
        ),
        Plot(
            "angles",
            {"zh": "关节角", "en": "Joint angles"},
            "°",
            (
                Series("q1", {"zh": "肩 θ1", "en": "theta1"}, "ink"),
                Series("q2", {"zh": "肘 θ2", "en": "theta2"}, "violet"),
            ),
        ),
    )
    steps = STEPS

    def __init__(self) -> None:
        self.ik: PlanarIK | None = None
        self.visited: list[tuple[float, float]] = []
        self.outside_since: float | None = None

    def build(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.modelname = "ik_lesson"
        spec.compiler.degree = False  # joint ranges below are in radians
        spec.option.timestep = 0.002
        # Servo damping on a light forearm is stiff for explicit Euler; implicit integration keeps it stable.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        G = mujoco.mjtGeom
        w = spec.worldbody
        w.add_light(pos=[0.5, -1.0, 2.5], dir=[-0.2, 0.4, -1.0], diffuse=[0.8, 0.8, 0.8])
        spec.add_material(name="rf_floor", rgba=[0.62, 0.66, 0.7, 1])
        for name in ("rf_dark", "rf_metal", "rf_shell"):
            spec.add_material(name=name)
        w.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[3, 3, 0.05], material="rf_floor")
        sx, sy, sz = SHOULDER
        w.add_geom(
            name="foot",
            type=G.mjGEOM_CYLINDER,
            size=[0.16, 0.02, 0],
            pos=[sx, 0.03, 0.02],
            rgba=[0.16, 0.18, 0.21, 1],
            material="rf_dark",
        )
        w.add_geom(
            name="pillar",
            type=G.mjGEOM_BOX,
            size=[0.035, 0.035, sz / 2],
            pos=[sx, 0.03, sz / 2],
            rgba=[0.78, 0.8, 0.83, 1],
            material="rf_metal",
        )
        dark, shell, metal = [0.16, 0.18, 0.21, 1], [0.95, 0.96, 0.97, 1], [0.85, 0.87, 0.9, 1]
        side = [0.7071068, 0.7071068, 0, 0]  # cylinders lying along y, facing the camera
        upper = w.add_body(name="upper", pos=list(SHOULDER), gravcomp=1)
        upper.add_joint(
            name="shoulder",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, -1, 0],
            damping=0.05,
            armature=0.01,
            range=[-2.1, 3.5],
            limited=True,
        )
        upper.add_geom(
            name="shoulder_motor",
            type=G.mjGEOM_CYLINDER,
            size=[0.045, 0.03, 0],
            quat=side,
            rgba=dark,
            material="rf_dark",
            contype=0,
            conaffinity=0,
            mass=0.05,
        )
        upper.add_geom(
            name="upper_link",
            type=G.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, L1, 0, 0],
            size=[0.02, 0, 0],
            rgba=shell,
            material="rf_shell",
            contype=0,
            conaffinity=0,
            mass=0.15,
        )
        fore = upper.add_body(name="fore", pos=[L1, 0, 0], gravcomp=1)
        fore.add_joint(
            name="elbow",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, -1, 0],
            damping=0.05,
            armature=0.01,
            range=[-2.7, 2.7],
            limited=True,
        )
        fore.add_geom(
            name="elbow_motor",
            type=G.mjGEOM_CYLINDER,
            size=[0.035, 0.028, 0],
            quat=side,
            rgba=dark,
            material="rf_dark",
            contype=0,
            conaffinity=0,
            mass=0.04,
        )
        fore.add_geom(
            name="fore_link",
            type=G.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, L2, 0, 0],
            size=[0.016, 0, 0],
            rgba=shell,
            material="rf_shell",
            contype=0,
            conaffinity=0,
            mass=0.1,
        )
        fore.add_geom(
            name="hand",
            type=G.mjGEOM_SPHERE,
            size=[0.022, 0, 0],
            pos=[L2, 0, 0],
            rgba=metal,
            material="rf_metal",
            contype=0,
            conaffinity=0,
            mass=0.03,
        )
        fore.add_site(name="hand", pos=[L2, 0, 0], size=[0.01, 0, 0])
        for name, lo, hi in (("shoulder", -2.1, 3.5), ("elbow", -2.7, 2.7)):
            servo = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            servo.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            servo.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            servo.gainprm[0] = 30.0
            servo.biasprm[:3] = [0.0, -30.0, -2.0]
            servo.ctrllimited = True
            servo.ctrlrange = [lo, hi]
        spec.add_key(name="home", qpos=[math.radians(10), math.radians(60)])
        return spec.compile()

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        if data.time == 0 and model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
        self.ik = PlanarIK(model, data, ctx)
        return {"lesson": self.ik}

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        assert self.ik is not None
        if control == "iterate":
            if int(ctx.values["method"]) != 2:
                ctx.values["method"] = 2.0  # stepping only makes sense in the iterative method
            self.ik.pending += 1

    def drain_samples(self) -> list[dict[str, float]]:
        if self.ik is None:
            return []
        out = list(self.ik.samples)
        self.ik.samples.clear()
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        assert self.ik is not None
        q = self.ik.actual(data)
        elbow, hand = forward(*q)
        target = self.ik.target()
        err = float(np.linalg.norm(target - hand))
        d = float(np.linalg.norm(target))
        self._track(ctx, data.time, target, err, d)
        other = analytic(target[0], target[1], not bool(ctx.values["elbow_up"]))
        other_elbow, other_hand = forward(*other)
        return {
            "q1": round(math.degrees(q[0]), 1),
            "q2": round(math.degrees(q[1]), 1),
            "target": [round(float(x), 4) for x in target],
            "d": round(d, 3),
            "reachable": INNER <= d <= REACH,
            "err_cm": round(err * 100, 2),
            "other_elbow": [round(float(x), 4) for x in other_elbow],
            "other_hand": [round(float(x), 4) for x in other_hand],
            "iterations": self.ik.iterations,
        }

    def _track(self, ctx: LessonContext, t: float, target: np.ndarray, err: float, d: float) -> None:
        """Remember what the learner has shown so far: targets reached, the outside, both elbows."""
        method = int(ctx.values["method"])
        if method == 1 and err < NEAR:
            point = (float(target[0]), float(target[1]))
            if all(math.dist(point, p) >= 0.1 for p in self.visited):
                self.visited.append(point)
                ctx.results["visited"] = [list(p) for p in self.visited]
            ctx.events.add(f"elbow:{int(ctx.values['elbow_up'])}")
        if method == 1 and d > REACH + 0.02:
            if self.outside_since is None:
                self.outside_since = t
            elif t - self.outside_since > 0.5:
                ctx.events.add("outside")
        else:
            self.outside_since = None
        if method == 2:
            ctx.results["iterate"] = {"n": self.ik.iterations if self.ik else 0, "err_cm": round(err * 100, 2)}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "planar_arm",
            "shoulder": list(SHOULDER),
            "x_axis": [1, 0, 0],
            "z_axis": [0, 0, 1],
            "l1": L1,
            "l2": L2,
            "upper": int(model.body("upper").id),
            "fore": int(model.body("fore").id),
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [0.12, -1.75, 0.68], "target": [0.1, 0.0, 0.55]}
