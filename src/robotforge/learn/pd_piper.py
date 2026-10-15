"""PID II: the same loop on a real arm's shoulder, the AgileX PiPER.

The PiPER comes from MuJoCo Menagerie. Its shoulder (joint2) is handed to the learner's PID as a plain
torque motor; the other joints keep the model's own position servos at the home pose. Two things the file
does for convenience are undone here, because a real arm has neither: `gravcomp` (MuJoCo cancelling
gravity on every link) is switched off, and the gripper holds a box whose load the controller's model
does not know about.

The story: gains tuned on our small arm sag on the PiPER (more gravity), overshoot once gravity is
compensated (more inertia), and come right when the damping ratio zeta = Kd / (2 sqrt(Kp I)) is matched,
the one thing that carries over between robots. The unknown load is left for the integral.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import mujoco
import numpy as np

from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Control, LessonContext, Quiz, Step, Sweep
from robotforge.learn.pd import ANGLE_PLOTS, JointLesson, JointPD, _runs, _text, step_bench

ROBOT = "piper"
JOINT = f"{ROBOT}/joint2"  # the shoulder
TOOL_BODY = f"{ROBOT}/link6"
PAYLOAD = 0.5  # kg in the box when it is full
EMPTY = 1e-4  # kg: an empty box
START = 60.0  # degrees in the lesson's frame (180 - joint2): the arm leans forward...
TARGET = 100.0  # ...and lifts to here
SIGN, OFFSET = -1.0, math.pi  # lesson angle = pi - joint2, so lifting is a bigger angle
SMALL_ARM_INERTIA = 0.076  # kg*m^2 about the small arm's shoulder (PID I), for the comparison
HOLD_STIFFER = 5.0  # the held joints' servo stiffness, times the file's


class ModelGravity:
    """What the controller believes gravity takes: the same robot with an empty box. A real controller
    has a model of the arm, not of whatever it happens to be holding."""

    def __init__(self, model: mujoco.MjModel, dof: int, payload_body: int) -> None:
        self.model = copy.deepcopy(model)
        self.model.body_mass[payload_body] = EMPTY
        self.data = mujoco.MjData(self.model)
        self.dof = dof
        self.out = np.zeros((self.model.nv, 1))

    def __call__(self, data: mujoco.MjData) -> float:
        m, d = self.model, self.data
        d.qpos[:] = data.qpos  # qvel stays zero: gravity only
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, self.out)
        return SIGN * float(self.out[self.dof, 0])


def _inertia(model: mujoco.MjModel, data: mujoco.MjData, dof: int) -> float:
    """The shoulder's moment of inertia in the current pose: the diagonal of the joint-space mass matrix."""
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    return float(full[dof, dof])


def _zeta(kp: float, kd: float, inertia: float) -> float | None:
    return kd / (2 * math.sqrt(kp * inertia)) if kp > 0 and inertia > 0 else None


def _plain(ctx: LessonContext) -> bool:
    return bool(_runs(ctx, gravity=lambda x: x == 0, ki=lambda x: x == 0, kp=lambda x: x > 0))


def _compensated(ctx: LessonContext) -> bool:
    return bool(_runs(ctx, gravity=lambda x: x > 0, ki=lambda x: x == 0, payload=lambda x: x == 0))


def _matched(ctx: LessonContext) -> bool:
    runs = _runs(ctx, gravity=lambda x: x > 0, kp=lambda x: x >= 60, payload=lambda x: x == 0)
    return any(
        r["metrics"]["overshoot_pct"] <= 5 and r["metrics"]["settle_s"] is not None and r["metrics"]["settle_s"] <= 0.4
        for r in runs
    )


def _loaded(ctx: LessonContext) -> bool:
    runs = _runs(ctx, gravity=lambda x: x > 0, ki=lambda x: x > 0, payload=lambda x: x > 0)
    return any(r["metrics"]["settle_s"] is not None and (r["metrics"]["end_err"] or 99) <= 0.5 for r in runs)


def _challenge(ctx: LessonContext) -> bool:
    test = ctx.results.get("test", {})
    values = ctx.values
    return test.get("stars", 0) >= 3 and bool(values.get("payload")) and not values.get("gravity")


STEPS = (
    Step(
        id="copy",
        title=_text("照搬小臂的增益", "The small arm's gains, copied over"),
        body=_text(
            "这是 AgileX PiPER 的真实模型（来自 MuJoCo Menagerie）。只有肩关节交给你的 PID，其他关节被按在原位。"
            "增益先照搬上一课小臂上调好的 Kp 40、Kd 3。测试：手臂从 60° 往上抬到 100°。\n"
            "先猜，再点「试一次」。",
            "This is the AgileX PiPER's real model (from MuJoCo Menagerie). Only the shoulder is driven by your "
            "PID; the other joints are held where they are. The gains start as the ones that worked on the "
            "small arm in the last lesson: Kp 40, Kd 3. The test lifts the arm from 60 to 100 degrees.\n"
            "Guess first, then press Run test.",
        ),
        done=_text(
            "最后差了 8.7°，小臂上只差 2.7°。PiPER 的肩要托起整条臂和 1.2 kg 的夹爪，重力要 6 N·m 左右"
            "（看公式），小臂只要 1.8。e = 重力 ÷ Kp，要的力矩大了三倍，误差也就大了三倍。\n"
            "顺便一提：Menagerie 的文件给每个连杆开了 gravcomp，让仿真直接把重力抵消掉。这一课把它关了，"
            "跟真机一样，重力得由电机自己扛。",
            "It ends 8.7 degrees short; the small arm was only 2.7 short. The PiPER's shoulder holds up the "
            "whole arm and a 1.2 kg gripper: gravity needs about 6 N*m (see the formula), against 1.8 on the "
            "small arm. e = gravity / Kp, so three times the torque means three times the error.\n"
            "By the way: the Menagerie file switches on gravcomp for every link, so the simulation simply "
            "cancels gravity. This lesson turns it off; like on the real arm, the motor has to carry it.",
        ),
        quiz=Quiz(
            question=_text(
                "小臂上 0.4 秒到位的增益，放到 PiPER 上会怎样？",
                "Gains that settle in 0.4 s on the small arm. On the PiPER?",
            ),
            options=(
                _text("一样好，0.4 秒到位", "Just as good: on target in 0.4 s"),
                _text("停在目标下面好几度", "It stops several degrees short"),
                _text("来回晃个不停", "It keeps wobbling"),
            ),
            answer=1,
            explain=_text(
                "PiPER 重得多，P 项要更大的误差才托得住。",
                "The PiPER is much heavier, so P needs a bigger error to hold it.",
            ),
            predict=True,
        ),
        check=_plain,
        unlock=("kp", "kd", "test"),
        show=("scale", "target", "error", "torque", "f_p", "f_d", "f_g"),
    ),
    Step(
        id="ff",
        title=_text("打开重力补偿", "Switch on gravity compensation"),
        body=_text(
            "PiPER 的模型很准，那就让控制器用它把重力那份提前算好加上。增益还是 Kp 40、Kd 3。先猜，再试一次。",
            "The PiPER's model is accurate, so let the controller work out gravity's share from it and add it "
            "up front. Gains stay at Kp 40, Kd 3. Guess, then run the test again.",
        ),
        done=_text(
            "误差没了，却冲过头四分之一（我们测到 25%）。之前是重力一直往回拽，看上去才没冲。"
            "在小臂上刚好刹住的 Kd 3，到 PiPER 上刹不住了。为什么？下一步看。",
            "The error is gone, but it overshoots by a quarter (25% in our runs). Before, gravity kept dragging it "
            "back, which hid the overshoot. The Kd 3 that braked the small arm just right is not enough for "
            "the PiPER. Why? Next step.",
        ),
        quiz=Quiz(
            question=_text("开了前馈，Kp 40、Kd 3 会怎样？", "With feedforward on, what do Kp 40 and Kd 3 do?"),
            options=(
                _text("稳稳到位，和小臂一样", "Settle neatly, like the small arm"),
                _text("到位了，但会冲过头", "Reach the target but overshoot"),
                _text("还是差好几度", "Still end several degrees short"),
            ),
            answer=1,
            explain=_text(
                "重力那份补上了，剩下的是一个刹车不够的弹簧。",
                "Gravity is taken care of; what is left is a spring with too little brake.",
            ),
            predict=True,
        ),
        check=_compensated,
        unlock=("gravity",),
        show=("f_i",),
    ),
    Step(
        id="zeta",
        title=_text("阻尼比 ζ：能搬到别的机器人上的那个数", "The damping ratio: the number that carries over"),
        body=_text(
            "刹车够不够，要拿 Kd 和手臂有多“沉”来比：ζ = Kd ÷ (2√(Kp × I))，I 是肩关节的转动惯量，公式里实时算。"
            "ζ 接近 1 刚好不冲；小于 1 会冲，大于 1 偏慢。\n"
            f"小臂的 I 约 {SMALL_ARM_INERTIA:g}，Kp 40、Kd 3 时 ζ 是 0.86；"
            "PiPER 的肩 I 约 0.37，同样的增益 ζ 只有 0.39。\n"
            "把 Kp 调到 80，再调 Kd 让 ζ 接近 1，试一次（也可以先扫一遍 Kd）。",
            "Whether the brake is enough depends on Kd against how heavy the arm is to turn: zeta = Kd / "
            "(2 sqrt(Kp * I)), where I is the shoulder's moment of inertia, worked out live in the formula. "
            "Zeta near 1 just avoids overshoot; below 1 overshoots, above 1 is slow.\n"
            f"The small arm's I is about {SMALL_ARM_INERTIA:g}: Kp 40 and Kd 3 give zeta 0.86. The PiPER's "
            "shoulder has I of about 0.37, so the same gains give only 0.39.\n"
            "Set Kp to 80, then pick Kd so zeta is close to 1 and run the test (or sweep Kd first).",
        ),
        done=_text(
            "ζ 接近 1，0.3 秒就稳稳到位。增益本身搬不到别的机器人上，阻尼比可以：先估出惯量，再按 ζ 定 Kd。\n"
            "Menagerie 文件里 PiPER 肩的位置伺服是 Kp 80、Kd 5，ζ 约 0.45，"
            "在开着 gravcomp 的仿真里同样的测试会冲过头 15%。",
            "With zeta near 1 it settles neatly in 0.3 s. Gains do not carry over between robots, but the "
            "damping ratio does: estimate the inertia, then choose Kd for the zeta you want.\n"
            "The Menagerie file's own shoulder servo is Kp 80, Kd 5: zeta about 0.45. In the simulation with "
            "gravcomp on, the same test overshoots by 15%.",
        ),
        check=_matched,
        show=("f_z",),
    ),
    Step(
        id="payload",
        title=_text("手里拿着东西", "Holding something"),
        body=_text(
            f"打开「盒子装满（{PAYLOAD:g} kg）」。控制器用的模型以为盒子是空的。前馈开着，先猜，再试一次。\n"
            "然后把 Ki 调到 20 左右再试。",
            f"Switch on Box full ({PAYLOAD:g} kg). The controller's model thinks the box is empty. With "
            "feedforward on, guess, then run the test.\n"
            "Then set Ki to about 20 and run it again.",
        ),
        done=_text(
            "只靠前馈差了约 2°：模型不知道盒子的重量。加上积分，剩下的误差被慢慢吃掉。"
            "前馈管算得出来的部分，积分管模型不知道的部分：负载、摩擦、磨损。工厂里的机械臂两个一起用。",
            "Feedforward alone ends about 2 degrees short: the model does not know the box's weight. With the "
            "integral the rest is worn away. Feedforward covers what can be worked out; the integral covers what "
            "the model does not know: loads, friction, wear. Factory arms use both.",
        ),
        quiz=Quiz(
            question=_text("盒子装满，只靠前馈会怎样？", "Box full, feedforward only: what happens?"),
            options=(
                _text("前馈会自动算上盒子", "The feedforward allows for the box by itself"),
                _text("又会差一点", "It ends a little short again"),
                _text("开始来回晃", "It starts to wobble"),
            ),
            answer=1,
            explain=_text("前馈只知道模型里有的东西。", "Feedforward only knows what is in the model."),
            predict=True,
        ),
        check=_loaded,
        unlock=("payload", "ki"),
    ),
    Step(
        id="challenge",
        title=_text("挑战：没有模型，满载三颗星", "Challenge: no model, fully loaded, three stars"),
        body=_text(
            "假如手边没有准确的模型：关掉重力补偿，盒子装满，只靠 PID 拿到三颗星"
            "（1 秒内稳定在 ±1° 以内，超调不超过 5%）。\n"
            "提示：重力和盒子都得靠积分去抵，Kp、Kd 也可以更大，用 ζ 管住超调。目标角度也解锁了。",
            "Suppose there is no accurate model at hand: switch gravity compensation off, keep the box full, and "
            "earn three stars with PID alone (within +-1 degree in 1 s, overshooting by at most 5%).\n"
            "Hint: the integral now has to carry gravity and the box; Kp and Kd can go higher too, with zeta "
            "keeping the overshoot in check. The target slider is unlocked.",
        ),
        done=_text(
            "没有模型也能做到，代价是更大的增益和积分（我们测到 Ki 要 60 以上，Ki 20 只有一颗星）。"
            "有模型时前馈把大头扛了，反馈只用收拾小误差，所以真机器人都尽量把模型做准。",
            "It can be done without a model, at the price of bigger gains and integral (in our runs Ki needed "
            "60 or more; Ki 20 earned one star). With a model the feedforward carries the bulk and feedback "
            "only tidies up small errors, which is why real robots work hard on accurate models.",
        ),
        check=_challenge,
        unlock=("target", "kick"),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text(
            "换一台转动惯量是 PiPER 四倍的大机械臂，Kp 也加到四倍。",
            "Picture a big arm whose shoulder is four times as heavy to turn as the PiPER's, "
            "with Kp also four times higher.",
        ),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("Kd 要变成几倍，ζ 才不变？", "How many times bigger must Kd be to keep the same zeta?"),
            options=(_text("2 倍", "2x"), _text("4 倍", "4x"), _text("16 倍", "16x")),
            answer=1,
            explain=_text(
                "ζ = Kd ÷ (2√(Kp × I))：Kp × I 变成 16 倍，开根号是 4 倍，所以 Kd 也要 4 倍。",
                "zeta = Kd / (2 sqrt(Kp * I)): Kp * I grows 16 times, its square root 4 times, so Kd must too.",
            ),
        ),
    ),
)


class PiperPIDLesson(JointLesson):
    id = "pid_piper"
    chapter = _text("控制原理", "Control")
    title = _text("PID（二）：在真机 PiPER 的肩关节上调", "PID II: tuning the real PiPER's shoulder")
    summary = _text(
        "小臂上的增益搬到 PiPER 上为什么不灵：重力、转动惯量和阻尼比 ζ，再加上模型不知道的负载。",
        "Why the small arm's gains fail on the PiPER: gravity, inertia and the damping ratio, "
        "plus a load the model does not know.",
    )
    minutes = 15
    test_start = START
    test_target = TARGET
    kick_speed = 2.0
    controls = (
        Control("target", "slider", _text("目标角度", "Target angle"), 30, 150, 1, TARGET, "°"),
        Control("kp", "slider", _text("Kp 比例", "Kp proportional"), 0, 200, 1, 40),
        Control("kd", "slider", _text("Kd 微分", "Kd derivative"), 0, 20, 0.1, 3),
        Control("ki", "slider", _text("Ki 积分", "Ki integral"), 0, 100, 1, 0),
        Control("gravity", "toggle", _text("重力补偿（前馈）", "Gravity compensation (feedforward)")),
        Control("payload", "toggle", _text(f"盒子装满（{PAYLOAD:g} kg）", f"Box full ({PAYLOAD:g} kg)")),
        Control("kick", "button", _text("推一下", "Push")),
        Control("test", "button", _text("试一次", "Run test")),
    )
    plots = ANGLE_PLOTS
    steps = STEPS
    bench = step_bench(
        (
            Sweep("kp", (20, 40, 60, 80, 100, 140, 200), ("overshoot_pct", "settle_s")),
            Sweep("kd", (2, 4, 6, 8, 10, 12, 15, 20), ("overshoot_pct", "settle_s")),
        ),
        TARGET,
        extra=(("payload", _text("盒子", "box")),),
    )

    def __init__(self) -> None:
        super().__init__()
        self.payload_body = -1
        self.dof = -1

    def build(self) -> mujoco.MjModel:
        world = World.model_validate(
            {"name": "pid_piper", "floor": {"style": "concrete"}, "robots": [{"name": ROBOT, "library": "piper"}]}
        )
        spec = compile_world(world).spec
        tcp = spec.site(f"{ROBOT}/tcp")
        box = spec.body(TOOL_BODY).add_body(name="payload", pos=list(tcp.pos))
        box.mass = EMPTY
        box.inertia = [2e-4, 2e-4, 2e-4]
        box.explicitinertial = True
        box.add_geom(
            name="payload_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.022, 0.022, 0.022],
            contype=0,
            conaffinity=0,
            rgba=[0.95, 0.66, 0.0, 1],
        )
        model = spec.compile()
        model.body_gravcomp[:] = 0.0  # a real arm gets no free gravity cancelling
        act = model.actuator(JOINT).id  # the shoulder's position servo becomes a plain torque motor
        model.actuator_gainprm[act, :] = 0.0
        model.actuator_gainprm[act, 0] = 1.0
        model.actuator_biasprm[act, :] = 0.0
        # The other joints' servos were tuned with gravity cancelled; stiffen them (same damping ratio) so
        # they hold the pose under real gravity and only the shoulder moves.
        for other in range(model.nu):
            if other != act and model.actuator_biastype[other] == mujoco.mjtBias.mjBIAS_AFFINE:
                model.actuator_gainprm[other, 0] *= HOLD_STIFFER
                model.actuator_biasprm[other, 1] *= HOLD_STIFFER
                model.actuator_biasprm[other, 2] *= math.sqrt(HOLD_STIFFER)
        limit = float(model.actuator_forcerange[act, 1])
        model.actuator_ctrlrange[act] = [-limit, limit]
        model.actuator_ctrllimited[act] = 1
        self.torque_max = limit
        self.payload_body = model.body("payload").id
        self.dof = int(model.joint(JOINT).dofadr[0])
        self.model = model
        return model

    def make_pd(self, model: mujoco.MjModel, ctx: LessonContext) -> JointPD:
        return JointPD(
            model,
            ctx,
            joint=JOINT,
            actuator=JOINT,
            torque_max=self.torque_max,
            sign=SIGN,
            offset=OFFSET,
            gravity=ModelGravity(model, self.dof, self.payload_body),
        )

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self._load(model, ctx)
        return super().attach(model, data, ctx)

    def _load(self, model: mujoco.MjModel, ctx: LessonContext) -> None:
        model.body_mass[self.payload_body] = PAYLOAD if ctx.values.get("payload") else EMPTY

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control == "payload":
            assert self.model is not None
            self._load(self.model, ctx)
            if self.pd is not None:
                self.pd.restart_integral()
            return
        super().on_control(control, value, data, ctx)

    def _finish_test(self) -> None:
        super()._finish_test()
        assert self.pd is not None
        ctx = self.pd.ctx
        if ctx.step == "challenge" and not _challenge(ctx):  # stars earned the easy way do not count here
            ctx.results["test"]["note"] = _text(
                "这次不算：挑战要关掉重力补偿、盒子装满。",
                "This one does not count: the challenge is with gravity compensation off and the box full.",
            )

    def formula(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> list[dict[str, Any]]:
        rows = super().formula(model, data, ctx)
        v = ctx.values
        inertia = _inertia(model, data, self.dof)
        zeta = _zeta(v["kp"], v["kd"], inertia)
        rows.append(
            {
                "show": "f_z",
                "label": _text("阻尼比 ζ", "Damping ratio ζ"),
                "expr": f"Kd ÷ (2√(Kp × I)) = {v['kd']:g} ÷ (2√({v['kp']:g} × {inertia:.2f}))",
                "value": "-" if zeta is None else f"{zeta:.2f}",
                "color": "amber",
            }
        )
        for row in rows:
            if row["show"] == "f_g":
                box = PAYLOAD if v.get("payload") else 0.0
                row["expr"] = _text(f"托住手臂和盒子（{box:g} kg）要的力矩", f"to hold the arm and box ({box:g} kg)")
        return rows

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        """The dial in the shoulder's plane: 0 degrees where joint2 is at 180, 90 where it is at 90."""
        data = self.fresh_data(model)
        jid = model.joint(JOINT).id
        qadr = int(model.joint(JOINT).qposadr[0])
        link = int(model.jnt_bodyid[jid])

        def arm_direction(q2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            data.qpos[qadr] = q2
            mujoco.mj_kinematics(model, data)
            anchor, axis = data.xanchor[jid].copy(), data.xaxis[jid].copy()
            reach = data.xpos[self.payload_body] - anchor
            reach -= axis * float(reach @ axis)  # in the plane the shoulder turns in
            return anchor, reach, data.xmat[link].reshape(3, 3).copy(), data.xpos[link].copy()

        anchor, zero, _, _ = arm_direction(math.pi)
        _, swing, _, _ = arm_direction(math.pi / 2)
        _, reach, rot, origin = arm_direction(OFFSET - math.radians(TARGET))
        weight = rot.T @ (data.xpos[self.payload_body] - origin)
        return {
            "kind": "joint",
            "pivot": anchor.round(4).tolist(),
            "axis": data.xaxis[jid].round(4).tolist(),
            "zero": (zero / np.linalg.norm(zero)).round(4).tolist(),
            "swing": (swing / np.linalg.norm(swing)).round(4).tolist(),
            "local": (rot.T @ reach / np.linalg.norm(reach)).round(4).tolist(),  # the arm's direction in its body
            "length": round(float(np.linalg.norm(reach)), 4),
            "body": link,
            "weight": weight.round(4).tolist(),
            "range": [15, 165],  # the dial around the shoulder, clear of the floor
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [0.3, -2.3, 0.8], "target": [0.22, 0.0, 0.36]}
