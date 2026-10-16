"""Lesson: imitation learning. The learner shows a desk arm how to pick a cube, and it learns by copying.

The learner steers the gripper (drag the amber ball, the sliders, or W A S D) to a red cube, closes it, lifts
the cube over a blue bin and lets go. Every demonstration is recorded as pairs of what the arm saw and what the
learner did next (learn/cloning.py). Then the arm tries alone, with two learners in turn: k nearest neighbours
(copy the closest moments) and a small neural network; near a demonstrated spot, far from all of them, and at 24
spread-out spots on a copy of the world, with success plotted against the distance to the nearest demonstration
and against how many demonstrations it learned from. The hand-written IK pick runs on the same spots last.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

from robotforge.learn import cloning as bc
from robotforge.learn.engine import (
    Bench,
    Control,
    Lesson,
    LessonContext,
    LessonError,
    Metric,
    Plot,
    Quiz,
    Series,
    Step,
    Sweep,
    Text,
)

METHODS = ("knn", "mlp", "ik")
WHERE = ("near", "far", "anywhere")
NEAR_OFFSET = 0.01  # m: "near a demonstration" puts the cube this far from a demonstrated spot
NEAR_TESTS = 8  # a test adds one near spot for each of the latest demonstrations, up to this many
MAX_DEMOS = 30
USE_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)  # the sweep's demonstration counts
DEMOS_WANTED = 5  # the "show it more" step
DEMO_SEED = 41  # the cubes of the learner's demonstrations come from this seed, so a replay meets the same spots
KEY_SPEED = 0.2  # m/s at a held key's full share (the viewer sends 70% of it)
SLIDERS = {"tx": (0.0, 0.45), "ty": (-0.35, 0.4), "tz": (0.01, 0.3)}  # m from the arm's base, height over the desk
TRY_PATH = 120  # gripper positions of a try sent to the viewer (every DT: 12 s)
PANEL_LAG = 10  # samples: the panel's "sees -> does" pair is this many DT old


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


def _cm(v: float) -> str:
    return f"{100 * v:.0f}"


# ---- the step checks ----------------------------------------------------------------------------------------


def _demonstrated(ctx: LessonContext) -> bool:
    return ctx.results.get("demos", 0) >= 1


def _copied_near(ctx: LessonContext) -> bool:
    return "try:near:ok" in ctx.events


def _tried_far(ctx: LessonContext) -> bool:
    return any(e.startswith("try:far:") for e in ctx.events)


def _demonstrated_more(ctx: LessonContext) -> bool:
    return ctx.results.get("demos", 0) >= DEMOS_WANTED


def _tested(method: str):
    def check(ctx: LessonContext) -> bool:
        return f"tested:{method}" in ctx.events

    check.__name__ = f"_tested_{method}"
    check.__doc__ = f"a test at the 24 spots with the {method} method has finished"
    return check


def _swept(ctx: LessonContext) -> bool:
    return "swept:done" in ctx.events


STEPS = (
    Step(
        id="show",
        title=_text("示范一次", "Show it once"),
        body=_text(
            "这节课你来当老师：亲手带机械臂抓一次方块，它看着学。\n"
            "拖动琥珀色小球（或用下面的滑条，或按 W A S D）把手爪移到红方块正上方，再往下放，让两根手指夹在方块两边；"
            "打开「手爪合上」；把方块提起来，移到蓝盒子上方，再关掉「手爪合上」，让它掉进盒子。\n"
            "你做的每一步都会被记下来：每 0.1 秒记一次方块在哪、手爪在哪、手指张多大（它“看到”的），"
            "和你接着让手爪去哪、合不合上（它该“做”的）。",
            "In this lesson you are the teacher: guide the arm through one pick and it learns by watching.\n"
            "Drag the amber ball (or use the sliders, or W A S D) to bring the gripper right over the red cube, then "
            "down so its two fingers are either side of it; turn on Gripper closed; lift the cube, move it over the "
            "blue bin and turn Gripper closed off so it drops in.\n"
            "Everything you do is recorded: every 0.1 s, where the cube is, where the gripper is and how far its "
            "fingers are open (what the robot sees), and where you sent the gripper next and whether you closed it "
            "(what it should do).",
        ),
        done=_text(
            "录下了一次示范：几十对“看到什么 → 做什么”。表格里是你的示范，桌上蓝色的线是手爪走过的路。"
            "模仿学习（也叫行为克隆）能用的，就只有这些数据。",
            "One demonstration recorded: a few dozen pairs of 'what it saw -> what it did'. The table lists your "
            "demonstrations, and the blue line over the desk is the path the gripper took. That is all the data "
            "imitation learning (also called behaviour cloning) gets.",
        ),
        check=_demonstrated,
        unlock=("tx", "ty", "tz", "grip", "new_cube"),
        show=("target", "region", "demos", "f_sample"),
    ),
    Step(
        id="copy",
        title=_text("照着做", "Copy it"),
        body=_text(
            "现在让它自己来。它用的办法叫 k 近邻：每 0.1 秒，在你的示范里找出和眼前最像的 3 个时刻，"
            "照你当时的做法动。\n"
            "「方块放在」选“示范过的地方”（离你示范时的方块 1 厘米），先猜，再点「让它自己做」。"
            "如果没抓到，多半是那次示范碰歪了方块：点「新方块」再示范一次，再试。",
            "Now let it try alone. It uses k nearest neighbours: every 0.1 s it finds the 3 moments of your "
            "demonstrations most like what it sees now, and does what you did then.\n"
            "With Cube at set to 'where you showed it' (1 cm from where the cube was), guess, then press Let it "
            "try. If it misses, your demonstration probably nudged the cube: press New cube, show it again, and "
            "try once more.",
        ),
        done=_text(
            "它把你的动作几乎原样做了一遍。这就是行为克隆：不用写一条规则，示范一次，在示范过的地方它就会了。",
            "It did what you did, almost move for move. That is behaviour cloning: not one rule written, one "
            "demonstration, and where you showed it, it can do it.",
        ),
        quiz=Quiz(
            question=_text(
                "方块就在你示范过的位置旁边，它能自己抓起来放进盒子吗？",
                "The cube is right beside where you showed it. Can the arm pick it up and put it in the bin alone?",
            ),
            options=(_text("能", "Yes"), _text("不能", "No")),
            answer=0,
            explain=_text(
                "眼前的情形和你示范时几乎一样，照着做就对了。", "What it sees is nearly what you saw: copying works."
            ),
            predict=True,
        ),
        check=_copied_near,
        unlock=("try", "where"),
        show=("target", "region", "demos", "trial", "f_sample"),
    ),
    Step(
        id="far",
        title=_text("换个远处", "Somewhere else"),
        body=_text(
            "「方块放在」改成“离所有示范都远”：桌上能放方块的地方里，离你示范过的位置最远的那一点。先猜，再点「让它自己做」。",
            "Set Cube at to 'far from every demonstration': of all the places a cube can be, the one furthest from "
            "where you showed it. Guess, then press Let it try.",
        ),
        done=_text(
            "它沿着你示范的路线，去了你示范时方块所在的地方，在空中合上手爪。从这一步起，它碰到的情形你一次也没示范过："
            "手里没东西，方块还在桌上……它只能照搬最像的时刻，一错再错。"
            "这叫协变量偏移（covariate shift）：学的时候只见过示范里的情形，自己做时一偏，就进了没见过的情形；"
            "错误还会越滚越大，叫复合误差。偶尔它乱碰几下把方块推到手爪下面，也会碰巧成功。",
            "It went along your path to where the cube was when you showed it, and closed the gripper on air. From "
            "then on it met situations you never showed it: nothing in its hand, the cube still on the desk... it "
            "can only repeat the closest moment it has, and each wrong move leads to another. This is covariate "
            "shift: it learned only from the situations in your demonstrations, and once it strays it is in ones "
            "it never saw; the errors snowball (compounding errors). Now and then it bumps the cube under its "
            "fingers and succeeds by luck.",
        ),
        quiz=Quiz(
            question=_text(
                "方块离你示范的地方十几厘米，它还能抓到吗？",
                "The cube is a dozen centimetres from where you showed it. Will it still get it?",
            ),
            options=(
                _text("能，它学会了抓方块", "Yes, it has learned to pick cubes"),
                _text("多半抓不到，它只会照着做", "Most likely not: it only copies"),
            ),
            answer=1,
            explain=_text(
                "它没学会“抓方块”这件事，只记住了你在哪里做了什么。",
                "It never learned 'pick the cube', only what you did where.",
            ),
            predict=True,
        ),
        check=_tried_far,
        show=("target", "region", "demos", "trial", "f_sample"),
    ),
    Step(
        id="more",
        title=_text("多示范几次", "Show it more"),
        body=_text(
            "一次示范只教会了一个位置。点「新方块」（方块出现在桌上随便一个地方，机械臂回到起点），"
            f"再示范几次，凑满 {DEMOS_WANTED} 次。",
            f"One demonstration taught it one place. Press New cube (the cube appears somewhere on the desk and the "
            f"arm goes back to its start) and show it again, until you have {DEMOS_WANTED} demonstrations.",
        ),
        done=_text(
            "桌上的蓝圈是你示范过的方块位置，蓝线是每次手爪走的路。几次示范，就是几百对“看到 → 做”。",
            "The blue rings on the desk are where the cube was in your demonstrations, the blue lines the "
            "gripper's paths: a few demonstrations make a few hundred 'see -> do' pairs.",
        ),
        check=_demonstrated_more,
        show=("target", "region", "demos", "trial", "f_sample"),
    ),
    Step(
        id="test",
        title=_text("考一考", "Put it to the test"),
        body=_text(
            "一次一次试太慢。「考 24 处」在桌上 24 个散开的位置各放一次方块，再在你每次示范的方块旁 1 厘米各放一次，"
            "让它每次都从头做（在另一份一模一样的仿真里，不打扰眼前这台）。成功的位置画绿点，失败的画红点；"
            "实验记录里的曲线是“成功率 对 离最近一次示范多远”。先猜，再点。",
            "Trying once at a time is slow. Test at 24 spots puts the cube at 24 spread-out places on the desk, and "
            "1 cm from each of your demonstrated spots, and lets it start from scratch every time (in an identical "
            "copy of the simulation, so the arm in front of you is left alone). Green dots mark successes, red "
            "ones failures; the experiment record plots success against the distance to the nearest "
            "demonstration. Guess, then press it.",
        ),
        done=_text(
            "离示范 2 厘米以内几乎都成功，再远就几乎都失败：照抄的本事，只覆盖每次示范周围一小圈。",
            "Within 2 cm of a demonstration nearly every try works; further away nearly every one fails. Copying "
            "covers only a small circle around each demonstration.",
        ),
        quiz=Quiz(
            question=_text(
                "照抄最像的时刻，散开的 24 处能成功多少？",
                "Copying the closest moments: how many of the 24 spread-out spots will work?",
            ),
            options=(
                _text("几乎全部", "Nearly all"),
                _text("一半左右", "About half"),
                _text("只有离示范很近的几处", "Only the few close to a demonstration"),
            ),
            answer=2,
            explain=_text(
                "照抄只在示范过的地方附近管用；离开两三厘米，手爪去的位置就偏了。",
                "Copying works only next to where you showed it; two or three centimetres off, the gripper goes "
                "to the wrong place.",
            ),
            predict=True,
        ),
        check=_tested("knn"),
        unlock=("test",),
        show=("target", "region", "demos", "tests", "trial", "f_sample"),
    ),
    Step(
        id="network",
        title=_text("小神经网络", "A small neural network"),
        body=_text(
            "换一种学法：「方法」选“小神经网络”。它不再翻查示范，而是用全部数据训练出一个函数，"
            "从看到的直接算出该怎么动（3 个 7→64→64→4 的小网络取平均，只用 numpy，训练一两秒）。"
            "先猜，再点「考 24 处」。",
            "Another way to learn: set Method to 'small neural network'. Instead of looking moments up, it trains "
            "one function on all the data that turns what it sees straight into what to do (three small 7-64-64-4 "
            "networks, averaged; numpy only, a second or two of training). Guess, then press Test at 24 spots.",
        ),
        done=_text(
            "两次示范之间的位置它也常能抓到：它学到的是一个平滑的函数，不是一张表，所以能在示范之间“插值”。"
            "训练时还把手爪的位置随机挪开一两厘米、目标不动，它就学会了从偏一点的地方回到路上，这是对付协变量偏移的老办法。"
            "离所有示范都远的地方（桌子边、角落）它还是会失败，那是“外推”，它只能瞎猜。示范越少，它越容易学歪。",
            "Between two demonstrations it now often succeeds: it learned a smooth function, not a table, so it can "
            "interpolate between demonstrations. In training the gripper's position was also nudged a centimetre or "
            "two at random with the target left where it was, so it learned to get back to the path from a little "
            "off it: an old remedy for covariate shift. Far from all demonstrations (the edges, the corners) it "
            "still fails: that is extrapolation, and there it can only guess. The fewer the demonstrations, the "
            "more it learns the wrong thing.",
        ),
        quiz=Quiz(
            question=_text("和照抄比，神经网络会？", "Compared with copying, the network will be?"),
            options=(
                _text("更差", "Worse"),
                _text(
                    "在示范之间也常做对，离示范太远还是不行",
                    "Often right between demonstrations, still wrong far from them",
                ),
                _text("哪里都能做对", "Right everywhere"),
            ),
            answer=1,
            explain=_text(
                "它能在示范之间插值，但没见过的地方它也只是在猜。",
                "It interpolates between demonstrations, but where it has seen nothing it is guessing too.",
            ),
            predict=True,
        ),
        check=_tested("mlp"),
        unlock=("method",),
        show=("target", "region", "demos", "tests", "trial", "f_sample"),
    ),
    Step(
        id="how_many",
        title=_text("要示范多少次", "How many demonstrations?"),
        body=_text(
            "示范越多越好吗？在实验记录里点「扫一遍 示范」：只用前 1 次、前 2 次……的示范训练，每次都考那 24 处，"
            "画出成功率随示范次数怎么变（用的是现在选的方法）。先猜。想看得更清楚，可以先多示范几次。",
            "Do more demonstrations help? In the experiment record press Sweep demos: it trains on your first "
            "demonstration only, then the first two, and so on, tests each at the 24 spots, and plots success "
            "against the number of demonstrations (with the method chosen now). Guess first. To see it more "
            "clearly, show it a few more times before you sweep.",
        ),
        done=_text(
            "示范少，成功率掉得很快；多、而且分散，才学得好。真实机器人的模仿学习常常要几十到几百次示范，就是这个原因。"
            "另一个办法是让它自己做，做错的地方由人接过来纠正，把纠正也加进数据（DAgger）。",
            "With few demonstrations success drops fast; many, spread out, is what works. That is why imitation "
            "learning on real robots often takes tens to hundreds of demonstrations. Another fix: let it act, take "
            "over where it goes wrong, and add your corrections to the data (DAgger).",
        ),
        quiz=Quiz(
            question=_text("示范少一半，成功率会？", "With half the demonstrations, success will?"),
            options=(_text("差不多", "Stay about the same"), _text("明显下降", "Drop clearly")),
            answer=1,
            explain=_text(
                "示范覆盖的地方少了，它没见过的位置就多了。",
                "Fewer demonstrations cover fewer places, so more spots are ones it never saw.",
            ),
            predict=True,
        ),
        check=_swept,
        unlock=("use",),
        show=("target", "region", "demos", "tests", "trial", "f_sample"),
    ),
    Step(
        id="ik",
        title=_text("手写的抓取", "The hand-written pick"),
        body=_text(
            "最后比一比：「方法」选“手写的 IK 抓取”。它不看示范，只用方块和盒子的位置算出手爪该去哪，"
            "再用逆运动学求出关节角（就是让机械臂把红方块放进盒子的那段代码）。先猜，再点「考 24 处」。",
            "Last, a comparison: set Method to 'hand-written IK pick'. It ignores the demonstrations: from where "
            "the cube and the bin are it works out where the gripper goes, and inverse kinematics finds the joint "
            "angles (the same code that puts the red cube in the bin when you ask an agent). Guess, then press "
            "Test at 24 spots.",
        ),
        done=_text(
            "处处都成功。能把规则写出来的事，写出来最可靠；模仿学习的用处，在那些说不清规则、只能示范的事上："
            "叠衣服、插线缆、做饭。",
            "It works everywhere. When you can write the rule down, the written rule is the most reliable; "
            "imitation learning is for the jobs whose rules nobody can write but anyone can show: folding "
            "clothes, plugging in a cable, cooking.",
        ),
        quiz=Quiz(
            question=_text(
                "手写的抓取在这 24 处能成功多少？", "How many of the 24 spots will the hand-written pick get?"
            ),
            options=(
                _text("一半左右", "About half"),
                _text("几乎全部", "Nearly all"),
                _text("比神经网络少", "Fewer than the network"),
            ),
            answer=1,
            explain=_text(
                "它不靠见过什么，而是每次都按方块的真实位置算。",
                "It does not depend on what it has seen: it works from where the cube really is, every time.",
            ),
            predict=True,
        ),
        check=_tested("ik"),
        show=("target", "region", "demos", "tests", "trial", "f_sample"),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text(
            "照抄在离示范远的地方会失败……",
            "Copying fails far from the demonstrations...",
        ),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("为什么会失败？", "Why does it fail there?"),
            options=(
                _text("电机力气不够", "The motors are too weak"),
                _text(
                    "它遇到了示范里没有的情形，只能照搬最像的做法，一步错，步步错",
                    "It meets situations no demonstration had, can only repeat the closest one it knows, and one "
                    "wrong move leads to the next",
                ),
                _text("方块太重", "The cube is too heavy"),
            ),
            answer=1,
            explain=_text(
                "对，这就是协变量偏移和复合误差。更多、更分散的示范，能插值的模型，在它出错的地方补上示范：都是为了让它见过更多情形。",
                "Right: covariate shift and compounding errors. More and more varied demonstrations, models that "
                "interpolate, demonstrations added where it goes wrong: all are ways to show it more situations.",
            ),
        ),
    ),
)


# ---- the live controller -------------------------------------------------------------------------------------


class Hands:
    """The lesson's arm controller: while the learner teaches, their target and grip, recorded every DT; during a
    try, a learned policy or the hand-written pick at the controls (the same code as the tests' roll-outs)."""

    rate_hz = bc.CONTROL_HZ

    def __init__(self, lesson: ImitationLesson, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.lesson = lesson
        self.scene = bc.Scene(model)
        self.follow = bc.Follow(model, self.scene)
        self.next = float(data.time)
        self.last: float | None = None
        self.pilot: bc.Pilot | bc.ScriptedPick | None = None

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        now = float(data.time)
        dt = 0.0 if self.last is None else max(0.0, now - self.last)
        self.last = now
        lesson = self.lesson
        if lesson.pending is not None:  # a try whose learner finished training on the lesson's thread
            self.start(data, *lesson.pending)
            lesson.pending = None
        if self.pilot is not None:
            self.pilot.update(model, data)
            if now + 1e-9 >= self.next:
                self.next = now + bc.DT
                lesson.sample(self.scene, data, self.follow.hand(data))
            if self.pilot.outcome is not None:
                if isinstance(self.pilot, bc.ScriptedPick):
                    self.follow.sync(data)  # the pick drove the motors: steer on from where it left the arm
                lesson.try_ended(self.pilot.outcome, self.follow.hand(data))
                self.pilot = None
            return
        busy = self.follow.busy(now)
        target, grip = lesson.steering(dt)
        self.follow.set(target, grip, now)
        if now + 1e-9 >= self.next:
            self.next = now + bc.DT
            obs = bc.observe(self.scene, self.follow, data)
            lesson.sample(self.scene, data, obs[3:6])
            lesson.taught(obs, self.follow, busy, data)
        self.follow.update(model, data)

    def start(self, data: mujoco.MjData, spot: np.ndarray, learner: bc.Copier | bc.Network | None) -> None:
        """A try: the arm at home, the cube at `spot`, and a learner (None: the hand-written pick) at the controls."""
        self.scene.start(data, spot)
        self.follow.reset()
        if learner is None:
            self.pilot = bc.ScriptedPick(self.scene.model, self.scene, data, spot)
        else:
            self.pilot = bc.Pilot(self.scene, self.follow, learner, float(data.time), spot)
        self.next = float(data.time)

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-imitation", "trying": self.pilot is not None}


class ImitationLesson(Lesson):
    id = "imitation"
    chapter = _text("模仿学习", "Imitation learning")
    title = _text("模仿学习：示范几次，教会机械臂抓方块", "Imitation learning: teach an arm a pick by showing it")
    summary = _text(
        "你来示范抓方块，机器人用 k 近邻和小神经网络照着学；换个位置就看出照抄能做什么、不能做什么，"
        "再和手写的 IK 抓取比一比。",
        "You demonstrate a pick and the robot copies it with k nearest neighbours and a small neural network. "
        "Move the cube to see what copying can and cannot do, then compare with a hand-written IK pick.",
    )
    minutes = 15
    # W S: the target forward and back, A D: left and right
    drive = {"max_speed": KEY_SPEED, "turn": KEY_SPEED, "label": _text("移动手爪", "Move the gripper"), "robot": False}
    controls = (
        Control("tx", "slider", _text("手爪：往前", "Gripper: forward"), *SLIDERS["tx"], 0.005, 0.22, " m"),
        Control("ty", "slider", _text("手爪：左右", "Gripper: sideways"), *SLIDERS["ty"], 0.005, 0.0, " m"),
        Control(
            "tz",
            "slider",
            _text("手爪：离桌面高度", "Gripper: height over the desk"),
            *SLIDERS["tz"],
            0.005,
            0.11,
            " m",
        ),
        Control("grip", "toggle", _text("手爪合上", "Gripper closed")),
        Control("new_cube", "button", _text("新方块", "New cube")),
        Control(
            "where",
            "choice",
            _text("方块放在", "Cube at"),
            options=(
                _text("示范过的地方", "where you showed it"),
                _text("离所有示范都远", "far from every demonstration"),
                _text("随便哪里", "anywhere"),
            ),
        ),
        Control("try", "button", _text("让它自己做", "Let it try")),
        Control("test", "button", _text("考 24 处", "Test at 24 spots")),
        Control(
            "method",
            "choice",
            _text("方法", "Method"),
            options=(
                _text("照抄最像的时刻（k 近邻）", "copy the closest moments (k nearest neighbours)"),
                _text("小神经网络", "small neural network"),
                _text("手写的 IK 抓取", "hand-written IK pick"),
            ),
        ),
        Control("use", "slider", _text("用几次示范", "Demonstrations used"), 1, MAX_DEMOS, 1, MAX_DEMOS),
    )
    plots = (
        Plot(
            "gripper",
            _text("手爪", "Gripper"),
            "cm",
            (
                Series("height", _text("离桌面高度", "height over the desk"), "blue"),
                Series("open", _text("手指张开", "finger opening"), "amber"),
            ),
            window=10.0,
            span=8.0,
        ),
    )
    steps = STEPS
    bench = Bench(
        title=_text("实验记录：考 24 处的成功率", "Experiment record: success at the test spots"),
        x=_text("离最近一次示范", "distance to the nearest demonstration"),
        x_unit="cm",
        y=_text("成功率", "success"),
        y_unit="%",
        params=("method", "use"),
        short=(_text("方法", "method"), _text("示范", "demos")),
        metrics=(
            Metric("success", _text("24 处成功", "success at 24"), "%", lower_is_better=False),
            Metric("near", _text("示范旁", "beside a demo"), "%", lower_is_better=False),
            Metric("far", _text("离示范 6 厘米外", "6 cm or more away"), "%", lower_is_better=False),
            Metric("train", _text("训练", "training"), "s", ranked=False),
        ),
        reference=100.0,
        sweeps=(Sweep("use", USE_VALUES, ("success",)),),
    )

    def __init__(self) -> None:
        self.model: mujoco.MjModel | None = None
        self.ctx: LessonContext | None = None
        self.hands: Hands | None = None
        self.demos: list[bc.Demo] = []
        self.recording: bc.Recording | None = None
        self.in_bin_since: float | None = None
        self.keys = (0.0, 0.0)  # m/s forward and left from W A S D
        self.trial: dict[str, Any] | None = None  # the live try: where, its path, how it ended
        self.tries = 0
        self.pending: tuple[np.ndarray, bc.Copier | bc.Network] | None = None  # a try waiting for its learner
        self.policy: tuple[tuple[str, int], bc.Copier | bc.Network] | None = None  # ((method, demos), learner)
        self.tests: list[dict[str, Any]] = []  # the last test's spots and results, for the drawing
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.progress: Text | None = None
        self.samples: deque[dict[str, float]] = deque(maxlen=512)
        self.last_pair: tuple[np.ndarray, np.ndarray] | None = None  # the newest recorded sample and its action
        self.drawing_version = 0
        self.drawing_sent = -1

    # ---- the world ----------------------------------------------------------------------------------------

    def build(self) -> mujoco.MjModel:
        self.model = bc.build_model()
        return self.model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        """The lesson starts, or the learner pressed Reset: the arm at home, the cube where the keyframe has it."""
        self.ctx = ctx
        self.hands = Hands(self, model, data)
        ctx.results.setdefault("demos", len(self.demos))
        self._home_values(ctx)
        self._record(self.hands.scene.cube(data)[:2], float(data.time))
        self.trial = None
        self._redraw()
        return {"lesson": self.hands}

    def close(self) -> None:
        self.stopping.set()

    def _home_values(self, ctx: LessonContext) -> None:
        assert self.hands is not None
        home = self.hands.follow.home_point
        ctx.values.update(tx=round(float(home[0]), 3), ty=round(float(home[1]), 3), tz=round(home[2] - bc.DESK, 3))
        ctx.values["grip"] = 0.0
        self.keys = (0.0, 0.0)

    def _record(self, spot: Sequence[float], now: float) -> None:
        self.recording = bc.Recording(np.array(spot[:2], float), now)
        self.in_bin_since = None
        self.last_pair = None

    # ---- teaching ---------------------------------------------------------------------------------------------

    def steering(self, dt: float) -> tuple[np.ndarray, float]:
        """The learner's target in the world and grip, the held keys moving the target along the desk."""
        ctx = self.ctx
        assert ctx is not None
        v = ctx.values
        fwd, left = self.keys
        if fwd or left:
            v["tx"] = float(np.clip(v["tx"] + fwd * dt, *SLIDERS["tx"]))
            v["ty"] = float(np.clip(v["ty"] + left * dt, *SLIDERS["ty"]))
        return np.array([v["tx"], v["ty"], bc.DESK + v["tz"]]), float(v["grip"])

    def on_drive(self, speed: float, turn: float, ctx: LessonContext) -> None:
        self.keys = (float(speed), float(turn))

    def taught(self, obs: np.ndarray, follow: bc.Follow, busy: bool, data: mujoco.MjData) -> None:
        """A sample while the learner steers (every DT): record it, and keep the demonstration once the cube is in
        the bin and the gripper open."""
        ctx, rec = self.ctx, self.recording
        if ctx is None or rec is None:
            return
        rec.add(obs, follow.target, follow.grip, busy)
        # The panel shows a pair from a second ago: a sample's action is what the learner did after it.
        X, Y = bc.label(rec.rows[-PANEL_LAG:])
        if len(X):
            self.last_pair = (X[0], Y[0])
        scene = self.hands.scene if self.hands else None
        if scene is None:
            return
        now = float(data.time)
        if scene.fell(data):
            self.recording = None
            ctx.results["summary"] = _text(
                "方块掉到桌子下面了，这次不算。点「新方块」再来。",
                "The cube fell off the desk; this one does not count. Press New cube to start again.",
            )
            return
        done = scene.in_bin(data) and follow.grip < 0.5 and not follow.busy(now)
        if not done:
            self.in_bin_since = None
            return
        if self.in_bin_since is None:
            self.in_bin_since = now
        if now - self.in_bin_since < bc.SETTLE:
            return
        demo = rec.finish(now)
        self.recording = None
        if len(self.demos) >= MAX_DEMOS:
            ctx.results["summary"] = _text(
                f"已经有 {MAX_DEMOS} 次示范了，这次没有再存。",
                f"You already have {MAX_DEMOS} demonstrations; not kept.",
            )
            return
        self.demos.append(demo)
        ctx.results["demos"] = len(self.demos)
        ctx.events.add("demo_saved")
        ctx.results["summary"] = _text(
            f"示范 #{len(self.demos)} 录好了：{len(demo.X)} 对“看到 → 做”，{demo.seconds:.1f} 秒。",
            f"Demonstration #{len(self.demos)} recorded: {len(demo.X)} 'see -> do' pairs in {demo.seconds:.1f} s.",
        )
        self._redraw()

    # ---- buttons -------------------------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control == "new_cube":
            self._new_cube(data, ctx)
        elif control == "try":
            self._try(data, ctx)
        elif control == "test":
            self._start_test(ctx)

    def _new_cube(self, data: mujoco.MjData, ctx: LessonContext) -> None:
        hands = self.hands
        assert hands is not None
        if hands.pilot is not None:
            hands.pilot = None
        rng = np.random.default_rng(DEMO_SEED + len(self.demos))
        spot = bc.random_spot(rng)
        hands.scene.start(data, spot)
        hands.follow.reset()
        self._home_values(ctx)
        self._record(spot, float(data.time))
        self.trial = None
        ctx.results.pop("summary", None)
        self._redraw()

    def _try(self, data: mujoco.MjData, ctx: LessonContext) -> None:
        hands = self.hands
        assert hands is not None
        if hands.pilot is not None or self.pending is not None or self.busy:
            raise LessonError("wait for the try or the test that is running to end")
        where = WHERE[int(ctx.values["where"])]
        method = METHODS[int(ctx.values["method"])]
        spots = [d.spot for d in self.demos]
        if method != "ik" and not spots:
            raise LessonError("show it once first: there is nothing to copy yet")
        self.tries += 1
        if where == "near" and spots:
            angle = 2 * math.pi * (self.tries * 0.618 % 1.0)  # round the spot, a new side every try
            spot = spots[-1] + NEAR_OFFSET * np.array([math.cos(angle), math.sin(angle)])
        elif where == "far":
            spot = bc.far_spot(spots)
        else:
            spot = bc.random_spot(np.random.default_rng(1000 + self.tries))
        self.recording = None
        self.trial = {
            "where": where,
            "method": method,
            "spot": [round(float(v), 3) for v in spot],
            "distance": round(bc.nearest(spot, spots), 3) if spots else None,
            "path": [],
            "ok": None,
        }
        use = len(self.demos)
        cached = self._cached(method, use)
        if method == "ik" or cached is not None:
            hands.start(data, spot, cached)
            ctx.results["summary"] = _text("它在自己做……", "It is trying on its own...")
            return
        # A network trains for a second or two: on the lesson's thread, so the viewer does not stop meanwhile.
        demos = list(self.demos)

        def train() -> None:
            self.progress = _text("训练中……", "Training...")
            learner = bc.learn(method, demos)
            self.policy = ((method, use), learner)
            self.progress = None
            if not self.stopping.is_set():
                self.pending = (spot, learner)

        ctx.results["summary"] = _text("先训练，再让它自己做……", "Training first, then it tries...")
        self.thread = threading.Thread(target=train, daemon=True)
        self.thread.start()

    def _cached(self, method: str, use: int) -> bc.Copier | bc.Network | None:
        """The learner trained on the first `use` demonstrations, if there is one (kNN is made on the spot)."""
        if method == "ik":
            return None
        key = (method, min(use, len(self.demos)))
        if self.policy is not None and self.policy[0] == key:
            return self.policy[1]
        if method == "knn":
            learner = bc.learn(method, self.demos[: key[1]])
            self.policy = (key, learner)
            return learner
        return None

    def try_ended(self, outcome: bc.Outcome, hand: np.ndarray) -> None:
        ctx, trial = self.ctx, self.trial
        if ctx is None or trial is None:
            return
        trial["ok"] = outcome.success
        trial["path"] = outcome.path[-TRY_PATH:]
        ctx.events.add(f"try:{trial['where']}:{'ok' if outcome.success else 'missed'}")
        far = trial["distance"]
        away = (
            _text(f"（离最近的示范 {_cm(far)} 厘米）", f" ({_cm(far)} cm from the nearest demonstration)")
            if far is not None
            else _text("", "")
        )
        verdict = _text("成功", "Success") if outcome.success else _text("失败", "Failed")
        ctx.results["summary"] = _text(
            f"{verdict['zh']}{away['zh']}：{outcome.why['zh']}，用了 {outcome.seconds:.1f} 秒。",
            f"{verdict['en']}{away['en']}: {outcome.why['en']} in {outcome.seconds:.1f} s.",
        )
        # The ball and the sliders come to where the gripper stopped, so steering on starts from there.
        ctx.values.update(tx=round(float(hand[0]), 3), ty=round(float(hand[1]), 3))
        ctx.values["tz"] = round(float(np.clip(hand[2] - bc.DESK, *SLIDERS["tz"])), 3)
        ctx.values["grip"] = float(self.hands.follow.grip) if self.hands else 0.0

    # ---- learning and testing ------------------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _start_test(self, ctx: LessonContext) -> None:
        if self.busy or self.pending is not None or (self.hands is not None and self.hands.pilot is not None):
            raise LessonError("wait for the try or the test that is running to end")
        method = METHODS[int(ctx.values["method"])]
        if method != "ik" and not self.demos:
            raise LessonError("show it once first: there is nothing to learn from yet")
        use = min(int(ctx.values["use"]), len(self.demos)) if method != "ik" else len(self.demos)
        demos = list(self.demos)
        assert self.model is not None
        model = bc.private(self.model)
        self.thread = threading.Thread(target=self._test, args=(model, method, demos, use, ctx), daemon=True)
        self.thread.start()

    def _test(self, model: mujoco.MjModel, method: str, demos: list[bc.Demo], use: int, ctx: LessonContext) -> None:
        self.progress = _text("训练中……", "Training...")
        learner = None if method == "ik" else bc.learn(method, demos[:use])
        if learner is not None and len(self.demos) == len(demos):
            self.policy = ((method, use), learner)  # a try right after uses it, untrained again
        spots_demo = [d.spot for d in demos[:use]] if method != "ik" else [d.spot for d in demos]
        fixed = bc.test_spots()
        beside = [
            d.spot + bc.CUBE / 4 * np.array([math.cos(1.3 * k), math.sin(1.3 * k)])
            for k, d in enumerate(demos[:use][-NEAR_TESTS:])
        ]
        spots = [*fixed, *beside]
        results: list[dict[str, Any]] = []
        for k, spot in enumerate(spots):
            if self.stopping.is_set():
                return
            self.progress = _text(f"考第 {k + 1} 处，共 {len(spots)} 处……", f"Testing spot {k + 1} of {len(spots)}...")
            out = bc.scripted_rollout(model, spot) if learner is None else bc.rollout(model, learner, spot)
            results.append(
                {
                    "spot": [round(float(v), 3) for v in spot],
                    "ok": out.success,
                    "cm": round(100 * bc.nearest(spot, spots_demo), 1) if spots_demo else None,
                    "fixed": k < len(fixed),
                }
            )
            with self.lock:
                self.tests = list(results)
            self.drawing_version += 1
        self.progress = None
        pairs = [(r["cm"] if r["cm"] is not None else math.inf, r["ok"]) for r in results]
        xs, ys = bc.by_distance(pairs)
        metrics = {
            "success": bc.rate([r["ok"] for r in results if r["fixed"]]),
            "near": bc.rate([r["ok"] for r in results if not r["fixed"]]),
            "far": bc.rate([ok for d, ok in pairs if d >= bc.FAR_CM]),
            "train": round(learner.seconds, 1) if learner is not None else None,
        }
        settings = {"method": float(METHODS.index(method)), "use": float(use)}
        # The record's curve leaves out the bands no spot fell in (the engine keeps numbers, not gaps).
        kept = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
        ctx.record_run(settings, metrics, [x for x, _ in kept], [y for _, y in kept])
        ctx.results["test"] = {"method": method, "use": use, **metrics}
        ctx.results["summary"] = _text(
            f"{_METHOD_ZH[method]}：24 处成功 {metrics['success']:g}%，示范旁 {_pct(metrics['near'])}，"
            f"离示范 6 厘米外 {_pct(metrics['far'])}。",
            f"{_METHOD_EN[method]}: {metrics['success']:g}% of the 24 spots, {_pct(metrics['near'])} beside a "
            f"demonstration, {_pct(metrics['far'])} 6 cm or more away.",
        )
        ctx.events.add(f"tested:{method}")

    def run_sweep(self, sweep: Sweep, ctx: LessonContext) -> None:
        """Train on the first 1, 2, 3... demonstrations and test each at the 24 spots, on a thread: the points
        come in one by one."""
        if self.busy:
            raise LessonError("a test is still running")
        method = METHODS[int(ctx.values["method"])]
        if method == "ik":
            raise LessonError("the hand-written pick learns from no demonstrations: pick a learned method to sweep")
        counts = [n for n in sweep.values if n <= len(self.demos)]
        if len(counts) < 2:
            raise LessonError("show it at least twice before sweeping")
        assert self.model is not None
        model = bc.private(self.model)
        demos = list(self.demos)
        ctx.events.discard("swept:done")
        self.thread = threading.Thread(target=self._sweep, args=(model, method, demos, counts, ctx), daemon=True)
        self.thread.start()
        return None

    def _sweep(self, model: mujoco.MjModel, method: str, demos: list[bc.Demo], counts: list[int], ctx) -> None:
        fixed = bc.test_spots()
        points: list[dict[str, Any]] = []
        for n in counts:
            if self.stopping.is_set():
                return
            self.progress = _text(f"用 {n} 次示范训练……", f"Training on {n} demonstrations...")
            learner = bc.learn(method, demos[:n])
            oks = []
            for k, spot in enumerate(fixed):
                if self.stopping.is_set():
                    return
                self.progress = _text(
                    f"用 {n} 次示范：考第 {k + 1} 处，共 {len(fixed)} 处……",
                    f"{n} demonstrations: testing spot {k + 1} of {len(fixed)}...",
                )
                oks.append(bc.rollout(model, learner, spot).success)
            points.append({"x": float(n), "success": bc.rate(oks)})
            ctx.sweep = {"control": "use", "points": list(points)}
            ctx.bench_version += 1
        self.progress = None
        ctx.events.add("swept:done")
        lo, hi = points[0], points[-1]
        ctx.results["summary"] = _text(
            f"{_METHOD_ZH[method]}：{lo['x']:g} 次示范成功 {lo['success']:g}%，"
            f"{hi['x']:g} 次示范成功 {hi['success']:g}%。",
            f"{_METHOD_EN[method]}: {lo['success']:g}% with {lo['x']:g} demonstration{'s' if lo['x'] != 1 else ''}, "
            f"{hi['success']:g}% with {hi['x']:g}.",
        )

    # ---- what the viewer shows ------------------------------------------------------------------------------------

    def sample(self, scene: bc.Scene, data: mujoco.MjData, hand: np.ndarray) -> None:
        with self.lock:
            self.samples.append(
                {
                    "t": float(data.time),
                    "height": round(100 * (float(hand[2]) - bc.DESK), 2),
                    "open": round(100 * scene.opening(data), 2),
                }
            )

    def drain_samples(self) -> list[dict[str, float]]:
        with self.lock:
            out = list(self.samples)
            self.samples.clear()
        return out

    def _redraw(self) -> None:
        self.drawing_version += 1

    def describe(self, model: mujoco.MjModel) -> dict[str, Any]:
        self.drawing_sent = -1  # a viewer that (re)connects gets the drawing with its next update
        return super().describe(model)

    def drawing(self) -> dict[str, Any]:
        with self.lock:
            tests = list(self.tests)
        return {
            "demos": [
                {"spot": [round(float(v), 3) for v in d.spot], "path": d.path[::2], "n": k + 1}
                for k, d in enumerate(self.demos)
            ],
            "tests": [{"spot": t["spot"], "ok": t["ok"]} for t in tests],
        }

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        hands = self.hands
        assert hands is not None
        trying = hands.pilot is not None
        target = hands.follow.hand(data) if isinstance(hands.pilot, bc.ScriptedPick) else hands.follow.target
        out: dict[str, Any] = {
            "target": [round(float(v), 4) for v in target],
            "hand": [round(float(v), 4) for v in hands.follow.hand(data)],
            "cube": [round(float(v), 4) for v in hands.scene.cube(data)],
            "trying": trying,
            "recording": self.recording is not None,
            "readout": self._readout(ctx, trying),
            "table": self._table(),
            "formula": self._formula(),
        }
        if self.trial is not None:
            path = hands.pilot.path if hands.pilot is not None else self.trial["path"]
            out["trial"] = {"spot": self.trial["spot"], "path": path[-TRY_PATH:], "ok": self.trial["ok"]}
        if self.drawing_version != self.drawing_sent:
            self.drawing_sent = self.drawing_version
            out["drawing"] = self.drawing()
        return out

    def _readout(self, ctx: LessonContext, trying: bool) -> Text:
        n = len(self.demos)
        if self.progress is not None:
            return self.progress
        if trying:
            return _text(f"它在自己做（学了 {n} 次示范）……", f"Trying on its own (learned from {n} demonstrations)...")
        if self.recording is not None:
            return _text(f"● 正在记录第 {n + 1} 次示范", f"● Recording demonstration {n + 1}")
        return _text(f"已有 {n} 次示范。点「新方块」再示范一次", f"{n} demonstrations. Press New cube to show it again")

    def _table(self) -> dict[str, Any] | None:
        if not self.demos:
            return None
        columns = [
            {"label": _text("方块在", "cube at"), "color": "blue", "sub": _text("厘米", "cm")},
            {"label": _text("样本", "samples"), "color": "ink", "sub": _text("看到 → 做", "see -> do")},
            {"label": _text("用时", "time"), "color": "ink", "sub": _text("秒", "s")},
        ]
        rows: list[dict[str, Any]] = [{"heading": _text("你的示范", "Your demonstrations")}]
        for k, d in enumerate(self.demos[-8:]):
            info = d.describe(len(self.demos) - min(8, len(self.demos)) + k)
            rows.append(
                {
                    "label": _text(f"#{info['n']}", f"#{info['n']}"),
                    "cells": [f"{info['spot'][0]}, {info['spot'][1]}", str(info["kept"]), f"{info['seconds']}"],
                    "best": None,
                }
            )
        if len(self.demos) > 8:
            rows.insert(
                1, {"heading": _text(f"（最近 8 次，共 {len(self.demos)} 次）", f"(latest 8 of {len(self.demos)})")}
            )
        return {"columns": columns, "rows": rows}

    def _formula(self) -> list[dict[str, Any]]:
        """One recorded pair, as the learner made it a second ago: what the arm saw, and what it should do."""
        pair = self.last_pair
        cube = hand = done = "-"
        if pair is not None:
            obs, action = pair
            up = np.array([0.0, 0.0, bc.DESK])
            cube = _xyz(obs[:3] - up)
            hand = f"{_xyz(obs[3:6] - up)} · {100 * obs[6]:.1f}"
            grip = _text("合上", "close") if action[3] >= 0.5 else _text("张开", "open")
            done = {lang: f"{_xyz(action[:3] - up)} · {grip[lang]}" for lang in ("zh", "en")}
        return [
            {
                "show": "f_sample",
                "label": _text("看到方块", "Sees cube"),
                "expr": _text("1 秒前，厘米", "1 s ago, cm"),
                "value": cube,
                "color": "blue",
            },
            {
                "show": "f_sample",
                "label": _text("看到手爪", "Sees gripper"),
                "expr": _text("位置 · 手指张开", "where · finger gap"),
                "value": hand,
                "color": "blue",
            },
            {
                "show": "f_sample",
                "label": _text("该做", "Should do"),
                "expr": _text("手爪接着去哪 · 抓握", "gripper goes to · grip"),
                "value": done,
                "color": "amber",
            },
        ]

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "imitate",
            "base": [0.0, 0.0, bc.DESK],
            "desk": bc.DESK,
            "region": {"r": list(bc.REGION_R), "deg": list(bc.REGION_DEG)},
            "bin": [*bc.BIN_XY, bc.DESK],
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [0.78, -0.62, 1.22], "target": [0.15, 0.02, 0.79]}


_METHOD_ZH = {"knn": "照抄（k 近邻）", "mlp": "小神经网络", "ik": "手写的 IK 抓取"}
_METHOD_EN = {"knn": "Copying (k nearest neighbours)", "mlp": "Small neural network", "ik": "Hand-written IK pick"}


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:g}%"


def _xyz(v: Sequence[float]) -> str:
    return ", ".join(f"{100 * float(x):.0f}" for x in v)
