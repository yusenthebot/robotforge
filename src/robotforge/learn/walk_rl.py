"""Lesson: reinforcement learning II. Any body learns to walk: a rhythm per joint, tuned by random search.

Our template dog, a three-legged starfish built from parts (no hand-written gait knows it), our template
humanoid and, when the learner came from a world with a robot of their own, that robot, stand in lanes.
"Learn" trains the chosen one from standing still, in its own private copy of the world on a background
thread, while the viewer replays the newest rhythm every four seconds. Each training lands in the record
as a learning curve. Knobs: the rhythm's frequency, how much the reward charges for energy, how long to
train. The search itself is control/cpg.py. What the learner's own robot learned can be kept in their
world's folder, and `move` and the keyboard then walk it that way (behaviors/rhythm_walk.py).
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import Kind, rhythm_walk, robot_kind
from robotforge.behaviors.rhythm_walk import (
    SavedWalk,
    heading_drift,
    joint_names,
    learn_steering,
    learn_turns,
    save_walk,
)
from robotforge.control.cpg import (
    DRIFT_WEIGHT,
    HEADING_WEIGHT,
    Body,
    CPGError,
    Reward,
    Rhythm,
    RhythmController,
    Trainer,
)
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Bench, Control, Lesson, LessonContext, Metric, Plot, Quiz, Series, Step, Text

SETTINGS = ("robot", "energy", "freq", "iterations", "reflexes", "shove")  # what each training records
START_X = -0.8
LANE = 0.9
ATTEMPT = 4.0  # s: the viewer replays the newest rhythm from the start line this often (and trains on this long)
ENERGY = (0.0, 0.1, 0.3)  # reward weight on energy, per (m g)
SHOVE = 0.4  # m/s: the random shove in training, in the push test after it and in the replay
SHOVE_AT = 2.0  # s into each replay
PUSH_TRIALS = 12
SAMPLE_DT = 0.02
KEEP_MIN = 0.1  # m in an attempt: a walk that gets less than this is not worth keeping
STAR = {
    "color": "purple",
    "base": {"shape": "cylinder", "size": [0.2, 0.05], "sensors": ["imu"]},
    "parts": [
        {
            "name": "hip",
            "at": [0.1, 0, 0],
            "joint": {"axis": "z", "range": [-40, 40]},
            "shape": "box",
            "size": [0.05, 0.03, 0.025],
            "direction": "forward",
            "repeat": {"count": 3, "around": "z"},
        },
        {
            "name": "leg",
            "parent": "hip",
            "at": "end",
            "joint": {"range": [-60, 90], "home": 40},
            "size": [0.02, 0.1],
            "direction": "forward",
            "foot": True,
        },
    ],
}
BUILT_IN = (
    ("dog", ("模板狗", "Template dog"), {"template": "quadruped", "params": {"color": "orange"}}),
    ("star", ("三脚海星（零件拼的）", "Three-legged starfish (built from parts)"), {"build": STAR}),
    ("humanoid", ("人形", "Humanoid"), {"template": "humanoid"}),
)


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


def _tops(model: mujoco.MjModel, robots: list[str]) -> dict[str, float]:
    """How high each robot reaches standing at home (m), so its name tag floats just above it."""
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key("home").qpos
    mujoco.mj_kinematics(model, data)
    out = {}
    for robot in robots:
        prefix = f"{robot}/"
        tops = []
        for g in range(model.ngeom):
            if not model.body(int(model.geom_bodyid[g])).name.startswith(prefix):
                continue
            rot = data.geom_xmat[g].reshape(3, 3)
            center, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            tops.append(float((data.geom_xpos[g] + rot @ center)[2] + np.abs(rot[2]) @ half))
        out[robot] = round(max(tops), 3) if tops else 0.5
    return out


def _turn_seconds(trained: dict[str, Any]) -> int:
    """About how long learning what follows a kept walk takes, from the robot's own training pace (a round of
    ATTEMPT-second tries): its steering searches its rounds with tries STEER_SECONDS long toward each aim, the turns
    as many rounds as two sides times their starts, then each side is tuned twice to begin out of the walk. With the
    sim running at real time, as in the viewer, an MJCF critter that trained 15 rounds in 4.4 s learned all that in
    109 s (this says 120), and an imported A1 that trained 30 rounds in 11.2 s in 139 s (this says 150). When the sim
    runs as fast as it can while the robot trains, as the tests drive it, a round takes longer and this says too
    much."""
    steer = rhythm_walk.STEER_ROUNDS * len(rhythm_walk.STEER_AIMS) * rhythm_walk.STEER_SECONDS / ATTEMPT
    turns = 2 * rhythm_walk.TURN_STARTS * rhythm_walk.TURN_ROUNDS + 2 * 2 * rhythm_walk.WALKING_ROUNDS
    per_round = trained["seconds"] / max(1, trained["rounds"])
    return max(10, int(round(per_round * (steer + turns) / 10.0)) * 10)


def lesson_reward(energy: float, shove: bool) -> Reward:
    """What a training in this lesson is rewarded for. Shoved, the heading does not count: a shove turns the robot
    whatever its rhythm does, and counting it made the search worse (humanoid with reflexes, 80 rounds: a steady walk
    in 1 of 12 trainings instead of 5, a fall in the replay in 8 instead of 3)."""
    return Reward(energy=energy, survive=shove, heading=0.0 if shove else HEADING_WEIGHT)


def _runs(ctx: LessonContext, **want: Any) -> list[dict[str, Any]]:
    return [r for r in ctx.runs if all(test(r["params"].get(k, 0.0)) for k, test in want.items())]


def _dog_learned(ctx: LessonContext) -> bool:
    return bool(_runs(ctx, robot=lambda v: v == 0, energy=lambda v: v == 0, iterations=lambda v: v >= 20))


def _reward_changed(ctx: LessonContext) -> bool:
    dog = _runs(ctx, robot=lambda v: v == 0)
    return {r["params"]["energy"] for r in dog} >= {1.0, 2.0}


def _star_learned(ctx: LessonContext) -> bool:
    return any(r["metrics"]["distance"] > 0.3 for r in _runs(ctx, robot=lambda v: v == 1))


def _humanoid_tried(ctx: LessonContext) -> bool:
    return bool(_runs(ctx, robot=lambda v: v == 2))


def _reflexes_tried(ctx: LessonContext) -> bool:
    """The humanoid trained with shoves, with and without reflexes."""
    shoved = _runs(ctx, robot=lambda v: v == 2, shove=lambda v: v > 0)
    return {r["params"]["reflexes"] for r in shoved} >= {0.0, 1.0}


STEPS = (
    Step(
        id="rhythm",
        title=_text("节律加随机搜索", "A rhythm and a random search"),
        body=_text(
            "这次不写步态，让机器人自己学。每个关节按一个正弦节律来回摆："
            "目标角 = 初始角 + 偏置 + 幅度 × sin(2π·f·t + 相位)。"
            "节律是现成的，要学的只有每个关节的偏置、幅度和相位。\n"
            "学法和倒立摆那课一样是随机搜索：每一轮把这些数往 8 个随机方向各推一下、正反各试一次，"
            "谁走得远就往谁那边挪。机器人每 4 秒从起跑线重放一次最新学到的节律。\n"
            "先猜，再用模板狗点「开始学」（30 轮，十几秒）。",
            "No gait is written this time: the robot learns one. Every joint swings to a sine rhythm: target = home + "
            "offset + amplitude * sin(2 pi f t + phase). The rhythm is given; only each joint's offset, amplitude "
            "and phase are learned.\n"
            "The learning is the random search from the cart-pole lesson: every round nudges these numbers in 8 "
            "random directions, tries each both ways, and moves toward whatever walked furthest. Every 4 s the "
            "robot replays the newest rhythm from the start line.\n"
            "Guess, then press Learn with the template dog (30 rounds, a dozen seconds).",
        ),
        done=_text(
            "30 轮之后，4 秒能走 1.2 到 1.8 米（我们从 4 个不同的随机起点各学了一次），和手写小跑的最快速度"
            "（0.36 m/s，4 秒 1.4 米）差不多，学 60 轮能到 1.8 到 2.2 米。"
            "姿势有点怪：奖励只说了“走得远、走得直”，没说“走得好看”。"
            "每次学都从一个新的随机起点开始，所以结果会不一样。",
            "After 30 rounds it covers 1.2 to 1.8 m in 4 s (we learned from 4 different random starts), about as fast "
            "as the hand-written trot's top speed (0.36 m/s, 1.4 m in 4 s); 60 rounds reach 1.8 to 2.2 m. It looks "
            "odd: the reward only said 'get far and straight', not 'look nice'. Every training starts from a new "
            "random seed, so results differ.",
        ),
        quiz=Quiz(
            question=_text("学 30 轮以后，4 秒能走多远？", "After 30 rounds, how far does it get in 4 s?"),
            options=(
                _text("还不会动", "It does not move yet"),
                _text("半米左右", "About half a metre"),
                _text("一米多，和手写的小跑差不多快", "Over a metre, about as fast as the hand-written trot"),
            ),
            answer=2,
            explain=_text("只学 36 个数，搜索很快就找到了。", "Only 36 numbers to learn: the search finds them fast."),
            predict=True,
        ),
        check=_dog_learned,
        unlock=("train", "stop", "iterations"),
        show=("labels", "f_rhythm", "f_search"),
    ),
    Step(
        id="reward",
        title=_text("奖励说了算", "The reward decides"),
        body=_text(
            f"奖励 = 走的距离 × (1 − {HEADING_WEIGHT:g} × 转向) − {DRIFT_WEIGHT:g} × 横着偏的距离 − w × 用掉的能量 ÷ "
            "(体重)。转向是朝向转了多少（弧度）：走成弧线就少算。现在 w 是 0：怎么费劲都行。\n"
            "先猜，再把「能耗的代价」调到 0.1 学一次，再调到 0.3 学一次。",
            f"Reward = distance * (1 - {HEADING_WEIGHT:g} * turn) - {DRIFT_WEIGHT:g} * sideways drift - w * energy "
            "spent / (weight), where turn is how far its heading turned, in radians: a walk that curves counts for "
            "less. Now w is 0: any effort goes.\n"
            "Guess, then set Energy cost to 0.1 and learn again, then 0.3.",
        ),
        done=_text(
            "w = 0.1：走 1.0 到 1.4 米，能量只用了 40 到 60 J（不算能量时是 75 到 140 J），步子变小。"
            "w = 0.3：我们学了 4 次，4 次都干脆学会了站着不动：挪一步的代价比走出的距离还大。"
            "机器人只会照着奖励去做，奖励没写清楚，它就会钻空子。",
            "w = 0.1: 1.0 to 1.4 m on 40 to 60 J (75 to 140 J when energy was free), with smaller steps. "
            "w = 0.3: in all 4 of our trainings it learned to stand still: a step costs more than the distance earns. "
            "A robot does exactly what the reward says; if the reward is loose, it finds the loophole.",
        ),
        quiz=Quiz(
            question=_text("能耗的代价调到 0.1，会学出什么？", "With energy cost 0.1, what does it learn?"),
            options=(
                _text("和原来一样", "The same as before"),
                _text("走得近一点，但省一大半力气", "A shorter walk on far less energy"),
                _text("跑得更快", "It runs faster"),
            ),
            answer=1,
            explain=_text("距离和能量在奖励里拔河。", "Distance and energy pull against each other in the reward."),
            predict=True,
        ),
        check=_reward_changed,
        unlock=("energy",),
        show=("f_reward",),
    ),
    Step(
        id="body",
        title=_text("换一个身体", "Another body"),
        body=_text(
            "「哪个机器人」选三脚海星：三条腿、每条两个关节，是用零件拼出来的。"
            "我们的步态程序只认 4 条或 6 条三关节的腿，"
            "它不会走。「能耗的代价」调回 0，不然它多半学会站着不动。\n"
            "如果你来这里之前的世界里有你自己的机器人（自己拼的，或者从文件导入的），它也在列表里。"
            "让它学会以后点「把这个走法存给我的机器人」，"
            "回到你的世界，在聊天里叫它往前走，或者选中它按 W，它就用学到的走法走。\n先猜，再让海星学。",
            "Set Which robot to the three-legged starfish: three legs of two joints each, built from parts. Our gait "
            "code only knows four or six legs of three joints, so it cannot walk this one. Set Energy cost back to "
            "0, or it will most likely learn to stand still.\n"
            "If the world you came from has a robot of your own (built from parts or imported from a file), it is in "
            "the list too. Once it has learned, press Keep this walk for my robot: back in your world, ask for it to "
            "walk in the chat, or select it and press W, and it walks the way it learned.\nGuess, then let the "
            "starfish learn.",
        ),
        done=_text(
            "它也学会了往前挪（我们测到 4 秒 0.55 到 0.75 米）。"
            "同一套方法，不管几条腿、腿长什么样，都能自己找到一种走法。"
            "这就是学习比手写强的地方。",
            "It learns to shuffle forward too (0.55 to 0.75 m in 4 s in our runs). The same method finds a way to walk "
            "for any number and shape of legs, which is where learning beats writing gaits by hand.",
        ),
        quiz=Quiz(
            question=_text("三脚海星能学会走吗？", "Can the starfish learn to walk?"),
            options=(_text("不能，没有现成的步态", "No, there is no gait for it"), _text("能", "Yes")),
            answer=1,
            explain=_text("学习不需要现成的步态。", "Learning needs no ready-made gait."),
            predict=True,
        ),
        check=_star_learned,
        unlock=("robot", "freq", "keep"),
    ),
    Step(
        id="two_legs",
        title=_text("两条腿呢", "And on two legs?"),
        body=_text(
            "选人形，再学一次。先猜。",
            "Pick the humanoid and learn again. Guess first.",
        ),
        done=_text(
            "它学会了往前一扑：扑出去三分之一米左右就倒（我们学了 6 次，6 次都是这样）。"
            "两条腿一碰就倒，每轮试的大多都会倒，搜索只能在倒法里挑扑得最远的。"
            "奖励看的是走了多远，倒下只扣一个固定的分，早倒晚倒都一样，又被钻了空子。"
            "下一步给它两样东西：站住也算分，和能看身体倾斜的“反射”。",
            "It learns to lunge: a third of a metre forward, then down (all 6 of our trainings). Two legs topple at "
            "the slightest nudge, so most tries fall, and of those the search keeps whichever falls furthest forward: "
            "another loophole: the reward is about distance, and a fall costs the same however early it comes. Next it "
            "gets two things: points for staying up, and reflexes that feel the body tilt.",
        ),
        quiz=Quiz(
            question=_text("人形用这个方法会学出什么？", "What does the humanoid learn this way?"),
            options=(
                _text("像人一样走", "To walk like a person"),
                _text("往前一扑，然后倒", "To lunge forward, then fall"),
                _text("站着不动", "To stand still"),
            ),
            answer=1,
            explain=_text(
                "奖励看的是走了多远，倒下扣的分早倒晚倒都一样；搜索碰到的大多是倒法，扑得最远的那种分最高。",
                "The reward is about distance, and a fall costs the same however early it comes: of the falls the "
                "search meets, a lunge scores best.",
            ),
            predict=True,
        ),
        check=_humanoid_tried,
    ),
    Step(
        id="reflexes",
        title=_text("给节律加上反射", "Give the rhythm reflexes"),
        body=_text(
            "节律是“盲”的，从不看传感器。反射让每个关节再多看 4 个数：身体的横滚、俯仰和它们的变化速度，"
            "每个数乘一个也要学的系数，加到关节目标上。\n"
            "打开「训练时推它，站住也算分」：每次试都被随机推一把（0.4 m/s），倒了就扣掉没站住的那几秒；"
            "朝向不再算分，推一把本来就会让它转。"
            "先关着反射学一次，再打开反射学一次（各 80 轮，二十秒左右）。学完会自动推 12 次，看站住几次。先猜。",
            "The rhythm is blind: it never reads a sensor. Reflexes give every joint 4 more numbers to look at: "
            "the body's roll, pitch and how fast they change, each times a gain that is learned too, added to the "
            "joint's target.\n"
            "Switch on Shove it while training, points for staying up: every try gets one random shove (0.4 m/s), "
            "and a fall costs the seconds it did not stay up; the heading no longer counts, as a shove turns it "
            "anyway. Learn once without reflexes, then once with them (80 rounds each, about twenty seconds). "
            "Afterwards it is shoved 12 times to count how often it stays up. Guess first.",
        ),
        done=_text(
            "不带反射：它学会了几乎站着不动，推 12 次站住 2 到 10 次（原地站着不学是 8 次）。"
            "带反射：我们学了 6 次，4 次学出了边走边稳的步子，走 0.3 到 0.55 米不倒，推 12 次站住 6 到 12 次；"
            "另外 2 次学崩了，一推就倒。反射让它能应付意外，可要学的数从 60 个变成 140 个，随机搜索就不稳了。"
            "G1 的神经网络有几万个数，所以要用梯度、在仿真里试几千万次：这就是深度强化学习。",
            "Without reflexes it learns to stand nearly still and stays up for 2 to 10 of 12 shoves (standing "
            "untrained manages 8). With reflexes, 4 of our 6 trainings found a steady walk, 0.3 to 0.55 m without "
            "falling, staying up for 6 to 12 of 12 shoves; the other 2 collapsed and fell at the first shove. "
            "Reflexes let it handle surprises, but the numbers to learn grow from 60 to 140 and random search gets "
            "unreliable. The G1's network has tens of thousands of numbers, so it is trained with gradients over "
            "tens of millions of tries: deep reinforcement learning.",
        ),
        quiz=Quiz(
            question=_text("加了反射，被推的时候会怎样？", "With reflexes, what happens when it is shoved?"),
            options=(
                _text("一定更稳", "It is always steadier"),
                _text("常常更稳，但有时学崩", "Often steadier, but some trainings collapse"),
                _text("没区别", "No difference"),
            ),
            answer=1,
            explain=_text(
                "反馈有用，可要学的数多了，搜索就难了。", "Feedback helps, but more numbers make the search harder."
            ),
            predict=True,
        ),
        check=_reflexes_tried,
        unlock=("reflexes", "shove"),
        show=("f_reflex",),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text("节律加随机搜索又快又简单。", "A rhythm plus random search is quick and simple."),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("G1 为什么不用这种方法走路？", "Why does the G1 not walk this way?"),
            options=(
                _text("太慢了", "It is too slow"),
                _text(
                    "它要随时看传感器调整，节律是盲的",
                    "It must correct from its sensors all the time; a rhythm is blind",
                ),
                _text("G1 的关节太多", "The G1 has too many joints"),
            ),
            answer=1,
            explain=_text(
                "对。会看传感器的策略才能站稳，这就需要神经网络和深度强化学习。",
                "Right. Only a policy that reads its sensors can stay up, and that takes a neural network and deep RL.",
            ),
        ),
    ),
)


class Watch:
    """Replays the newest rhythm from the start line every ATTEMPT seconds and samples the live walk."""

    rate_hz = 50.0

    def __init__(self, lesson: LearnToWalkLesson) -> None:
        self.lesson = lesson
        self.next_sample = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.lesson.tick(model, data)
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            self.lesson.samples.append(self.lesson.sample(data))

    def status(self) -> dict[str, Any]:
        return {"controller": "learn-to-walk-watch"}


class LearnToWalkLesson(Lesson):
    id = "learn_to_walk"
    chapter = _text("强化学习", "Reinforcement learning")
    title = _text("强化学习（二）：任何形状的机器人都能学会走", "RL II: any body can learn to walk")
    summary = _text(
        "不写步态：狗、零件拼的三脚海星、你自己的机器人，各自学出一种走法；再看奖励怎么左右它，两条腿为什么学不会。",
        "No gait written: a dog, a starfish built from parts and your own robot each learn a way to walk; then see "
        "how the reward steers it, and why two legs will not learn this way.",
    )
    minutes = 15
    plots = (
        Plot(
            "walk",
            _text("这一趟走了多远", "This attempt's distance"),
            "m",
            (Series("x", _text("前进", "forward"), "ink"),),
            window=6.0,
            span=0.5,
        ),
    )
    steps = STEPS
    bench = Bench(
        title=_text("实验记录：学习曲线", "Experiment record: learning curves"),
        x=_text("训练轮数", "round"),
        x_unit="",
        y=_text("4 秒走了", "distance in 4 s"),
        y_unit="m",
        params=SETTINGS,
        short=(
            _text("机器人", "robot"),
            _text("能耗代价", "energy"),
            _text("频率", "freq"),
            _text("轮", "rounds"),
            _text("反射", "reflexes"),
            _text("推", "shove"),
        ),
        metrics=(
            Metric("distance", _text("走了", "covered"), "m", lower_is_better=False),
            Metric("survived", _text(f"推 {PUSH_TRIALS} 次站住", f"up of {PUSH_TRIALS}"), "", lower_is_better=False),
            Metric("energy", _text("能量", "energy"), "J", ranked=False),
            Metric("seconds", _text("学了", "trained"), "s", ranked=False),
        ),
    )

    def __init__(self) -> None:
        self.playground = None
        self.ctx: LessonContext | None = None
        self.robots: list[tuple[str, tuple[str, str], dict[str, Any]]] = list(BUILT_IN)
        self.own: tuple[str, Path] | None = None  # the learner's robot: (its name in their world, the world folder)
        self.turning: threading.Thread | None = None  # learning to turn after a walk was kept
        self.controls = self._controls()
        self.model: mujoco.MjModel | None = None
        self.start: np.ndarray | None = None
        self.bodies: dict[str, Body] = {}
        self.homes: dict[str, np.ndarray] = {}
        self.rhythms: dict[str, Rhythm] = {}
        self.players: dict[str, RhythmController] = {}
        self.attempt_t0 = 0.0
        self.attempt_x0 = 0.0
        self.samples: list[dict[str, float]] = []
        self.solo: dict[str, mujoco.MjModel] = {}
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.progress: dict[str, Any] | None = None  # {"robot", "round", "of", "distance"} while training
        self.trainings = 0
        self.replays = 0
        self.shoved = False
        self.trained: dict[str, dict[str, Any]] = {}  # robot key -> its last training's result

    def _controls(self) -> tuple[Control, ...]:
        keep = (Control("keep", "button", _text("把这个走法存给我的机器人", "Keep this walk for my robot")),)
        return (
            Control("train", "button", _text("开始学", "Learn")),
            Control("stop", "button", _text("停", "Stop")),
            Control(
                "robot",
                "choice",
                _text("哪个机器人", "Which robot"),
                options=tuple(_text(*label) for _, label, _ in self.robots),
            ),
            Control(
                "energy",
                "choice",
                _text("能耗的代价 w", "Energy cost w"),
                options=tuple(_text(f"{w:g}", f"{w:g}") for w in ENERGY),
            ),
            Control("freq", "slider", _text("节律频率", "Rhythm frequency"), 1.0, 3.0, 0.5, 2.0, "Hz"),
            Control("iterations", "slider", _text("学多少轮", "Rounds"), 10, 100, 5, 30),
            Control("reflexes", "toggle", _text("反射（看身体倾斜）", "Reflexes (feel the body tilt)")),
            Control(
                "shove", "toggle", _text("训练时推它，站住也算分", "Shove it while training, points for staying up")
            ),
            *(keep if self.own is not None else ()),
        )

    # ---- the world -------------------------------------------------------------------------------------

    def build(self) -> mujoco.MjModel:
        own = self._own_robot()
        if own is not None:
            self.robots.append(own)
            self.controls = self._controls()
            self.steps = (self._your_robot_first(STEPS[0]), *STEPS[1:])
        robots = [
            {"name": key, **entry, "pos": [START_X, (i - (len(self.robots) - 1) / 2) * LANE]}
            for i, (key, _, entry) in enumerate(self.robots)
        ]
        world = World.model_validate({"name": "learn_to_walk", "floor": {"style": "concrete"}, "robots": robots})
        self.model = compile_world(world, self._world_dir()).model
        return self.model

    def _world_dir(self) -> Path | None:
        """The learner's world folder, where a robot they imported from a file finds it (a relative path in their
        world.yaml means one next to it), as their world itself compiles."""
        return self.own[1] if self.own is not None else None

    def _your_robot_first(self, step: Step) -> Step:
        """A learner who came to teach their own robot (an agent sends them here) can from the first step."""
        assert self.own is not None
        name = self.own[0]
        return replace(
            step,
            body=_text(
                f"{step.body['zh']}\n你自己的机器人 {name} 也在「哪个机器人」里：想直接教它，选它、点「开始学」，"
                "学好了点「把这个走法存给我的机器人」。",
                f"{step.body['en']}\nYour own robot, {name}, is under Which robot too: to teach it right away, pick "
                "it, press Learn, and once it walks press Keep this walk for my robot.",
            ),
            unlock=(*step.unlock, "robot", "keep"),
        )

    def _own_robot(self) -> tuple[str, tuple[str, str], dict[str, Any]] | None:
        """A robot from the world the learner came from that can walk by rhythm, if there is one; one that no
        gait code knows comes first, since that is the one that needs to learn. Robots imported from a file count
        too: their world compiles them against its folder, and so does the lesson."""
        store = getattr(self.playground, "store", None)
        if store is None:
            return None
        try:
            world = store.load()
        except (OSError, ValueError):
            return None
        found = []
        for entry in world.robots:
            spec = entry.model_dump(exclude={"name", "pos", "yaw"}, exclude_none=True)
            try:
                solo = compile_world(
                    World.model_validate({"name": "solo", "robots": [{"name": "yours", **spec}]}), store.dir
                ).model
                Body.find(solo, "yours")
            except (CPGError, ValueError):
                continue
            found.append((robot_kind(solo, "yours", entry.template) != Kind.OTHER, entry.name, spec, solo))
        if not found:
            return None
        _, name, spec, solo = min(found, key=lambda f: f[0])  # stable: the first gait-less one, else the first
        self.solo["yours"] = solo
        self.own = (name, store.dir)
        return "yours", (f"你的机器人：{name}", f"Your robot: {name}"), spec

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.start = data.qpos.copy()
        home = model.key("home").qpos if model.nkey else model.qpos0
        self.bodies = {key: Body.find(model, key) for key, _, _ in self.robots}
        self.homes = {key: home[b.qadr].copy() for key, b in self.bodies.items()}
        for key, b in self.bodies.items():
            self.rhythms.setdefault(key, Rhythm.still(b.joints))
        self.players = {key: RhythmController(model, key, self.rhythms[key], self.homes[key]) for key in self.bodies}
        self._new_attempt(data)
        return {**{key: p for key, p in self.players.items()}, "watch": Watch(self)}

    @property
    def active(self) -> str:
        assert self.ctx is not None
        return self.robots[int(self.ctx.values["robot"])][0]

    # ---- the live replay ---------------------------------------------------------------------------------

    def _new_attempt(self, data: mujoco.MjData) -> None:
        """The robot being taught back on its start line, still; its rhythm starts over."""
        assert self.start is not None
        b = self.bodies[self.active]
        data.qpos[b.qpos] = self.start[b.qpos]
        data.qvel[b.qvel] = 0.0
        self.attempt_t0 = float(data.time)
        self.attempt_x0 = float(self.start[b.qpos[0]])  # the free joint's x comes first
        self.players[self.active].restart(self.attempt_t0)
        self.shoved = False
        self.replays += 1

    def tick(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        elapsed = float(data.time) - self.attempt_t0
        if elapsed >= ATTEMPT:
            self._new_attempt(data)
            mujoco.mj_forward(model, data)
        elif self.ctx is not None and self.ctx.values.get("shove") and not self.shoved and elapsed >= SHOVE_AT:
            b = self.bodies[self.active]  # the replay gets shoved too, so you can see what the reflexes do
            angle = self.replays * 2.399  # a different direction every time (the golden angle)
            data.qvel[b.qvel[0]] += SHOVE * np.cos(angle)
            data.qvel[b.qvel[1]] += SHOVE * np.sin(angle)
            self.shoved = True

    def sample(self, data: mujoco.MjData) -> dict[str, float]:
        b = self.bodies[self.active]
        return {"t": float(data.time), "x": float(data.xpos[b.trunk][0]) - self.attempt_x0}

    def drain_samples(self) -> list[dict[str, float]]:
        out, self.samples = self.samples, []
        return out

    # ---- training --------------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control == "stop":
            self.stopping.set()
        elif control == "keep":
            ctx.results["summary"] = self._keep(ctx)
        elif control == "robot":
            self._new_attempt(data)
        elif control == "train" and (self.thread is None or not self.thread.is_alive()):
            key = self.active
            settings = {k: float(ctx.values[k]) for k in SETTINGS}
            # Every training starts from standing still.
            rhythm = Rhythm.still(self.bodies[key].joints, settings["freq"], reflexes=bool(settings["reflexes"]))
            self.rhythms[key] = rhythm
            self.players[key].rhythm = rhythm
            self.stopping.clear()
            self.progress = {"robot": key, "round": 0, "of": int(settings["iterations"]), "distance": 0.0}
            self.thread = threading.Thread(target=self._train, args=(key, rhythm, settings, ctx), daemon=True)
            self.thread.start()

    def _solo(self, key: str) -> mujoco.MjModel:
        """The robot alone at the origin, for training away from the live world."""
        if key not in self.solo:
            entry = next(e for k, _, e in self.robots if k == key)
            self.solo[key] = compile_world(
                World.model_validate({"name": "solo", "robots": [{"name": key, **entry}]}), self._world_dir()
            ).model
        return self.solo[key]

    def _train(self, key: str, rhythm: Rhythm, settings: dict[str, float], ctx: LessonContext) -> None:
        shove = bool(settings["shove"])
        reward = lesson_reward(ENERGY[int(settings["energy"])], shove)
        trainer = Trainer(self._solo(key), key, reward, ATTEMPT, rhythm.freq)
        curve: list[float] = []

        def progress(round_: int, walk: dict[str, float]) -> None:
            curve.append(walk["distance"])
            self.progress = {
                "robot": key,
                "round": round_,
                "of": int(settings["iterations"]),
                "distance": walk["distance"],
            }

        self.trainings += 1  # a new random start every time: results differ, as they do in real training
        log = trainer.train(
            rhythm,
            iterations=int(settings["iterations"]),
            seed=self.trainings,
            pushes=SHOVE if shove else 0.0,
            callback=progress,
            stop=self.stopping.is_set,
        )
        final = trainer.rollout(rhythm)
        shoved = trainer.push_test(rhythm, SHOVE, PUSH_TRIALS)  # how it copes with surprises it may never have met
        self.progress = None
        result = {
            "robot": key,
            "distance": round(final["distance"], 2),
            "energy": round(final["energy"], 1),
            "fell": bool(final["fell"]),
            "seconds": round(log.seconds, 1),
            "rounds": len(curve),
            "survived": int(shoved["survived"]),
        }
        ctx.results["training"] = result
        fell = _text("，摔倒了", ", and fell") if result["fell"] else _text("", "")
        # Outside the reward step, where standing still is the point, a walk lost to the energy cost is a slip
        # (the cost carried over from that step), so say how to get the walk back.
        lost_to_energy = result["distance"] < KEEP_MIN and reward.energy > 0 and ctx.step != "reward"
        hint = (
            _text(
                "。能耗有代价时，站着不动最划算：把「能耗的代价」调回 0 再学。",
                ". With an energy cost, standing still pays best: set Energy cost back to 0 and learn again.",
            )
            if lost_to_energy
            else _text("", "")
        )
        ctx.results["summary"] = {
            "zh": f"学了 {result['rounds']} 轮（{result['seconds']:g} 秒）：4 秒走 {result['distance']:g} 米，"
            f"用了 {result['energy']:g} J{fell['zh']}；推 {PUSH_TRIALS} 次站住 {result['survived']} 次{hint['zh']}",
            "en": f"{result['rounds']} rounds ({result['seconds']:g} s): {result['distance']:g} m in 4 s "
            f"on {result['energy']:g} J{fell['en']}; stayed up for {result['survived']} of {PUSH_TRIALS} shoves"
            f"{hint['en']}",
        }
        metrics = {k: result[k] for k in ("distance", "survived", "energy", "seconds")}
        ctx.record_run(settings, metrics, list(range(1, len(curve) + 1)), curve)
        self.trained[key] = result

    def _keep(self, ctx: LessonContext) -> Text:
        """Write what the learner's robot learned into their world's folder, or say why not yet."""
        if self.own is None or self.active != "yours":
            return _text(
                "先在「哪个机器人」里选你的机器人，让它学一次。",
                "Pick your robot under Which robot and let it learn first.",
            )
        if self.thread is not None and self.thread.is_alive():
            return _text("等它学完再存。", "Wait until it has finished learning.")
        if self.turning is not None and self.turning.is_alive():
            return _text(
                "它还在学转弯，学完会自动存好。", "It is still learning to turn; that is kept by itself when done."
            )
        result = self.trained.get("yours")
        if result is None:
            return _text("你的机器人还没学过，先点「开始学」。", "Your robot has not learned yet: press Learn first.")
        if result["distance"] < KEEP_MIN:
            return _text(
                f"它学到的走法 4 秒只走了 {result['distance']:g} 米，存了也走不动。多学几轮再存。",
                f"What it learned covers only {result['distance']:g} m in 4 s, too little to keep. Learn for more "
                "rounds first.",
            )
        name, folder = self.own
        rhythm = self.rhythms["yours"]
        assert self.model is not None
        joints = joint_names(self.model, self.bodies["yours"])
        walk = SavedWalk(
            robot=name,
            joints=joints,
            # What it learned around, so it plays the same if the world's home pose changes (an MJCF robot's did).
            home={j: float(v) for j, v in zip(joints, self.homes["yours"], strict=True)},
            freq=rhythm.freq,
            reflexes=rhythm.reflexes,
            theta=[float(v) for v in rhythm.theta],
            speed=max(0.0, result["distance"]) / ATTEMPT,
            distance=result["distance"],
            seconds=ATTEMPT,
            survived=result["survived"],
            shoves=PUSH_TRIALS,
            switching="fade",
        )
        try:
            path = save_walk(folder, walk)
        except OSError as exc:
            return _text(f"没存上：{exc}", f"Could not keep it: {exc}")
        ctx.results["kept"] = {"robot": name, "file": str(path), "speed": round(walk.speed, 3), "turns": False}
        drift = heading_drift(self._solo("yours"), "yours", walk)
        self.turning = threading.Thread(target=self._learn_turns, args=(walk, folder, ctx), daemon=True)
        self.turning.start()
        wait = _turn_seconds(result)
        curve, check = abs(math.degrees(drift or 0.0)), rhythm_walk.STEER_CHECK
        then = _text(
            f"它先学怎么保持方向（不管的话，它的走法 {check:.0f} 秒会拐 {curve:.0f} 度），再学原地转弯，",
            f"It first learns to hold its heading (left alone, its walk turns {curve:.0f} degrees in {check:.0f} s), "
            "then ",
        )
        if drift is None:
            then = _text("它先学怎么保持方向，再学原地转弯，", "It first learns to hold its heading, then ")
        return _text(
            f"存好了：{name} 学会的走法（4 秒 {result['distance']:g} 米，约 {walk.speed:.2f} 米/秒）放进了你的世界。"
            f"{then['zh']}向左、向右各学两次留下好的那次，再练从走路中直接起转（一共约 {wait} 秒），学完也会存好。"
            f"退出课程回到你的世界，在聊天里让 {name} 走，或者选中它按 W 往前、A 和 D 原地转。",
            f"Kept: {name}'s walk ({result['distance']:g} m in 4 s, about {walk.speed:.2f} m/s) is in your world now. "
            f"{then['en']}goes on to learn to turn on the spot, each way twice keeping the better, then to start "
            f"each turn straight out of its walk (about {wait} s in all), kept as well when done. "
            f"Leave the lesson, then ask for {name} to walk in the chat, or select it and press W to go forward and "
            "A or D to turn on the spot.",
        )

    def _learn_turns(self, walk: SavedWalk, folder: Path, ctx: LessonContext) -> None:
        """After the walk is kept: it learns to hold its heading, then the robot learns to turn on the spot each way,
        on the robot alone, each kept as soon as it is learned."""
        try:
            walk = learn_steering(self._solo("yours"), "yours", walk, seed=self.trainings)
            save_walk(folder, walk)
            if walk.steer is not None:
                ctx.results["kept"] = {**ctx.results.get("kept", {}), "holds_heading": True}
            turned = learn_turns(self._solo("yours"), "yours", walk, seed=self.trainings)
            save_walk(folder, turned)
        except (OSError, ValueError, CPGError) as exc:
            ctx.results["summary"] = _text(f"转弯没学成：{exc}", f"Turning was not learned: {exc}")
            return
        rate = math.degrees(turned.turn_rate or 0.0)
        ctx.results["kept"] = {**ctx.results.get("kept", {}), "turns": True, "turn_deg_s": round(rate)}
        ctx.results["summary"] = _text(
            f"{walk.robot} 也学会了原地转弯：慢的那边每秒约 {rate:.0f} 度。回到你的世界，选中它按 A、D 就能转。",
            f"{walk.robot} also learned to turn on the spot: about {rate:.0f} degrees a second on its slower side. "
            "Back in your world, select it and press A or D to turn.",
        )

    def close(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=10.0)

    # ---- what the panel shows ------------------------------------------------------------------------------

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        key = self.active
        b = self.bodies[key]
        progress = self.progress
        readout = _text("", "")
        if progress is not None:
            readout = _text(
                f"正在学：第 {progress['round']} / {progress['of']} 轮，最新的节律走 {progress['distance']:.2f} 米",
                f"Learning: round {progress['round']} of {progress['of']}, "
                f"newest rhythm walks {progress['distance']:.2f} m",
            )
        joints, w = b.joints, ENERGY[int(ctx.values["energy"])]
        heading = lesson_reward(w, bool(ctx.values["shove"])).heading
        distance = (
            _text(f"距离 × (1 − {heading:g} × 转向)", f"distance * (1 - {heading:g} * turn)")
            if heading
            else _text("距离", "distance")  # shoved, the heading does not count
        )
        return {
            "follow": b.trunk,
            "readout": readout,
            "formula": [
                {
                    "show": "f_rhythm",
                    "label": _text("关节目标", "Joint target"),
                    "expr": _text(
                        "初始 + 偏置 + 幅度 × sin(2π·f·t + 相位)", "home + offset + amplitude * sin(2 pi f t + phase)"
                    ),
                    "value": f"f = {ctx.values['freq']:g} Hz",
                    "color": "ink",
                },
                {
                    "show": "f_rhythm",
                    "label": _text("要学的数", "Numbers to learn"),
                    "expr": _text(f"3 × {joints} 个关节", f"3 * {joints} joints"),
                    "value": f"{3 * joints}",
                    "color": "amber",
                },
                {
                    "show": "f_search",
                    "label": _text("每轮试几次", "Tries per round"),
                    "expr": _text("8 个方向 × 正反 2 次 + 1 次看效果", "8 directions * both ways + 1 to check"),
                    "value": "17",
                    "color": "blue",
                },
                {
                    "show": "f_reflex",
                    "label": _text("反射", "Reflexes"),
                    "expr": _text(
                        "关节修正 = 系数 × (横滚, 俯仰, 横滚速度, 俯仰速度)",
                        "joint correction = gains * (roll, pitch, roll rate, pitch rate)",
                    ),
                    "value": _text(f"多 {4 * joints} 个数", f"+{4 * joints} numbers")
                    if ctx.values["reflexes"]
                    else _text("关", "off"),
                    "color": "green",
                },
                {
                    "show": "f_reward",
                    "label": _text("奖励", "Reward"),
                    "expr": _text(
                        f"{distance['zh']} − {DRIFT_WEIGHT:g} × 横偏 − {w:g} × 能量 ÷ 体重",
                        f"{distance['en']} - {DRIFT_WEIGHT:g} * drift - {w:g} * energy / weight",
                    ),
                    "value": f"w = {w:g}",
                    "color": "violet",
                },
            ],
        }

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        bodies = {key: Body.find(model, key) for key, _, _ in self.robots}
        tops = _tops(model, [key for key, _, _ in self.robots])
        return {
            "kind": "faceoff",
            "entrants": [
                {
                    "name": key,
                    "label": _text(*label),
                    "sub": None,
                    "color": ("amber", "violet", "blue", "green")[i % 4],
                    "base": [START_X, (i - (len(self.robots) - 1) / 2) * LANE, 0.0],
                    "body": bodies[key].trunk,
                    "top": tops[key],  # name tags just above each
                    "cloud": [],
                }
                for i, (key, label, _) in enumerate(self.robots)
            ],
            "targets": [],
        }

    def camera(self) -> dict[str, list[float]]:
        # Side on to the lane of the robot being taught (the dog at first), a little ahead of it; the camera then
        # walks with whichever robot learns. A page reloaded mid-lesson gets the lane of the one picked by then.
        index = int(self.ctx.values["robot"]) if self.ctx is not None else 0
        y = (index - (len(self.robots) - 1) / 2) * LANE
        return {"eye": [START_X + 0.3, y - 2.6, 1.3], "target": [START_X + 0.3, y, 0.15]}
