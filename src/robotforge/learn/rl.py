"""Lesson: reinforcement learning, watched live.

A cart-pole learns to balance with ARS (random search on a linear policy). Nobody tells it how to push:
it only gets a score for staying up. The viewer shows every attempt with the newest policy while the
training curve climbs, round by round.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any

import mujoco
import numpy as np

from robotforge.control.actuation import ActionScale
from robotforge.control.ars import LinearPolicy, train_ars
from robotforge.control.env import RobotEnv
from robotforge.control.tasks import make_task
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Bench, Control, Lesson, LessonContext, Metric, Plot, Quiz, Series, Step

EPISODE_SECONDS = 5.0  # training episodes
SHOW_SECONDS = 8.0  # live attempts end after this long (a good policy just keeps going)
FALLEN_RAD = 0.6  # pole angle at which an attempt counts as fallen
ITERATIONS = 30
PACE = 0.7  # s per training round, so the learner can watch it improve
TRAINED_RETURN = 8.0  # mean score per training round that counts as "learned" (10 is a full episode)
PARAMS = ("lr", "noise", "dirs")  # the training knobs each recorded run remembers
PUSH_SPEED = 1.2  # rad/s added to the pole by Push
PUSH_AFTER = 1.0  # s into a fresh attempt: a push waiting for the cart to start over comes then


class Live:
    """Runs the current policy on the live cart-pole, starting a new attempt whenever the pole falls."""

    rate_hz = 100.0

    def __init__(self, lesson: RLLesson, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.lesson = lesson
        self.task = make_task("balance", model, "pole")
        self.scale = ActionScale(model, self.task.actuators)
        self.hinge = int(model.joint("pole/hinge").qposadr[0])
        self.hinge_dof = int(model.joint("pole/hinge").dofadr[0])
        self.slider = int(model.joint("pole/slider").qposadr[0])
        self.home = data.qpos.copy()
        self.rng = np.random.default_rng(0)
        self.started = float(data.time)
        self.attempt = 0
        self.last_survived = 0.0
        self.pending_restart = True
        self.pushed_at: float | None = None
        self.push_in: float | None = None  # s into the next attempt to push, when the push waits for a restart

    def restart(self, data: mujoco.MjData) -> None:
        data.qpos[:] = self.home
        data.qpos[self.hinge] += self.rng.uniform(-0.1, 0.1)  # a small random tilt, like the training episodes
        data.qvel[:] = 0
        self.started = float(data.time)
        self.attempt += 1
        self.pending_restart = False

    def angle(self, data: mujoco.MjData) -> float:
        return float(data.qpos[self.hinge])

    def push(self, data: mujoco.MjData) -> None:
        data.qvel[self.hinge_dof] += PUSH_SPEED
        self.pushed_at = float(data.time)
        self.push_in = None

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.pending_restart:
            self.restart(data)
        elapsed = float(data.time) - self.started
        if self.push_in is not None and elapsed >= self.push_in:
            self.push(data)
        # A push gets its full answer before the attempt can time out: did the policy catch it or not?
        just_pushed = self.pushed_at is not None and float(data.time) - self.pushed_at < 4.0
        timed_out = elapsed > SHOW_SECONDS and not just_pushed
        if abs(self.angle(data)) > FALLEN_RAD or abs(float(data.qpos[self.slider])) > 0.9 or timed_out:
            self.last_survived = elapsed
            self.lesson.attempt_ended(elapsed, abs(self.angle(data)) > FALLEN_RAD)
            self.restart(data)
            return
        policy = self.lesson.policy
        if policy is None:  # no controller: the motor does nothing
            data.ctrl[self.task.actuators] = 0.0
            return
        action = policy.act(policy.weights, self.task.observation(data))
        data.ctrl[self.task.actuators] = self.scale.to_ctrl(np.asarray(action, float))

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-rl", "attempt": self.attempt}


def _saw_it_fall(ctx: LessonContext) -> bool:
    return "fell_uncontrolled" in ctx.events


def _tried_random(ctx: LessonContext) -> bool:
    return ctx.results.get("random_tries", 0) >= 3


def _trained(ctx: LessonContext) -> bool:
    return ctx.results.get("train", {}).get("mean", 0.0) >= TRAINED_RETURN


def _compared_settings(ctx: LessonContext) -> bool:
    """At least two training runs in the record, with different settings."""
    settings = {tuple(sorted(run["params"].items())) for run in ctx.runs}
    return len(settings) >= 2


def _survived_push(ctx: LessonContext) -> bool:
    return "survived_push" in ctx.events


STEPS = (
    Step(
        id="fall",
        title={"zh": "不控制会怎样", "en": "What happens without control"},
        body={
            "zh": "这是一个倒立摆：小车上立着一根杆，小车左右动才能让杆不倒。\n"
            "现在没有任何控制。点「放手」看看会发生什么。",
            "en": "This is a cart-pole: a pole stands on a cart, and only moving the cart left and right keeps it up.\n"
            "Right now nothing controls it. Press Let go and watch.",
        },
        done={
            "zh": "杆一下就倒了。倒立摆天生不稳：稍微一歪，重力就让它越歪越快。",
            "en": "It falls at once. A cart-pole is unstable by nature: "
            "the slightest lean, and gravity makes it worse.",
        },
        check=_saw_it_fall,
        unlock=("release",),
    ),
    Step(
        id="reward",
        title={"zh": "奖励：只告诉它什么算好", "en": "Reward: say what is good, not how"},
        body={
            "zh": "强化学习不教机器人“该往哪推”，只给它打分（奖励）。\n下面哪个打分方式能教会它平衡？",
            "en": "Reinforcement learning never tells the robot which way to push; it only gives a score (a reward).\n"
            "Which score would teach it to balance?",
        },
        done={
            "zh": "机器人自己去找“怎样得分最高”。分数设计得好，它就能学会；设计得不好，它会学会钻空子。",
            "en": "The robot works out for itself how to score the most. "
            "A good score teaches the skill; a careless one teaches it to game the score.",
        },
        quiz=Quiz(
            question={"zh": "选一个打分方式", "en": "Pick a score"},
            options=(
                {"zh": "杆倒下时给 +10 分", "en": "+10 when the pole falls"},
                {
                    "zh": "杆每多立住一刻就 +1 分，小车跑太远扣分",
                    "en": "+1 for every moment the pole stays up, minus a little when the cart wanders off",
                },
                {"zh": "每一刻告诉它该往左还是往右推", "en": "Tell it every moment whether to push left or right"},
            ),
            answer=1,
            explain={
                "zh": "对：奖励只说“立住越久越好”，不说怎么做到。第三个选项其实是监督学习，需要有人知道正确答案。",
                "en": "Right: the reward only says 'up longer is better', not how. The third option is supervised "
                "learning, which needs someone who already knows the answer.",
            },
        ),
    ),
    Step(
        id="random",
        title={"zh": "瞎试行不行", "en": "Could random guessing work?"},
        body={
            "zh": "它的“大脑”很简单：推力 = 一组权重 × 它看到的东西（杆的角度、角速度、小车位置和速度）。\n"
            "点三次「随机试一次」：每次随便挑一组权重，看看能立住多久。",
            "en": "Its 'brain' is simple: push = a set of weights times what it senses (pole angle and spin, cart "
            "position and speed).\nPress Try random weights three times: each picks weights at random. "
            "How long does the pole stay up?",
        },
        done={
            "zh": "随便碰运气几乎不可能成功：可能的权重组合太多了。需要一个“每次往好一点的方向挪”的办法。",
            "en": "Pure luck almost never works: there are far too many possible weights. It needs a way to move a "
            "little toward better each time.",
        },
        check=_tried_random,
        unlock=("random",),
    ),
    Step(
        id="train",
        title={"zh": "开始训练：随机搜索 (ARS)", "en": "Train it: random search (ARS)"},
        body={
            "zh": "点「开始训练」。每一轮，它把权重往 8 个随机方向各挪一点、正反各试一次（16 次尝试），"
            "留下得分最高的几个方向，再朝它们走一小步。\n"
            "右边的曲线是每一轮的平均得分（满分 10）；画面里的小车一直在用最新学到的权重。",
            "en": "Press Start training. Each round it nudges the weights in 8 random directions, tries each both "
            "ways (16 attempts), keeps the best directions and takes a small step toward them.\n"
            "The curve on the right is each round's average score (10 is perfect); the cart you see always uses "
            "the newest weights.",
        },
        done={
            "zh": "几十轮之后它学会了平衡。没人教它怎么推，它只知道“立住越久分越高”。这就是强化学习。",
            "en": "After a few dozen rounds it balances. Nobody showed it how to push; it only knew that staying up "
            "longer scores more. That is reinforcement learning.",
        },
        check=_trained,
        unlock=("train",),
        show=("training",),
    ),
    Step(
        id="tune",
        title={"zh": "调一调超参数", "en": "Tune the hyperparameters"},
        body={
            "zh": "训练本身也有旋钮，叫“超参数”。打开「快进」省得等，然后换一个设置再训练一次，"
            "右下角的实验记录会把每次的训练曲线叠在一起：\n"
            "· 步长 α：每轮朝好方向走多大一步。\n"
            "· 探索噪声 ν：每次试的时候把权重晃多大。\n"
            "· 方向数：每轮试几个方向（每个方向正反各试一次）。\n"
            "先猜一猜：步长调到 0.02 会怎样？噪声调到 1.0 呢？",
            "en": "Training has knobs of its own, called hyperparameters. Turn on Fast-forward to skip the waiting, "
            "then train again with another setting; the experiment record at the bottom right lays every "
            "training curve over the others:\n"
            "· Step size α: how far each round moves toward the good directions.\n"
            "· Exploration noise ν: how much the weights are shaken for each try.\n"
            "· Directions: how many directions each round tries (each one both ways).\n"
            "Guess first: what happens with a step size of 0.02? With noise at 1.0?",
        },
        done={
            "zh": "步长太小，30 轮都学不会；太大，得分忽高忽低、反而学得晚。噪声太大，每次尝试都太随机，"
            "分不出哪个方向真的好。方向越多，每一轮的判断越准，但要跑的仿真也成倍增加。"
            "调超参数没有标准答案，就是这样一次次做实验、对比曲线。",
            "en": "Too small a step and 30 rounds are not enough; too big and the score jumps around and it learns "
            "later, not sooner. Too much noise and every try is so random that no direction stands out. More "
            "directions make each round's judgement better but multiply the simulation to run. There is no "
            "formula for hyperparameters: you run experiments like these and compare the curves.",
        },
        check=_compared_settings,
        unlock=("train", "lr", "noise", "dirs", "fast"),
        show=("training",),
    ),
    Step(
        id="push",
        title={"zh": "推一下试试", "en": "Give it a push"},
        body={
            "zh": "点「推一下」。学到的策略能不能接住？\n"
            "如果最新一次训练没学会（曲线一直在虚线下面），推之前小车会先换回你学得最好的那组权重。",
            "en": "Press Push. Can the learned policy catch it?\n"
            "If your newest training did not learn (its curve stayed under the dashed line), the cart first goes "
            "back to the best weights you have trained.",
        },
        done={
            "zh": "它能接住小的推力：训练时每次开局都有随机的倾斜，所以它学会的是“怎么救回来”，而不只是“怎么站着”。",
            "en": "It catches small pushes: every training episode started with a random lean, so it learned how to "
            "recover, not only how to stand still.",
        },
        check=_survived_push,
        unlock=("push",),
    ),
    Step(
        id="scale",
        title={"zh": "想一想", "en": "Think about it"},
        body={
            "zh": "倒立摆几十轮、十几万步就学会了。robotforge 自带的 Go2 走路策略却训练了 2460 万步，"
            "还用了神经网络和 GPU。为什么差这么多？",
            "en": "The cart-pole learned in a few dozen rounds and about a hundred thousand steps. The Go2 walk that "
            "ships with robotforge took 24.6 million steps, a neural network and a GPU. Why such a difference?",
        },
        done={
            "zh": "所以学走路要用更大的网络、更多的并行仿真和更聪明的算法（比如 PPO）。"
            "robotforge 的 RL 工具和 UniLab 就是干这个的。",
            "en": "That is why walking needs bigger networks, many simulations in parallel and smarter algorithms "
            "(such as PPO). robotforge's RL tools and UniLab are there for exactly that.",
        },
        quiz=Quiz(
            question={"zh": "选一个最好的解释", "en": "Pick the best explanation"},
            options=(
                {"zh": "Go2 比倒立摆重", "en": "The Go2 is heavier"},
                {
                    "zh": "四足有 12 个电机、几十个输入，还要处理脚和地面的接触，简单的线性策略学不会",
                    "en": "A quadruped has 12 motors, dozens of inputs and feet hitting the ground; a simple linear "
                    "policy cannot capture that",
                },
                {"zh": "四足不需要奖励", "en": "Quadrupeds need no reward"},
            ),
            answer=1,
            explain={
                "zh": "对：动作和状态越多、动力学越复杂，需要探索的空间就越大，策略也要更有表达力。",
                "en": "Right: more actions, more inputs and messier dynamics mean a far bigger space to explore and "
                "a policy that can express much more.",
            },
        ),
    ),
)


class RLLesson(Lesson):
    id = "reinforcement_learning"
    chapter = {"zh": "强化学习", "en": "Reinforcement learning"}
    title = {"zh": "强化学习（一）：让倒立摆自己学会平衡", "en": "RL I: a cart-pole teaches itself to balance"}
    summary = {
        "zh": "倒立摆从一碰就倒，到自己学会平衡：看奖励、试错和训练曲线是怎么回事。",
        "en": "A cart-pole goes from falling at once to balancing by itself: "
        "rewards, trial and error, training curves.",
    }
    minutes = 10
    controls = (
        Control("release", "button", {"zh": "放手", "en": "Let go"}),
        Control("random", "button", {"zh": "随机试一次", "en": "Try random weights"}),
        Control("train", "button", {"zh": "开始训练", "en": "Start training"}),
        Control("lr", "slider", {"zh": "步长 α", "en": "Step size α"}, 0.02, 1.0, 0.02, 0.1),
        Control("noise", "slider", {"zh": "探索噪声 ν", "en": "Exploration noise ν"}, 0.05, 1.0, 0.05, 0.3),
        Control("dirs", "slider", {"zh": "每轮方向数", "en": "Directions per round"}, 2, 16, 2, 8),
        Control("fast", "toggle", {"zh": "快进（不放慢）", "en": "Fast-forward (no slow motion)"}),
        Control("push", "button", {"zh": "推一下", "en": "Push"}),
    )
    bench = Bench(
        title={"zh": "实验记录：训练曲线", "en": "Experiment record: training curves"},
        x={"zh": "训练轮数", "en": "training round"},
        x_unit="",
        y={"zh": "平均得分", "en": "average score"},
        y_unit="",
        params=("lr", "noise", "dirs"),
        short=({"zh": "α", "en": "α"}, {"zh": "ν", "en": "ν"}, {"zh": "方向", "en": "dirs"}),
        metrics=(
            Metric("learned_at", {"zh": "第几轮学会", "en": "learned at round"}),
            Metric("final", {"zh": "最后得分", "en": "final score"}, lower_is_better=False),
            Metric("steps", {"zh": "仿真步数", "en": "simulation steps"}, ranked=False),
        ),
        reference=TRAINED_RETURN,
    )
    plots = (
        Plot(
            "training",
            {"zh": "训练：每一轮的得分", "en": "Training: score per round"},
            "",
            (
                Series("best", {"zh": "这一轮最好", "en": "best this round"}, "muted", dashed=True),
                Series("mean", {"zh": "这一轮平均", "en": "average this round"}, "amber"),
            ),
            window=float(ITERATIONS),
        ),
    )
    steps = STEPS

    def __init__(self) -> None:
        self.model: mujoco.MjModel | None = None
        self.live_ctrl: Live | None = None
        self.policy: LinearPolicy | None = None  # what the live cart-pole runs; None = no control
        self.score: float | None = None  # the final score of the training that made `policy`, once it is done
        self.newest: LinearPolicy | None = None  # the latest training's weights
        self.best: tuple[float, LinearPolicy, int] | None = None  # the best finished training: score, weights, run
        self.ctx: LessonContext | None = None
        self.samples: deque[dict[str, float]] = deque(maxlen=256)
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.lock = threading.Lock()

    def build(self) -> mujoco.MjModel:
        world = World.model_validate({"name": "rl_lesson", "robots": [{"name": "pole", "template": "cartpole"}]})
        self.model = compile_world(world).model
        return self.model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.live_ctrl = Live(self, model, data)
        return {"lesson": self.live_ctrl}

    def attempt_ended(self, seconds: float, fell: bool) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        ctx.results["last_attempt"] = round(seconds, 1)
        if fell and self.policy is None and "released" in ctx.events:
            ctx.events.add("fell_uncontrolled")

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        assert self.live_ctrl is not None and self.model is not None
        if control == "release":
            ctx.events.add("released")
            self.policy = self.score = None
            self.live_ctrl.pending_restart = True
        elif control == "random":
            n_obs = 4
            rng = np.random.default_rng()
            self.policy = LinearPolicy(rng.normal(0, 1.0, size=(1, n_obs + 1)), np.ones(n_obs))
            self.score = None
            ctx.results["random_tries"] = ctx.results.get("random_tries", 0) + 1
            self.live_ctrl.pending_restart = True
        elif control == "train" and not self.training:
            self.policy = LinearPolicy.zeros(4, 1)  # every run starts from scratch, so runs compare fairly
            self.newest, self.score = self.policy, None
            with self.lock:
                self.samples.append({"t": 0.0})  # time goes back: the plot starts a fresh curve
            self.thread = threading.Thread(target=self._train, args=(self.policy, ctx), daemon=True)
            self.thread.start()
        elif control == "push":
            self._push(data, ctx)

    @property
    def training(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _push(self, data: mujoco.MjData, ctx: LessonContext) -> None:
        """Push the pole. Weights that never learned to balance cannot catch it, and following the lesson (a step
        size of 0.02 in the tuning step) leaves the cart on exactly such weights: then the cart first goes back to
        the best weights trained so far, and the push comes once it has started over."""
        assert self.live_ctrl is not None
        best = self.best
        learned = self.score is not None and self.score >= TRAINED_RETURN
        if learned or best is None or self.training:
            self.live_ctrl.push(data)
            return
        self.score, self.policy, run = best
        self.live_ctrl.pending_restart = True
        self.live_ctrl.push_in = PUSH_AFTER
        ctx.results["summary"] = {
            "zh": f"最新的权重没学会平衡，先换回学得最好的那组（第 {run} 次训练，最后得分 {self.score:.1f}），再推。",
            "en": f"The newest weights never learned to balance, so the cart goes back to the best ones (training "
            f"{run}, final score {self.score:.1f}) before the push.",
        }

    def _train(self, policy: LinearPolicy, ctx: LessonContext) -> None:
        env = RobotEnv(make_task("balance", self.model, "pole"), episode_seconds=EPISODE_SECONDS)
        settings = {k: float(ctx.values[k]) for k in PARAMS}
        rounds: list[float] = []
        means: list[float] = []

        def round_done(iteration: int, entry: dict[str, float]) -> None:
            with self.lock:
                self.samples.append({"t": float(iteration), "mean": entry["mean_return"], "best": entry["best_return"]})
            rounds.append(float(iteration))
            means.append(entry["mean_return"])
            ctx.results["train"] = {"round": iteration, "mean": round(entry["mean_return"], 2)}
            if entry["mean_return"] >= TRAINED_RETURN:
                ctx.events.add("trained")
            if not ctx.values.get("fast"):
                self.stopping.wait(PACE)  # slow enough to watch; returns at once when the lesson closes

        dirs = int(settings["dirs"])
        _, log = train_ars(
            env,
            iterations=ITERATIONS,
            directions=dirs,
            top=max(1, dirs // 2),
            step_size=settings["lr"],
            noise=settings["noise"],
            seed=0,
            callback=round_done,
            policy=policy,
            stop=self.stopping.is_set,
        )
        if len(means) == ITERATIONS:  # a run cut short by leaving the lesson is not a result
            learned = next((r for r, m in zip(rounds, means, strict=True) if m >= TRAINED_RETURN), None)
            metrics = {"learned_at": learned, "final": round(means[-1], 1), "steps": log.env_steps}
            ctx.record_run(settings, metrics, rounds, means)
            final = means[-1]
            if self.policy is policy:
                self.score = final
            if final >= TRAINED_RETURN and (self.best is None or final > self.best[0]):
                self.best = (final, policy, ctx.run_count)

    def close(self) -> None:
        self.stopping.set()

    def drain_samples(self) -> list[dict[str, float]]:
        with self.lock:
            out = list(self.samples)
            self.samples.clear()
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        live = self.live_ctrl
        assert live is not None
        up = float(data.time) - live.started
        pushed = live.pushed_at
        if pushed is not None and live.started <= pushed and data.time - pushed > 3.0 and "trained" in ctx.events:
            ctx.events.add("survived_push")  # the attempt that took the push is still standing 3 s later
        train = ctx.results.get("train")
        if self.policy is None:
            mode = {"zh": "没有控制", "en": "no control"}
        elif self.policy is self.newest:
            mode = {"zh": "最新学到的权重", "en": "newest learned weights"}
        elif self.best is not None and self.policy is self.best[1]:
            mode = {"zh": "学得最好的权重", "en": "best learned weights"}
        else:
            mode = {"zh": "随机权重", "en": "random weights"}
        zh = f"第 {live.attempt} 次  立住 {up:.1f} 秒  上次 {ctx.results.get('last_attempt', 0):.1f} 秒  {mode['zh']}"
        en = f"attempt {live.attempt}  up {up:.1f} s  last {ctx.results.get('last_attempt', 0):.1f} s  {mode['en']}"
        if train:
            zh += f"  训练第 {train['round']} 轮 平均 {train['mean']:.1f}"
            en += f"  round {train['round']} average {train['mean']:.1f}"
        return {"readout": {"zh": zh, "en": en}, "angle": round(math.degrees(live.angle(data)), 1)}

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [0.0, -2.6, 1.0], "target": [0.0, 0.0, 0.55]}
