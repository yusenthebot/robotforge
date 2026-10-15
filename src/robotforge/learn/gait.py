"""Lesson: walking gaits. Which legs step together, how long a step takes, and what that does.

Our template dog and the Unitree Go2 stand in two lanes and run the same gait code (behaviors/quadruped).
Every test walks the chosen dog for five seconds from its start line and lands in the experiment record:
distance over time, how fast it got, how much the body rolled and pitched. A footfall chart shows which
feet are down, the floor shows the support line or polygon under the dog with its centre of mass, and a
live formula works out the stride and the top speed from the step period.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import quadruped
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import (
    Bench,
    Control,
    Lesson,
    LessonContext,
    Metric,
    Plot,
    Quiz,
    Series,
    Step,
    Text,
)

DOGS = ("ours", "go2")
LANE = 0.5  # m from the middle to each dog's lane
START_X = -1.2
TEST_SECONDS = 5.0
SAMPLE_DT = 0.01
DOWN = 0.5  # cosine of the trunk's tilt below which a dog has fallen
PATTERN_KEYS = ("trot", "pace", "bound")
LEGS = ("front_left", "front_right", "rear_left", "rear_right")
LEG_KEYS = {"front_left": "fl", "front_right": "fr", "rear_left": "rl", "rear_right": "rr"}
LEG_NAMES = {
    "front_left": ("左前", "front left"),
    "front_right": ("右前", "front right"),
    "rear_left": ("左后", "rear left"),
    "rear_right": ("右后", "rear right"),
}


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


def _runs(ctx: LessonContext, **want: Any) -> list[dict[str, Any]]:
    """Recorded runs whose settings pass every predicate in `want`."""
    return [r for r in ctx.runs if all(test(r["params"].get(k, 0.0)) for k, test in want.items())]


def _ran_pattern(index: int):
    return lambda ctx: bool(_runs(ctx, dog=lambda d: d == 0, pattern=lambda p: p == index))


def _periods_compared(ctx: LessonContext) -> bool:
    """Trots asked for 0.5 m/s or more, with a quick period and a slow one."""
    fast = _runs(ctx, dog=lambda d: d == 0, pattern=lambda p: p == 0, speed=lambda v: v >= 0.5)
    periods = [r["params"]["period"] for r in fast]
    return bool(periods) and min(periods) <= 0.3 and max(periods) >= 0.45


def _go2_walked(ctx: LessonContext) -> bool:
    return bool(_runs(ctx, dog=lambda d: d == 1, pattern=lambda p: p == 0))


def _challenge(ctx: LessonContext) -> bool:
    walk = ctx.results.get("walk", {})
    return walk.get("dog") == "go2" and not walk.get("fell") and walk.get("distance", 0.0) >= CHALLENGE_DISTANCE


CHALLENGE_DISTANCE = 2.7  # m in five seconds
CHALLENGE_SWAY = 2.0  # degrees, roll and pitch (RMS)


def _challenge_steady(ctx: LessonContext) -> bool:
    walk = ctx.results.get("walk", {})
    return _challenge(ctx) and max(walk.get("roll", 99.0), walk.get("pitch", 99.0)) <= CHALLENGE_SWAY


STEPS = (
    Step(
        id="trot",
        title=_text("四条腿，谁和谁一起迈", "Four legs: which step together"),
        body=_text(
            "左边是我们的模板狗，右边是 Unitree Go2，用的是同一套走路程序。先看我们的狗。\n"
            "走路时，每条腿轮流“着地往后蹬”和“抬起往前迈”，一来一回叫一个步子周期 T。"
            "右边的“着地”图里，每条腿一行，涂满的时候脚在地上。\n先猜，再点「走 5 秒」。",
            "On the left is our template dog, on the right the Unitree Go2; both run the same walking code. "
            "Our dog goes first.\n"
            "While walking, each leg takes turns pushing back on the ground and swinging forward in the air; one "
            "round of that is a step period T. In the footfall chart on the right each leg has a row, filled "
            "while that foot is down.\nGuess, then press Walk 5 s.",
        ),
        done=_text(
            "左前和右后一起着地，右前和左后一起着地：对角的两条腿一组，这叫小跑（trot）。"
            "地上那条线连着着地的两只脚，黄点是重心的投影：它一直落在这条对角线附近，所以只靠两条腿也站得住。",
            "Front left lands with rear right, front right with rear left: diagonal pairs, a trot. The line on "
            "the floor joins the two feet that are down, and the amber dot is the centre of mass seen from "
            "above: it stays close to that diagonal line, which is why two feet are enough.",
        ),
        quiz=Quiz(
            question=_text("小跑时，哪两条腿同时着地？", "In a trot, which two legs are down together?"),
            options=(
                _text("对角的两条：左前和右后", "Diagonal ones: front left and rear right"),
                _text("同一侧的两条：左前和左后", "Same side: front left and rear left"),
                _text("两条前腿", "The two front legs"),
            ),
            answer=0,
            explain=_text(
                "对角线两头的脚把机身夹在中间。", "Feet at both ends of a diagonal hold the body between them."
            ),
            predict=True,
        ),
        check=_ran_pattern(0),
        unlock=("walk",),
        show=("labels", "support", "f_legs", "f_stride"),
    ),
    Step(
        id="pace",
        title=_text("换成同侧两条腿一起迈", "Same-side legs together"),
        body=_text(
            "「步态」选「踱步」：左边两条腿一组、右边两条腿一组（骆驼和长颈鹿这样走）。先猜，再走 5 秒。",
            "Set Gait to Pace: both left legs as one pair, both right legs as the other (camels and giraffes "
            "walk like this). Guess, then walk 5 s.",
        ),
        done=_text(
            "机身左右摇：只有左边或右边的脚着地时，支撑线在身体一侧，重心悬在线外，身体就往空的那边倒，"
            "下一步再被接住。实验记录里的“左右晃”大约是小跑的 7 倍（我们测到 4.2° 对 0.6°）。",
            "The body rocks side to side: with only the left or the right feet down, the support line runs "
            "along one side, the centre of mass hangs outside it, and the body tips toward the empty side "
            "until the next step catches it. 'Roll' in the record is about seven times the trot's (4.2 against "
            "0.6 degrees in our runs).",
        ),
        quiz=Quiz(
            question=_text("同侧两条腿一起迈，机身会怎样？", "Same-side pairs: what does the body do?"),
            options=(
                _text("左右摇", "It rocks side to side"),
                _text("前后点头", "It nods front to back"),
                _text("和小跑一样稳", "It is as steady as the trot"),
            ),
            answer=0,
            explain=_text("支撑线跑到身体一侧去了。", "The support line moves to one side of the body."),
            predict=True,
        ),
        check=_ran_pattern(1),
        unlock=("pattern",),
        show=("f_com",),
    ),
    Step(
        id="bound",
        title=_text("两条前腿一起，两条后腿一起", "Front pair, then rear pair"),
        body=_text(
            "「步态」选「跳跑」：两条前腿一组、两条后腿一组（兔子和全速奔跑的狗这样跑）。先猜，再走 5 秒。",
            "Set Gait to Bound: the two front legs as one pair, the two rear legs as the other (rabbits, and "
            "dogs at full gallop). Guess, then walk 5 s.",
        ),
        done=_text(
            "机身前后点头，而且几乎走不动（我们测到 5 秒只走了 -0.1 米，小跑走了 1 米）。"
            "支撑线横在身体前面或后面，重心总在线外。动物跳跑靠的是腾空和很大的蹬地力，"
            "我们这套慢慢迈步的程序做不到。所以四足机器人走路默认用小跑：对角支撑线一直穿过身体中间，又稳又快。",
            "The body nods front to back and hardly gets anywhere (-0.1 m in 5 s in our runs, against 1 m for "
            "the trot). The support line lies across the front or the back of the body, so the centre of mass "
            "is always off it. Animals bound with flight phases and big pushes, which this slow stepping code "
            "does not do. That is why legged robots walk with a trot by default: the diagonal support line "
            "always runs through the middle of the body.",
        ),
        quiz=Quiz(
            question=_text("两条前腿一起迈呢？", "And the two front legs together?"),
            options=(
                _text("左右摇", "It rocks side to side"),
                _text("前后点头，走得还慢", "It nods front to back, and goes slowly"),
                _text("最快", "Fastest of all"),
            ),
            answer=1,
            explain=_text("支撑线横在身体前后。", "The support line lies across the front or the back."),
            predict=True,
        ),
        check=_ran_pattern(2),
    ),
    Step(
        id="period",
        title=_text("步子周期 T：快步还是大步", "Step period: quick steps or long ones"),
        body=_text(
            "换回「小跑」，把速度调到 0.6 m/s。看公式：每步迈多远 = 速度 × T × ½，可腿只能伸那么长，"
            "所以 T 决定了最快能走多快。\n先猜，再分别用 T 0.5 和 T 0.25 各走一次。",
            "Back to Trot, and set the speed to 0.6 m/s. Look at the formula: stride = speed * T * 1/2, but a "
            "leg only reaches so far, so T sets the top speed.\nGuess, then walk once with T 0.5 and once with "
            "T 0.25.",
        ),
        done=_text(
            "T 0.25 的快步 5 秒走了 2.6 米，T 0.5 只走了 1.3 米：步子再大也有上限，要更快只能迈得更勤。"
            "代价是费劲：调参台里测过，T 0.25 每米耗能是 T 0.5 的 1.6 倍。",
            "Quick steps at T 0.25 covered 2.6 m in 5 s, T 0.5 only 1.3 m: strides cannot grow past the leg's "
            "reach, so going faster means stepping more often. It costs energy: the tuning bench measured 1.6 "
            "times the energy per metre at T 0.25 as at T 0.5.",
        ),
        quiz=Quiz(
            question=_text(
                "要 0.6 m/s，T 0.5 和 T 0.25 谁走得远？", "Asked for 0.6 m/s, which goes further: T 0.5 or T 0.25?"
            ),
            options=(
                _text("T 0.5：步子大", "T 0.5: longer steps"),
                _text("T 0.25：步子勤", "T 0.25: more steps"),
                _text("一样远，速度是一样的", "The same: the speed is the same"),
            ),
            answer=1,
            explain=_text(
                "步幅到顶以后，只有缩短 T 才能更快。", "Once the stride is at its limit, only a shorter T goes faster."
            ),
            predict=True,
        ),
        check=_periods_compared,
        unlock=("period", "speed"),
        show=("f_max",),
    ),
    Step(
        id="go2",
        title=_text("换成真机：Unitree Go2", "Now a real robot: the Unitree Go2"),
        body=_text(
            "「哪只狗」选 Go2。同样是小跑、T 0.36、0.6 m/s。先猜，再走 5 秒。",
            "Set Which dog to the Go2. Same trot, T 0.36, 0.6 m/s. Guess, then walk 5 s.",
        ),
        done=_text(
            "Go2 走了 2.7 米，我们的狗 1.8 米。公式没变，变的是腿长：Go2 的腿 0.28 米，我们的 0.21 米，"
            "最大步幅跟着变大，同样的 T 就能走得更快。好的步态参数是按腿长来定的。",
            "The Go2 covered 2.7 m, our dog 1.8 m. The formula is the same; the legs are not: 0.28 m on the Go2 "
            "against 0.21 m, so the longest stride grows and the same T goes faster. Good gait settings scale "
            "with leg length.",
        ),
        quiz=Quiz(
            question=_text("同样的设置，Go2 会走多远？", "Same settings: how far does the Go2 get?"),
            options=(
                _text("和我们的狗差不多", "About as far as our dog"),
                _text("远得多", "Much further"),
                _text("近得多，它更重", "Much less: it is heavier"),
            ),
            answer=1,
            explain=_text("腿更长，最大步幅更大。", "Longer legs, a longer top stride."),
            predict=True,
        ),
        check=_go2_walked,
        unlock=("dog",),
    ),
    Step(
        id="challenge",
        title=_text("挑战：Go2 又快又稳", "Challenge: the Go2, fast and steady"),
        body=_text(
            f"让 Go2 在 5 秒里走过 {CHALLENGE_DISTANCE:g} 米，机身左右晃和前后晃都不超过 {CHALLENGE_SWAY:g}°。"
            "步态、T 和速度都可以调。",
            f"Make the Go2 cover {CHALLENGE_DISTANCE:g} m in 5 s with roll and pitch both under "
            f"{CHALLENGE_SWAY:g} degrees. Gait, T and speed are all yours.",
        ),
        done=_text(
            "你给一台真机器狗调出了一套又快又稳的步态。",
            "You tuned a fast, steady gait for a real robot dog.",
        ),
        check=_challenge_steady,
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text("马、狗和四足机器人都会换步态。", "Horses, dogs and legged robots all change gaits."),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("四足机器人为什么默认用小跑走路？", "Why do legged robots walk with a trot by default?"),
            options=(
                _text("小跑最好看", "A trot looks best"),
                _text(
                    "对角两条腿着地，支撑线穿过身体中间，机身不摇，速度也上得去",
                    "With diagonal feet down the support line runs through the body: no rocking, and it can go fast",
                ),
                _text("小跑最省电", "A trot uses the least power"),
            ),
            answer=1,
            explain=_text(
                "对。踱步左右摇、跳跑前后点头，只有小跑的支撑线一直在身体中间。",
                "Right. A pace rocks sideways and a bound nods; only the trot keeps its support line in the middle.",
            ),
        ),
    ),
)


class Watch:
    """Measures the walking dog every physics step and samples the plots at 100 Hz."""

    rate_hz = 500.0

    def __init__(self, lesson: GaitLesson) -> None:
        self.lesson = lesson
        self.next_sample = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.lesson.measure(model, data)
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            self.lesson.samples.append(self.lesson.sample(model, data))

    def status(self) -> dict[str, Any]:
        return {"controller": "gait-watch"}


class GaitLesson(Lesson):
    id = "gait"
    chapter = _text("走路", "Walking")
    title = _text("四足步态：腿怎么轮流迈", "Quadruped gaits: how the legs take turns")
    summary = _text(
        "小跑、踱步、跳跑，步子周期和速度：先在我们的模板狗上试，再换到真机 Go2。",
        "Trot, pace and bound, step period and speed: first on our template dog, then on the real Go2.",
    )
    minutes = 15
    controls = (
        Control("walk", "button", _text("走 5 秒", "Walk 5 s")),
        Control(
            "pattern",
            "choice",
            _text("步态", "Gait"),
            options=(
                _text("小跑（对角）", "Trot (diagonal)"),
                _text("踱步（同侧）", "Pace (same side)"),
                _text("跳跑（前后）", "Bound (front, rear)"),
            ),
        ),
        Control(
            "period", "slider", _text("步子周期 T", "Step period T"), 0.2, 0.6, 0.01, quadruped.TEMPLATE_PERIOD, "s"
        ),
        Control("speed", "slider", _text("速度", "Speed"), 0.1, 0.8, 0.05, 0.2, "m/s"),
        Control(
            "dog",
            "choice",
            _text("哪只狗", "Which dog"),
            options=(_text("我们的模板", "Our template"), _text("Unitree Go2", "Unitree Go2")),
        ),
    )
    plots = (
        Plot(
            "footfall",
            _text("着地（涂满 = 脚在地上）", "Footfall (filled = foot down)"),
            "",
            tuple(
                Series(
                    LEG_KEYS[leg], _text(*LEG_NAMES[leg]), "blue" if leg in ("front_left", "rear_right") else "violet"
                )
                for leg in LEGS
            ),
            window=3.0,
            kind="footfall",
        ),
        Plot(
            "sway",
            _text("机身晃动", "Body sway"),
            "°",
            (
                Series("roll", _text("左右晃", "roll"), "blue"),
                Series("pitch", _text("前后晃", "pitch"), "violet"),
            ),
            window=4.0,
        ),
    )
    steps = STEPS
    bench = Bench(
        title=_text("实验记录：5 秒走了多远", "Experiment record: distance in 5 s"),
        x=_text("开始走后", "since the start"),
        x_unit="s",
        y=_text("走过的距离", "distance"),
        y_unit="m",
        params=("dog", "pattern", "period", "speed"),
        short=(_text("狗", "dog"), _text("步态", "gait"), _text("T", "T"), _text("速度", "speed")),
        metrics=(
            Metric("distance", _text("走了", "covered"), "m", lower_is_better=False),
            Metric("roll", _text("左右晃", "roll"), "°"),
            Metric("pitch", _text("前后晃", "pitch"), "°"),
        ),
    )

    def __init__(self) -> None:
        self.playground = None
        self.ctx: LessonContext | None = None
        self.start: np.ndarray | None = None
        self.gaits: dict[str, quadruped.QuadrupedGait] = {}
        self.trunks: dict[str, int] = {}
        self.feet: dict[str, dict[str, tuple[int, np.ndarray]]] = {}  # dog -> leg -> (body, point)
        self.floor = -1
        self.samples: list[dict[str, float]] = []
        self.test: dict[str, Any] | None = None

    # ---- the world ---------------------------------------------------------------------------------

    def build(self) -> mujoco.MjModel:
        world = World.model_validate(
            {
                "name": "gait",
                "floor": {"style": "concrete"},
                "robots": [
                    {"name": "ours", "template": "quadruped", "params": {"color": "orange"}, "pos": [START_X, -LANE]},
                    {"name": "go2", "library": "go2", "pos": [START_X, LANE]},
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
        self.gaits = {dog: self._gait(model, dog, ctx) for dog in DOGS}
        self.trunks = {dog: int(g.trunk) for dog, g in self.gaits.items()}
        self.feet = {
            dog: {leg.name: (int(leg.chain.body), leg.chain.local_point.copy()) for leg in g.legs}
            for dog, g in self.gaits.items()
        }
        return {**{dog: g for dog, g in self.gaits.items()}, "watch": Watch(self)}

    def _gait(self, model: mujoco.MjModel, dog: str, ctx: LessonContext) -> quadruped.QuadrupedGait:
        home = model.key("home").qpos
        gait = quadruped.for_template(model, dog, home) if dog == "ours" else quadruped.for_robot(model, dog, home)
        gait.set_pattern(quadruped.PATTERNS[PATTERN_KEYS[int(ctx.values["pattern"])]])
        gait.set_period(float(ctx.values["period"]))
        return gait

    @property
    def active(self) -> str:
        assert self.ctx is not None
        return DOGS[int(self.ctx.values["dog"])]

    # ---- the walk test ---------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control != "walk":
            return
        assert self.start is not None and self.playground is not None
        session = self.playground.session
        model = session.model
        data.qpos[:] = self.start  # both dogs back on their start line, still
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for dog in DOGS:  # fresh gaits with the current settings: every test starts the same
            self.gaits[dog] = self._gait(model, dog, ctx)
            session.set_controller(dog, self.gaits[dog])
        gait = self.gaits[self.active]
        gait.command(float(ctx.values["speed"]))
        trunk = self.trunks[self.active]
        self.test = {
            "dog": self.active,
            "t0": float(data.time),
            "x0": float(data.xpos[trunk][0]),
            "times": [],
            "distances": [],
            "roll": [],
            "pitch": [],
            "fell": False,
            "settings": {k: float(ctx.values[k]) for k in ("dog", "pattern", "period", "speed")},
            "commanded": gait.command_speed,
        }

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        test = self.test
        if test is None:
            return
        t = float(data.time) - test["t0"]
        trunk = self.trunks[test["dog"]]
        if float(data.xmat[trunk][8]) < DOWN:
            test["fell"] = True
        if t >= 1.0:  # sway once the gait has got going
            roll, pitch = quadruped._roll_pitch(data.xquat[trunk])
            test["roll"].append(math.degrees(roll))
            test["pitch"].append(math.degrees(pitch))
        if t >= TEST_SECONDS or test["fell"]:
            self._finish(data)

    def _finish(self, data: mujoco.MjData) -> None:
        test, self.test = self.test, None
        assert test is not None and self.ctx is not None
        self.gaits[test["dog"]].command(0.0)
        trunk = self.trunks[test["dog"]]

        def rms(values: list[float]) -> float:
            return round(float(np.sqrt(np.mean(np.square(values)))), 2) if values else 0.0

        distance = round(float(data.xpos[trunk][0]) - test["x0"], 2)
        result = {
            "dog": test["dog"],
            "distance": distance,
            "speed": round(distance / TEST_SECONDS, 3),
            "roll": rms(test["roll"]),
            "pitch": rms(test["pitch"]),
            "fell": test["fell"],
            "commanded": round(test["commanded"], 3),
        }
        self.ctx.results["walk"] = result
        fell = _text("，摔倒了", ", and fell") if result["fell"] else _text("", "")
        self.ctx.results["summary"] = {
            "zh": f"走了 {distance:g} 米，左右晃 {result['roll']:g}°，前后晃 {result['pitch']:g}°{fell['zh']}",
            "en": f"Covered {distance:g} m, roll {result['roll']:g}°, pitch {result['pitch']:g}°{fell['en']}",
        }
        metrics = {k: result[k] for k in ("distance", "roll", "pitch")}
        self.ctx.record_run(test["settings"], metrics, test["times"], test["distances"])

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        dog = self.active if self.test is None else self.test["dog"]
        trunk = self.trunks[dog]
        roll, pitch = quadruped._roll_pitch(data.xquat[trunk])
        down = self._down(model, data, dog)
        out = {"t": float(data.time), "roll": math.degrees(roll), "pitch": math.degrees(pitch)}
        out.update({LEG_KEYS[leg]: float(leg in down) for leg in LEGS})
        test = self.test
        if test is not None:
            test["times"].append(round(float(data.time) - test["t0"], 3))
            test["distances"].append(float(data.xpos[trunk][0]) - test["x0"])
        return out

    def drain_samples(self) -> list[dict[str, float]]:
        out, self.samples = self.samples, []
        return out

    # ---- what the overlays and the formula show --------------------------------------------------------

    def _down(self, model: mujoco.MjModel, data: mujoco.MjData, dog: str) -> set[str]:
        """Legs whose foot touches the floor right now."""
        bodies = {body: leg for leg, (body, _) in self.feet[dog].items()}
        down: set[str] = set()
        for contact in data.contact[: data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.floor not in (g1, g2):
                continue
            other = g2 if g1 == self.floor else g1
            leg = bodies.get(int(model.geom_bodyid[other]))
            if leg is not None:
                down.add(leg)
        return down

    def _foot(self, data: mujoco.MjData, dog: str, leg: str) -> np.ndarray:
        body, point = self.feet[dog][leg]
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ point

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        dog = self.active if self.test is None else self.test["dog"]
        down = sorted(self._down(model, data, dog), key=LEGS.index)
        points = np.array([self._foot(data, dog, leg)[:2] for leg in down]).reshape(-1, 2)
        feet = (_hull(points) if len(points) > 2 else points).round(4).tolist()  # in order around the patch
        com = data.subtree_com[self.trunks[dog]][:2]
        readout = _text("", "")
        if self.test is not None:
            left = max(0.0, TEST_SECONDS - (float(data.time) - self.test["t0"]))
            readout = _text(f"走着呢，还剩 {left:.1f} 秒", f"Walking, {left:.1f} s to go")
        return {
            "follow": self.trunks[dog],  # the camera walks along with the dog being tested
            "support": feet,
            "com": com.round(4).tolist(),
            "formula": self.formula(dog, down, feet, com, ctx),
            "readout": readout,
        }

    def formula(
        self, dog: str, down: list[str], feet: list[list[float]], com: np.ndarray, ctx: LessonContext
    ) -> list[dict[str, Any]]:
        gait = self.gaits[dog]
        asked = float(ctx.values["speed"])
        v = min(asked, gait.max_speed)  # what the gait will actually step for
        T = gait.period
        legs_zh = "、".join(LEG_NAMES[leg][0] for leg in down) or "没有"
        legs_en = ", ".join(LEG_NAMES[leg][1] for leg in down) or "none"
        rows: list[dict[str, Any]] = [
            {
                "show": "f_legs",
                "label": _text("着地的脚", "Feet down"),
                "expr": _text(legs_zh, legs_en),
                "value": f"{len(down)}",
                "color": "blue",
            },
            {
                "show": "f_stride",
                "label": _text("每步迈多远", "Stride"),
                "expr": f"v × T × ½ = {v:.2f} × {T:g} × 0.5",
                "value": f"{100 * v * T * gait.duty:.1f} cm",
                "color": "ink",
            },
            {
                "show": "f_max",
                "label": _text("最快能走", "Top speed"),
                "expr": _text(
                    f"最大步幅 {100 * gait.max_stride:.1f} cm ÷ (T × ½) × 0.7",
                    f"longest stride {100 * gait.max_stride:.1f} cm / (T * 1/2) * 0.7",
                ),
                "value": f"{gait.max_speed:.2f} m/s",
                "color": "amber",
            },
        ]
        if asked > gait.max_speed + 1e-6:
            rows[-1]["note"] = _text(
                f"要 {asked:g} m/s，超过了这个 T 的上限，只能走 {gait.max_speed:.2f}",
                f"{asked:g} m/s is past this T's limit; it walks at {gait.max_speed:.2f}",
            )
        rows.append(
            {
                "show": "f_com",
                "label": _text("重心离支撑", "Centre of mass off support"),
                "expr": _text("黄点到地上那条线或那块面", "amber dot to the line or patch on the floor"),
                "value": "-" if len(feet) < 2 else f"{100 * _off_support(np.array(feet), com):.1f} cm",
                "color": "violet",
            }
        )
        return rows

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "gait",
            "entrants": [
                {
                    "name": dog,
                    "label": label,
                    "color": color,
                    "real": dog == "go2",
                    "base": [START_X, y, 0.0],
                    "body": _trunk(model, dog),
                    "top": top,
                    "cloud": [],
                }
                for dog, label, color, y, top in (
                    ("ours", _text("模板狗", "Template dog"), "amber", -LANE, 0.5),
                    ("go2", _text("Unitree Go2", "Unitree Go2"), "blue", LANE, 0.62),
                )
            ],
            "targets": [],
        }

    def camera(self) -> dict[str, list[float]]:
        # Side on, a little ahead of the dogs; the camera then walks along with the dog being tested.
        return {"eye": [START_X + 0.4, -3.3, 1.35], "target": [START_X + 0.4, 0.0, 0.15]}


def _trunk(model: mujoco.MjModel, dog: str) -> int:
    """The body under the dog's free joint."""
    for j in range(model.njnt):
        body = int(model.jnt_bodyid[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and model.body(body).name.startswith(f"{dog}/"):
            return body  # free joints in library models often have no name of their own
    raise ValueError(f"{dog} has no free joint")


def _off_support(feet: np.ndarray, com: np.ndarray) -> float:
    """How far the centre of mass is from the support: the distance to the segment between two feet, or
    to the polygon of three or four (0 inside)."""
    if len(feet) == 2:
        return _to_segment(com, feet[0], feet[1])
    hull = _hull(feet)
    if _inside(com, hull):
        return 0.0
    return min(_to_segment(com, a, b) for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True))


def _to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _hull(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
    return points[order]


def _inside(p: np.ndarray, hull: np.ndarray) -> bool:
    signs = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        signs.append(cross >= 0)
    return all(signs) or not any(signs)
