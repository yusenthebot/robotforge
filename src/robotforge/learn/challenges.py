"""Challenges: goals to beat, with a timer, stars and a personal best.

They run on the lesson engine (same panel, controls, plots and checks) but in the real scenes with real
robots, driven by the same controllers the rest of robotforge uses.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from typing import Any

import mujoco
import numpy as np

from robotforge.design import scenes, terrain
from robotforge.design.world import RobotEntry, World, compile_world, spawn_height
from robotforge.learn.engine import Control, Lesson, LessonContext, Plot, Series, Step

FALLEN_UPRIGHT = 0.5  # cosine of the trunk's tilt below which a legged robot has fallen


def scene_model(preset: str, robot: dict[str, Any], seed: int = 0) -> tuple[mujoco.MjModel, scenes.Content]:
    """A scene preset with one robot on its first place, compiled without touching the learner's world."""
    content = scenes.build(preset, seed)
    entry = RobotEntry(**robot)
    scenes.place([entry], content, lambda e: spawn_height(e, None))
    world = World.model_validate(
        {
            "name": f"challenge_{preset}",
            "floor": content.floor,
            "terrain": [terrain.parse(t) for t in content.terrain],
            "objects": content.objects,
            "robots": [entry],
        }
    )
    return compile_world(world).model, content


def _stars(value: float, thresholds: tuple[float, float, float], lower_is_better: bool = False) -> int:
    if lower_is_better:
        return sum(value <= t for t in thresholds)
    return sum(value >= t for t in thresholds)


class Challenge(Lesson):
    """One goal step plus a scoring run; subclasses say what is measured."""

    kind = "challenge"
    robot = ""

    def __init__(self) -> None:
        self.playground = None  # set by the runner before attach: challenges drive real behaviors
        self.started: float | None = None
        self.finished: float | None = None

    def on_drive(self, speed: float, turn: float, ctx: LessonContext) -> None:
        if self.playground is None or self.finished is not None:
            return
        if self.started is None and (speed or turn):
            self.started = float(self.playground.session.data.time)  # the clock starts with the first move
        self.playground.behaviors.move(self.robot, speed, turn)

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.started = self.finished = None  # after a reset the run starts over
        ctx.results.pop("run", None)
        return {}

    def trunk(self, model: mujoco.MjModel) -> int:
        return int(model.body(f"{self.robot}/{self.trunk_name}").id)

    trunk_name = "trunk"

    def fallen(self, data: mujoco.MjData, trunk: int) -> bool:
        return float(data.xmat[trunk][8]) < FALLEN_UPRIGHT

    def retry(self, readout: dict[str, str]) -> dict[str, str]:
        """After a run, say how to go again: Reset puts everything back and starts a fresh attempt."""
        if self.finished is None:
            return readout
        return {"zh": readout["zh"] + "  （按「重置」再来一次）", "en": readout["en"] + "  (press Reset to go again)"}

    def elapsed(self, data: mujoco.MjData) -> float:
        if self.started is None:
            return 0.0
        return (self.finished if self.finished is not None else float(data.time)) - self.started


def _run_scored(ctx: LessonContext) -> bool:
    return ctx.results.get("run", {}).get("stars", 0) >= 1


# ---- maze: find the way out -------------------------------------------------------------------------

MAZE_ROWS = MAZE_COLS = 5
MAZE_CELL = 1.0
DRIVE_SPEED = 0.35  # m/s a learner averages on the keys, turns included: sets the par time


def maze_route(seed: int) -> list[tuple[float, float]]:
    """Cell centres from the entrance mark to the exit mark through the seeded maze (the scene's layout)."""
    open_walls = scenes.carve_maze(MAZE_ROWS, MAZE_COLS, random.Random(seed))
    start, goal = (0, 0), (MAZE_ROWS - 1, MAZE_COLS - 1)
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    todo = deque([start])
    while todo:
        r, c = todo.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= nr < MAZE_ROWS and 0 <= nc < MAZE_COLS) or (nr, nc) in previous:
                continue
            a, b = sorted([(r, c), (nr, nc)])
            if (a[0], a[1], b[0], b[1]) in open_walls:
                previous[(nr, nc)] = (r, c)
                todo.append((nr, nc))
    cells: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        cells.append(node)
        node = previous[node]
    x0, y0 = -MAZE_COLS * MAZE_CELL / 2, -MAZE_ROWS * MAZE_CELL / 2
    centre = [(x0 + (c + 0.5) * MAZE_CELL, y0 + (r + 0.5) * MAZE_CELL) for r, c in reversed(cells)]
    entrance = (x0 + 0.5 * MAZE_CELL, y0 - 0.5 * MAZE_CELL)
    exit_mark = (x0 + (MAZE_COLS - 0.5) * MAZE_CELL, y0 + (MAZE_ROWS + 0.5) * MAZE_CELL)
    return [entrance, *centre, exit_mark]


def maze_path_length(seed: int) -> float:
    """Length of the way from the entrance to the exit, in meters."""
    route = maze_route(seed)
    return sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False))


class MazeChallenge(Challenge):
    id = "maze_escape"
    chapter = {"zh": "路径规划", "en": "Path planning"}
    title = {"zh": "迷宫逃脱", "en": "Maze escape"}
    summary = {
        "zh": "开小车从黄色起点找到绿色出口，越快星越多。每次都是新迷宫。",
        "en": "Drive the rover from the yellow start to the green exit; faster earns more stars. A new maze each time.",
    }
    minutes = 5
    robot = "rover"
    trunk_name = "chassis"
    drive = {"max_speed": 0.7, "turn": 2.6, "follow": "rover/chassis"}
    plots = (
        Plot(
            "goal",
            {"zh": "离出口的距离", "en": "Distance to the exit"},
            "m",
            (Series("d", {"zh": "距离", "en": "distance"}, "amber"),),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.seed = random.randrange(10_000)
        self.goal = (0.0, 0.0)
        self.par = 30.0
        self.samples: deque[dict[str, float]] = deque(maxlen=512)
        self.next_sample = 0.0

    @property
    def steps(self) -> tuple[Step, ...]:  # type: ignore[override]
        par = self.par
        return (
            Step(
                id="escape",
                title={"zh": "找到出口", "en": "Find the way out"},
                body={
                    "zh": "用 W A S D 开小车穿过迷宫，到达绿色出口就算完成。计时从你第一次按键开始。\n"
                    f"这个迷宫的标准时间是 {par:.0f} 秒：一颗星是到达，两颗星是 {par * 1.5:.0f} 秒以内，"
                    f"三颗星是 {par:.0f} 秒以内。\n小技巧：沿着一面墙一直走，总能走出去（右手法则）。",
                    "en": "Drive the rover through the maze with W A S D and reach the green exit. "
                    "The clock starts with your first key.\n"
                    f"Par for this maze is {par:.0f} s: one star for getting out, two within "
                    f"{par * 1.5:.0f} s, three within {par:.0f} s.\n"
                    "Tip: keep one hand on a wall and you will always get out (the wall follower rule).",
                },
                done={
                    "zh": "出来了！想更快？试着先在脑子里规划路线：这正是机器人里“路径规划”要解决的问题。",
                    "en": "You are out! Want to be faster? Plan the route in your head first: that is the path "
                    "planning problem robots solve.",
                },
                check=_run_scored,
            ),
        )

    def build(self) -> mujoco.MjModel:
        model, content = scene_model("maze", {"name": "rover", "template": "rover"}, self.seed)
        goal = next(o for o in content.objects if o["name"] == "goal")
        self.goal = (float(goal["pos"][0]), float(goal["pos"][1]))
        self.par = round(maze_path_length(self.seed) / DRIVE_SPEED)
        return model

    def drain_samples(self) -> list[dict[str, float]]:
        out = list(self.samples)
        self.samples.clear()
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        body = self.trunk(model)
        pos = data.xpos[body][:2]
        d = float(math.dist(pos, self.goal))
        if data.time >= self.next_sample:
            self.next_sample = float(data.time) + 0.05
            self.samples.append({"t": float(data.time), "d": d})
        if self.finished is None and self.started is not None and d < 0.45:
            self.finished = float(data.time)
            t = self.elapsed(data)
            stars = 1 + _stars(t, (self.par * 1.5, self.par), lower_is_better=True)
            ctx.results["run"] = {"time": round(t, 1), "stars": min(stars, 3)}
            ctx.results["summary"] = {"zh": f"{t:.1f} 秒逃出迷宫", "en": f"Out in {t:.1f} s"}
            ctx.results["stars"] = min(stars, 3)
            self.playground.behaviors.stop(self.robot)
        t = self.elapsed(data)
        return {
            "readout": self.retry(
                {
                    "zh": f"用时 {t:.1f} s  离出口 {d:.1f} m  标准 {self.par:.0f} s",
                    "en": f"time {t:.1f} s  to exit {d:.1f} m  par {self.par:.0f} s",
                }
            ),
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [-1.2, -8.4, 6.0], "target": [0.0, -1.0, 0.0]}


# ---- obstacle course: how far can it walk -----------------------------------------------------------


class CourseChallenge(Challenge):
    id = "obstacle_course"
    chapter = {"zh": "运动", "en": "Locomotion"}
    title = {"zh": "障碍赛道：能走多远", "en": "Obstacle course: how far can it go"}
    summary = {
        "zh": "驾驶机器狗走过斜坡、台阶和碎石路。走得越远星越多，掉下去或摔倒就结束。",
        "en": "Walk the robot dog over a ramp, stairs and rough ground. "
        "The further, the more stars; a fall ends the run.",
    }
    minutes = 5
    robot = "dog"
    # Holding W walks at 70% of max_speed (the viewer's share). At 0.35 m/s the dog fell on the stairs in all 8 of
    # our key-driven runs (W held, A and D tapped to stay in the middle), and the keys have no slower setting; at
    # 0.3 m/s 6 of 8 got past them and 3 finished (a wider sweep of tap timing: 13 of 20 past, 7 finished). Only
    # distance earns stars, so the slower walk costs nothing.
    drive = {"max_speed": 0.43, "turn": 1.2, "follow": "dog/trunk"}
    plots = (
        Plot(
            "progress",
            {"zh": "走了多远", "en": "Distance along the course"},
            "m",
            (Series("x", {"zh": "距离", "en": "distance"}, "amber"),),
            window=30.0,
            span=1.0,  # standing on the start pad it plotted its settling on a -0.002 m axis
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.marks: list[tuple[str, float]] = []
        self.best = 0.0
        self.samples: deque[dict[str, float]] = deque(maxlen=1024)
        self.next_sample = 0.0

    @property
    def steps(self) -> tuple[Step, ...]:  # type: ignore[override]
        names = {
            "ramp": ("斜坡", "the ramp"),
            "stairs": ("台阶", "the stairs"),
            "rough": ("碎石路", "the rough ground"),
        }
        zh = "、".join(f"过了{names[k][0]}（{x:.1f} 米）" for k, x in self.marks)
        en = ", ".join(f"past {names[k][1]} ({x:.1f} m)" for k, x in self.marks)
        return (
            Step(
                id="course",
                title={"zh": "走得越远越好", "en": "Go as far as you can"},
                body={
                    "zh": "用 W A S D 驾驶机器狗沿赛道往前走（往 +x，红色终点在 17.6 米）。"
                    "摔倒就结束，按「重置」再来。\n"
                    f"星星：{zh}。\n提示：上台阶前把方向摆正，直着上，不要斜着上。",
                    "en": "Walk the robot dog along the course with W A S D (toward +x; the red finish is at 17.6 m). "
                    f"A fall ends the run; press Reset to try again.\nStars: {en}.\n"
                    "Tip: line up before the stairs and take them straight, not at an angle.",
                },
                # Not "out of reach": with keys alone 7 of our 20 runs finished (7 fell on the stairs, 6 on the curbs
                # and stones), and a learner who just finished reads this next to "Finished!".
                done={
                    "zh": "台阶、路沿和梅花桩对内置步态来说很难：它按固定节奏抬腿，不会“看路”，"
                    "脚能不能跨过边沿有一部分靠运气。走完全程是做得到的，只是不能每次都成功。"
                    "能看地形的控制器，或者在崎岖地面上用强化学习训练过的控制器，才是真实四足机器人稳稳走过这种路的办法。",
                    "en": "The stairs, curbs and stepping stones are hard for the built-in gait: it lifts its feet on "
                    "a fixed rhythm and does not look at the ground, so whether a foot clears an edge is partly luck. "
                    "The course can be finished, just not every time. A controller that sees the terrain, or one "
                    "trained on rough ground with reinforcement learning, is how real legged robots get across "
                    "reliably.",
                },
                check=_run_scored,
            ),
        )

    def build(self) -> mujoco.MjModel:
        model, content = scene_model("obstacle_course", {"name": "dog", "template": "quadruped"})
        ends: dict[str, float] = {}
        for patch in content.terrain:
            parsed = terrain.parse(patch)
            if parsed.type in ("ramp", "stairs", "rough") and parsed.type not in ends:
                ends[parsed.type] = float(parsed.pos[0] + parsed.footprint()[0] / 2)
        self.marks = sorted(ends.items(), key=lambda kv: kv[1])
        return model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.best = 0.0
        return super().attach(model, data, ctx)

    def drain_samples(self) -> list[dict[str, float]]:
        out = list(self.samples)
        self.samples.clear()
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        body = self.trunk(model)
        x = float(data.xpos[body][0])
        if data.time >= self.next_sample:
            self.next_sample = float(data.time) + 0.1
            self.samples.append({"t": float(data.time), "x": x})
        if self.started is not None and self.finished is None:
            self.best = max(self.best, x)
            down = self.fallen(data, body)
            if down or x >= 17.6:
                self.finished = float(data.time)
                stars = _stars(self.best, tuple(end for _, end in self.marks))  # type: ignore[arg-type]
                if not down:
                    stars = 3
                ctx.results["stars"] = stars
                ctx.results["run"] = {"distance": round(self.best, 2), "stars": stars, "finished": not down}
                done = "走完全程！" if not down else "摔倒了，"
                ctx.results["summary"] = {
                    "zh": f"{done}走了 {self.best:.1f} 米，用时 {self.elapsed(data):.0f} 秒",
                    "en": f"{'Finished!' if not down else 'Fell over.'} {self.best:.1f} m "
                    f"in {self.elapsed(data):.0f} s",
                }
                self.playground.behaviors.stop(self.robot)
        shown = round(x, 1) + 0.0  # settling a millimetre back on the start pad read "-0.0 m"
        return {
            "readout": self.retry(
                {
                    "zh": f"距离 {shown:.1f} m  最远 {self.best:.1f} m  用时 {self.elapsed(data):.0f} s",
                    "en": f"distance {shown:.1f} m  best {self.best:.1f} m  time {self.elapsed(data):.0f} s",
                }
            ),
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [-1.6, -2.4, 1.4], "target": [0.6, 0.0, 0.2]}


# ---- shove survival: which controller keeps its feet ----------------------------------------------

SHOVE_EVERY = 2.5  # s between shoves
SHOVE_STEP = 0.2  # m/s of sideways kick added per level
SHOVE_LEVELS = 12


class ShoveChallenge(Challenge):
    id = "shove_survival"
    chapter = {"zh": "控制", "en": "Control"}
    title = {"zh": "推不倒", "en": "Don't fall over"}
    summary = {
        "zh": "机器狗每 2.5 秒被推一下，一次比一次猛。选一个控制器，看它能扛几级。",
        "en": "The robot dog gets shoved every 2.5 s, harder each time. "
        "Pick a controller and see how many levels it takes.",
    }
    minutes = 4
    robot = "dog"
    controls = (
        Control(
            "controller",
            "choice",
            {"zh": "控制器", "en": "Controller"},
            options=(
                {"zh": "保持姿势", "en": "Hold pose"},
                {"zh": "原地踏步", "en": "Step in place"},
                {"zh": "MPC 站稳", "en": "Stand with MPC"},
            ),
        ),
        Control("start", "button", {"zh": "开始推", "en": "Start shoving"}),
    )
    plots = (
        Plot(
            "tilt", {"zh": "机身倾斜", "en": "Body tilt"}, "°", (Series("tilt", {"zh": "倾斜", "en": "tilt"}, "ink"),)
        ),
    )
    steps = (
        Step(
            id="survive",
            title={"zh": "扛住越来越猛的推", "en": "Take harder and harder shoves"},
            body={
                "zh": "先选控制器，再点「开始推」。每 2.5 秒推一次，每次更猛，摔倒就结束。\n"
                "星星：扛过 3 级、5 级、7 级。\n"
                "三个控制器都试一试：「保持姿势」只让每个电机守住角度；「原地踏步」一直在倒腾脚，"
                "被推时有机会接住；「MPC 站稳」每 20 毫秒预演很多种未来，主动调整身体。",
                "en": "Pick a controller, then press Start shoving. A shove every 2.5 s, each one harder; a fall ends "
                "the run.\nStars: survive levels 3, 5 and 7.\nTry all three: Hold pose only keeps every motor at its "
                "angle; Step in place keeps moving its feet and can catch a shove; Stand with MPC rehearses many "
                "futures every 20 ms and actively shifts its body.",
            },
            done={
                "zh": "同一台机器人、同样的电机，只换控制器就能多扛几级：保持姿势最先倒，会迈步的好一些，"
                "会预演未来的 MPC 通常最稳。差距不算大，真正抗推的四足会把“预测”和“迈步”结合起来，"
                "这正是研究的前沿。",
                "en": "Same robot, same motors, and a different controller survives more: holding a pose falls "
                "first, stepping does a little better, and MPC, which looks ahead, usually lasts longest. "
                "The gap is not huge; legged robots that really shrug off shoves combine prediction with "
                "stepping, which is research today.",
            },
            check=_run_scored,
            unlock=("controller", "start"),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.level = 0
        self.next_shove: float | None = None
        self.rng = random.Random(0)
        self.samples: deque[dict[str, float]] = deque(maxlen=1024)
        self.next_sample = 0.0

    def build(self) -> mujoco.MjModel:
        world = World.model_validate(
            {
                "name": "challenge_shove",
                "floor": {"style": "concrete"},
                "robots": [{"name": "dog", "template": "quadruped"}],
            }
        )
        return compile_world(world).model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        super().attach(model, data, ctx)
        self.level = 0
        self.next_shove = None
        return self._controller(model, data, int(ctx.values.get("controller", 0)))

    def _controller(self, model: mujoco.MjModel, data: mujoco.MjData, choice: int) -> dict[str, Any]:
        from robotforge.control.methods import make_controller
        from robotforge.control.tasks import make_task

        if choice == 1:  # the trot gait with zero speed keeps stepping
            self.playground.behaviors.move(self.robot, 0.0, 0.0)
            return {self.robot: self.playground.session.controllers[self.robot]}
        snapshot = copy.copy(data)
        task = make_task("stand", model, self.robot)
        task.begin(snapshot)
        return {self.robot: make_controller("hold" if choice == 0 else "mpc", task, model, snapshot, None)}

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        session = self.playground.session
        if control == "controller":
            for name, ctrl in self._controller(session.model, data, int(value)).items():
                session.set_controller(name, ctrl)
        elif control == "start" and self.next_shove is None and self.finished is None:
            self.started = float(data.time)
            self.next_shove = float(data.time) + 1.0

    def drain_samples(self) -> list[dict[str, float]]:
        out = list(self.samples)
        self.samples.clear()
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        body = self.trunk(model)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body][8])))))
        if data.time >= self.next_sample:
            self.next_sample = float(data.time) + 0.05
            self.samples.append({"t": float(data.time), "tilt": tilt})
        if self.next_shove is not None and self.finished is None:
            if self.fallen(data, body):
                self._finish(ctx, data, survived=self.level - 1)
            elif data.time >= self.next_shove:
                if self.level >= SHOVE_LEVELS:
                    self._finish(ctx, data, survived=SHOVE_LEVELS)
                else:
                    self.level += 1
                    self._shove(model, data, body)
                    self.next_shove = float(data.time) + SHOVE_EVERY
        state = "" if self.next_shove is not None or self.finished is not None else "（点「开始推」）"
        return {
            "readout": self.retry(
                {
                    "zh": f"第 {self.level} 级  倾斜 {tilt:.0f}°{state}",
                    "en": f"level {self.level}  tilt {tilt:.0f}°" + ("" if not state else " (press Start shoving)"),
                }
            ),
            "level": self.level,
        }

    def _shove(self, model: mujoco.MjModel, data: mujoco.MjData, body: int) -> None:
        jnt = int(model.body_jntadr[body])
        dof = int(model.jnt_dofadr[jnt])
        side = 1.0 if self.rng.random() < 0.5 else -1.0
        kick = np.array([0.15 * self.rng.uniform(-1, 1), side, 0.0]) * SHOVE_STEP * self.level
        data.qvel[dof : dof + 3] += kick  # a sideways shove on the free joint's linear velocity

    def _finish(self, ctx: LessonContext, data: mujoco.MjData, survived: int) -> None:
        self.finished = float(data.time)
        survived = max(0, survived)
        stars = _stars(survived, (3, 5, 7))
        ctx.results["stars"] = stars
        ctx.results["run"] = {"levels": survived, "stars": stars}
        ctx.results["summary"] = {"zh": f"扛过了 {survived} 级", "en": f"Survived {survived} levels"}

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [-0.9, -1.5, 0.75], "target": [0.0, 0.0, 0.15]}
