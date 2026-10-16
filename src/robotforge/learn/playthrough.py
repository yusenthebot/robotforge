"""Playthroughs: what a learner does in every lesson, step by step, and a player that does it headless.

Every step of a lesson is checked on the physics, so a step that no sequence of moves can pass is a lesson
a learner gets stuck in, and nothing else would notice. A recipe here is the moves a step's text asks for
(make the guess, turn the knobs, press the buttons, let time pass); `play` makes them on a real Playground
the way the viewer does: physics in 50 ms slices with a lesson update after each, and every step must pass
within its time. scripts/playtest.py plays every lesson and prints the table; tests/test_playthrough.py
keeps them passing.
"""

from __future__ import annotations

import inspect
import json
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robotforge.learn import autopilot, balance, challenges, gait, ik, imitation, pd, sensors, twins
from robotforge.learn.engine import LessonError, Step

if TYPE_CHECKING:
    from robotforge.playground import Playground

TICK = 0.05  # s of physics between lesson updates: the viewer sends them 20 times a second
GRACE = 1.0  # s a step may take to notice it passed once its moves are made


class TimeUp(Exception):
    """A step used up its time."""


class Player:
    """Makes a learner's moves in the running lesson and plays the physics in between."""

    def __init__(self, playground: Playground) -> None:
        self.pg = playground
        self.update: dict[str, Any] = {}
        self.sim = 0.0  # s of physics played so far
        self.limits = (math.inf, math.inf)  # what the current step may take: s of physics, s on the clock
        self._sim_end = math.inf
        self._wall_end = math.inf

    @property
    def lesson(self) -> Any:
        return self.pg.lessons.lesson

    @property
    def ctx(self) -> Any:
        return self.pg.lessons.ctx

    @property
    def session(self) -> Any:
        return self.pg.session

    def budget(self, sim: float, wall: float) -> None:
        self.limits = (sim, wall)
        self._sim_end = self.sim + sim
        self._wall_end = time.perf_counter() + wall

    def handle(self, action: str, **fields: Any) -> None:
        self.pg.lessons.handle({"action": action, **fields})

    def play(self, seconds: float, realtime: bool = False) -> dict[str, Any]:
        """Physics and lesson updates as the viewer runs them. `realtime` holds the pace at one simulated second
        per second, as in the viewer, so a lesson's background training gets the machine it would have there."""
        session = self.session
        for _ in range(max(1, round(seconds / TICK))):
            if self.sim >= self._sim_end:
                raise TimeUp(f"out of time: the step's {self.limits[0]:g} s of physics ran out")
            if time.perf_counter() >= self._wall_end:
                raise TimeUp(f"out of time: the step's {self.limits[1]:g} s on the clock ran out")
            start = time.perf_counter()
            session.step(max(1, round(TICK / session.model.opt.timestep)))
            self.update = self.pg.lessons.tick() or {}
            self.sim += TICK
            if realtime:
                time.sleep(max(0.0, TICK - (time.perf_counter() - start)))
        return self.update

    def passed(self, step: Step) -> bool:
        return step.id in self.pg.lessons.passed

    def wait_for(self, name: str) -> None:
        """Play until a FINISHED condition holds: a test, a training, a planned move has ended."""
        finished = FINISHED[name]
        realtime = name == "training"
        self.play(0.2, realtime)
        while not finished(self):
            self.play(TICK if realtime else 0.25, realtime)

    def wait_to_pass(self, step: Step, seconds: float) -> None:
        end = self.sim + seconds
        while not self.passed(step) and self.sim < end:
            self.play(TICK)


FINISHED: dict[str, Callable[[Player], bool]] = {
    "test": lambda p: p.lesson.test is None,  # a face-off's or a bench's test has ended
    "training": lambda p: p.lesson.thread is None or not p.lesson.thread.is_alive(),
    "move": lambda p: p.lesson.run is None or bool(p.lesson.run.get("finished")),  # a planned arm move
    "shoves": lambda p: p.lesson.next_shove is None,  # a round of ever harder shoves
    "run": lambda p: "run" in p.ctx.results,  # a challenge's scored run
    # the imitation lesson's try: the arm on its own, after any training it waited for
    "try": lambda p: p.lesson.hands.pilot is None and p.lesson.pending is None and not p.lesson.busy,
}


# ---- moves -----------------------------------------------------------------------------------------------


class Move:
    """One thing a learner does: click, slide, wait."""

    def apply(self, p: Player) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Answer(Move):
    choice: int

    def apply(self, p: Player) -> None:
        p.handle("answer", choice=self.choice)
        p.play(TICK)

    def __str__(self) -> str:
        return f"answer {self.choice}"


class Set(Move):
    """Turn knobs, in the order given."""

    def __init__(self, **values: float) -> None:
        self.values = values

    def apply(self, p: Player) -> None:
        for control, value in self.values.items():
            p.handle("set", control=control, value=value)
        p.play(TICK)

    def __str__(self) -> str:
        return "set " + ", ".join(f"{k}={v:g}" for k, v in self.values.items())


@dataclass(frozen=True)
class Press(Move):
    """Press a button, wait for what it started to end (`until`, a FINISHED name), then `seconds` more."""

    control: str
    seconds: float = 0.1
    until: str | None = None

    def apply(self, p: Player) -> None:
        p.handle("press", control=self.control)
        if self.until is not None:
            p.wait_for(self.until)
        p.play(self.seconds)

    def __str__(self) -> str:
        wait = f"until the {self.until} ends, then " if self.until else ""
        return f"press {self.control}, {wait}{self.seconds:g} s"


@dataclass(frozen=True)
class Wait(Move):
    seconds: float

    def apply(self, p: Player) -> None:
        p.play(self.seconds)

    def __str__(self) -> str:
        return f"wait {self.seconds:g} s"


@dataclass(frozen=True)
class Sweep(Move):
    control: str
    until: str | None = None  # a sweep that runs on the lesson's own thread: wait for it to end (a FINISHED name)

    def apply(self, p: Player) -> None:
        p.handle("sweep", control=self.control)
        if self.until is not None:
            p.wait_for(self.until)
        p.play(TICK)

    def __str__(self) -> str:
        return f"sweep {self.control}" + (f", until the {self.until} ends" if self.until else "")


@dataclass(frozen=True)
class Drag(Move):
    """Grab a body with the mouse (at a site on it, or its origin), pull it to `to`, hold, and let go."""

    body: str
    to: tuple[float, float, float]
    site: str = ""
    seconds: float = 1.0

    def apply(self, p: Player) -> None:
        session = p.session
        grab = session.data.site(self.site).xpos if self.site else session.data.body(self.body).xpos
        session.start_drag(session.model.body(self.body).id, grab.copy())
        session.move_drag(np.array(self.to, float))
        try:
            p.play(self.seconds)
        finally:
            session.end_drag()
        p.play(0.2)

    def __str__(self) -> str:
        return f"drag {self.body} to {list(self.to)} for {self.seconds:g} s"


@dataclass(frozen=True)
class Do(Move):
    """Moves that need eyes on the screen, such as driving to a spot: a function that plays them."""

    what: Callable[[Player], None]
    label: str

    def apply(self, p: Player) -> None:
        self.what(p)

    def __str__(self) -> str:
        return self.label


# ---- playing a lesson -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    steps: dict[str, tuple[Move, ...]]
    slow: bool = False  # trains on a background thread or runs long tests: out of the quick test run
    sim_limit: float = 120.0  # s of physics one step may take
    wall_limit: float = 120.0  # s on the clock one step may take


@dataclass
class StepReport:
    id: str
    passed: bool
    sim: float
    wall: float
    tried: list[str] = field(default_factory=list)
    failed: str = ""  # why it did not pass: the check that never held, or what the lesson refused
    state: str = ""  # the lesson's results and last recorded run when it got stuck


@dataclass
class LessonReport:
    lesson: str
    total: int
    steps: list[StepReport] = field(default_factory=list)
    error: str = ""  # the lesson could not even start
    results: dict[str, Any] = field(default_factory=dict)  # the lesson's results when it was left

    @property
    def passed(self) -> int:
        return sum(s.passed for s in self.steps)

    @property
    def ok(self) -> bool:
        return not self.error and self.passed == self.total

    @property
    def sim(self) -> float:
        return sum(s.sim for s in self.steps)

    @property
    def wall(self) -> float:
        return sum(s.wall for s in self.steps)

    @property
    def stuck(self) -> StepReport | None:
        return next((s for s in self.steps if not s.passed), None)

    def explain(self) -> str:
        """What went wrong, for a failed test or the playtest's notes."""
        if self.error:
            return f"{self.lesson}: {self.error}"
        stuck = self.stuck
        if stuck is None:
            return f"{self.lesson}: all {self.total} steps passed"
        lines = [
            f"{self.lesson}: stuck at step {stuck.id!r} ({self.passed} of {self.total} passed)",
            f"  tried: {'; '.join(stuck.tried) or '-'}",
            f"  failed: {stuck.failed}",
        ]
        if stuck.state:
            lines.append(f"  state: {stuck.state}")
        return "\n".join(lines)


def manual(step: Step) -> bool:
    """A step that only asks the learner to read on: Next finishes it."""
    return step.check is None and step.quiz is None


def describe_check(step: Step) -> str:
    if step.check is None:
        assert step.quiz is not None
        return f"the quiz wants answer {step.quiz.answer}"
    check = step.check
    if check.__name__ == "<lambda>":
        try:
            return inspect.getsource(check).strip().removeprefix("check=").rstrip(",")
        except (OSError, TypeError):
            return "a lambda"
    doc = (check.__doc__ or "").strip().splitlines()
    return f"{check.__name__}" + (f" ({doc[0]})" if doc else "")


def _state(p: Player) -> str:
    ctx = p.ctx
    if ctx is None:
        return ""
    results = {k: v for k, v in ctx.results.items() if k != "summary"}
    out = "results " + json.dumps(results, default=str, ensure_ascii=False)[:400]
    if ctx.runs:
        last = ctx.runs[-1]
        out += f"; last run {last['params']} -> {json.dumps(last['metrics'], default=str)[:200]}"
    return out


def play(playground: Playground, lesson_id: str, recipe: Recipe | None = None) -> LessonReport:
    """Play one lesson from its first step to its last, as far as it goes."""
    recipe = recipe or RECIPES[lesson_id]
    runner = playground.lessons
    try:
        runner.start(lesson_id)
    except (LessonError, ValueError, OSError) as exc:
        return LessonReport(lesson_id, 0, error=f"could not start: {exc}")
    steps = tuple(runner.lesson.steps)
    report = LessonReport(lesson_id, len(steps))
    player = Player(playground)
    try:
        for step in steps:
            result = _play_step(player, step, recipe)
            report.steps.append(result)
            if not result.passed:
                break
        if runner.ctx is not None:  # None when the lesson ended under the player (its world was swapped)
            report.results = json.loads(json.dumps(runner.ctx.results, default=str))
    finally:
        runner.exit()
    return report


def _play_step(p: Player, step: Step, recipe: Recipe) -> StepReport:
    moves = recipe.steps.get(step.id)
    sim0, wall0 = p.sim, time.perf_counter()
    p.budget(recipe.sim_limit, recipe.wall_limit)
    tried: list[str] = []
    failed = ""
    if moves is None:
        failed = "the recipe has no moves for this step"
    else:
        try:
            for move in moves:
                tried.append(str(move))
                move.apply(p)
            if manual(step):
                p.handle("next")
            else:
                p.wait_to_pass(step, GRACE)
        except LessonError as exc:
            failed = f"the lesson refused: {exc}"
        except TimeUp as exc:
            failed = str(exc)
        except Exception as exc:  # a lesson that crashes on a learner's move is a finding, not the end of the run
            where = traceback.extract_tb(exc.__traceback__)[-1]
            failed = f"crashed: {type(exc).__name__}: {exc} ({Path(where.filename).name}:{where.lineno})"
    passed = p.passed(step)
    if passed and not manual(step):
        p.handle("next")
    elif not passed and not failed:
        failed = f"check never passed: {describe_check(step)}"
    return StepReport(
        id=step.id,
        passed=passed,
        sim=round(p.sim - sim0, 2),
        wall=round(time.perf_counter() - wall0, 2),
        tried=tried,
        failed="" if passed else failed,
        state="" if passed else _state(p),
    )


# ---- driving on the keys -----------------------------------------------------------------------------------

KEY_SHARE = 0.7  # a held key drives a lesson's robot at this share of its top speed and turn (app.js sendDrive)


def _hold(p: Player, keys: str) -> None:
    """Hold these of W A S D and let go of the rest, sent the way the viewer sends them to a lesson."""
    drive = p.lesson.drive
    fwd = ("W" in keys) - ("S" in keys)
    left = ("A" in keys) - ("D" in keys)
    p.handle("drive", speed=fwd * drive["max_speed"] * KEY_SHARE, turn=left * drive["turn"] * KEY_SHARE)


@dataclass(frozen=True)
class Keys(Move):
    """Hold keys of W A S D for a while, then let go."""

    keys: str
    seconds: float

    def apply(self, p: Player) -> None:
        _hold(p, self.keys)
        p.play(self.seconds)
        _hold(p, "")
        p.play(0.3)

    def __str__(self) -> str:
        return f"hold {self.keys} for {self.seconds:g} s"


def _bearing(p: Player, body: int, point: tuple[float, float]) -> tuple[float, float]:
    """How far the robot must turn to face `point` (rad, left positive) and how far away it is (m)."""
    data = p.session.data
    x, y = data.xpos[body][:2]
    heading = math.atan2(data.xmat[body][3], data.xmat[body][0])
    want = math.atan2(point[1] - y, point[0] - x)
    return math.atan2(math.sin(want - heading), math.cos(want - heading)), math.dist((x, y), point)


def _steer(error: float, drive_within: float = 0.3, straight_within: float = 0.1) -> str:
    """Keys toward a point `error` rad off: turn on the spot when far off, W with a nudge when close, else W."""
    turn = "A" if error > 0 else "D"
    if abs(error) > drive_within:
        return turn
    return "W" + turn if abs(error) > straight_within else "W"


def _face(p: Player, body: int, point: tuple[float, float]) -> None:
    for _ in range(200):
        error, _ = _bearing(p, body, point)
        if abs(error) < math.radians(4):
            break
        _hold(p, "A" if error > 0 else "D")
        p.play(TICK)
    _hold(p, "")
    p.play(0.3)


def _rover(p: Player) -> int:
    return p.session.model.body("rover").id


def _ahead(p: Player) -> float:
    """The lidar's straight-ahead reading in the panel (-1 when nothing is in range)."""
    reading = p.update["live"]["ranges"][0]
    return reading if reading >= 0 else math.inf


def _drive_to_the_wall(p: Player) -> None:
    """Face the near wall and hold W until the lidar reads under 32 cm (the rover stops within a centimetre)."""
    body = _rover(p)
    x, _ = p.session.data.xpos[body][:2]
    _face(p, body, (float(x), -sensors.ARENA))
    _hold(p, "W")
    for _ in range(300):
        p.play(TICK)
        if _ahead(p) < 0.32:
            break
    _hold(p, "")
    p.play(2.5)


def _bump_the_crate(p: Player) -> None:
    body = _rover(p)
    Keys("S", 1.0).apply(p)
    _face(p, body, sensors.CRATE)
    _hold(p, "W")
    for _ in range(200):
        if p.play(TICK)["live"].get("touch"):
            break
    _hold(p, "")
    p.play(0.2)


def _find_the_ball(p: Player) -> None:
    body = _rover(p)
    Keys("S", 1.0).apply(p)
    _face(p, body, sensors.BALL)
    for _ in range(400):
        error, distance = _bearing(p, body, sensors.BALL)
        if distance < 0.9:
            break
        _hold(p, _steer(error))
        p.play(TICK)
    _hold(p, "")
    p.play(0.5)


def _out_of_the_maze(p: Player) -> None:
    """Drive the way out cell by cell on the keys: turn on the spot at every corner, then W."""
    body = p.session.model.body("rover/chassis").id
    for waypoint in challenges.maze_route(p.lesson.seed)[1:]:
        for _ in range(400):
            error, distance = _bearing(p, body, waypoint)
            if distance < 0.2:
                break
            _hold(p, _steer(error))
            p.play(TICK)
            if "run" in p.ctx.results:
                return
    _hold(p, "")
    p.play(0.5)


def _along_the_course(p: Player, every: float = 0.25, off_line: float = 0.2) -> None:
    """Hold W down the middle of the course, tapping A or D whenever the dog drifts off the centre line. A learner
    looks again `every` s and steers once the drift passes `off_line`; small changes to either change the run."""
    body = p.session.model.body("dog/trunk").id
    for _ in range(round(150 / every)):
        data = p.session.data
        heading = math.atan2(data.xmat[body][3], data.xmat[body][0])
        back = -2 * float(data.xpos[body][1]) - 1.5 * heading  # turn this way to come back to the line
        _hold(p, "W" + ("A" if back > off_line else "D" if back < -off_line else ""))
        p.play(every)
        if "run" in p.ctx.results:
            break
    _hold(p, "")


def _step_until_close(p: Player) -> None:
    """Press Step, watch the error, and press again until it reads under 1 cm (seven presses in our runs)."""
    for _ in range(15):
        Press("iterate", 0.4).apply(p)
        if p.update["live"]["err_cm"] < 1.0:
            break


# ---- teaching by showing ----------------------------------------------------------------------------------

DRAG_SPEED = 0.2  # m/s: the scripted learner drags the ball this fast (the arm follows at up to 0.25)
HIGH = 0.1  # m over the desk: carried high enough to clear the bin's rim
GRASP = 0.025  # m over the desk: the cube's middle, and 5 mm more
FIND_THE_TOGGLE = 0.5  # s a learner takes to move from the ball to the grip toggle


def _drag(p: Player, tx: float, ty: float, tz: float) -> None:
    """Drag the amber ball to (tx, ty, tz), in small moves as the viewer sends them, then wait (at most 3 s) for
    the gripper to get there, as a learner watching the arm does."""
    v = p.ctx.values
    start = np.array([v["tx"], v["ty"], v["tz"]], float)
    goal = np.array([tx, ty, tz], float)
    moves = max(1, math.ceil(float(np.linalg.norm(goal - start)) / (DRAG_SPEED * TICK)))
    for k in range(1, moves + 1):
        x, y, z = start + (goal - start) * k / moves
        Set(tx=x, ty=y, tz=z).apply(p)
    for _ in range(round(3.0 / TICK)):
        live = p.update["live"]
        if np.linalg.norm(np.subtract(live["hand"], live["target"])) < 0.01:
            break
        p.play(TICK)


def _show_a_pick(p: Player) -> None:
    """One demonstration with the lesson's own controls: the gripper over the cube where the screen shows it,
    down, close, up, over the bin, open; then wait until the record has kept it."""
    before = p.ctx.results.get("demos", 0)
    p.play(TICK)
    x, y, _ = p.update["live"]["cube"]  # the arm's base is at the desk's origin: x, y are the sliders' values
    bin_x, bin_y, _ = p.lesson.rig(p.session.model)["bin"]
    _drag(p, x, y, HIGH)
    _drag(p, x, y, GRASP)
    p.play(FIND_THE_TOGGLE)
    Set(grip=1).apply(p)
    p.play(FIND_THE_TOGGLE)
    _drag(p, x, y, HIGH)
    _drag(p, bin_x, bin_y, HIGH)
    p.play(FIND_THE_TOGGLE)
    Set(grip=0).apply(p)
    for _ in range(round(3.0 / TICK)):
        if p.ctx.results.get("demos", 0) > before:
            break
        p.play(TICK)


def _show_more(p: Player) -> None:
    while p.ctx.results.get("demos", 0) < imitation.DEMOS_WANTED:
        Press("new_cube", 0.3).apply(p)
        _show_a_pick(p)


# ---- the recipes -----------------------------------------------------------------------------------------

JOINT_TEST = pd.TEST_SECONDS + 0.4  # a Run test and its result coming in
FK_ANGLES = tuple(round(math.degrees(q)) for q in ik.analytic(*ik.START_TARGET, elbow_up=True))
CLEAN = autopilot.RUN_SECONDS + 0.6
WALK = gait.TEST_SECONDS + 0.4
PUSH = balance.PUSH_AT + balance.WATCH + 0.3
CART_PUSH = twins.WATCH_SECONDS + 0.4
LEARN = Press("train", 0.5, until="training")


def _push(strength: float) -> tuple[Move, ...]:
    return (Set(strength=strength), Press("push", PUSH))


RECIPES: dict[str, Recipe] = {
    "pd_control": Recipe(
        {
            "feel": (Drag("arm", (0.3, -0.06, 0.62), site="tip"),),
            "p": (Answer(1), Set(kp=10), Press("test", JOINT_TEST), Set(kp=40), Press("test", JOINT_TEST)),
            "d": (Answer(1), Sweep("kd"), Set(kd=3), Press("test", JOINT_TEST)),
            "why_low": (Answer(0), Set(kp=80, kd=5), Press("test", JOINT_TEST)),
            "ff_i": (
                Answer(0),
                Set(gravity=1),
                Press("test", JOINT_TEST),
                Set(gravity=0, ki=10),
                Press("test", JOINT_TEST),
            ),
            "challenge": (Set(ki=0, gravity=1), Press("test", JOINT_TEST)),
        }
    ),
    "pid_piper": Recipe(
        {
            "copy": (Answer(1), Press("test", JOINT_TEST)),
            "ff": (Answer(1), Set(gravity=1), Press("test", JOINT_TEST)),
            "zeta": (Set(kp=80, kd=10), Press("test", JOINT_TEST)),
            "payload": (Answer(1), Set(payload=1), Press("test", JOINT_TEST), Set(ki=20), Press("test", JOINT_TEST)),
            "challenge": (Set(gravity=0, ki=60), Press("test", JOINT_TEST)),
            "quiz": (Answer(1),),
        }
    ),
    "inverse_kinematics": Recipe(
        {
            "fk": (Set(theta1=FK_ANGLES[0], theta2=FK_ANGLES[1]), Wait(1.0)),
            "ik": (
                Set(method=1),
                Set(tx=0.35, tz=0.05),
                Wait(1.0),
                Set(tx=0.05, tz=0.4),
                Wait(1.0),
                Set(tx=-0.25, tz=0.25),
                Wait(1.0),
            ),
            "workspace": (Set(tx=0.6, tz=0.45), Wait(1.2)),
            "elbow": (Set(tx=0.3, tz=0.05), Wait(1.0), Set(elbow_up=0), Wait(1.2)),
            "iterate": (Set(method=2), Wait(0.5), Do(_step_until_close, "press Step until the error reads under 1 cm")),
            "why_iterate": (Answer(1),),
        }
    ),
    "ik_arm": Recipe(
        {
            "drag": (Set(tx=0.35, ty=0.1, tz=0.2), Wait(1.5), Set(tx=0.2, ty=-0.15, tz=0.35), Wait(1.5)),
            "ellipsoid": (Set(tx=0.6, ty=0.0, tz=0.2), Wait(2.0)),
            "damping": (
                Set(damping=0, tx=0.3),
                Wait(1.5),
                Set(tx=0.85),
                Wait(2.0),
                Set(damping=3, tx=0.3),
                Wait(1.5),
                Set(tx=0.85),
                Wait(2.0),
            ),
            "down": (Set(down=1, tx=0.3, ty=0.0, tz=0.1), Wait(2.0)),
            "piper": (Set(down=0, arm=1, tx=0.55, ty=0.0, tz=0.25), Wait(3.0)),
            "quiz": (Answer(1),),
        }
    ),
    "arm_planning": Recipe(
        {
            "straight": (Answer(1), Press("straight", until="move")),
            "rrt": (Answer(0), Press("plan", until="move")),
            "smooth": (
                Answer(1),
                Set(smooth=0),
                Press("plan", until="move"),
                Set(smooth=1),
                Press("plan", until="move"),
            ),
            "step": (
                Answer(1),
                Set(step=0),
                Press("plan", until="move"),
                Set(step=3),
                Press("plan", until="move"),
            ),
            "piper": (Set(arm=1), Press("straight", until="move"), Press("plan", until="move")),
            "quiz": (Answer(2),),
        }
    ),
    "sensors": Recipe(
        {
            "encoders": (Keys("W", 3.3),),
            "imu": (Keys("A", 4.5),),
            "lidar": (Do(_drive_to_the_wall, "face the near wall, hold W until the lidar reads 32 cm"),),
            "bumper": (Do(_bump_the_crate, "back off, face the crate, W into it"),),
            "camera": (Do(_find_the_ball, "back off, turn to the ball, drive up to it"),),
            "drift": (Answer(1),),
        }
    ),
    "lidar_autopilot": Recipe(
        {
            "stop": (Wait(0.5), Press("run", CLEAN)),
            "timid": (Set(stop_a=3), Press("run", CLEAN)),
            "speed": (Set(stop_a=1, stop_b=1, speed_a=2), Press("run", CLEAN)),
            "noise": (Set(noise_a=2, noise_b=2, filter_b=1, speed_a=1, speed_b=1), Press("run", CLEAN)),
            "quiz": (Answer(1),),
        }
    ),
    "gait": Recipe(
        {
            "trot": (Answer(0), Press("walk", WALK)),
            "pace": (Answer(0), Set(pattern=1), Press("walk", WALK)),
            "bound": (Answer(1), Set(pattern=2), Press("walk", WALK)),
            "period": (
                Answer(1),
                Set(pattern=0, speed=0.6, period=0.5),
                Press("walk", WALK),
                Set(period=0.25),
                Press("walk", WALK),
            ),
            "go2": (Answer(1), Set(period=0.36, dog=1), Press("walk", WALK)),
            "challenge": (Set(period=0.25), Press("walk", WALK)),
            "quiz": (Answer(1),),
        }
    ),
    "balance": Recipe(
        {
            "stand": (Wait(1.5), Answer(0), *_push(0.1)),
            "edge": (Answer(1), *_push(0.3), *_push(0.4)),
            "side": (Answer(0), Set(direction=1), *_push(0.5)),
            "crouch": (Answer(1), Set(direction=0, crouch=0.3), *_push(0.4)),
            "g1": (Answer(1), Set(robot=1, crouch=0), *_push(0.8)),
            "quiz": (Answer(0),),
        }
    ),
    "reinforcement_learning": Recipe(
        {
            "fall": (Press("release", 3.0),),
            "reward": (Answer(1),),
            "random": (Press("random", 1.0), Press("random", 1.0), Press("random", 1.0)),
            "train": (LEARN,),
            "tune": (Set(lr=0.02, fast=1), LEARN),
            "push": (Press("push", 4.0),),
            "scale": (Answer(1),),
        },
        slow=True,
        sim_limit=math.inf,  # training runs on the clock (0.7 s a round, so it can be watched), not on the physics
        wall_limit=180.0,
    ),
    "learn_to_walk": Recipe(
        {
            "rhythm": (Answer(2), LEARN),
            "reward": (Answer(1), Set(energy=1), LEARN, Set(energy=2), LEARN),
            "body": (Answer(1), Set(robot=1, energy=0), LEARN),
            "two_legs": (Answer(1), Set(robot=2), LEARN),
            "reflexes": (Answer(1), Set(shove=1, iterations=80), LEARN, Set(reflexes=1), LEARN),
            "quiz": (Answer(1),),
        },
        slow=True,
        sim_limit=math.inf,
        wall_limit=600.0,
    ),
    "imitation": Recipe(
        {
            "show": (Do(_show_a_pick, "steer the gripper to the cube, close it, carry it over the bin, open"),),
            "copy": (Answer(0), Press("try", until="try")),
            "far": (Answer(1), Set(where=1), Press("try", until="try")),
            "more": (Do(_show_more, f"New cube, then show it again, until {imitation.DEMOS_WANTED} are kept"),),
            "test": (Answer(2), Press("test", until="training")),
            "network": (Answer(1), Set(method=1), Press("test", until="training")),
            "how_many": (Answer(1), Sweep("use", until="training")),
            "ik": (Answer(1), Set(method=2), Press("test", until="training")),
            "quiz": (Answer(1),),
        },
        wall_limit=240.0,  # the sweep trains a network five times on a busy machine
    ),
    "maze_escape": Recipe({"escape": (Do(_out_of_the_maze, "drive the way out, cell by cell"),)}, sim_limit=240.0),
    "shove_survival": Recipe({"survive": (Wait(0.5), Set(controller=2), Wait(0.5), Press("start", 0.2, until="run"))}),
    "obstacle_course": Recipe(
        {"course": (Do(_along_the_course, "hold W down the middle, tapping A or D back to the centre line"),)},
        sim_limit=180.0,
    ),
    "arm_faceoff": Recipe(
        {
            "meet": (Set(workspace=1), Wait(0.2)),
            "reach": (Press("reach", 0.2, until="test"),),
            "gravity": (Set(gravcomp=1), Press("reach", 0.2, until="test")),
            "why7": (Answer(1),),
            "pick": (Press("pick", 0.2, until="test"),),
        }
    ),
    "humanoid_faceoff": Recipe(
        {
            "meet": (Wait(1.0),),
            "walk": (Press("walk", 0.2, until="test"),),
            "push": (Press("push", 0.2, until="test"),),
            "how": (Answer(1),),
        }
    ),
    "pid_twins": Recipe(
        {
            "compare": (Wait(0.5), Press("test", JOINT_TEST)),
            "tune_a": (Set(kp_a=40, kd_a=3), Press("test", JOINT_TEST)),
            "feedforward": (Set(kp_a=10, kd_a=0.3, gravity=1), Press("test", JOINT_TEST)),
        }
    ),
    "lqr_twins": Recipe(
        {
            "trade": (Wait(1.0), Press("push", CART_PUSH)),
            "weights": (Set(r_b=1, q_b=4, q_a=0), Wait(1.0), Press("push", CART_PUSH)),
            "quiz": (Answer(1),),
        }
    ),
    "mpc_twins": Recipe(
        {
            "bold": (Wait(0.5), Press("shove", 0.2, until="shoves")),
            "horizon": (Set(noise_a=3, horizon_a=0), Press("shove", 0.2, until="shoves")),
            "cost": (Answer(1),),
        },
        slow=True,
    ),
    "gait_twins": Recipe(
        {
            "rhythm": (Wait(0.5), Press("flat", 0.2, until="test")),
            "lift": (
                Set(period_a=1, period_b=1, lift_a=0, lift_b=2),
                Press("flat", 0.2, until="test"),
                Press("curb", 0.2, until="test"),
            ),
        }
    ),
}
