"""Lessons: short interactive courses that run inside the viewer.

A lesson loads its own small rig, attaches a controller whose knobs the learner turns, and walks
through steps. Each step's check runs on the physics, so "done" means the robot really did it.
The viewer draws what the lesson declares (controls, plots, overlays, quizzes); physics, checks and
grading stay here, the single source of truth.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mujoco

if TYPE_CHECKING:
    from robotforge.playground import Playground

Text = dict[str, str]  # {"zh": ..., "en": ...}
SAMPLE_SECONDS = 12.0  # how much plot history the lesson keeps for its checks


class LessonError(ValueError):
    pass


@dataclass(frozen=True)
class Control:
    id: str
    kind: Literal["slider", "toggle", "button"]
    label: Text
    min: float = 0.0
    max: float = 1.0
    step: float = 0.1
    value: float = 0.0
    unit: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "value": self.value,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class Series:
    key: str
    label: Text
    color: Literal["ink", "amber", "blue", "violet", "green", "muted"] = "ink"
    dashed: bool = False


@dataclass(frozen=True)
class Plot:
    id: str
    title: Text
    unit: str
    series: tuple[Series, ...]
    window: float = 6.0  # seconds shown

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "unit": self.unit,
            "window": self.window,
            "series": [{"key": s.key, "label": s.label, "color": s.color, "dashed": s.dashed} for s in self.series],
        }


@dataclass(frozen=True)
class Quiz:
    question: Text
    options: tuple[Text, ...]
    answer: int
    explain: Text


@dataclass(frozen=True)
class Step:
    id: str
    title: Text
    body: Text  # what to try, and the idea behind it
    done: Text  # what the learner just saw, shown once the check passes
    check: Callable[[LessonContext], bool] | None = None  # None: finished by pressing Next
    quiz: Quiz | None = None
    unlock: tuple[str, ...] = ()  # controls usable from this step on
    show: tuple[str, ...] = ()  # overlays switched on from this step on

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "title": self.title, "body": self.body, "done": self.done}
        if self.quiz is not None:
            q = self.quiz
            out["quiz"] = {"question": q.question, "options": list(q.options)}
        out["manual"] = self.check is None and self.quiz is None
        return out


@dataclass
class LessonContext:
    """What a step's check can look at: the learner's settings, what happened, and recent samples."""

    values: dict[str, float]
    events: set[str] = field(default_factory=set)
    samples: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=4096))
    answers: dict[str, int] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)  # e.g. the last challenge run

    def recent(self, seconds: float) -> list[dict[str, float]]:
        if not self.samples:
            return []
        end = self.samples[-1]["t"]
        return [s for s in self.samples if s["t"] >= end - seconds]


class Lesson:
    """One course. Subclasses fill in the class attributes and the rig hooks."""

    id: str = ""
    chapter: Text = {}
    title: Text = {}
    summary: Text = {}
    minutes: int = 10
    controls: tuple[Control, ...] = ()
    plots: tuple[Plot, ...] = ()
    steps: tuple[Step, ...] = ()

    def build(self) -> mujoco.MjModel:
        raise NotImplementedError

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        """Return {controller name: controller} to run while the lesson is open."""
        raise NotImplementedError

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        """A slider moved or a button was pressed; settings are already in ctx.values."""

    def drain_samples(self) -> list[dict[str, float]]:
        """Plot samples recorded by the controller since the last call."""
        return []

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        """Extra numbers the viewer shows or draws with (overlays), sent with every lesson update."""
        return {}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        """Static facts the viewer needs to draw overlays (pivot points, body ids, lengths)."""
        return {}

    def camera(self) -> dict[str, list[float]] | None:
        return None

    def describe(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "id": self.id,
            "chapter": self.chapter,
            "title": self.title,
            "summary": self.summary,
            "controls": [c.describe() for c in self.controls],
            "plots": [p.describe() for p in self.plots],
            "steps": [s.describe() for s in self.steps],
            "rig": self.rig(model),
            "camera": self.camera(),
        }

    def card(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "chapter": self.chapter,
            "title": self.title,
            "summary": self.summary,
            "minutes": self.minutes,
        }


class LessonRunner:
    """Runs at most one lesson at a time on the playground's session.

    The lesson replaces the open world while it runs; leaving it brings the world back.
    """

    def __init__(self, playground: Playground, lessons: dict[str, type[Lesson]]) -> None:
        self.playground = playground
        self.lessons = lessons
        self.lesson: Lesson | None = None
        self.ctx: LessonContext | None = None
        self.step = 0
        self.passed: set[str] = set()
        self._controllers: dict[str, Any] = {}
        self._version = -1
        self._saved: tuple[Any, mujoco.MjModel, str] | None = None  # (store, model, source) to return to
        self._lock = threading.RLock()

    # ---- catalog and lifecycle -----------------------------------------------------

    def catalog(self) -> list[dict[str, Any]]:
        return [cls().card() for cls in self.lessons.values()]

    @property
    def active(self) -> bool:
        return self.lesson is not None

    def start(self, lesson_id: str) -> dict[str, Any]:
        cls = self.lessons.get(lesson_id)
        if cls is None:
            raise LessonError(f"unknown lesson {lesson_id!r}; lessons: {', '.join(self.lessons)}")
        pg = self.playground
        session = pg.session
        lesson = cls()
        model = lesson.build()
        with self._lock:
            if self.lesson is None:  # switching lessons keeps the world we came from
                self._saved = (pg.store, session.model, session.source)
            pg.store = None
            pg.modes.clear()
            session.load_model(model, f"lesson:{lesson.id}")
            self.lesson = lesson
            self.ctx = LessonContext(values={c.id: c.value for c in lesson.controls if c.kind != "button"})
            self.step = 0
            self.passed = set()
            self._attach()
            self._version = session.version
            return lesson.describe(model)

    def exit(self) -> dict[str, Any]:
        with self._lock:
            if self.lesson is None:
                return {"left": None}
            left = self.lesson.id
            self._close()
            saved, self._saved = self._saved, None
        if saved is not None:
            store, model, source = saved
            pg = self.playground
            if store is not None:
                pg.store = store
                pg.reload_world()
            else:
                pg.session.load_model(model, source)
        return {"left": left}

    def _close(self) -> None:
        self.lesson = None
        self.ctx = None
        self._controllers = {}

    def _attach(self) -> None:
        """(Re)attach the lesson's controllers, e.g. after the learner pressed Reset."""
        assert self.lesson is not None and self.ctx is not None
        session = self.playground.session
        with session.lock:
            self._controllers = self.lesson.attach(session.model, session.data, self.ctx)
            for name, ctrl in self._controllers.items():
                session.set_controller(name, ctrl)

    def after_reset(self) -> None:
        """Session reset hook: a reset clears controllers, so put the lesson's back."""
        with self._lock:
            if self.lesson is not None and self.playground.session.version == self._version:
                self._attach()

    # ---- learner input ---------------------------------------------------------------

    def handle(self, msg: dict[str, Any]) -> None:
        action = msg.get("action")
        with self._lock:
            if self.lesson is None or self.ctx is None:
                raise LessonError("no lesson is running")
            if action == "set":
                self._set(str(msg["control"]), float(msg["value"]))
            elif action == "press":
                self._press(str(msg["control"]))
            elif action == "answer":
                self._answer(int(msg["choice"]))
            elif action == "next":
                self._advance(force=True)
            elif action == "goto":
                self._goto(int(msg["step"]))
            else:
                raise LessonError(f"unknown lesson action {action!r}")

    def _control(self, control_id: str) -> Control:
        assert self.lesson is not None
        for c in self.lesson.controls:
            if c.id == control_id:
                return c
        raise LessonError(f"this lesson has no control {control_id!r}")

    def _unlocked(self) -> set[str]:
        assert self.lesson is not None
        return {cid for s in self.lesson.steps[: self.step + 1] for cid in s.unlock}

    def _set(self, control_id: str, value: float) -> None:
        c = self._control(control_id)
        if c.kind == "button":
            raise LessonError(f"{control_id} is a button; press it instead")
        if control_id not in self._unlocked():
            raise LessonError(f"{control_id} unlocks in a later step")
        value = min(max(value, c.min), c.max) if c.kind == "slider" else float(bool(value))
        assert self.ctx is not None and self.lesson is not None
        self.ctx.values[control_id] = value
        with self.playground.session.lock:
            self.lesson.on_control(control_id, value, self.playground.session.data, self.ctx)

    def _press(self, control_id: str) -> None:
        c = self._control(control_id)
        if c.kind != "button":
            raise LessonError(f"{control_id} is not a button")
        if control_id not in self._unlocked():
            raise LessonError(f"{control_id} unlocks in a later step")
        assert self.ctx is not None and self.lesson is not None
        self.ctx.events.add(f"pressed:{control_id}")
        with self.playground.session.lock:
            self.lesson.on_control(control_id, 1.0, self.playground.session.data, self.ctx)

    def _answer(self, choice: int) -> None:
        assert self.lesson is not None and self.ctx is not None
        step = self.lesson.steps[self.step]
        if step.quiz is None:
            raise LessonError("this step has no question")
        if not 0 <= choice < len(step.quiz.options):
            raise LessonError("no such answer")
        self.ctx.answers[step.id] = choice
        if choice == step.quiz.answer:
            self.passed.add(step.id)

    def _advance(self, force: bool) -> None:
        assert self.lesson is not None
        step = self.lesson.steps[self.step]
        if step.id not in self.passed and not (force and step.check is None and step.quiz is None):
            raise LessonError("finish this step first")
        self.passed.add(step.id)
        if self.step + 1 < len(self.lesson.steps):
            self.step += 1

    def _goto(self, index: int) -> None:
        assert self.lesson is not None
        reachable = max([i for i, s in enumerate(self.lesson.steps) if s.id in self.passed] + [-1]) + 1
        if not 0 <= index <= min(reachable, len(self.lesson.steps) - 1):
            raise LessonError("that step is not open yet")
        self.step = index

    # ---- the live update the viewer receives ----------------------------------------------

    def tick(self) -> dict[str, Any] | None:
        """Called by the viewer hub ~20 times a second while a lesson runs."""
        with self._lock:
            if self.lesson is None or self.ctx is None:
                return None
            session = self.playground.session
            if session.version != self._version:  # the world was swapped under the lesson
                self._close()
                self._saved = None
                return {"type": "lesson", "ended": True}
            new = self.lesson.drain_samples()
            self.ctx.samples.extend(new)
            if session.drag is not None:
                self.ctx.events.add("dragged")
            step = self.lesson.steps[self.step]
            if step.check is not None and step.id not in self.passed:
                with session.lock:
                    if step.check(self.ctx):
                        self.passed.add(step.id)
            with session.lock:
                live = self.lesson.live(session.model, session.data, self.ctx)
            quiz = None
            if step.quiz is not None and step.id in self.ctx.answers:
                chosen = self.ctx.answers[step.id]
                quiz = {"chosen": chosen, "right": chosen == step.quiz.answer, "explain": step.quiz.explain}
            return {
                "type": "lesson",
                "lesson": self.lesson.id,
                "step": self.step,
                "passed": [s.id for s in self.lesson.steps if s.id in self.passed],
                "unlocked": sorted(self._unlocked()),
                "shown": sorted({o for s in self.lesson.steps[: self.step + 1] for o in s.show}),
                "values": dict(self.ctx.values),
                "samples": [_round(s) for s in new],
                "live": live,
                "results": self.ctx.results,
                "quiz": quiz,
            }

    def describe_active(self) -> dict[str, Any] | None:
        with self._lock:
            if self.lesson is None:
                return None
            return self.lesson.describe(self.playground.session.model)


def _round(sample: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in sample.items()}
