"""Run a task with a controller and measure it; run several controllers and compare them in a table.

The same metrics for every algorithm, so a beginner can see the trade-offs:

- reward: the task's score integrated over the run (higher is better),
- success: is the goal met at the end; failed: did the robot fall / drop the pole,
- effort: mechanical work the actuators did, in joules (sum of |force x speed| dt),
- wall/sim: computer seconds per simulated second (below 1 = faster than real time).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.control.methods import make_controller
from robotforge.control.tasks import Task
from robotforge.sim.session import SimSession

MONITOR = "__monitor__"


class EpisodeMonitor:
    """A controller that does not control: it scores the run at every physics step."""

    def __init__(self, task: Task, model: mujoco.MjModel, data: mujoco.MjData):
        self.task = task
        self.rate_hz = 1.0 / model.opt.timestep
        self.dt = model.opt.timestep
        self.t0 = float(data.time)
        self.reward = 0.0
        self.cost = 0.0
        self.effort = 0.0
        self.failed_at: float | None = None
        self.first_success: float | None = None
        self.steps = 0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        cost = self.task.step_cost(data)
        self.cost += cost * self.dt
        self.reward += (self.task.alive_bonus - cost) * self.dt
        acts = self.task.actuators
        power = np.abs(data.actuator_force[acts] * data.actuator_velocity[acts]).sum()
        self.effort += float(power) * self.dt
        t = float(data.time) - self.t0
        if self.failed_at is None and self.task.failed(data):
            self.failed_at = t
        if self.first_success is None and self.task.success(data):
            self.first_success = t
        self.steps += 1

    def status(self) -> dict[str, Any]:
        return {"controller": "monitor", "reward": round(self.reward, 3)}


@dataclass
class RunResult:
    method: str
    sim_seconds: float
    reward: float
    cost: float
    success: bool
    failed: bool
    failed_at: float | None
    first_success_at: float | None
    effort_j: float
    wall_seconds: float
    report: dict[str, Any] = field(default_factory=dict)
    controller: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    @property
    def wall_per_sim(self) -> float:
        return self.wall_seconds / self.sim_seconds if self.sim_seconds > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["wall_per_sim"] = round(self.wall_per_sim, 3)
        for key in ("reward", "cost", "effort_j", "wall_seconds", "sim_seconds"):
            out[key] = round(out[key], 3)
        return out


def run(
    task: Task,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    method: str,
    seconds: float,
    params: dict[str, Any] | None = None,
    push_at: float | None = None,
    push_strength: float = 1.0,
) -> RunResult:
    """Run `method` on `task` for `seconds` of simulated time from the state in `data` (modified in place)."""
    task.begin(data)
    controller = make_controller(method, task, model, data, params)
    session = SimSession(model=model, data=data)
    monitor = EpisodeMonitor(task, model, data)
    session.set_controller(task.robot.name, controller)
    session.set_controller(MONITOR, monitor)
    events: list[str] = []
    started = time.perf_counter()
    t0 = float(data.time)

    def advance_to(sim_time: float) -> None:
        n = round((sim_time - (data.time - t0)) / model.opt.timestep)
        if n > 0:
            session.step(n)

    if push_at is not None and 0 < push_at < seconds:
        advance_to(push_at)
        events.append(f"t={push_at:.1f}s: {task.push(data, push_strength)}")
    advance_to(seconds)
    wall = time.perf_counter() - started
    if not np.isfinite(data.qpos).all():
        events.append("the simulation became unstable")
    return RunResult(
        method=method,
        sim_seconds=float(data.time - t0),
        reward=monitor.reward,
        cost=monitor.cost,
        success=bool(task.success(data)) and monitor.failed_at is None,
        failed=monitor.failed_at is not None,
        failed_at=monitor.failed_at,
        first_success_at=monitor.first_success,
        effort_j=monitor.effort,
        wall_seconds=wall,
        report=task.report(data),
        controller=controller.status(),
        events=events,
    )


def compare(
    make_task: Callable[[], Task],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    methods: list[str],
    seconds: float,
    params: dict[str, dict[str, Any]] | None = None,
    push_at: float | None = None,
    push_strength: float = 1.0,
    seed: int | None = None,
) -> list[RunResult]:
    """Run each method from the same starting state (a copy of `data`) and return the results.

    With `seed`, the task first randomizes that start (e.g. a tilted pole), identically for every method."""
    results = []
    for method in methods:
        task = make_task()
        start = copy.copy(data)
        if seed is not None:
            task.reset(start, np.random.default_rng(seed))
            mujoco.mj_forward(model, start)
        results.append(run(task, model, start, method, seconds, (params or {}).get(method), push_at, push_strength))
    return results


def table(results: list[RunResult]) -> str:
    """A Markdown table, one row per method."""
    headline = _headline_metric(results)
    head = ["method", "reward", "success", headline or "result", "effort (J)", "wall/sim", "notes"]
    rows = []
    for r in results:
        notes = []
        if r.failed_at is not None:
            notes.append(f"failed at {r.failed_at:.2f}s")
        elif r.first_success_at is not None and r.first_success_at > 0.01:
            notes.append(f"goal reached at {r.first_success_at:.2f}s")
        if "plan_hz" in r.controller:
            notes.append(f"{r.controller['plan_hz']:.0f} plans/s")
        if "train_seconds" in r.controller:
            notes.append(f"learned in {r.controller['train_seconds']:.1f}s")
        notes.extend(e.split(": ", 1)[-1] for e in r.events if "unstable" in e)
        value = r.report.get(headline, "") if headline else ""
        rows.append(
            [
                r.method,
                f"{r.reward:.2f}",
                "yes" if r.success else "no",
                str(value),
                f"{r.effort_j:.2f}",
                f"{r.wall_per_sim:.2f}",
                "; ".join(notes),
            ]
        )
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(head)]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(head), sep, *(line(row) for row in rows)])


def _headline_metric(results: list[RunResult]) -> str | None:
    """The number that best says how well the task went, e.g. distance_m for reach."""
    if not results:
        return None
    for key in ("distance_m", "pole_angle_deg", "avg_speed_m_s", "height_m"):
        if key in results[0].report:
            return key
    return None


@dataclass
class Comparison:
    task: dict[str, Any]
    results: list[RunResult]

    @property
    def table(self) -> str:
        return table(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "table": self.table, "results": [r.to_dict() for r in self.results]}


def compare_methods(
    task: str,
    methods: list[str] | None = None,
    seconds: float | None = None,
    *,
    task_params: dict[str, Any] | None = None,
    method_params: dict[str, dict[str, Any]] | None = None,
    push_at: float | None = None,
    push_strength: float = 1.0,
    model: mujoco.MjModel | None = None,
    robot: str | None = None,
    data: mujoco.MjData | None = None,
    seed: int | None = 0,
) -> Comparison:
    """Run several methods on the same task from the same start and tabulate the results.

    Without `model`, uses the task's small default world (see env.task_world). With `model`
    and `robot`, starts from `data` (e.g. the live world's current state; it is not modified).
    `seed` randomizes the start the same way for every method (None = start exactly from `data`)."""
    from robotforge.control.env import task_world
    from robotforge.control.tasks import make_task

    if model is None:
        model, default_robot = task_world(task)
        robot = robot or default_robot
    if robot is None:
        raise ValueError("say which robot to use (robot='name')")
    if data is None:
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
    probe = make_task(task, model, robot, task_params)
    methods = list(methods or probe.methods)
    seconds = float(seconds or probe.default_seconds)
    results = compare(
        lambda: make_task(task, model, robot, task_params),
        model,
        data,
        methods,
        seconds,
        method_params,
        push_at,
        push_strength,
        seed,
    )
    return Comparison(task=probe.describe(), results=results)
