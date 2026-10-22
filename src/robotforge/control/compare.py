"""Run a task with a controller and measure it; run several controllers and compare them in a table.

The same metrics for every algorithm, so a beginner can see the trade-offs:

- reward: the task's score integrated over the run (higher is better),
- success: is the goal met at the end; failed: did the robot fall / drop the pole,
- effort: mechanical work the actuators did, in joules (sum of |force x speed| dt),
- wall/sim: computer seconds per simulated second (below 1 = faster than real time).
"""

from __future__ import annotations

import copy
import math
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.control.methods import check, make_controller
from robotforge.control.tasks import Task, TaskError
from robotforge.control.tasks.base import divergences
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
        self.unstable_at: float | None = None
        self.steps = 0
        self._divergences = divergences(data)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.task.observe(data)
        cost = self.task.step_cost(data)
        self.cost += cost * self.dt
        self.reward += (self.task.alive_bonus - cost) * self.dt
        acts = self.task.actuators
        power = np.abs(data.actuator_force[acts] * data.actuator_velocity[acts]).sum()
        self.effort += float(power) * self.dt
        t = float(data.time) - self.t0
        if self.unstable_at is None and divergences(data) > self._divergences:
            self.unstable_at = t
            self.failed_at = t if self.failed_at is None else self.failed_at
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
    stopped_early: str | None = None

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
    deadline: float | None = None,
    *,
    _experiment: Any = None,
) -> RunResult:
    """Run `method` on `task` for `seconds` of simulated time from the state in `data` (modified in place).

    `deadline` (a time.monotonic() value) stops the run early if the computer is too slow for it."""
    check_push(push_at, seconds)
    task.begin(data)
    if _experiment is not None:
        _experiment.update(
            task={
                "start_description": _record_values(task.describe())[0],
                "resolved_goal": _record_values(
                    {key: getattr(task, key) for key in ("target", "goal") if hasattr(task, key)}
                )[0],
            }
        )
    controller = make_controller(method, task, model, data, params)
    session = SimSession(model=model, data=data)
    monitor = EpisodeMonitor(task, model, data)
    session.set_controller(task.robot.name, controller)
    session.set_controller(MONITOR, monitor)
    events: list[str] = []
    started = time.perf_counter()
    t0 = float(data.time)

    dt = model.opt.timestep
    chunk = max(1, round(0.1 / dt))
    stopped: str | None = None

    def advance_to(sim_time: float) -> None:
        nonlocal stopped
        while stopped is None:
            n = round((sim_time - (data.time - t0)) / dt)
            if n <= 0:
                return
            if deadline is not None and time.monotonic() > deadline:
                stopped = f"stopped after {data.time - t0:.1f} simulated seconds: out of computer time"
                return
            session.step(min(n, chunk))

    if push_at is not None:
        advance_to(push_at)
        if stopped is None:
            events.append(f"t={push_at:.1f}s: {task.push(data, push_strength)}")
    advance_to(seconds)
    if stopped is not None:
        events.append(stopped)
    wall = time.perf_counter() - started
    if monitor.unstable_at is not None or not np.isfinite(data.qpos).all():
        events.append(f"t={monitor.unstable_at or 0:.2f}s: the simulation became unstable")
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
        stopped_early=stopped,
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
    wall_budget: float | None = None,
    *,
    record_dir: str | Path | None = None,
    _recording: _ComparisonRecording | None = None,
) -> list[RunResult]:
    """Run each method from the same starting state (a copy of `data`) and return the results.

    With `seed`, the task first randomizes that start (e.g. a tilted pole), identically for every method.
    With `wall_budget` (seconds of computer time for everything), methods that no longer fit are skipped
    and a run that overruns stops early."""
    try:
        probe = make_task()
        for method in methods:  # refuse a bad list before spending time on the good ones
            check(probe, method)
        check_push(push_at, seconds)
    except BaseException as exc:
        if _recording:
            status = "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            _recording.finish([], 0.0, status, f"{type(exc).__name__}: {exc}")
        raise
    recording = _recording
    if recording is None and record_dir is not None:
        recording = _ComparisonRecording(
            record_dir,
            probe,
            model,
            data,
            methods,
            seconds,
            params,
            seed,
            wall_budget,
            push_at,
            push_strength,
            replayable=False,
        )
    deadline = time.monotonic() + wall_budget if wall_budget is not None else None
    results = []
    started = time.perf_counter()
    active = None
    try:
        for index, method in enumerate(methods):
            active = recording.child(index, method) if recording else None
            if deadline is not None and time.monotonic() > deadline:
                if active:
                    active.finish("stopped", reason="not run: out of computer time")
                    active = None
                    continue
                break
            task = make_task()
            start = copy.copy(data)
            if seed is not None:
                task.reset(start, np.random.default_rng(seed))
                mujoco.mj_forward(model, start)
            if active:
                active.snapshot_environment(
                    model, start, world=recording.world, scope="comparison_method", robot=task.robot.name
                )
                active.update(task={"start_description": _record_values(task.describe())[0]})
            method_params = (params or {}).get(method)
            execution_params = (
                _policy_parameters(recording.parent.record, recording.parent.folder, method, method_params)
                if recording
                else nullcontext(method_params)
            )
            with execution_params as captured_params:
                result = run(
                    task,
                    model,
                    start,
                    method,
                    seconds,
                    captured_params,
                    push_at,
                    push_strength,
                    deadline,
                    **({"_experiment": active} if active else {}),
                )
            results.append(result)
            if active:
                _final_state(active, model, start)
                values, limitations = _record_values(result.to_dict())
                active.update(
                    metrics={**values, "unavailable_metrics": limitations},
                    budget_used={"sim_seconds": result.sim_seconds, "wall_seconds": result.wall_seconds},
                )
                active.finish("stopped" if result.stopped_early else "completed", reason=result.stopped_early)
                active = None
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        if active:
            active.finish(status, error=f"{type(exc).__name__}: {exc}")
        if recording:
            for index in range(len(recording.children), len(methods)):
                child = recording.child(index, methods[index])
                child.finish("cancelled", reason="not run: an earlier method failed")
            recording.finish(results, time.perf_counter() - started, status, f"{type(exc).__name__}: {exc}")
        raise
    if recording:
        stopped = len(results) < len(methods) or any(r.stopped_early for r in results)
        recording.finish(results, time.perf_counter() - started, "stopped" if stopped else "completed")
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
        if r.stopped_early:
            notes.append("stopped early (too slow)")
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
    widths = [max([len(h), *(len(row[i]) for row in rows)]) for i, h in enumerate(head)]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(head), sep, *(line(row) for row in rows)])


def _headline_metric(results: list[RunResult]) -> str | None:
    """The number that best says how well the task went, e.g. distance_m for reach."""
    if not results:
        return None
    for key in (
        "distance_m",
        "pole_angle_deg",
        "lean_deg",
        "avg_speed_m_s",
        "height_m",
        "force_mean_n",  # wipe
        "settled_n",  # press
        "depth_mm",  # insert
    ):
        if key in results[0].report:
            return key
    return None


def check_push(push_at: float | None, seconds: float) -> None:
    if push_at is not None and not 0 < push_at < seconds:
        raise TaskError(f"push_at must be after the start and before the end (0 < push_at < {seconds:g})")


@dataclass
class Comparison:
    task: dict[str, Any]
    results: list[RunResult]
    skipped: list[str] = field(default_factory=list)
    experiment: dict[str, str] | None = None

    @property
    def table(self) -> str:
        return table(self.results)

    def to_dict(self) -> dict[str, Any]:
        out = {"task": self.task, "table": self.table, "results": [r.to_dict() for r in self.results]}
        if self.experiment is not None:
            out["experiment"] = self.experiment
        if self.skipped:
            out["skipped"] = {m: "not run: out of computer time" for m in self.skipped}
        return out


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
    wall_budget: float | None = None,
    world: Any = None,
    world_dir: Path | None = None,
    record_dir: str | Path | None = None,
) -> Comparison:
    """Run several methods on the same task from the same start and tabulate the results.

    Without `model`, uses the task's small default world (see env.task_world). With `model`
    and `robot`, starts from `data` (e.g. the live world's current state; it is not modified).
    `seed` randomizes the start the same way for every method (None = start exactly from `data`)."""
    from robotforge.control.env import task_world
    from robotforge.control.methods import resolve_method
    from robotforge.control.runtime import bind_control, effective_task_params
    from robotforge.control.task_registry import bind_task, describe_task, resolve_task

    task = resolve_task(task, world).name
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
    probe = bind_task(task, model, robot, task_params, world=world)
    methods = [resolve_method(m) for m in (methods if methods is not None else probe.methods)]
    if not methods:
        raise TaskError(f"{task} has no live control methods; choose a trained policy")
    settings = [effective_task_params(task, m, task_params, world_dir) for m in methods if m.startswith("policy:")]
    if settings:
        # Compare controllers on the stored goal too; multiple policies must agree.
        canonical = []
        for values in settings:
            candidate = bind_task(task, model, robot, values, world=world)
            canonical.append(getattr(candidate, "settings", candidate.params.model_dump()))
        if any(value != canonical[0] for value in canonical[1:]):
            raise TaskError(
                "policies have different saved task settings; give explicit task_params for a fair comparison"
            )
        task_params = settings[0]
    probe = bind_task(task, model, robot, task_params, world=world)
    if not seconds and getattr(probe, "contact", False):
        probe.begin(data)  # adaptive duration depends on the actual starting height
    given = {resolve_method(m): p for m, p in (method_params or {}).items()}
    unknown = set(given) - set(methods)
    if unknown:
        raise TaskError(f"method_params names methods not selected: {', '.join(sorted(unknown))}")
    method_params = {
        m: bind_control(
            model, copy.copy(data), robot, task, m, task_params, given.get(m), world=world, world_dir=world_dir
        ).params
        for m in methods
    }
    seconds = float(seconds or probe.default_seconds)
    recording = None
    if record_dir is not None:
        check_push(push_at, seconds)
        recording = _ComparisonRecording(
            record_dir,
            probe,
            model,
            data,
            methods,
            seconds,
            method_params,
            seed,
            wall_budget,
            push_at,
            push_strength,
            world=world,
        )
    results = compare(
        lambda: bind_task(task, model, robot, task_params, world=world),
        model,
        data,
        methods,
        seconds,
        method_params,
        push_at,
        push_strength,
        seed,
        wall_budget,
        **({"_recording": recording} if recording else {}),
    )
    done = {r.method for r in results}
    return Comparison(
        task=describe_task(probe),
        results=results,
        skipped=[m for m in methods if m not in done],
        experiment=recording.parent.reference() if recording else None,
    )


def _record_values(value: Any, path: str = "") -> tuple[Any, dict[str, str]]:
    """Keep legacy return values intact while writing finite, explicit JSON records."""
    unavailable: dict[str, str] = {}

    def clean(item: Any, key: str) -> Any:
        if isinstance(item, np.ndarray):
            return clean(item.tolist(), key)
        if isinstance(item, np.generic):
            return clean(item.item(), key)
        if isinstance(item, Path):
            return str(item)
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if math.isfinite(item):
                return item
            unavailable[key] = "non-finite value (wall_per_sim is undefined when simulated time is zero)"
            return None
        if isinstance(item, (list, tuple)):
            return [clean(v, f"{key}[{i}]") for i, v in enumerate(item)]
        if isinstance(item, dict) and all(isinstance(k, str) for k in item):
            return {k: clean(v, f"{key}.{k}" if key else k) for k, v in item.items()}
        unavailable[key] = f"{type(item).__name__} cannot be reconstructed from JSON"
        return None

    return clean(value, path), unavailable


def _final_state(handle: Any, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Retain full integration state as a replay check without changing simulation state."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "final-state.npz"
        np.savez(path, state=state, state_spec=int(spec))
        handle.artifact(path, "final_state")


def _resolved_method_params(task: Task, method: str, supplied: dict[str, Any] | None) -> dict[str, Any]:
    from robotforge.control.method_params import parameter_model
    from robotforge.control.methods import _LQR_WEIGHTS

    values = parameter_model(method, task).model_validate(supplied or {}).model_dump(mode="python")
    if method == "lqr" and not hasattr(task, "lqr_controller"):
        values.update({**_LQR_WEIGHTS.get(task.name, {}), **(supplied or {})})
    return values


@contextmanager
def _policy_parameters(
    record: Any,
    record_folder: Path,
    method: str,
    params: dict[str, Any] | None,
) -> Iterator[dict[str, Any] | None]:
    """Execute the archived policy bytes, even if training replaces the original files."""
    if not method.startswith("policy:"):
        yield params
        return
    from robotforge.control.rl.store import check_name
    from robotforge.experiments.schema import ArtifactRef
    from robotforge.experiments.store import verified_artifact

    # Recorded names must be validated before they can select any output path.
    name = check_name(method.removeprefix("policy:"))
    policy = record.algorithm.get("policies", {}).get(method)
    if not policy:
        raise ValueError(f"comparison has no archived policy for {method}")
    with tempfile.TemporaryDirectory(prefix="robotforge-policy-") as directory:
        folder = Path(directory) / name
        folder.mkdir()
        for filename in ("policy.json", "policy.npz"):
            ref = ArtifactRef.model_validate(policy[filename])
            if ref not in record.artifacts:
                raise ValueError(f"comparison policy artifact is missing from the record: {filename}")
            source = verified_artifact(record_folder, ref)
            (folder / filename).write_bytes(source.read_bytes())
        yield {**(params or {}), "path": str(folder)}


class _ComparisonRecording:
    """A single writer for the parent and its method records; no controller is built here."""

    def __init__(
        self,
        root: str | Path,
        task: Task,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        methods: list[str],
        seconds: float,
        params: dict[str, dict[str, Any]] | None,
        seed: int | None,
        wall_budget: float | None,
        push_at: float | None,
        push_strength: float,
        *,
        world: Any = None,
        replayable: bool = True,
    ):
        from robotforge.control.task_registry import describe_task, registered_task_name
        from robotforge.experiments import ExperimentStore

        self.store = ExperimentStore(root)
        self.world = world
        self.children: list[dict[str, str]] = []
        self.methods = methods
        self.params, limitations = _record_values(params or {})
        resolved, other = _record_values({m: _resolved_method_params(task, m, (params or {}).get(m)) for m in methods})
        limitations.update(other)
        reasons = [f"{key}: {reason}" for key, reason in limitations.items()]
        for callback in ("control", "passive", "sensor", "act_dyn", "act_gain", "act_bias", "contactfilter"):
            getter = getattr(mujoco, f"get_mjcb_{callback}", None)
            if getter is not None and getter() is not None:
                reasons.append(f"MuJoCo Python callback {callback} is not captured")
        if not replayable:
            reasons.append("low-level compare received a Python task factory; its closure cannot be reconstructed")
        settings = getattr(task, "settings", task.params.model_dump(mode="python"))
        task_info = {
            "registered_name": registered_task_name(task),
            "implementation_name": task.name,
            "robot": task.robot.name,
            "params": _record_values(settings)[0],
            "description": _record_values(describe_task(task))[0],
        }
        algorithm = {
            "domain": "control",
            "name": "comparison",
            "methods": methods,
            "method_params": resolved,
            "supplied_params": self.params,
            "push": {"at_sim_seconds": push_at, "strength": push_strength},
            "reproduction": {"entrypoint": "robotforge.control.compare.reproduce_comparison", "limitations": reasons},
        }
        self.budget = {
            "requested": {"sim_seconds_per_method": seconds, "wall_seconds": wall_budget},
            "units": {"sim_seconds_per_method": "simulated seconds", "wall_seconds": "wall-clock seconds"},
        }
        self.parent = self.store.begin(
            "comparison",
            task=task_info,
            algorithm=algorithm,
            seeds={"reset": seed},
            budget=self.budget,
        )
        try:
            for plugin in self.parent.record.version.get("plugins", []):
                if plugin.get("source_state") != "current_file_not_loaded_bytecode":
                    reasons.append(f"plugin {plugin.get('name')}: source is not verifiable for reproduction")
            if reasons:
                self.parent.update(algorithm={"reproduction": {**algorithm["reproduction"], "limitations": reasons}})
            self.parent.snapshot_environment(
                model, data, world=world, scope="comparison_initial", robot=task.robot.name
            )
            policies = {}
            for method in methods:
                if method.startswith("policy:"):
                    folder = Path(self.params[method]["path"])
                    policies[method] = {
                        name: self.parent.artifact(folder / name, f"{method}/{name}").model_dump()
                        for name in ("policy.json", "policy.npz")
                    }
            if policies:
                self.parent.update(algorithm={"policies": policies})
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            self.parent.finish(status, error=f"{type(exc).__name__}: {exc}")
            raise

    def child(self, index: int, method: str) -> Any:
        record = self.parent.record
        child = self.store.begin(
            "control",
            parent=self.parent.reference(),
            task=record.task,
            algorithm={
                "domain": "control",
                "name": method,
                "params": record.algorithm["method_params"][method],
                "supplied_params": self.params.get(method, {}),
                "push": record.algorithm["push"],
                "comparison_index": index,
                "reproduction": {"from": "parent initial state", "reset": "once using parent seed"},
            },
            seeds=record.seeds,
            budget={
                "requested": {"sim_seconds": self.budget["requested"]["sim_seconds_per_method"]},
                "shared_wall_seconds": self.budget["requested"]["wall_seconds"],
                "units": {"sim_seconds": "simulated seconds", "shared_wall_seconds": "wall-clock seconds"},
            },
        )
        self.children.append(child.reference())
        return child

    def finish(self, results: list[RunResult], wall: float, status: str, error: str | None = None) -> None:
        values, limitations = _record_values([r.to_dict() for r in results], "results")
        self.parent.update(
            metrics={"results": values, "children": self.children, "unavailable_metrics": limitations},
            budget_used={
                "sim_seconds": sum(r.sim_seconds for r in results),
                "wall_seconds": wall,
                "methods_run": len(results),
                "methods_requested": len(self.methods),
            },
        )
        self.parent.finish(status, reason="out of computer time" if status == "stopped" else None, error=error)


def reproduce_comparison(record_path: str | Path, *, record_dir: str | Path | None = None) -> Comparison:
    """Execute a recorded comparison from its captured model and initial state.

    A method child uses its parent's initial state and resets exactly once, which also
    reconstructs randomized goals kept inside task adapters. Wall-clock deadlines are
    repeated as requested, so machine-dependent cutoffs and timing are not promised equal.
    Plugin code must already be installed and trusted; records never execute archived code.
    """
    from robotforge.control.task_registry import bind_task, describe_task, resolve_task
    from robotforge.design.world import World
    from robotforge.experiments import load_record, provenance, restore_environment
    from robotforge.experiments.schema import check_id
    from robotforge.experiments.store import record_path as locate_record
    from robotforge.experiments.store import verified_artifact

    path = locate_record(record_path)
    record = load_record(path)
    selected = None
    if record.kind == "control" and record.parent and record.parent.get("kind") == "comparison":
        selected = record.algorithm["name"]
        # Parent references are descriptive; require the related record beside this child.
        parent_id = check_id(record.parent["id"])
        parent_path = path.parent.parent / parent_id / "experiment.json"
        record = load_record(parent_path)
        if record.id != parent_id:
            raise ValueError("comparison parent identity does not match the method record")
        path = parent_path
    if record.kind != "comparison":
        raise ValueError("reproduce_comparison needs a comparison experiment or one of its method records")
    reasons = record.algorithm.get("reproduction", {}).get("limitations", [])
    if reasons:
        raise ValueError("comparison cannot be reproduced: " + "; ".join(reasons))
    world = None
    for artifact in record.artifacts:
        if artifact.role == "world_specification":
            world = World.model_validate_json(verified_artifact(path.parent, artifact).read_text(encoding="utf-8"))
    resolve_task(record.task["registered_name"], world)  # Load normal installed plugins before comparing provenance.
    current = provenance()
    for key in ("active_source_sha256", "mujoco", "numpy", "python", "platform", "plugins"):
        if record.version.get(key) != current.get(key):
            raise ValueError(f"comparison code/plugin compatibility mismatch: {key}")
    if any(p.get("source_state") != "current_file_not_loaded_bytecode" for p in current.get("plugins", [])):
        raise ValueError("comparison cannot reproduce in-memory plugins without a verifiable source file")
    for callback in ("control", "passive", "sensor", "act_dyn", "act_gain", "act_bias", "contactfilter"):
        getter = getattr(mujoco, f"get_mjcb_{callback}", None)
        if getter is not None and getter() is not None:
            raise ValueError(f"comparison cannot reproduce with active MuJoCo Python callback {callback}")
    model, data = restore_environment(path)
    methods = [selected] if selected else record.algorithm["methods"]
    params = copy.deepcopy(record.algorithm["supplied_params"])
    task_name, robot = record.task["registered_name"], record.task["robot"]
    task_params = record.task["params"]
    seconds = record.budget["requested"]["sim_seconds_per_method"]
    wall_budget = record.budget["requested"].get("wall_seconds")
    push = record.algorithm["push"]

    def make_task() -> Task:
        return bind_task(task_name, model, robot, task_params, world=world)

    with ExitStack() as bundles:
        for method in methods:
            params[method] = bundles.enter_context(_policy_parameters(record, path.parent, method, params.get(method)))
        probe = make_task()
        recording = None
        if record_dir is not None:
            recording = _ComparisonRecording(
                record_dir,
                probe,
                model,
                data,
                methods,
                seconds,
                params,
                record.seeds.get("reset"),
                wall_budget,
                push["at_sim_seconds"],
                push["strength"],
                world=world,
            )
            recording.parent.update(origins=[{"kind": "reproduction", "experiment_id": record.id}])
        results = compare(
            make_task,
            model,
            data,
            methods,
            seconds,
            params,
            push["at_sim_seconds"],
            push["strength"],
            record.seeds.get("reset"),
            wall_budget,
            _recording=recording,
        )
    done = {r.method for r in results}
    return Comparison(
        task=describe_task(probe),
        results=results,
        skipped=[m for m in methods if m not in done],
        experiment=recording.parent.reference() if recording else None,
    )
