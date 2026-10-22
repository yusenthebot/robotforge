"""Training runs: a folder under policies/, a process that trains in it, and a status file everyone reads.

A run is one or more trials (one algorithm with one set of settings each) on one environment. `start_run` checks
the request and launches `python -m robotforge.training job <folder>`, a process of its own so the viewer and the
agent stay smooth. The job trains the trials, several at a time in worker processes, optionally as a search
(random or successive halving), and keeps `status.json` current: every trial's curve, episode returns, settings
and score, and which trial leads. When it ends, the best trial's policy is `policy.npz` in the folder, exactly
where the policies of control/rl are, so `policy:<name>`, `list_policies` and the viewer's play button find it.

    <folder>/run.json        what was asked
    <folder>/status.json     live state (see `status_document`)
    <folder>/policy.json     metadata of the best policy (like control/rl), policy.npz its network
    <folder>/trials/<id>/    each trial's policy.npz, state.npz (to continue it) and progress.json
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robotforge.control.methods import base_method
from robotforge.control.rl import store
from robotforge.control.rl.jobs import JOBS
from robotforge.control.rl.runtime import MlpPolicy
from robotforge.control.task_registry import registered_task_name
from robotforge.control.tasks import TaskError
from robotforge.control.tasks.base import readable
from robotforge.training import algorithms, search
from robotforge.training.envs import ObservedTask, build_env, describe
from robotforge.training.search import SearchSpec, TrialPlan
from robotforge.training.trial import (
    HELD_OUT_SEED,
    POLICY_FILE,
    PROGRESS_FILE,
    Stopped,
    _load_plugins,
    evaluate_policy,
    thin,
)

if TYPE_CHECKING:
    from robotforge.experiments import ExperimentHandle

RUN_FILE = "run.json"
TRIALS_DIR = "trials"
HEARTBEAT_SECONDS = 5.0
DEFAULT_STEPS = 60_000
MAX_STEPS = 20_000_000
MAX_PARALLEL = 4
SHOWN = 80
HELD_OUT_EPISODES = 50  # the winner of several trials is scored again on this many fresh starts


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str
    env: dict[str, Any]
    algos: list[str] = Field(min_length=1)
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)  # fixed settings, by algorithm
    steps: int = Field(DEFAULT_STEPS, gt=0, le=MAX_STEPS, description="environment steps a trial may use")
    minutes: float | None = Field(None, gt=0, le=240)
    search: SearchSpec | None = None
    seed: int = 0
    parallel: int = Field(1, ge=0, le=MAX_PARALLEL)  # 0: in this process, one at a time (tests)
    eval_episodes: int = Field(5, ge=1, le=50)
    held_out_episodes: int | None = Field(
        None,
        ge=0,
        le=200,
        description="episodes the winner is scored on, on starts no trial was ranked on; None: "
        f"{HELD_OUT_EPISODES} when several trials compete, none for a single trial; 0: never",
    )
    notes: list[str] = Field(default_factory=list)  # things the person starting the run should know, kept with it


def plan_for(spec: RunSpec) -> list[TrialPlan]:
    names = [algorithms.get(name).name for name in spec.algos]
    fixed = {}
    for given, params in spec.params.items():
        algo = algorithms.get(given)
        if algo.name not in names:
            raise TaskError(f"settings for {given!r} have no chosen algorithm; choose one of: {', '.join(names)}")
        if algo.name in fixed:
            raise TaskError(f"settings for {algo.name!r} were supplied more than once through aliases")
        checked = algo.params(params)
        fixed[algo.name] = {key: checked[key] for key in params}
    if spec.search is not None:
        return search.plan_trials(spec.search, names, fixed)
    plans = [search.single_trial(name, fixed.get(name), spec.seed + i)[0] for i, name in enumerate(names)]
    return [TrialPlan(f"t{i + 1:02d}", p.algo, p.params, p.seed) for i, p in enumerate(plans)]


def default_parallel(trials: int) -> int:
    return max(1, min(MAX_PARALLEL, (os.cpu_count() or 2) - 2, trials))


# ---- starting --------------------------------------------------------------------------------------------


def make_run_spec(name: str, env: dict[str, Any], **kwargs: Any) -> RunSpec:
    try:
        return RunSpec(name=name, env=env, **kwargs)
    except ValidationError as exc:
        raise TaskError(readable(exc, "training settings")) from exc


def start_run(spec: RunSpec, folder: Path, replace: bool = False) -> dict[str, Any]:
    """Check the request and launch the training process. Returns at once."""
    folder = Path(folder)
    algos = [algorithms.get(n) for n in spec.algos]
    plans = plan_for(spec)  # checks the settings and the search space: a wrong value is refused here, not later
    for algo in algos:
        algorithms.need(algo)
    meta = policy_meta(spec)  # builds the environment once: fails here, readably, if the task does not fit
    spec = spec.model_copy(update={"notes": [*spec.notes, *speed_notes(spec, meta)]})
    with JOBS._lock:
        busy = JOBS.running([folder.parent, store.policies_home(None)])
        if len(busy) >= 1:
            raise TaskError(
                f"{busy[0].name!r} is still training and uses the computer's cores; wait for it (training_status) "
                "or stop it (stop_training) before starting another"
            )
        if (folder / store.POLICY_FILE).exists() or (folder / RUN_FILE).exists():
            state = JOBS.status(folder).get("state")
            if state not in (*store.FINAL_STATES, "unknown"):
                raise TaskError(f"{folder.name!r} is already training ({state})")
            if not replace:
                raise TaskError(
                    f"a policy named {folder.name!r} already exists ({state}); pick another name or replace it"
                )
        return _launch(spec, folder, plans, meta)


def speed_notes(spec: RunSpec, meta: dict[str, Any]) -> list[str]:
    """A warning when the observation holds a camera: it draws a picture at every step, which is the slow part."""
    slow, rate = meta.get("slow_sensors"), meta.get("steps_per_second")
    if not slow:
        return []
    names = ", ".join(repr(n) for n in slow)
    note = f"the depth camera {names} draws a picture at every step, which makes the simulation many times slower"
    if rate:
        minutes = spec.steps / rate / 60.0
        note += (
            f" (measured {rate:.0f} steps a second here, so {spec.steps:,} steps take about {minutes:.0f} min "
            "for each trial, and trials side by side share the cores)"
        )
    return [note + "; fewer timesteps, a lidar or observation='task' are faster"]


def _measured_rate(env: Any, steps: int = 25) -> float:
    """Environment steps a second, measured with a few idle steps (only worth doing when something is slow)."""
    env.reset(seed=0)
    idle = np.zeros(env.action_space.shape[0], dtype=np.float32)
    started = time.perf_counter()
    done = 0
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(idle)
        done += 1
        if terminated or truncated:
            break
    return done / max(time.perf_counter() - started, 1e-6)


def policy_meta(spec: RunSpec) -> dict[str, Any]:
    """What policy.json says about a run before it ends: the environment, its spaces, how to run a policy."""
    env = build_env(spec.env)
    try:
        facts = describe(env)
        task = env.task
        slow = list(task.sensors.renders) if isinstance(task, ObservedTask) and task.sensors is not None else []
        return {
            **({"slow_sensors": slow, "steps_per_second": round(_measured_rate(env), 1)} if slow else {}),
            "name": spec.name,
            "robot": spec.env["robot"],
            "task": registered_task_name(task),
            **({"implementation_task": task.name} if registered_task_name(task) != task.name else {}),
            "action_mode": spec.env["action_mode"],
            "residual_scale": spec.env["residual_scale"],
            "env": {**spec.env, "task": registered_task_name(task)},
            "observation": task.observation_labels(),
            "observation_layout": facts["observation"]["sources"],
            "left_out": facts["observation"].get("left_out", []),
            "action": facts["action"],
            "actuators": facts["action"]["actuators"],
            "control_dt": env.control_dt,
            "physics_timestep": float(env.model.opt.timestep),
            "episode_seconds": env.episode_seconds,
            "base_method": base_method(task) if env.action_mode == "residual" else None,
        }
    finally:
        env.close()


def _begin_experiment(spec: RunSpec, folder: Path) -> ExperimentHandle:
    handle = store.experiment_store(folder).begin(
        "training_run",
        environment={"env_spec": spec.env, "scope": spec.env.get("scope"), "robot": spec.env["robot"]},
        task={
            "registered_name": spec.env["task"],
            "implementation_name": None,
            "params": spec.env.get("task_params", {}),
        },
        algorithm={
            "domain": "training",
            "name": spec.algos,
            "params": spec.params,
            "run_spec": spec.model_dump(mode="json"),
            "settings_source": "RunSpec",
        },
        seeds={
            "run": spec.seed,
            "search": spec.search.seed if spec.search else None,
            "ranking": {"base": 10_000, "episodes": spec.eval_episodes, "protocol": "base + episode"},
            "held_out": {
                "base": HELD_OUT_SEED,
                "episodes": spec.held_out_episodes
                if spec.held_out_episodes is not None
                else (HELD_OUT_EPISODES if len(plan_for(spec)) > 1 else 0),
                "protocol": "base + episode",
            },
        },
        budget={
            "steps_per_trial": spec.steps,
            "requested_wall_seconds": spec.minutes * 60 if spec.minutes else None,
            "semantics": (
                "whole algorithm iterations may overshoot; training steps exclude ranking and held-out "
                "evaluation; wall includes queue and evaluation"
            ),
        },
    )
    handle.update(metrics={"phase": "queued"})
    return handle


def _launch(spec: RunSpec, folder: Path, plans: list[TrialPlan], meta: dict[str, Any]) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    for old in (store.STOP_FILE, store.HEARTBEAT_FILE, POLICY_FILE):
        (folder / old).unlink(missing_ok=True)
    shutil.rmtree(folder / TRIALS_DIR, ignore_errors=True)
    store.write_json(folder / RUN_FILE, spec.model_dump(mode="json"))
    experiment = _begin_experiment(spec, folder)
    algo = plans[0].algo if len({p.algo for p in plans}) == 1 else "+".join(spec.algos)
    store.write_json(
        folder / store.POLICY_FILE,
        {
            **meta,
            "algo": algo,
            "trials_planned": len(plans),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": experiment.reference(),
        },
    )
    starting = {"state": "starting", "name": spec.name, "timesteps": 0, "progress": 0.0, "updated": time.time()}
    starting["experiment"] = experiment.reference()
    store.write_json(folder / store.STATUS_FILE, starting)
    try:
        with (folder / store.LOG_FILE).open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "robotforge.training", "job", str(folder)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                **_detached(),
            )
    except OSError as exc:
        JOBS._mark_failed(folder, starting, f"could not start the training process: {exc}")
        raise TaskError(f"could not start the training process: {exc}") from exc
    JOBS.procs[folder] = proc
    observation = {
        "size": len(meta["observation"]),
        "sources": meta["observation_layout"],
        "left_out": meta["left_out"],
    }
    return {"folder": str(folder), "pid": proc.pid, "trials": [vars(p) for p in plans], "observation": observation,
            "action": meta["action"], "notes": spec.notes, "experiment": experiment.reference()}  # fmt: skip


def _detached() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        }
    return {"start_new_session": True}


# ---- the training process --------------------------------------------------------------------------------


class _Trial:
    def __init__(self, plan: TrialPlan) -> None:
        self.plan = plan
        self.state = "pending"
        self.rung = -1
        self.steps = 0
        self.score: dict[str, Any] | None = None
        self.error: str | None = None
        self.timed_out = False
        self.seconds = 0.0
        self.experiments: list[dict[str, Any]] = []
        self.live: dict[str, Any] = {}
        self.history: dict[str, Any] = {"curve": [], "returns": [], "metrics": {}}

    def absorb(self, progress: dict[str, Any]) -> None:
        """A finished segment's series join the history (the next segment starts its file afresh)."""
        self.history["curve"] += progress.get("curve", [])
        self.history["returns"] += progress.get("returns", [])
        for name, series in progress.get("metrics", {}).items():
            self.history["metrics"].setdefault(name, []).extend(series)
        self.live = {}

    def view(self) -> dict[str, Any]:
        live = self.live
        curve = thin(self.history["curve"] + live.get("curve", []), SHOWN)
        returns = thin(self.history["returns"] + live.get("returns", []), SHOWN)
        names = {*self.history["metrics"], *live.get("metrics", {})}
        metrics = {
            n: thin(self.history["metrics"].get(n, []) + live.get("metrics", {}).get(n, []), SHOWN) for n in names
        }
        out: dict[str, Any] = {
            "id": self.plan.id,
            "experiments": self.experiments,
            "algo": self.plan.algo,
            "settings": self.plan.params,
            "seed": self.plan.seed,
            "state": self.state,
            "rung": max(self.rung, 0),
            "steps": max(self.steps, int(live.get("steps", 0))),
            "mean_return": live.get("mean_return"),
            "best_return": live.get("best_return"),
            "success_rate": live.get("success_rate"),
            "curve": curve,
            "returns": returns,
            "metrics": metrics,
        }
        if self.score is not None:
            out["score"] = self.score
        if self.error:
            out["error"] = self.error
        if self.timed_out:
            out["timed_out"] = True
        return out

    def standing(self) -> float:
        """What ranks it: its last test return (fixed starts, so trials compare), never a training return."""
        return float(self.score["mean_return"]) if self.score is not None else float("-inf")


class Job:
    """Runs a RunSpec in `folder`: segments in workers, status.json kept current, the best policy kept."""

    def __init__(self, spec: RunSpec, folder: Path) -> None:
        self.spec, self.folder = spec, folder
        self.trials = [_Trial(p) for p in plan_for(spec)]
        method = spec.search.method if spec.search else "random"
        eta = spec.search.eta if spec.search else 3
        self.rungs = search.rungs(method, len(self.trials), eta)
        self.eta = eta
        self.started = time.monotonic()
        self.deadline = self.started + 60.0 * spec.minutes if spec.minutes else None
        self.stop_file = folder / store.STOP_FILE
        self.last_write = 0.0
        self.stop_seen = False
        self.timed_out = False
        self.planned = self._planned_steps()
        self.rung_now = 0
        self.facts = store.read_json(folder / store.POLICY_FILE)
        self.experiment = store.training_experiment(folder)
        if self.experiment is None or self.experiment.record.status != "running":
            folder.mkdir(parents=True, exist_ok=True)
            self.experiment = _begin_experiment(spec, folder)
            self.facts["experiment"] = self.experiment.reference()
            store.write_json(folder / store.POLICY_FILE, self.facts)
        self.experiment.update(metrics={"phase": "running"}, budget={"planned_training_steps": self.planned})

    def _planned_steps(self) -> int:
        total, before = 0, 0.0
        for share, alive in self.rungs:
            total += int(alive * (share - before) * self.spec.steps)
            before = share
        return max(total, 1)

    # ---- status ----

    def leader(self) -> _Trial | None:
        """The trial with the best test return among those that went furthest; none before the first test."""
        scored = [t for t in self.trials if t.score is not None and t.state != "failed"]
        if not scored:
            return None
        furthest = max(t.rung for t in scored)
        return max(
            (t for t in scored if t.rung == furthest),
            key=lambda t: (t.standing(), (t.score or {}).get("success_rate", 0), t.plan.id),
        )

    def _front(self) -> _Trial | None:
        """Before any test there is no leader; the trial furthest along stands in for the plain numbers."""
        live = [t for t in self.trials if t.state != "failed" and t.live.get("mean_return") is not None]
        return max(live, key=lambda t: float(t.live["mean_return"])) if live else None

    def document(self, state: str = "running", **more: Any) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        steps = sum(max(t.steps, int(t.live.get("steps", 0))) for t in self.trials)
        progress = min(1.0, steps / self.planned)
        lead = self.leader()
        doc: dict[str, Any] = {
            "state": state,
            "name": self.spec.name,
            "experiment": self.experiment.reference(),
            "task": self.spec.env["task"],
            "robot": self.spec.env["robot"],
            "algo": self.facts.get("algo"),
            "timesteps": int(steps),
            "timesteps_planned": self.planned,
            "progress": round(progress, 4),
            "elapsed_s": round(elapsed, 1),
            "pid": os.getpid(),
            "updated": time.time(),
            "search": {
                "method": self.spec.search.method if self.spec.search else None,
                "trials": len(self.trials),
                "rungs": [{"budget_share": round(s, 4), "trials": n} for s, n in self.rungs],
                "rung": self.rung_now,
                "steps_per_trial": self.spec.steps,
            },
            "trials": [t.view() for t in self.trials],
            "leader": lead.plan.id if lead else None,
        }
        if state == "running" and progress > 0.02:
            doc["eta_s"] = round(elapsed * (1.0 - progress) / progress, 1)
        notes = list(self.spec.notes)
        if self.timed_out:
            doc["timed_out"] = True
            notes.append("time budget reached; queued trials were cancelled and active trials stopped cooperatively")
        if notes:
            doc["notes"] = notes
        shown = lead or self._front()
        if shown is not None:
            view = shown.view()
            doc["curve"] = view["curve"]  # the same keys control/rl's training_status reads
            if view["mean_return"] is not None:
                doc["mean_episode_reward"] = view["mean_return"]
            if view["success_rate"] is not None:
                doc["success_rate"] = view["success_rate"]
        doc.update(more)
        return doc

    def write(self, state: str = "running", force: bool = False, **more: Any) -> None:
        now = time.monotonic()
        if not force and now - self.last_write < 0.8:
            return
        self.last_write = now
        document = self.document(state, **more)
        store.write_json(self.folder / store.STATUS_FILE, document)
        if state in store.FINAL_STATES:
            store.finish_experiment(self.folder, document)

    # ---- running ----

    def trial_dir(self, trial: _Trial) -> Path:
        return self.folder / TRIALS_DIR / trial.plan.id

    def job_for(self, trial: _Trial, budget: int, resume: bool) -> dict[str, Any]:
        algo = algorithms.get(trial.plan.algo)
        child = store.experiment_store(self.folder).begin(
            "training_segment",
            parent=self.experiment.reference(),
            origins=[trial.experiments[-1]] if resume and trial.experiments else [],
            environment={"env_spec": self.spec.env},
            task={
                "registered_name": self.spec.env["task"],
                "implementation_name": None,
                "params": self.spec.env.get("task_params", {}),
            },
            algorithm={
                "name": algo.name,
                "domain": "training",
                "params": trial.plan.params,
                "source": algo.source,
                "settings_source": "search" if self.spec.search else "RunSpec.params and defaults",
                "trial": trial.plan.id,
                "rung": trial.rung,
                "resume": resume,
            },
            seeds={
                "training": trial.plan.seed,
                "ranking": {"base": 10_000, "episodes": self.spec.eval_episodes, "protocol": "base + episode"},
            },
            budget={
                "training_steps": budget,
                "requested_run_wall_seconds": self.spec.minutes * 60 if self.spec.minutes else None,
                "semantics": "incremental whole-iteration training budget; evaluation steps excluded",
            },
        )
        child.update(metrics={"phase": "queued"})
        trial.experiments.append(child.reference())
        return {
            "experiment": child.reference(),
            "parent_experiment": self.experiment.reference(),
            "id": trial.plan.id,
            "algo": trial.plan.algo,
            "params": trial.plan.params,
            "seed": trial.plan.seed,
            "env": self.spec.env,
            "world_dir": self.spec.env.get("world_dir"),
            "budget": budget,
            "resume": resume,
            "trial_dir": str(self.trial_dir(trial)),
            "stop_file": str(self.stop_file),
            "eval_episodes": self.spec.eval_episodes,
            "deadline": self.deadline,
        }

    def should_stop(self) -> bool:
        """Notice the shared time budget while waiting, between rungs and during final testing."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.timed_out = True
            self.stop_file.touch()
        if self.stop_file.exists():
            self.stop_seen = True
        return self.stop_seen

    def poll(self, active: list[_Trial]) -> None:
        for t in active:
            progress = store.read_json(self.trial_dir(t) / PROGRESS_FILE)
            if progress and progress.get("state") == "running":
                t.live = progress
        self.should_stop()
        self.write()

    def run(self) -> str:
        parallel = self.spec.parallel or 1
        pool: Any
        if self.spec.parallel == 0:
            pool = ThreadPoolExecutor(max_workers=1)
        else:
            import multiprocessing as mp

            pool = ProcessPoolExecutor(max_workers=min(parallel, len(self.trials)), mp_context=mp.get_context("spawn"))
        alive = list(self.trials)
        before = 0.0
        try:
            for r, (share, _count) in enumerate(self.rungs):
                self.rung_now = r
                increment = int(round((share - before) * self.spec.steps))
                before = share
                futures: dict[Future, _Trial] = {}
                for t in alive:
                    t.rung, t.state = r, "queued"
                    payload = self.job_for(t, max(increment, 1), resume=r > 0)
                    try:
                        future = pool.submit(_segment, payload)
                    except Exception as exc:
                        child = store.open_experiment(payload["experiment"])
                        child.finish("failed", reason="segment submission failed", error=f"{type(exc).__name__}: {exc}")
                        raise
                    futures[future] = t
                self._wait(futures)
                self._collect(alive)
                if self.should_stop() or r == len(self.rungs) - 1:
                    break
                keep = self.rungs[r + 1][1]
                ranked = search.survivors(
                    {
                        t.plan.id: (t.standing(), float((t.score or {}).get("success_rate", 0)))
                        for t in alive
                        if t.state == "done"
                    },
                    keep,
                )
                for t in alive:
                    if t.state == "done" and t.plan.id not in ranked:
                        t.state = "pruned"
                alive = [t for t in alive if t.plan.id in ranked]
                if not alive:
                    break
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            # Only after every worker has exited may the coordinator repair orphaned child records.
            for t in self.trials:
                if not t.experiments:
                    continue
                child = store.open_experiment(t.experiments[-1])
                if child.record.status == "running":
                    queued = child.record.metrics.get("phase") == "queued"
                    child.finish("cancelled" if queued else "failed", reason="coordinator closed the worker pool")
        return "stopped" if self.stop_seen else "done"

    def _wait(self, futures: dict[Future, _Trial]) -> None:
        pending = set(futures)
        while pending:
            if self.should_stop() and self.timed_out:
                for f in list(pending):
                    if f.cancel():
                        t = futures[f]
                        t.state, t.timed_out = "stopped", self.timed_out
                        if t.experiments:
                            child = store.open_experiment(t.experiments[-1])
                            child.update(
                                metrics={"phase": "cancelled", "started": False},
                                budget_used={"training_steps": 0, "wall_seconds": 0},
                            )
                            child.finish(
                                "cancelled",
                                reason="timeout before worker started"
                                if self.timed_out
                                else "stop requested before worker started",
                            )
                        pending.remove(f)
                if not pending:
                    break
            done, pending = wait(pending, timeout=0.8, return_when=FIRST_COMPLETED)
            for f in done:
                t = futures[f]
                try:
                    out = f.result()
                except Exception as exc:  # one trial failing must not end the others
                    t.state = "failed"
                    t.error = f"{type(exc).__name__}: {exc}"
                    # The future is finished: the coordinator can safely repair a worker that died.
                    if t.experiments:
                        child = store.open_experiment(t.experiments[-1])
                        if child.record.status == "running":
                            child.finish("failed", reason="worker failed", error=t.error)
                        t.steps = max(t.steps, int(child.record.budget_used.get("cumulative_training_steps", 0)))
                    print(f"trial {t.plan.id} failed:\n{''.join(traceback.format_exception(exc)[-6:])}", flush=True)
                else:
                    t.timed_out = out.get("timed_out", False)
                    t.state = "stopped" if out["stopped"] or t.timed_out else "done"
                    t.steps, t.score, t.seconds = out["steps"], out["score"], out["seconds"]
            running = [t for f, t in futures.items() if f in pending]
            for t in running:
                if t.state == "queued" and (self.trial_dir(t) / PROGRESS_FILE).exists():
                    t.state = "running"
            self.poll(running)

    def _collect(self, alive: list[_Trial]) -> None:
        for t in alive:
            progress = store.read_json(self.trial_dir(t) / PROGRESS_FILE)
            if progress.get("state") in ("done", "stopped"):
                t.absorb(progress)

    # ---- the outcome ----

    def finish(self, state: str) -> dict[str, Any]:
        ranked = [t for t in self.trials if t.score is not None and (self.trial_dir(t) / POLICY_FILE).exists()]
        if not ranked:
            if state == "stopped":
                # Keep a policy interrupted before its first complete test, without inventing a ranking score.
                ranked = [t for t in self.trials if (self.trial_dir(t) / POLICY_FILE).exists()]
                if not ranked:
                    return {"note": "stopped before a policy could be saved"}
            else:
                errors = "; ".join(f"{t.plan.id}: {t.error}" for t in self.trials if t.error) or "no trial finished"
                raise TaskError(f"training produced no policy ({errors})")
        last = max(t.rung for t in ranked)
        ranked = [t for t in ranked if t.rung == last]  # the trials that went furthest
        best = max(ranked, key=lambda t: (t.standing(), (t.score or {}).get("success_rate", 0), t.plan.id))
        source = self.trial_dir(best)
        shutil.copyfile(source / POLICY_FILE, self.folder / POLICY_FILE)
        meta = store.read_json(self.folder / store.POLICY_FILE)
        final = {
            "trial": best.plan.id,
            "algo": best.plan.algo,
            "settings": best.plan.params,
            "steps": best.steps,
            **(best.score or {}),
        }
        if best.score is None:
            final["score_note"] = "policy saved before a complete ranking evaluation; its quality is unmeasured"
        held_out = self._held_out(state)
        if held_out is not None:
            final["held_out"] = held_out
            final["ranking_score_note"] = (
                "mean_return, success_rate and mean_result are the winner's scores on the starts the trials were "
                "ranked on, which flatters a winner picked out of several; held_out is the fair result"
            )
        store.write_json(
            self.folder / store.POLICY_FILE,
            {
                **meta,
                "algo": best.plan.algo,
                "final": final,
                "timesteps": sum(t.steps for t in self.trials),
                "train_seconds": round(time.monotonic() - self.started, 1),
            },
        )
        self.experiment.artifact(self.folder / POLICY_FILE, "winning_policy")
        origin = store.policy_experiment_for(self.folder, self.folder / POLICY_FILE, best.experiments)
        self.experiment.update(
            origins=[origin] if origin else [],
            metrics={"winner": best.plan.id, "policy_origin_status": "verified" if origin else "unknown"},
        )
        if origin:
            self.experiment.update(task=store.open_experiment(origin).record.task)
        meta = store.read_json(self.folder / store.POLICY_FILE)
        meta.pop("policy_experiment", None)
        meta.pop("policy_origin_note", None)
        if origin:
            meta["policy_experiment"] = origin
        else:
            meta["policy_origin_note"] = "unknown: no verified segment policy artifact matches the saved weights"
        store.write_json(self.folder / store.POLICY_FILE, {**meta, "experiment": self.experiment.reference()})
        self.leader_id = best.plan.id
        return final

    def _held_out(self, state: str) -> dict[str, Any] | None:
        """The winner scored again on starts no trial was ranked on: picking the best of several trials by one
        set of test starts makes its score on those starts optimistic, so this is the number to quote."""
        asked = self.spec.held_out_episodes
        episodes = (HELD_OUT_EPISODES if len(self.trials) > 1 else 0) if asked is None else asked
        if episodes <= 0 or state == "stopped" or self.should_stop():
            return None
        self.write("testing", force=True)
        env = build_env(self.spec.env)
        try:
            policy = MlpPolicy.load(self.folder / POLICY_FILE)
            return {
                **evaluate_policy(env, policy, episodes, seed=HELD_OUT_SEED, should_stop=self.should_stop),
                "seed": HELD_OUT_SEED,
            }
        except Stopped:
            return None
        finally:
            env.close()

    def failed_all(self) -> bool:
        return all(t.state == "failed" for t in self.trials)


def _segment(job: dict[str, Any]) -> dict[str, Any]:
    from robotforge.training.trial import run_segment

    return run_segment(job)


def run_job(folder: Path) -> int:
    """The training process: run the spec in `folder`, record the outcome. Returns the exit code."""
    folder = Path(folder)
    if hasattr(os, "nice"):
        os.nice(5)  # training must never make the viewer or the agent stutter
    job = None
    try:
        spec = RunSpec.model_validate(store.read_json(folder / RUN_FILE))
        threading.Thread(target=_heartbeat, args=(folder,), daemon=True).start()
        _load_plugins({"world_dir": spec.env.get("world_dir")})
        job = Job(spec, folder)
        job.write("running", force=True)
        state = job.run()
        final = job.finish(state)
        if job.should_stop():
            state = "stopped"
    except Exception as exc:
        tail = traceback.format_exc().strip().splitlines()[-25:]
        store.write_json(
            folder / store.STATUS_FILE,
            {
                **(job.document("failed") if job is not None else {"state": "failed", "updated": time.time()}),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": tail,
            },
        )
        store.finish_experiment(folder, store.read_json(folder / store.STATUS_FILE))
        raise
    job.write(
        state,
        force=True,
        final=final,
        leader=getattr(job, "leader_id", None),
        progress=1.0 if state == "done" else None,
    )
    (folder / store.STOP_FILE).unlink(missing_ok=True)
    print(f"{state}: {final}", flush=True)
    return 0


def _heartbeat(folder: Path) -> None:
    beat = folder / store.HEARTBEAT_FILE
    while True:
        with contextlib.suppress(OSError):
            beat.touch()
        time.sleep(HEARTBEAT_SECONDS)


def run_inline(spec: RunSpec, folder: Path) -> dict[str, Any]:
    """Run a spec in this process, one trial at a time (parallel=0), and return the final document. For tests and
    scripts that want the result without a process of their own."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plan_for(spec)
    for name in spec.algos:
        algorithms.need(algorithms.get(name))
    meta = policy_meta(spec)
    if (folder / RUN_FILE).exists():
        (folder / store.STOP_FILE).unlink(missing_ok=True)
    (folder / POLICY_FILE).unlink(missing_ok=True)
    shutil.rmtree(folder / TRIALS_DIR, ignore_errors=True)
    experiment = _begin_experiment(spec, folder)
    store.write_json(folder / RUN_FILE, spec.model_dump(mode="json"))
    store.write_json(folder / store.POLICY_FILE, {**meta, "experiment": experiment.reference()})
    job = None
    try:
        job = Job(spec, folder)
        state = job.run()
        final = job.finish(state)
        if job.should_stop():
            state = "stopped"
        job.write(state, force=True, final=final, leader=getattr(job, "leader_id", None))
    except Exception as exc:
        document = job.document("failed") if job else {"state": "failed"}
        document["error"] = f"{type(exc).__name__}: {exc}"
        store.write_json(folder / store.STATUS_FILE, document)
        store.finish_experiment(folder, document)
        raise
    return store.read_json(folder / store.STATUS_FILE)


__all__ = ["Job", "RunSpec", "make_run_spec", "plan_for", "run_inline", "run_job", "start_run"]
