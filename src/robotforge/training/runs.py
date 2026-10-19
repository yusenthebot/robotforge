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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robotforge.control.rl import store
from robotforge.control.rl.jobs import JOBS
from robotforge.control.tasks import TaskError
from robotforge.control.tasks.base import readable
from robotforge.training import algorithms, search
from robotforge.training.envs import build_env, describe
from robotforge.training.search import SearchSpec, TrialPlan
from robotforge.training.trial import POLICY_FILE, PROGRESS_FILE, thin

RUN_FILE = "run.json"
TRIALS_DIR = "trials"
HEARTBEAT_SECONDS = 5.0
DEFAULT_STEPS = 60_000
MAX_STEPS = 20_000_000
MAX_PARALLEL = 4
SHOWN = 80


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


def plan_for(spec: RunSpec) -> list[TrialPlan]:
    if spec.search is not None:
        return search.plan_trials(spec.search, spec.algos, spec.params)
    plans = []
    for i, name in enumerate(spec.algos):
        plans.append(search.single_trial(name, spec.params.get(name), spec.seed + i)[0])
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
    for algo in algos:
        algorithms.need(algo)
    plans = plan_for(spec)  # also checks the settings and the search space
    meta = policy_meta(spec)  # builds the environment once: fails here, readably, if the task does not fit
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


def policy_meta(spec: RunSpec) -> dict[str, Any]:
    """What policy.json says about a run before it ends: the environment, its spaces, how to run a policy."""
    env = build_env(spec.env)
    try:
        facts = describe(env)
        task = env.task
        return {
            "name": spec.name,
            "robot": spec.env["robot"],
            "task": getattr(task, "name", spec.env["task"]),
            "action_mode": spec.env["action_mode"],
            "residual_scale": spec.env["residual_scale"],
            "env": spec.env,
            "observation": task.observation_labels(),
            "observation_layout": facts["observation"]["sources"],
            "left_out": facts["observation"].get("left_out", []),
            "action": facts["action"],
            "actuators": facts["action"]["actuators"],
            "control_dt": env.control_dt,
            "physics_timestep": float(env.model.opt.timestep),
            "episode_seconds": env.episode_seconds,
            "base_method": task.methods[0] if env.action_mode == "residual" else None,
        }
    finally:
        env.close()


def _launch(spec: RunSpec, folder: Path, plans: list[TrialPlan], meta: dict[str, Any]) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    for old in (store.STOP_FILE, store.HEARTBEAT_FILE, POLICY_FILE):
        (folder / old).unlink(missing_ok=True)
    shutil.rmtree(folder / TRIALS_DIR, ignore_errors=True)
    store.write_json(folder / RUN_FILE, spec.model_dump(mode="json"))
    algo = plans[0].algo if len({p.algo for p in plans}) == 1 else "+".join(spec.algos)
    store.write_json(
        folder / store.POLICY_FILE,
        {**meta, "algo": algo, "trials_planned": len(plans), "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    starting = {"state": "starting", "name": spec.name, "timesteps": 0, "progress": 0.0, "updated": time.time()}
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
            "action": meta["action"]}  # fmt: skip


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
        self.seconds = 0.0
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
        return out

    def standing(self) -> float:
        """What ranks it now: its last test score, else its running training return."""
        if self.score is not None:
            return float(self.score["mean_return"])
        value = self.live.get("mean_return")
        return float(value) if value is not None else float("-inf")


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
        self.planned = self._planned_steps()
        self.rung_now = 0
        self.facts = store.read_json(folder / store.POLICY_FILE)

    def _planned_steps(self) -> int:
        total, before = 0, 0.0
        for share, alive in self.rungs:
            total += int(alive * (share - before) * self.spec.steps)
            before = share
        return max(total, 1)

    # ---- status ----

    def leader(self) -> _Trial | None:
        ranked = [t for t in self.trials if t.state != "failed"]
        return max(ranked, key=_Trial.standing) if ranked else None

    def document(self, state: str = "running", **more: Any) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        steps = sum(max(t.steps, int(t.live.get("steps", 0))) for t in self.trials)
        progress = min(1.0, steps / self.planned)
        lead = self.leader()
        doc: dict[str, Any] = {
            "state": state,
            "name": self.spec.name,
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
        if lead is not None:
            view = lead.view()
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
        store.write_json(self.folder / store.STATUS_FILE, self.document(state, **more))

    # ---- running ----

    def trial_dir(self, trial: _Trial) -> Path:
        return self.folder / TRIALS_DIR / trial.plan.id

    def job_for(self, trial: _Trial, budget: int, resume: bool) -> dict[str, Any]:
        remaining = None if self.deadline is None else max(0.5, self.deadline - time.monotonic())
        return {
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
            "max_seconds": remaining,
        }

    def poll(self, active: list[_Trial]) -> None:
        for t in active:
            progress = store.read_json(self.trial_dir(t) / PROGRESS_FILE)
            if progress and progress.get("state") == "running":
                t.live = progress
        if self.stop_file.exists():
            self.stop_seen = True
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
                    futures[pool.submit(_segment, self.job_for(t, max(increment, 1), resume=r > 0))] = t
                self._wait(futures)
                self._collect(alive)
                if self.stop_seen or (self.deadline and time.monotonic() > self.deadline) or r == len(self.rungs) - 1:
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
        return "stopped" if self.stop_seen else "done"

    def _wait(self, futures: dict[Future, _Trial]) -> None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.8, return_when=FIRST_COMPLETED)
            for f in done:
                t = futures[f]
                try:
                    out = f.result()
                except Exception as exc:  # one trial failing must not end the others
                    t.state = "failed"
                    t.error = f"{type(exc).__name__}: {exc}"
                    print(f"trial {t.plan.id} failed:\n{''.join(traceback.format_exception(exc)[-6:])}", flush=True)
                else:
                    t.state = "stopped" if out["stopped"] else "done"
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
        self.leader_id = best.plan.id
        return final

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
    spec = RunSpec.model_validate(store.read_json(folder / RUN_FILE))
    threading.Thread(target=_heartbeat, args=(folder,), daemon=True).start()
    job = Job(spec, folder)
    try:
        job.write("running", force=True)
        state = job.run()
        final = job.finish(state)
    except Exception as exc:
        tail = traceback.format_exc().strip().splitlines()[-25:]
        store.write_json(
            folder / store.STATUS_FILE,
            {**job.document("failed"), "error": f"{type(exc).__name__}: {exc}", "traceback_tail": tail},
        )
        raise
    job.write(state, force=True, final=final, leader=job.leader_id, progress=1.0 if state == "done" else None)
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
    store.write_json(folder / RUN_FILE, spec.model_dump(mode="json"))
    store.write_json(folder / store.POLICY_FILE, policy_meta(spec))
    job = Job(spec, folder)
    state = job.run()
    final = job.finish(state)
    job.write(state, force=True, final=final, leader=job.leader_id)
    return store.read_json(folder / store.STATUS_FILE)


__all__ = ["Job", "RunSpec", "make_run_spec", "plan_for", "run_inline", "run_job", "start_run"]
