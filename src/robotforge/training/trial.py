"""One training segment: an algorithm, its settings and a step budget, run against an environment.

`TrainContext` is what an algorithm sees (from a plugin too): the environment, its settings, a random generator,
`rollout` / `evaluate` to try a policy, `has_budget()` to know when to stop, `log` to put numbers on the training
view, and `state` to keep arrays between segments (a search continues the trials it keeps). `run_segment` is the
function a worker process runs; it saves the policy, the state and a progress file the search and the viewer read.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotforge.control.rl import store
from robotforge.control.tasks import TaskError
from robotforge.training import algorithms
from robotforge.training.envs import TrainingEnv, build_env
from robotforge.training.policy import RunningStats, Shape

EVAL_SEED = 10_000  # test episodes use these seeds, so every trial is scored on the same starts
MAX_POINTS = 4000  # points kept per series in memory; more are thinned
SHOWN_POINTS = 80  # points per series written for the viewer and the agent
STOP_POLL_SECONDS = 0.5
PROGRESS_FILE = "progress.json"
STATE_FILE = "state.npz"
POLICY_FILE = "policy.npz"


class Stopped(Exception):
    """The run was asked to stop."""


@dataclass(frozen=True)
class Episode:
    ret: float
    length: int
    success: bool


def thin(points: list[list[float]], limit: int) -> list[list[float]]:
    """At most `limit` points, evenly picked, always keeping the last."""
    if len(points) <= limit:
        return points
    step = len(points) / limit
    picked = [points[int(i * step)] for i in range(limit - 1)]
    return [*picked, points[-1]]


class Sink:
    """What a segment reports as it runs: episode returns, a learning curve and named series; written to a file."""

    def __init__(self, folder: Path | None, budget: int, offset: int = 0) -> None:
        self.folder = folder
        self.budget = budget
        self.returns: list[list[float]] = []
        self.curve: list[list[float]] = []
        self.metrics: dict[str, list[list[float]]] = {}
        self.successes = 0
        self.episodes = 0
        self.recent: deque[float] = deque(maxlen=12)
        self.recent_ok: deque[bool] = deque(maxlen=12)
        self.best_return = -np.inf
        self._next_curve = offset
        self._every = max(1, budget // 60)
        self._flushed = 0.0
        self.extra: dict[str, Any] = {}
        self.steps = offset

    def episode(self, steps: int, ep: Episode) -> None:
        self.steps = steps
        self.episodes += 1
        self.successes += int(ep.success)
        self.recent.append(ep.ret)
        self.recent_ok.append(ep.success)
        self.best_return = max(self.best_return, ep.ret)
        self.returns.append([steps, round(ep.ret, 3)])
        if len(self.returns) > MAX_POINTS:
            self.returns = self.returns[::2]
        if steps >= self._next_curve:
            self._next_curve = steps + self._every
            self._series(self.curve, steps, float(np.mean(self.recent)))
        self.flush()

    def point(self, steps: int, **values: float) -> None:
        for name, value in values.items():
            if value is not None and np.isfinite(value):
                self._series(self.metrics.setdefault(name, []), steps, float(value))
        self.flush()

    @staticmethod
    def _series(series: list[list[float]], steps: int, value: float) -> None:
        series.append([steps, round(value, 4)])
        if len(series) > MAX_POINTS:
            del series[::2]

    def document(self, state: str = "running", **more: Any) -> dict[str, Any]:
        recent_ok = float(np.mean(self.recent_ok)) if self.recent_ok else 0.0
        return {
            "state": state,
            "steps": int(self.steps),
            "episodes": self.episodes,
            "mean_return": round(float(np.mean(self.recent)), 3) if self.recent else None,
            "best_return": round(float(self.best_return), 3) if self.episodes else None,
            "success_rate": round(recent_ok, 3),
            "returns": thin(self.returns, SHOWN_POINTS),
            "curve": thin(self.curve, SHOWN_POINTS),
            "metrics": {k: thin(v, SHOWN_POINTS) for k, v in self.metrics.items()},
            "updated": time.time(),
            **self.extra,
            **more,
        }

    def flush(self, force: bool = False, state: str = "running", **more: Any) -> None:
        now = time.monotonic()
        if self.folder is None or (not force and now - self._flushed < 1.0):
            return
        self._flushed = now
        store.write_json(self.folder / PROGRESS_FILE, self.document(state, **more))


class TrainContext:
    """What an algorithm gets. Everything an algorithm needs to try policies and report progress."""

    def __init__(
        self,
        env: TrainingEnv,
        params: dict[str, Any],
        *,
        seed: int,
        budget: int,
        state: dict[str, np.ndarray] | None = None,
        sink: Sink | None = None,
        folder: Path | None = None,
        stop_file: Path | None = None,
        max_seconds: float | None = None,
    ) -> None:
        self.env = env
        self.params = params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.budget = int(budget)
        self.state: dict[str, np.ndarray] = state if state is not None else {}
        self.sink = sink or Sink(None, self.budget)
        self.folder = folder  # this trial's folder, for algorithms that keep their own files
        self.obs_dim = int(env.observation_space.shape[0])
        self.act_dim = int(env.action_space.shape[0])
        self.shape = Shape(self.obs_dim, self.act_dim)
        self.stats = RunningStats.from_arrays(self.state, self.obs_dim)
        self.norm = (
            (self.state["norm_mean"], self.state["norm_var"])
            if "norm_mean" in self.state
            else self.stats.snapshot(center=False)
        )
        self.steps = int(self.state["steps"]) if "steps" in self.state else 0
        self._start = self.steps
        self._stop_file = stop_file
        self._stop_checked = 0.0
        self._stopped = False
        self._deadline = time.monotonic() + max_seconds if max_seconds else None

    # ---- budget and stopping ----

    @property
    def steps_used(self) -> int:
        """Environment steps spent in this segment."""
        return self.steps - self._start

    def stopped(self) -> bool:
        if self._stopped:
            return True
        now = time.monotonic()
        if self._stop_file is not None and now - self._stop_checked >= STOP_POLL_SECONDS:
            self._stop_checked = now
            self._stopped = self._stop_file.exists()
        return self._stopped

    def expired(self) -> bool:
        """True once the run was asked to stop or its time is up."""
        if self.stopped():
            return True
        return self._deadline is not None and time.monotonic() >= self._deadline

    def has_budget(self) -> bool:
        """True while the segment may go on: steps left, no stop request, time left."""
        return self.steps_used < self.budget and not self.expired()

    # ---- trying policies ----

    def new_shape(self, hidden: tuple[int, ...] = ()) -> Shape:
        """The layer sizes of the policy this algorithm trains (kept for building the final policy)."""
        self.shape = Shape(self.obs_dim, self.act_dim, tuple(int(h) for h in hidden))
        return self.shape

    def update_norm(self, center: bool = True) -> None:
        """Use the observations seen so far to normalize the policies built from now on."""
        self.norm = self.stats.snapshot(center=center)

    def policy(self, theta: np.ndarray):
        """The policy with flat parameters `theta` (layer sizes from new_shape), normalizing with the current norm."""
        return self.shape.policy(theta, *self.norm)

    def rollout(self, act, seed: int | None = None) -> Episode:
        """One episode with `act(obs) -> action`; counted in the budget and shown in the episode returns."""
        env = self.env
        obs, _ = env.reset(seed=seed)
        total, n, info = 0.0, 0, {}
        while True:
            self.stats.push(np.asarray(obs, dtype=np.float64))
            obs, reward, terminated, truncated, info = env.step(act(obs))
            total += reward
            n += 1
            if terminated or truncated:
                break
        return self.note_episode(total, n, bool(info.get("is_success")))

    def note_episode(self, total: float, length: int, success: bool) -> Episode:
        """Count an episode that something else ran (a library's own training loop) in the budget and the view."""
        self.steps += length
        ep = Episode(float(total), int(length), bool(success))
        self.sink.episode(self.steps, ep)
        return ep

    def evaluate(self, theta: np.ndarray, episodes: int = 1, seeds: list[int] | None = None) -> float:
        """The mean return of the policy `theta` over `episodes` episodes (same start seeds: pass `seeds`)."""
        policy = self.policy(theta)
        starts = seeds if seeds is not None else [int(s) for s in self.rng.integers(0, 2**31 - 1, episodes)]
        return float(np.mean([self.rollout(policy, seed=s).ret for s in starts]))

    def log(self, **values: float) -> None:
        """Put numbers on the training view (loss, step size, a population's spread ...), at the current step."""
        self.sink.point(self.steps, **values)

    # ---- saving ----

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self.state.items()}
        out.update(self.stats.to_arrays())
        out["norm_mean"], out["norm_var"] = (np.asarray(a, dtype=float) for a in self.norm)
        out["steps"] = np.array(self.steps)
        return out


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    tmp = store.temp_path(path)
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    store.replace_file(tmp, path)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as f:
            return {k: np.array(f[k]) for k in f.files}
    except (OSError, ValueError, KeyError):
        return {}


def evaluate_policy(env: TrainingEnv, policy, episodes: int, seed: int = EVAL_SEED) -> dict[str, Any]:
    """Test a policy on fixed starts, calm (no shoves): mean return, success rate and the task's own numbers."""
    push, env.push = env.push, 0.0
    try:
        returns, successes, reports = [], [], []
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            total, info = 0.0, {}
            while True:
                obs, reward, terminated, truncated, info = env.step(policy(obs))
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            successes.append(bool(info.get("is_success")))
            reports.append(env.task.report(env.data))
    finally:
        env.push = push
    numbers = {
        key: round(float(np.mean([r[key] for r in reports])), 4)
        for key, value in reports[0].items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    return {
        "episodes": episodes,
        "mean_return": round(float(np.mean(returns)), 3),
        "success_rate": round(float(np.mean(successes)), 3),
        "mean_result": numbers,
    }


def final_policy(ctx: TrainContext, result: Any):
    """The policy an algorithm returned: flat parameters (a policy of the shape it set), or a policy object."""
    if isinstance(result, np.ndarray | list | tuple):
        return ctx.policy(np.asarray(result, dtype=np.float64))
    if callable(result) and callable(getattr(result, "save", None)):
        return result
    raise TaskError(
        "an algorithm returns its flat parameter vector (a numpy array) or a policy with __call__(obs) and "
        f"save(path); this one returned {type(result).__name__}"
    )


def run_segment(job: dict[str, Any]) -> dict[str, Any]:
    """Train one trial for one segment (the function worker processes run). Returns the summary; the policy, the
    state to continue from and the progress are in the trial's folder."""
    folder = Path(job["trial_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    _load_plugins(job)
    algo = algorithms.get(job["algo"])
    env = build_env(job["env"])
    state = load_arrays(folder / STATE_FILE) if job.get("resume") else {}
    stop_file = Path(job["stop_file"]) if job.get("stop_file") else None
    budget = int(job["budget"])
    sink = Sink(folder, budget, offset=int(state["steps"]) if "steps" in state else 0)
    sink.extra = {"id": job.get("id"), "algo": job["algo"], "budget": budget + sink.steps}
    ctx = TrainContext(
        env,
        algo.params(job.get("params")),
        seed=int(job.get("seed", 0)),
        budget=budget,
        state=state,
        sink=sink,
        folder=folder,
        stop_file=stop_file,
        max_seconds=job.get("max_seconds"),
    )
    started = time.monotonic()
    algorithms.need(algo)
    try:
        policy = final_policy(ctx, algo.run(ctx))
        policy.save(folder / POLICY_FILE)
        save_arrays(folder / STATE_FILE, ctx.arrays())
        score = evaluate_policy(env, policy, int(job.get("eval_episodes", 5)))
    finally:
        env.close()
    out = {
        "id": job.get("id"),
        "steps": ctx.steps,
        "segment_steps": ctx.steps_used,
        "seconds": round(time.monotonic() - started, 2),
        "score": score,
        "stopped": ctx.stopped(),
    }
    sink.steps = ctx.steps
    sink.flush(force=True, state="stopped" if out["stopped"] else "done", score=score, seconds=out["seconds"])
    (folder / "segment.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def _load_plugins(job: dict[str, Any]) -> None:
    """A worker process starts with nothing loaded: the user's plugins, and the world's when they are trusted."""
    from robotforge.extend import ensure_user_plugins, load_world_plugins

    ensure_user_plugins()
    if job.get("world_dir"):
        load_world_plugins(Path(job["world_dir"]))
