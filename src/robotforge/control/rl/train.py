"""Train a policy with Stable-Baselines3. This runs inside the training process and imports PyTorch.

    python -m robotforge.control.rl job <policy folder>     # what train_policy starts

The trainer reads the spec (train.json), trains with several simulation processes in parallel,
normalizes observations (VecNormalize), checks the policy every so often on fixed test episodes
and keeps the best one, and writes its progress to status.json for the MCP server to read.
"""

from __future__ import annotations

import contextlib
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as BaseVecEnv
from torch import nn

from robotforge.control.rl import store, worker
from robotforge.control.rl.jobs import log_tail
from robotforge.control.rl.runtime import MlpPolicy
from robotforge.control.rl.spec import DEFAULTS, TrainSpec, algo_settings

STATUS_EVERY = 1.0  # seconds between status.json updates
EVALS = 8  # times the policy is tested during training
EVAL_EPISODES = 4
FINAL_EPISODES = 10
CURVE_POINTS = 200
TAIL_LINES = 25
HEARTBEAT_SECONDS = 5.0


class LeanSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv whose workers do not import PyTorch (see worker.py)."""

    def __init__(self, env_fns: list[Callable[[], Any]], start_method: str | None = None):
        import multiprocessing as mp

        self.waiting = False
        self.closed = False
        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns, strict=True):
            process = ctx.Process(target=worker.worker, args=(work_remote, remote, env_fn), daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        BaseVecEnv.__init__(self, len(env_fns), observation_space, action_space)


def export_policy(model: PPO | SAC, norm: VecNormalize) -> MlpPolicy:
    """The deterministic action network and the observation normalizer, as numpy arrays."""
    if isinstance(model, PPO):
        layers = [*model.policy.mlp_extractor.policy_net, model.policy.action_net]
        squash = "clip"
    else:
        layers = [*model.actor.latent_pi, model.actor.mu]
        squash = "tanh"
    weights, biases, kinds = [], [], set()
    for layer in layers:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy().astype(np.float64))
            biases.append(layer.bias.detach().cpu().numpy().astype(np.float64))
        else:
            kinds.add(type(layer).__name__.lower())
    if len(kinds) > 1:
        raise ValueError(f"cannot export a network that mixes activations: {sorted(kinds)}")
    activation = kinds.pop() if kinds else "identity"
    rms = norm.obs_rms
    return MlpPolicy(
        weights, biases, activation, squash, rms.mean.astype(float), rms.var.astype(float), norm.clip_obs, norm.epsilon
    )


class Evaluator:
    """Runs the exported policy on fixed test episodes in this process (same shoves as training)."""

    def __init__(self, spec: TrainSpec):
        self.env = worker.make_env(spec.model_dump(), rank=-1)
        self.inner = self.env.unwrapped.inner
        self.task = self.inner.task

    def run(self, net: MlpPolicy, episodes: int, seed: int = 10_000) -> dict[str, Any]:
        returns, successes, reports = [], [], []
        for i in range(episodes):
            obs, _ = self.env.reset(seed=seed + i)
            total, info = 0.0, {}
            while True:
                obs, reward, terminated, truncated, info = self.env.step(net(obs))
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            successes.append(bool(info.get("is_success")))
            reports.append(self.task.report(self.inner.data))
        numbers = {
            key: round(float(np.mean([r[key] for r in reports])), 4)
            for key, value in reports[0].items()
            if isinstance(value, int | float)
        }
        return {
            "episodes": episodes,
            "mean_return": round(float(np.mean(returns)), 3),
            "success_rate": round(float(np.mean(successes)), 3),
            "mean_result": numbers,
        }


class Progress(BaseCallback):
    """Writes status.json, tests the policy now and then, keeps the best, and stops on request."""

    def __init__(self, trainer: Trainer):
        super().__init__()
        self.t = trainer
        self.last_status = 0.0
        self.next_eval = 1
        self.stop_reason: str | None = None

    def _on_step(self) -> bool:
        t = self.t
        now = time.monotonic()
        progress = t.progress(self.num_timesteps)
        if progress >= self.next_eval / EVALS:
            self.next_eval = int(progress * EVALS) + 1
            t.evaluate_and_keep(self.num_timesteps)
        if now - self.last_status >= STATUS_EVERY:
            self.last_status = now
            t.write_status("running", self.num_timesteps)
            if (t.folder / store.STOP_FILE).exists():
                self.stop_reason = "stopped on request"
                return False
        if t.spec.max_minutes is not None and now - t.started > 60.0 * t.spec.max_minutes:
            self.stop_reason = "time budget used up"
            return False
        return True


class Trainer:
    def __init__(self, spec: TrainSpec, folder: Path):
        self.spec = spec
        self.folder = folder
        self.started = time.monotonic()
        self.curve: list[list[float]] = []
        self.evals: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.model: PPO | SAC | None = None
        self.norm: VecNormalize | None = None
        self.evaluator = Evaluator(spec)

    # ---- progress ----

    def progress(self, steps: int) -> float:
        done = steps / self.spec.timesteps
        if self.spec.max_minutes is not None:
            done = max(done, (time.monotonic() - self.started) / (60.0 * self.spec.max_minutes))
        return min(done, 1.0)

    def write_status(self, state: str, steps: int, **extra: Any) -> None:
        elapsed = time.monotonic() - self.started
        model = self.model
        episodes = list(model.ep_info_buffer)[-20:] if model is not None else []
        successes = list(model.ep_success_buffer)[-20:] if model is not None else []
        progress = self.progress(steps)
        status: dict[str, Any] = {
            "state": state,
            "name": self.spec.name,
            "task": self.spec.task,
            "algo": self.spec.algo,
            "timesteps": int(steps),
            "timesteps_planned": self.spec.timesteps,
            "progress": round(progress, 4),
            "elapsed_s": round(elapsed, 1),
            "steps_per_s": round(steps / elapsed, 1) if elapsed > 0 else 0.0,
            "pid": os.getpid(),
            "updated": time.time(),
        }
        if state == "running" and progress > 0.01:
            status["eta_s"] = round(elapsed * (1.0 - progress) / progress, 1)
        if episodes:
            control_dt = self.evaluator.inner.control_dt
            status["mean_episode_reward"] = round(float(np.mean([e["r"] for e in episodes])), 3)
            status["mean_episode_seconds"] = round(float(np.mean([e["l"] for e in episodes])) * control_dt, 2)
            if state == "running" and (not self.curve or self.curve[-1][0] < steps):
                self.curve.append([int(steps), status["mean_episode_reward"]])
                if len(self.curve) > CURVE_POINTS:
                    self.curve = self.curve[::2]
        if successes:
            status["success_rate"] = round(float(np.mean(successes)), 3)
        if self.best is not None:
            status["best_test"] = {k: self.best[k] for k in ("timesteps", "mean_return", "success_rate")}
        status["curve"] = self.curve
        status.update(extra)
        store.write_json(self.folder / store.STATUS_FILE, status)

    # ---- testing and keeping the best ----

    def evaluate_and_keep(self, steps: int, episodes: int = EVAL_EPISODES) -> dict[str, Any]:
        net = export_policy(self.model, self.norm)
        result = {"timesteps": int(steps), **self.evaluator.run(net, episodes)}
        self.evals.append(result)
        if self.best is None or result["mean_return"] > self.best["mean_return"]:
            self.best = result
            net.save(self.folder / store.WEIGHTS_FILE)
            self.model.save(self.folder / store.MODEL_FILE)
            self.norm.save(str(self.folder / store.NORM_FILE))
        print(
            f"test at {steps} steps: return {result['mean_return']:.2f}, success {result['success_rate']:.0%}"
            f"{' (best so far)' if self.best is result else ''}",
            flush=True,
        )
        return result

    # ---- training ----

    def make_envs(self) -> VecNormalize:
        spec = self.spec
        data = spec.model_dump()
        fns = [partial(worker.make_env, data, i) for i in range(spec.n_envs)]
        processes = spec.n_envs > 1 and not DEFAULTS[spec.task].inline
        venv: VecEnv = LeanSubprocVecEnv(fns) if processes else DummyVecEnv(fns)
        settings = algo_settings(spec)
        return VecNormalize(
            VecMonitor(venv), norm_obs=True, norm_reward=spec.algo == "ppo", gamma=settings.get("gamma", 0.99)
        )

    def make_model(self) -> PPO | SAC:
        spec = self.spec
        settings = algo_settings(spec)
        net = list(spec.net)
        if spec.algo == "ppo":
            kwargs = {"net_arch": {"pi": net, "vf": net}, "activation_fn": nn.Tanh}
            if spec.action_mode == "residual":
                kwargs["log_std_init"] = -1.0  # explore gently around a controller that already works
            return PPO("MlpPolicy", self.norm, policy_kwargs=kwargs, seed=spec.seed, device="cpu", **settings)
        kwargs = {"net_arch": {"pi": net, "qf": net}, "activation_fn": nn.ReLU}
        return SAC("MlpPolicy", self.norm, policy_kwargs=kwargs, seed=spec.seed, device="cpu", **settings)

    def run(self) -> dict[str, Any]:
        spec = self.spec
        print(f"training {spec.algo.upper()} on {spec.task} with {spec.n_envs} simulation processes", flush=True)
        self.write_status("starting", 0)
        self.norm = self.make_envs()
        try:
            self.model = self.make_model()
            callback = Progress(self)
            self.model.learn(total_timesteps=spec.timesteps, callback=callback)
            steps = int(self.model.num_timesteps)
            last = self.evaluate_and_keep(steps)
            self.write_status("testing", steps)
        finally:
            self.norm.close()
        # Score the kept policy on more episodes, then without shoves too when training used them.
        net = MlpPolicy.load(self.folder / store.WEIGHTS_FILE)
        final = {"kept_from_timesteps": self.best["timesteps"], **self.evaluator.run(net, FINAL_EPISODES, seed=20_000)}
        if spec.push > 0:
            calm = TrainSpec.model_validate({**spec.model_dump(), "push": 0.0})
            final["without_shoves"] = Evaluator(calm).run(net, FINAL_EPISODES, seed=20_000)
        stopped = callback.stop_reason == "stopped on request"
        return {
            "final": final,
            "last_test": last,
            "tests": self.evals,
            "timesteps": steps,
            "train_seconds": round(time.monotonic() - self.started, 1),
            "stopped_early": callback.stop_reason,
            "state": "stopped" if stopped else "done",
        }


def versions() -> dict[str, str]:
    import gymnasium
    import mujoco
    import stable_baselines3

    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
    }


def run_job(folder: Path) -> int:
    """The training process: train the spec in `folder`, record the outcome. Returns the exit code."""
    folder = Path(folder)
    started = time.monotonic()
    if hasattr(os, "nice"):
        os.nice(5)  # training must never make the viewer or the agent stutter
    torch.set_num_threads(1)  # small networks: one thread is fastest (measured), and leaves cores to physics
    spec = TrainSpec.model_validate(store.read_json(folder / store.SPEC_FILE))
    meta = store.read_json(folder / store.POLICY_FILE)
    threading.Thread(target=_heartbeat, args=(folder,), daemon=True).start()
    try:
        result = Trainer(spec, folder).run()
    except Exception as exc:  # recorded for the user, then re-raised into train.log
        tail = traceback.format_exc().strip().splitlines()[-TAIL_LINES:]
        if isinstance(exc, EOFError | BrokenPipeError | ConnectionError):
            # A simulation worker died; its own traceback went to train.log before this one.
            tail = [*log_tail(folder, TAIL_LINES), "---", *tail]
        store.write_json(
            folder / store.STATUS_FILE,
            {
                **store.read_json(folder / store.STATUS_FILE),
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": tail,
                "elapsed_s": round(time.monotonic() - started, 1),
                "updated": time.time(),
            },
        )
        raise
    state = result.pop("state")
    store.write_json(folder / store.POLICY_FILE, {**meta, **result, "versions": versions()})
    store.write_json(
        folder / store.STATUS_FILE,
        {
            **store.read_json(folder / store.STATUS_FILE),
            "state": state,
            "timesteps": result["timesteps"],
            "elapsed_s": result["train_seconds"],
            "final": result["final"],
            "stopped_early": result["stopped_early"],
            "updated": time.time(),
        },
    )
    (folder / store.STOP_FILE).unlink(missing_ok=True)
    print(f"{state}: {result['final']}", flush=True)
    return 0


def _heartbeat(folder: Path) -> None:
    """Touch the heartbeat file while this process lives, so a server can tell a long final test from a
    process that died (a status file alone goes quiet during tests)."""
    beat = folder / store.HEARTBEAT_FILE
    while True:
        with contextlib.suppress(OSError):  # the folder may be gone (replaced or deleted): nothing to tell
            beat.touch()
        time.sleep(HEARTBEAT_SECONDS)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_job(Path(sys.argv[1])))
