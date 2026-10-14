"""Training jobs: start a training process, follow it, stop it, notice when it died.

Training runs in its own process (`python -m robotforge.control.rl job <folder>`), never inside
the MCP server: PyTorch and the simulation workers would otherwise stall the server and freeze
the viewer. The process prints only to train.log; the MCP protocol on the server's stdout never
sees it. It is detached from the server, so it finishes (and saves the policy) even if the agent
app is closed; the status file says how it went.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from robotforge.control.rl import store
from robotforge.control.rl.spec import TrainSpec, describe_env, estimate_seconds
from robotforge.control.tasks import TaskError

RL_MODULES = ("gymnasium", "stable_baselines3", "torch")
INSTALL_HINT = (
    "training neural network policies needs the optional 'rl' extra (gymnasium, Stable-Baselines3, PyTorch CPU), "
    "which is not installed. Install it with: uv sync --extra rl (in a robotforge checkout) or "
    "pip install 'robotforge[rl]'. Without it, the 'ars' method still learns cart-pole balance with numpy only."
)
STALE_SECONDS = 120.0  # a running job that has not written its status for this long is gone
MAX_RUNNING = 1  # each job already uses most CPU cores


def missing_rl_modules() -> list[str]:
    return [m for m in RL_MODULES if importlib.util.find_spec(m) is None]


def require_rl() -> None:
    missing = missing_rl_modules()
    if missing:
        raise TaskError(f"{INSTALL_HINT} (missing: {', '.join(missing)})")


class Jobs:
    """The training processes this program started, by policy folder."""

    def __init__(self) -> None:
        self.procs: dict[Path, subprocess.Popen] = {}

    def running(self) -> list[Path]:
        """Jobs still training (a finished one may take a moment to exit after saying it is done)."""
        return [
            f
            for f, p in self.procs.items()
            if p.poll() is None and self.status(f).get("state") not in store.FINAL_STATES
        ]

    def start(self, spec: TrainSpec, folder: Path, replace: bool = False) -> dict[str, Any]:
        """Check the spec, prepare the folder and launch the training process. Returns at once."""
        require_rl()
        busy = self.running()
        if len(busy) >= MAX_RUNNING:
            raise TaskError(
                f"policy {busy[0].name!r} is still training and uses the computer's cores; wait for it "
                f"(training_status) or stop it (stop_training) before starting another"
            )
        folder = Path(folder)
        if (folder / store.POLICY_FILE).exists():
            state = self.status(folder).get("state")
            if state not in store.FINAL_STATES:
                raise TaskError(f"policy {folder.name!r} is already training ({state})")
            if not replace:
                raise TaskError(
                    f"a policy named {folder.name!r} already exists ({state}); pick another name, or replace it"
                )
        env_facts = describe_env(spec)  # fails here, readably, if the task does not fit the robot
        folder.mkdir(parents=True, exist_ok=True)
        for old in (store.WEIGHTS_FILE, store.MODEL_FILE, store.NORM_FILE, store.STOP_FILE):
            (folder / old).unlink(missing_ok=True)
        estimate = estimate_seconds(spec)
        store.write_json(folder / store.SPEC_FILE, spec.model_dump())
        store.write_json(
            folder / store.POLICY_FILE,
            {
                "name": spec.name,
                "task": spec.task,
                "task_params": spec.task_params,
                "robot": spec.robot.name,
                "robot_spec": spec.robot.model_dump(exclude_none=True),
                "algo": spec.algo,
                "action_mode": spec.action_mode,
                "residual_scale": spec.residual_scale,
                "timesteps_planned": spec.timesteps,
                "max_minutes": spec.max_minutes,
                "n_envs": spec.n_envs,
                "seed": spec.seed,
                "push": spec.push,
                "net": spec.net,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                **env_facts,
            },
        )
        store.write_json(
            folder / store.STATUS_FILE,
            {"state": "starting", "name": spec.name, "timesteps": 0, "progress": 0.0, "updated": time.time()},
        )
        log = (folder / store.LOG_FILE).open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "robotforge.control.rl", "job", str(folder)],
                stdin=subprocess.DEVNULL,
                stdout=log,  # never the MCP server's stdout, which carries the protocol
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                **_detached(),
            )
        finally:
            log.close()  # the child has its own handle
        self.procs[folder] = proc
        return {"folder": str(folder), "pid": proc.pid, "estimate_seconds": round(estimate), **env_facts}

    def status(self, folder: Path) -> dict[str, Any]:
        """status.json, corrected when the process died without saying so."""
        folder = Path(folder)
        status = store.read_json(folder / store.STATUS_FILE)
        if not status:
            return {"state": "unknown"}
        state = status.get("state")
        if state in store.FINAL_STATES:
            return status
        proc = self.procs.get(folder)
        if proc is not None and proc.poll() is not None:
            # Read again: the process may have written its final status just before exiting.
            status = store.read_json(folder / store.STATUS_FILE)
            if status.get("state") in store.FINAL_STATES:
                return status
            return self._mark_failed(folder, status, f"the training process exited with code {proc.returncode}")
        if proc is None and time.time() - float(status.get("updated", 0)) > STALE_SECONDS:
            return self._mark_failed(
                folder, status, "the training process stopped without finishing (the computer slept or it was killed)"
            )
        return status

    def _mark_failed(self, folder: Path, status: dict[str, Any], why: str) -> dict[str, Any]:
        status = {**status, "state": "failed", "error": why, "traceback_tail": log_tail(folder), "updated": time.time()}
        store.write_json(folder / store.STATUS_FILE, status)
        return status

    def stop(self, folder: Path, wait: float = 20.0) -> dict[str, Any]:
        """Ask the training to stop; it keeps the best policy so far and runs its final test."""
        folder = Path(folder)
        status = self.status(folder)
        if status.get("state") in store.FINAL_STATES:
            return status
        (folder / store.STOP_FILE).write_text("stop\n", encoding="utf-8")
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            status = self.status(folder)
            if status.get("state") in store.FINAL_STATES:
                return status
            time.sleep(0.2)
        proc = self.procs.get(folder)
        if status.get("timesteps", 0) == 0 and proc is not None:
            proc.terminate()  # still starting up: nothing learned yet, nothing to keep
            proc.wait(timeout=10)
            return self._mark_failed(folder, status, "stopped before it had learned anything")
        return {**status, "note": "stopping: it is saving the best policy so far and running its final test"}

    def wait(self, folder: Path, on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Block until the job ends, calling on_progress with each new status."""
        last = None
        while True:
            status = self.status(folder)
            if on_progress is not None and status.get("updated") != last:
                last = status.get("updated")
                on_progress(status)
            if status.get("state") in store.FINAL_STATES:
                return status
            time.sleep(0.5)


def log_tail(folder: Path, lines: int = 25) -> list[str]:
    try:
        text = (Path(folder) / store.LOG_FILE).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.strip().splitlines()[-lines:]


def _detached() -> dict[str, Any]:
    """Keep training alive if the agent app closes, and keep Ctrl+C in the server's terminal away from it."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


JOBS = Jobs()
