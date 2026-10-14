"""UniLab as robotforge's external training backend.

UniLab (github.com/unilabsim/UniLab, Apache-2.0) trains robot policies with many parallel
MuJoCo simulations and a GPU-capable learner. It pins its own torch and MuJoCo builds, so it
lives in its own checkout and virtual environment, named by $ROBOTFORGE_UNILAB_DIR. robotforge
only ever talks to it through subprocesses and files:

- `check()` asks the checkout's python for its versions (unilab_tool.py info).
- `start_training()` launches unilab_tool.py train detached; it trains, exports ONNX with
  UniLab's own exporter, writes the policy card and reports progress in a status file.
- `job_status()` reads that file; a finished job's policy is installed under
  $ROBOTFORGE_HOME/policies, where move(walker="rl") finds it.

Nothing here writes to stdout, which carries the MCP protocol when robotforge runs as a server.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from robotforge.bridge import policies
from robotforge.bridge.spec import NAME_RE, SpecError, load_spec

ENV_VAR = "ROBOTFORGE_UNILAB_DIR"
TOOL = Path(__file__).resolve().parent / "unilab_tool.py"
INFO_TIMEOUT = 120.0  # s; the first import of torch in a fresh venv is slow
MAX_ITERATIONS = 5000
SETUP_HINT = (
    "Set it up once (outside robotforge's environment): git clone https://github.com/unilabsim/UniLab.git, "
    "then in that folder run `uv sync --extra mujoco --extra uni_rl`, and set ROBOTFORGE_UNILAB_DIR to the folder."
)
# Tasks robotforge can train and then play: UniLab task owner -> robotforge library robot.
TASKS: dict[str, dict[str, Any]] = {
    "go2_joystick_flat": {
        "robot": "go2",
        "algo": "ppo",
        "explain": "Unitree Go2 learns to follow forward, sideways and turning speed commands on flat ground.",
        "suggested_iterations": 1000,
        "seconds_per_iteration": 0.8,  # measured on an M5 Max, 1024 environments
    },
}


# Jobs started by this process: polling them reaps finished children, which would otherwise linger
# as zombies that still answer "alive" to a signal-0 probe.
_STARTED: dict[int, subprocess.Popen] = {}
# Status reads may install a finished job's policy; MCP runs sync tools on worker threads, so two
# calls at once must not both install it.
_STATUS_LOCK = threading.Lock()
JOB_RE = re.compile(r"\A\d{8}-\d{6}-[0-9a-f]{6}\Z")


class UniLabError(RuntimeError):
    """User-facing: what is missing or wrong, and how to fix it."""


def checkout() -> Path:
    """The configured UniLab checkout, or a UniLabError that explains how to set one up."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        raise UniLabError(f"UniLab is not configured: {ENV_VAR} is not set. {SETUP_HINT}")
    root = Path(raw).expanduser().resolve()
    project = root / "pyproject.toml"
    if not project.is_file() or 'name = "unilab"' not in project.read_text(encoding="utf-8"):
        raise UniLabError(f"{ENV_VAR}={root} is not a UniLab checkout (no pyproject.toml for unilab). {SETUP_HINT}")
    return root


def python_of(root: Path) -> Path:
    """The interpreter of the checkout's own virtual environment."""
    candidates = [root / ".venv" / "bin" / "python", root / ".venv" / "Scripts" / "python.exe"]
    for path in candidates:
        if path.is_file():
            return path
    raise UniLabError(f"{root} has no .venv yet. In that folder run `uv sync --extra mujoco --extra uni_rl`.")


def _run_tool(args: list[str], timeout: float) -> dict[str, Any]:
    root = checkout()
    try:
        done = subprocess.run(
            [str(python_of(root)), str(TOOL), *args],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise UniLabError(f"UniLab did not answer within {timeout:.0f} s ({' '.join(args[:1])})") from None
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-8:])
        raise UniLabError(f"UniLab's environment failed ({' '.join(args[:1])}, exit {done.returncode}):\n{tail}")
    lines = [ln for ln in done.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise UniLabError(f"UniLab gave no result for {args[0]}; output: {done.stdout[-500:]}")
    return json.loads(lines[-1])


def check() -> dict[str, Any]:
    """Is UniLab set up? Versions, devices and the tasks robotforge can train with it."""
    root = checkout()
    started = time.perf_counter()
    info = _run_tool(["info"], INFO_TIMEOUT)
    return {
        "configured": True,
        "checkout": str(root),
        "interpreter": str(python_of(root)),
        **{k: v for k, v in info.items() if k != "tasks"},
        "check_seconds": round(time.perf_counter() - started, 1),
        "trainable_tasks": {name: t["explain"] for name, t in TASKS.items()},
    }


def jobs_dir() -> Path:
    return policies.user_dir().parent / "unilab" / "jobs"


def start_training(task: str, iterations: int, name: str, num_envs: int | None = None) -> dict[str, Any]:
    """Launch a UniLab training run in the background; returns the job id and where to watch it."""
    if task not in TASKS:
        raise UniLabError(f"robotforge can train {', '.join(TASKS)} with UniLab, not {task!r}")
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise UniLabError(f"iterations must be 1..{MAX_ITERATIONS} (UniLab's default for {task} is 151)")
    if num_envs is not None and not 16 <= num_envs <= 16384:
        raise UniLabError("num_envs must be 16..16384")
    if not NAME_RE.match(name):
        raise UniLabError("policy names use lowercase letters, digits and _, start with a letter")
    root = checkout()
    python = python_of(root)
    job = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    folder = jobs_dir() / job
    folder.mkdir(parents=True)
    spec = TASKS[task]
    args = [
        str(python),
        str(TOOL),
        "train",
        "--task",
        task,
        "--algo",
        spec["algo"],
        "--iterations",
        str(iterations),
        "--log-root",
        str(folder / "runs"),
        "--status",
        str(folder / "status.json"),
        "--export-to",
        str(folder / "policy"),
        "--name",
        name,
        "--robot",
        spec["robot"],
    ]
    if num_envs:
        args += ["--num-envs", str(num_envs)]
    try:
        with (folder / "tool.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                args, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **_detached()
            )
    except OSError as exc:
        shutil.rmtree(folder, ignore_errors=True)  # no half-made job that looks like it is starting
        raise UniLabError(f"could not start UniLab's python ({python}): {exc}") from exc
    _STARTED[proc.pid] = proc
    job_info = {"job": job, "task": task, "iterations": iterations, "name": name, "pid": proc.pid, "command": args}
    (folder / "job.json").write_text(json.dumps(job_info, indent=1), encoding="utf-8")
    estimate = iterations * spec["seconds_per_iteration"]
    return {
        "job": job,
        "pid": proc.pid,
        "task": task,
        "iterations": iterations,
        "policy_name": name,
        "estimated_minutes": round(estimate / 60 + 0.5, 1),  # plus about 30 s to start and export
        "status_file": str(folder / "status.json"),
    }


def _detached() -> dict[str, Any]:
    """Popen options that let training outlive the robotforge process (and its stdin/stdout)."""
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        return {"creationflags": flags}
    return {"start_new_session": True}


def job_status(job: str | None = None) -> dict[str, Any]:
    """Progress of one job (the newest if None); installs the policy once the job is done."""
    folder = _job_folder(job)
    with _STATUS_LOCK:
        return _read_status(folder)


def _read_status(folder: Path) -> dict[str, Any]:
    status_file = folder / "status.json"
    running = ("starting", "training", "exporting")
    status = _load_json(status_file) or {"state": "starting", "pid": _load_json(folder / "job.json").get("pid")}
    if status.get("state") in running and not _alive(status.get("pid")):
        status = _load_json(status_file) or status  # it may have finished between the two reads
        if status.get("state") in running:
            status["state"] = "failed"
            status.setdefault("error", "the training process is gone (stopped or crashed); see the log")
            status["log_tail"] = _tail(folder / "tool.log", 8)
    status["job"] = folder.name
    if status.get("state") == "done" and status.get("policy_dir") and "installed" not in status:
        try:
            installed = policies.install(Path(status["policy_dir"]))
        except SpecError as exc:
            status.update(state="failed", error=f"the exported policy did not validate: {exc}")
        else:
            status["installed"] = installed.spec.name
            status["try_it"] = f"move(robot=..., speed=0.5, walker={installed.spec.name!r})"
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v for k, v in status.items() if k != "job"}, indent=1), encoding="utf-8")
        os.replace(tmp, status_file)
    return status


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _tail(path: Path, lines: int) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-lines:]


def cancel(job: str | None = None) -> dict[str, Any]:
    folder = _job_folder(job)
    status = job_status(folder.name)
    pid = status.get("pid")
    if status.get("state") not in ("starting", "training", "exporting") or not _alive(pid):
        return {"job": folder.name, "cancelled": False, "state": status.get("state")}
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)], stdin=subprocess.DEVNULL, capture_output=True, check=False
            )
        else:
            os.killpg(int(pid), signal.SIGTERM)  # its own process group: the tool, UniLab's CLI and its trainer
    except ProcessLookupError:
        return {"job": folder.name, "cancelled": False, "state": "finished before it could be stopped"}
    return {"job": folder.name, "cancelled": True}


def list_jobs(limit: int = 10) -> list[dict[str, Any]]:
    root = jobs_dir()
    if not root.is_dir():
        return []
    out = []
    for folder in sorted((p for p in root.iterdir() if JOB_RE.match(p.name)), reverse=True)[:limit]:
        status = job_status(folder.name)
        out.append(
            {k: status.get(k) for k in ("job", "state", "task", "iteration", "iterations", "mean_reward", "installed")}
        )
    return out


def export_run(run: Path, name: str, task: str | None = None) -> dict[str, Any]:
    """Export an existing UniLab run directory (e.g. one trained by hand) and install it."""
    if not (run / "run_config.json").is_file():
        raise UniLabError(f"{run} is not a UniLab run directory (it has no run_config.json)")
    if not NAME_RE.match(name):
        raise UniLabError("policy names use lowercase letters, digits and _, start with a letter")
    out = jobs_dir().parent / "exports" / name
    args = ["export", "--run", str(run.resolve()), "--out", str(out), "--name", name]
    if task:
        args += ["--task", task]
    result = _run_tool(args, timeout=600.0)
    installed = policies.install(out)
    load_spec(installed.directory)
    return {**result, "installed": installed.spec.name}


def _job_folder(job: str | None) -> Path:
    root = jobs_dir()
    if job is None:
        folders = sorted(p for p in root.iterdir() if JOB_RE.match(p.name)) if root.is_dir() else []
        if not folders:
            raise UniLabError("no UniLab training jobs yet; start one with unilab_train")
        return folders[-1]
    folder = root / job
    if not JOB_RE.match(job) or not folder.is_dir():
        raise UniLabError(f"no UniLab job {job!r}; jobs: {', '.join(j['job'] for j in list_jobs()) or 'none'}")
    return folder


def _alive(pid: Any) -> bool:
    """Is `pid` a training tool we started? A stale pid from before a reboot may belong to anyone now."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid in _STARTED:
        return _STARTED[pid].poll() is None
    if sys.platform == "win32":
        done = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        rows = [line.split('","') for line in done.stdout.splitlines() if line.startswith('"')]
        return any(len(r) > 1 and r[1] == str(pid) and r[0].lower().startswith('"python') for r in rows)
    try:
        if os.getpgid(pid) != pid:  # the tool leads its own session; anything else is not ours
            return False
    except (ProcessLookupError, PermissionError):
        return False
    done = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False
    )
    return TOOL.name in done.stdout
