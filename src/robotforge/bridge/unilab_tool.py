"""UniLab side of the bridge. Runs INSIDE UniLab's own Python environment, never robotforge's.

UniLab pins its own torch and MuJoCo build, so robotforge never imports it. Instead robotforge
runs this file with the python of a UniLab checkout:

    <unilab>/.venv/bin/python unilab_tool.py info
    ... unilab_tool.py train --task go2_joystick_flat --iterations 50 --log-root DIR --status FILE --export-to DIR
    ... unilab_tool.py export --run <run dir> --out DIR --name NAME --robot go2
    ... unilab_tool.py record --run <run dir> --steps 30 --out trace.npz

`export` reads the observation, action and PD contract from a live UniLab environment (not
from documentation) and writes it as a robotforge policy card next to a single-file ONNX model.
Only the standard library is imported at module level, so robotforge's tests can import the
pure helpers; UniLab, torch, onnx and MuJoCo are imported inside the commands that need them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CARD_FORMAT = "robotforge-policy/1"
# UniLab supports Python 3.10, which has no datetime.UTC.
UTC = timezone.utc  # noqa: UP017
# UniLab task owner -> (robotforge library robot, title, summary)
TASKS: dict[str, tuple[str, str, str]] = {
    "go2_joystick_flat": (
        "go2",
        "Go2 walking (UniLab PPO)",
        "Velocity-tracking walk for the Unitree Go2 on flat ground, trained with UniLab's PPO in MuJoCo.",
    ),
    "g1_walk_flat": (
        "g1",
        "G1 walking (UniLab)",
        "Velocity-tracking walk for the Unitree G1 humanoid on flat ground, trained with UniLab.",
    ),
}
ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
FIELD_RE = re.compile(r"^\s*(Mean reward|Mean episode length|Time elapsed|ETA|Total steps)\s*:\s*(\S+)")


# ---- progress parsing (pure, tested in robotforge) ------------------------------------------


def parse_progress(line: str, state: dict[str, Any]) -> bool:
    """Update `state` from one line of UniLab's training log; True when something changed."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = ITERATION_RE.search(clean)
    if m:
        state["iteration"] = int(m.group(1)) + 1  # UniLab counts from 0
        state["iterations"] = int(m.group(2))
        return True
    m = FIELD_RE.match(clean)
    if not m:
        return False
    key, value = m.group(1), m.group(2)
    if key in ("Time elapsed", "ETA"):
        state["elapsed_s" if key == "Time elapsed" else "eta_s"] = _clock_seconds(value)
    elif key == "Total steps":
        state["env_steps"] = int(float(value))
    else:
        with contextlib.suppress(ValueError):
            state["mean_reward" if key == "Mean reward" else "mean_episode_length"] = float(value)
    return True


def _clock_seconds(text: str) -> float | None:
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return total


def write_status(path: Path, status: dict[str, Any]) -> None:
    """Atomically replace the status file, so a reader never sees half a JSON document."""
    status = {**status, "updated": datetime.now(UTC).isoformat(timespec="seconds")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows refuses while robotforge has the file open for reading
            if attempt == 19:
                raise
            time.sleep(0.05)


# ---- UniLab environment helpers (need UniLab) -----------------------------------------------


def unilab_root() -> Path:
    import unilab

    return Path(unilab.__file__).resolve().parents[2]


def _load_cfg(run: Path):
    from omegaconf import OmegaConf

    path = run / "run_config.json"
    if not path.is_file():
        raise SystemExit(f"{run} has no run_config.json; is it a UniLab run directory?")
    cfg = OmegaConf.create(json.loads(path.read_text(encoding="utf-8"))["config"])
    cfg.training.device = "cpu"
    return cfg


def _make_env(cfg):
    from unilab.training import ensure_registries

    ensure_registries()
    from unilab.base.config_adapter import create_env
    from unilab.scripts.train_rsl_rl import build_ppo_play_env_cfg_override

    return create_env(cfg, num_envs=1, env_cfg_override=build_ppo_play_env_cfg_override(cfg))


def _owner_task(cfg) -> str:
    """The task owner directory (e.g. go2_joystick_flat) whose config produced this run."""
    algo, sim, name = str(cfg.algo.algo), str(cfg.training.sim_backend), str(cfg.training.task_name)
    for owner in sorted((unilab_root() / "src" / "unilab" / "conf" / algo / "task").glob(f"*/{sim}.yaml")):
        if re.search(rf"task_name:\s*{re.escape(name)}\s*$", owner.read_text(encoding="utf-8"), re.M):
            return owner.parent.name
    raise SystemExit(f"cannot find the UniLab task owner for {name} ({algo}, {sim}); pass --task")


def _func_name(func: Any) -> str:
    return getattr(func, "__name__", None) or type(func).__name__


def _sensor(model, name: str) -> tuple[int, str]:
    """(MuJoCo sensor type, site name) of a site sensor without a reference frame."""
    import mujoco

    s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if s < 0:
        raise SystemExit(f"sensor {name!r} not found in the UniLab model")
    if int(model.sensor_objtype[s]) != mujoco.mjtObj.mjOBJ_SITE or int(model.sensor_refid[s]) >= 0:
        raise SystemExit(f"sensor {name!r} is not a plain site sensor; robotforge cannot rebuild it")
    site = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(model.sensor_objid[s]))
    return int(model.sensor_type[s]), site


def _translate_terms(env, group: str, model, joints: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    """UniLab observation terms -> robotforge card terms, plus the IMU site the base terms use."""
    import mujoco

    om = env.observation_manager
    names = om._group_obs_term_names[group]
    cfgs = om._group_obs_term_cfgs[group]
    dims = om._group_obs_term_dim[group]
    sites: set[str] = set()
    body_frame = False
    out = []
    for name, cfg, dim in zip(names, cfgs, dims, strict=True):
        func = _func_name(cfg.func)
        params = dict(cfg.params or {})
        term: dict[str, Any]
        if func == "builtin_sensor":
            kind, site = _sensor(model, params["sensor_name"])
            mapping = {
                int(mujoco.mjtSensor.mjSENS_GYRO): "base_ang_vel",
                int(mujoco.mjtSensor.mjSENS_VELOCIMETER): "base_lin_vel",
            }
            if kind not in mapping:
                raise SystemExit(f"observation {name}: sensor type {kind} is not supported by robotforge yet")
            term = {"term": mapping[kind]}
            sites.add(site)
        elif func == "projected_gravity_from_sensor":
            kind, site = _sensor(model, params["sensor_name"])
            if kind != int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS):
                raise SystemExit(f"observation {name}: expected a framezaxis sensor, got type {kind}")
            term = {"term": "neg_up_axis_world"}
            sites.add(site)
        elif func in ("projected_gravity", "base_ang_vel", "base_lin_vel"):
            term, body_frame = {"term": func}, True
        elif func == "joint_pos_rel":
            if params.get("biased"):
                raise SystemExit(f"observation {name}: encoder-biased joint positions are not supported")
            term = {"term": "joint_pos_rel"}
        elif func == "joint_vel_rel":
            term = {"term": "joint_vel"}
        elif func == "last_action":
            term = {"term": "last_action"}
        elif func == "generated_commands":
            term = {"term": "velocity_command"}
        elif func == "G1GaitPhase":
            if params.get("init_mode", "offset_phase") != "offset_phase":
                raise SystemExit(f"observation {name}: only the offset_phase gait clock is supported")
            term = {
                "term": "gait_phase",
                "frequency": float(params.get("frequency", 1.5)),
                "offsets": [0.0, 0.5],  # left, right: half a cycle apart
                "unit": "radian",
            }
        elif func == "quadruped_gait_phase":
            offsets = list(params.get("phase_offsets", (0.0, 0.5, 0.5, 0.0)))
            term = {
                "term": "gait_phase",
                "frequency": float(params.get("frequency", 2.0)),
                "offsets": [float(o) for o in offsets],
                "always_advance": params.get("command_name") is None,
            }
        else:
            raise SystemExit(f"observation {name} ({func}) has no robotforge equivalent yet")
        if cfg.scale is not None:
            scale = cfg.scale
            term["scale"] = float(scale) if isinstance(scale, int | float) else [float(v) for v in scale]
        if cfg.clip is not None:
            lo, hi = (float(v) for v in cfg.clip)
            if lo != -hi:
                raise SystemExit(f"observation {name}: asymmetric clip {cfg.clip} is not supported")
            term["clip"] = hi
        if int(cfg.history_length or 0) > 1:
            if not cfg.flatten_history_dim:
                raise SystemExit(f"observation {name}: unflattened history is not supported")
            term["history"] = int(cfg.history_length)
        width = int(dim[0])
        expected = {"joint_pos_rel": len(joints), "joint_vel": len(joints), "last_action": len(joints)}.get(
            term["term"], len(term.get("offsets", [])) if term["term"] == "gait_phase" else 3
        ) * term.get("history", 1)
        if width != expected:
            raise SystemExit(f"observation {name}: UniLab says {width} values, robotforge would build {expected}")
        out.append(term)
    if len(sites) > 1 or (sites and body_frame):
        raise SystemExit(f"base terms are measured in more than one frame ({sorted(sites)}); not supported")
    return out, (sites.pop() if sites else None)


def _pd_gains(env, cfg, model, joints: list[str]) -> dict[str, Any]:
    """The joint PD loop during training: actuator gains, overridden by the pd_gains event's mean."""
    import mujoco
    import numpy as np
    from omegaconf import OmegaConf

    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    act_of = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    acts = [act_of[j] for j in jid]
    kp = np.array([model.actuator_gainprm[a, 0] for a in acts], float)
    kd = np.array([-model.actuator_biasprm[a, 2] for a in acts], float)
    events = OmegaConf.to_container(cfg.env.events, resolve=True) or {}
    pd = events.get("pd_gains")
    if pd:
        p = pd.get("params", {})
        op = p.get("operation", "abs")
        if op not in ("abs", "scale"):
            raise SystemExit(f"pd_gains operation {op!r} is not supported")
        if "kp_range" in p:
            mean_kp = float(np.mean(p["kp_range"]))
            kp = np.full_like(kp, mean_kp) if op == "abs" else kp * mean_kp
        if "kd_range" in p:
            mean_kd = float(np.mean(p["kd_range"]))
            kd = np.full_like(kd, mean_kd) if op == "abs" else kd * mean_kd
    limited = [bool(model.actuator_forcelimited[a]) for a in acts]
    return {
        "kp": [round(float(v), 6) for v in kp],
        "kd": [round(float(v), 6) for v in kd],
        "joint_damping": [round(float(model.dof_damping[model.jnt_dofadr[j]]), 6) for j in jid],
        "torque_limit": [float(np.abs(model.actuator_forcerange[a]).max()) for a in acts] if all(limited) else None,
    }


def build_card(run: Path, name: str, robot: str | None, title: str | None, task: str | None) -> dict[str, Any]:
    """Read the policy contract off a live UniLab environment built from the run's own config."""
    import numpy as np

    cfg = _load_cfg(run)
    task = task or _owner_task(cfg)
    env = _make_env(cfg)
    model = env._backend.get_playback_model(0)
    terms = env.action_manager.active_terms
    if terms != ["joint_pos"]:
        raise SystemExit(f"robotforge plays policies with one joint_pos action term; this run has {terms}")
    action = env.action_manager.get_term("joint_pos")
    joints = list(action._target_names)
    entity = env.scene["robot"]
    if list(entity.joint_names) != joints:
        raise SystemExit("the action joints differ from the robot's joint order; not supported yet")
    default = entity.data.default_joint_pos_torch("cpu")[0].numpy().astype(float)
    offset = action._offset.reshape(-1).numpy().astype(float) if action._offset.numel() > 1 else None
    if offset is not None and not np.allclose(offset, default):
        raise SystemExit("the action offset is not the default pose; not supported yet")
    scale = action._scale.reshape(-1).numpy().astype(float) if hasattr(action._scale, "numpy") else action._scale
    scale_out: Any = float(scale) if np.isscalar(scale) or np.ndim(scale) == 0 else [float(v) for v in scale]
    if isinstance(scale_out, list) and len(set(scale_out)) == 1:
        scale_out = scale_out[0]
    clip = None
    if getattr(action, "_clip", None) is not None:
        bounds = action._clip.reshape(-1, 2).numpy()
        if not (np.allclose(bounds[:, 0], -bounds[:, 1]) and np.allclose(bounds[:, 1], bounds[0, 1])):
            raise SystemExit("asymmetric or per-joint action clips are not supported")
        clip = float(bounds[0, 1])
    group = str(cfg.env.get("policy_observation_group", "policy"))
    observations, site = _translate_terms(env, group, model, joints)
    command = env.command_manager.get_term("twist").cfg
    if getattr(command, "heading_command", False):
        raise SystemExit("heading commands are not supported; robotforge commands a turn rate")
    ranges = command.ranges
    summary = (
        json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
        if (run / "run_summary.json").is_file()
        else {}
    )
    robot_default, title_default, summary_default = TASKS.get(task, (robot, task, f"UniLab task {task}"))
    card = {
        "format": CARD_FORMAT,
        "name": name,
        "title": title or title_default,
        "robot": robot or robot_default,
        "summary": summary_default,
        "onnx": "policy.onnx",
        "input_name": "obs",
        "output_name": None,
        "policy_dt": float(env.step_dt),
        "joints": joints,
        "default_joint_pos": [round(float(v), 6) for v in default],
        "action_scale": scale_out,
        "action_clip": clip,
        "pd": _pd_gains(env, cfg, model, joints),
        "base_site": site,
        "observations": observations,
        "commands": {
            "lin_vel_x": [float(v) for v in ranges.lin_vel_x],
            "lin_vel_y": [float(v) for v in ranges.lin_vel_y],
            "ang_vel_z": [float(v) for v in ranges.ang_vel_z],
            "planar_dead_zone": float(getattr(command, "planar_dead_zone", 0.0)),
        },
        "trained_with": {
            "framework": "UniLab",
            "unilab_version": _version("unilab"),
            "task": task,
            "algo": str(cfg.algo.algo),
            "sim": str(cfg.training.sim_backend),
            "sim_dt": float(model.opt.timestep),
            # UniLab's summary counts the last iteration's index, from 0
            "iterations": summary["completed_iterations"] + 1 if "completed_iterations" in summary else None,
            "env_steps": summary.get("total_env_steps"),
            "final_mean_reward": summary.get("final_mean_reward"),
            "training_wall_seconds": summary.get("training_wall_time_sec"),
            "learner_device": _run_info(run).get("device"),
            "run": run.name,
            "exported": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    env.close()
    return card


def _run_info(run: Path) -> dict[str, Any]:
    return json.loads((run / "run_config.json").read_text(encoding="utf-8")).get("run", {})


def _version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _child_options() -> dict[str, Any]:
    """Run UniLab's own commands with UTF-8 pipes and, on Windows, without a console window each."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    options: dict[str, Any] = {"env": env, "encoding": "utf-8", "errors": "replace"}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return options


def _export_onnx_with_unilab(run: Path, task: str, cfg) -> None:
    """UniLab writes policy.onnx during playback; run its eval for one step to get it."""
    command = [
        sys.executable,
        "-c",
        "import sys; from unilab.cli import eval_main; sys.exit(eval_main())",
        "--algo",
        str(cfg.algo.algo),
        "--task",
        task,
        "--sim",
        str(cfg.training.sim_backend),
        "--load-run",
        run.name,
        "--render-mode",
        "record",  # playback (and with it the export) is skipped in "none" mode
        f"training.log_root={run.parent.parent}",
        "training.play_steps=1",
        "training.play_env_num=1",
    ]
    done = subprocess.run(
        command, cwd=unilab_root(), stdin=subprocess.DEVNULL, capture_output=True, text=True, **_child_options()
    )
    if done.returncode != 0 or not (run / "policy.onnx").is_file():
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-15:])
        raise SystemExit(f"UniLab's eval did not export policy.onnx (exit {done.returncode}):\n{tail}")


def export(run: Path, out: Path, name: str, robot: str | None, title: str | None, task: str | None) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort

    cfg = _load_cfg(run)
    task = task or _owner_task(cfg)
    started = time.perf_counter()
    if _onnx_is_stale(run):
        _export_onnx_with_unilab(run, task, cfg)
    card = build_card(run, name, robot, title, task)
    out.mkdir(parents=True, exist_ok=True)
    # UniLab's exporter stores weights in policy.onnx.data; one self-contained file is easier to move around.
    model = onnx.load(str(run / "policy.onnx"))
    onnx.save_model(model, str(out / "policy.onnx"), save_as_external_data=False)
    session = ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
    width = session.get_inputs()[0].shape[-1]
    card["input_name"] = session.get_inputs()[0].name
    card["output_name"] = session.get_outputs()[0].name
    obs_dim = sum(_term_width(t, len(card["joints"])) for t in card["observations"])
    if width != obs_dim:
        raise SystemExit(f"the ONNX model reads {width} values but the observation terms make {obs_dim}")
    action = session.run(None, {card["input_name"]: np.zeros((1, obs_dim), np.float32)})[0]
    if action.shape[-1] != len(card["joints"]) or not np.all(np.isfinite(action)):
        raise SystemExit(f"the ONNX model returned {action.shape} for {len(card['joints'])} joints")
    (out / "policy.json").write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
    return {
        "policy_dir": str(out),
        "onnx_bytes": (out / "policy.onnx").stat().st_size,
        "export_seconds": round(time.perf_counter() - started, 1),
    }


def _onnx_is_stale(run: Path) -> bool:
    """policy.onnx is missing or older than the newest checkpoint (e.g. exported mid-training)."""
    onnx_file = run / "policy.onnx"
    checkpoints = list(run.glob("model_*.pt"))
    if not onnx_file.is_file():
        return True
    return bool(checkpoints) and max(c.stat().st_mtime for c in checkpoints) > onnx_file.stat().st_mtime


def _term_width(term: dict[str, Any], joints: int) -> int:
    base = {"joint_pos_rel": joints, "joint_vel": joints, "last_action": joints}.get(term["term"])
    if base is None:
        base = len(term["offsets"]) if term["term"] == "gait_phase" else 3
    return base * int(term.get("history", 1))


def record(run: Path, steps: int, out: Path) -> dict[str, Any]:
    """Roll the run's policy out in UniLab's own environment and save states, observations and actions."""
    import numpy as np
    import onnxruntime as ort
    import torch

    cfg = _load_cfg(run)
    env = _make_env(cfg)
    session = ort.InferenceSession(str(run / "policy.onnx"), providers=["CPUExecutionProvider"])
    obs = env.get_observations()
    key = next(iter(obs))  # the actor group comes first
    rows: dict[str, list[Any]] = {"phys": [], "obs": [], "act": [], "cmd": []}
    for _ in range(steps):
        o = obs[key].cpu().numpy().astype(np.float32)
        a = session.run(None, {session.get_inputs()[0].name: o})[0]
        rows["phys"].append(env.get_physics_state_snapshot().reshape(-1))
        rows["obs"].append(o[0])
        rows["act"].append(a[0])
        rows["cmd"].append(env.command_manager.get_command("twist").cpu().numpy()[0])
        obs = env.step(torch.as_tensor(a)).obs
    layout = env.get_physics_state_layout()
    np.savez_compressed(out, nq=layout.nq, nv=layout.nv, **{k: np.array(v, np.float32) for k, v in rows.items()})
    env.close()
    return {"trace": str(out), "steps": steps}


def info() -> dict[str, Any]:
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return {
        "unilab": _version("unilab"),
        "torch": torch.__version__,
        "mujoco": _version("mujoco"),
        "onnx": _version("onnx"),
        "python": sys.version.split()[0],
        "devices": devices,
        "root": str(unilab_root()),
        "tasks": sorted(TASKS),
    }


# ---- training -------------------------------------------------------------------------------


def train(args: argparse.Namespace) -> int:
    status_path = Path(args.status)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "state": "starting",
        "task": args.task,
        "algo": args.algo,
        "sim": args.sim,
        "iterations": args.iterations,
        "iteration": 0,
        "pid": os.getpid(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_status(status_path, state)
    overrides = [
        f"algo.max_iterations={args.iterations}",
        f"algo.save_interval={max(args.iterations, 1)}",
        "training.no_play=true",
        f"training.log_root={log_root}",
    ]
    if args.num_envs:
        overrides.append(f"algo.num_envs={args.num_envs}")
    command = [
        sys.executable,
        "-u",
        "-c",
        "import sys; from unilab.cli import train_main; sys.exit(train_main())",
        "--algo",
        args.algo,
        "--task",
        args.task,
        "--sim",
        args.sim,
        "--render-mode",
        "none",
        *overrides,
    ]
    started = time.perf_counter()
    log_path = status_path.with_name("train.log")
    state["state"], state["log"] = "training", str(log_path)
    write_status(status_path, state)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=unilab_root(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **_child_options(),
        )
        last_write = 0.0
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if parse_progress(line, state) and time.monotonic() - last_write > 1.0:
                write_status(status_path, state)
                last_write = time.monotonic()
        code = proc.wait()
    state["train_seconds"] = round(time.perf_counter() - started, 1)
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        state.update(state="failed", error=f"UniLab training exited with code {code}", log_tail=tail)
        write_status(status_path, state)
        return 1
    runs = sorted(
        (p for p in log_root.glob("*/*") if (p / "run_config.json").is_file()), key=lambda p: p.stat().st_mtime
    )
    if not runs:
        state.update(state="failed", error=f"training finished but no run directory appeared under {log_root}")
        write_status(status_path, state)
        return 1
    state["run_dir"] = str(runs[-1])
    if not args.export_to:
        state["state"] = "done"
        write_status(status_path, state)
        return 0
    state["state"] = "exporting"
    write_status(status_path, state)
    try:
        result = export(runs[-1], Path(args.export_to), args.name, args.robot, args.title, args.task)
    except SystemExit as exc:
        state.update(state="failed", error=f"export failed: {exc}")
        write_status(status_path, state)
        return 1
    state.update(result)
    state["state"] = "done"
    state["total_seconds"] = round(time.perf_counter() - started, 1)
    write_status(status_path, state)
    return 0


def _isolate() -> None:
    """Drop this file's folder from the import path: it holds robotforge's own unilab.py, which would
    shadow the real `unilab` package of the environment this script runs in."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]


def main(argv: list[str] | None = None) -> int:
    _isolate()
    parser = argparse.ArgumentParser(prog="unilab_tool", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    p_train = sub.add_parser("train")
    p_train.add_argument("--task", required=True)
    p_train.add_argument("--algo", default="ppo")
    p_train.add_argument("--sim", default="mujoco")
    p_train.add_argument("--iterations", type=int, required=True)
    p_train.add_argument("--num-envs", type=int, default=None)
    p_train.add_argument("--log-root", required=True)
    p_train.add_argument("--status", required=True)
    p_train.add_argument("--export-to", default=None)
    p_train.add_argument("--name", default="unilab_policy")
    p_train.add_argument("--robot", default=None)
    p_train.add_argument("--title", default=None)
    p_export = sub.add_parser("export")
    p_export.add_argument("--run", required=True, type=Path)
    p_export.add_argument("--out", required=True, type=Path)
    p_export.add_argument("--name", required=True)
    p_export.add_argument("--robot", default=None)
    p_export.add_argument("--title", default=None)
    p_export.add_argument("--task", default=None)
    p_record = sub.add_parser("record")
    p_record.add_argument("--run", required=True, type=Path)
    p_record.add_argument("--steps", type=int, default=30)
    p_record.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "train":
        return train(args)
    if args.command == "info":
        result = info()
    elif args.command == "export":
        result = export(args.run, args.out, args.name, args.robot, args.title, args.task)
    else:
        result = record(args.run, args.steps, args.out)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
