"""Reinforcement learning tier 1: policy folders, the numpy runtime, training jobs and their MCP tools.

The first half needs no RL libraries (CI installs only the core). The second half trains for real
with the optional `rl` extra and skips cleanly without it.
"""

import copy
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.control import TaskError, compare_methods, make_env, make_task
from robotforge.control.rl import JOBS, MlpPolicy, TrainedPolicy, list_policies, make_spec, store
from robotforge.control.rl.jobs import INSTALL_HINT, Jobs
from robotforge.control.rl.runtime import attach_policy
from robotforge.control.rl.spec import describe_env
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

HAVE_RL = all(importlib.util.find_spec(m) is not None for m in ("gymnasium", "stable_baselines3", "torch"))
needs_rl = pytest.mark.skipif(not HAVE_RL, reason="the optional rl extra is not installed (uv sync --extra rl)")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _random_net(obs_dim: int, act_dim: int, rng: np.random.Generator, squash: str = "clip") -> MlpPolicy:
    sizes = [obs_dim, 16, 16, act_dim]
    weights = [rng.normal(0, 0.3, (b, a)) for a, b in zip(sizes, sizes[1:], strict=False)]
    biases = [rng.normal(0, 0.1, b) for b in sizes[1:]]
    return MlpPolicy(weights, biases, "tanh", squash, rng.normal(0, 0.1, obs_dim), rng.uniform(0.5, 2, obs_dim))


def _fake_policy(folder, task: str, net: MlpPolicy, action_mode: str = "direct", **meta) -> None:
    """A policy folder as training leaves it, with a given network (no training needed)."""
    folder.mkdir(parents=True)
    spec = make_spec(folder.name, task, {"name": meta.pop("robot", "x"), "template": meta.pop("template")})
    facts = describe_env(spec.model_copy(update={"action_mode": action_mode}))
    store.write_json(
        folder / store.POLICY_FILE,
        {"name": folder.name, "task": task, "algo": "ppo", "action_mode": action_mode, **facts, **meta},
    )
    store.write_json(folder / store.STATUS_FILE, {"state": "done", "updated": time.time()})
    net.save(folder / store.WEIGHTS_FILE)


# ---- no RL libraries needed -----------------------------------------------------------------------


def test_numpy_policy_is_plain_math_and_survives_a_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    net = _random_net(4, 2, rng)
    obs = rng.normal(size=4)
    x = np.clip((obs - net.obs_mean) / np.sqrt(net.obs_var + 1e-8), -10, 10)
    for i, (w, b) in enumerate(zip(net.weights, net.biases, strict=True)):
        x = w @ x + b
        x = np.tanh(x) if i < 2 else np.clip(x, -1, 1)
    np.testing.assert_allclose(net(obs), x, rtol=1e-6)
    net.save(tmp_path / "p.npz")
    again = MlpPolicy.load(tmp_path / "p.npz")
    np.testing.assert_array_equal(again(obs), net(obs))
    (tmp_path / "bad.npz").write_bytes(b"not a policy")
    with pytest.raises(TaskError, match="could not read the trained network"):
        MlpPolicy.load(tmp_path / "bad.npz")


@pytest.mark.parametrize(("task", "mode"), [("balance", "direct"), ("walk", "residual"), ("reach", "residual")])
def test_a_trained_policy_runs_in_the_world_exactly_as_it_ran_in_training(tmp_path, task, mode):
    """Same start, same network: the live controller and the training env give the same trajectory."""
    env = make_env(task, action_mode=mode)
    net = _random_net(env.observation_space.shape[0], env.action_space.shape[0], np.random.default_rng(1))
    template = {"balance": "cartpole", "walk": "quadruped", "reach": "arm"}[task]
    _fake_policy(tmp_path / "p", task, net, mode, robot=env.task.robot.name, template=template)
    policy = TrainedPolicy.load(tmp_path / "p")

    obs, _ = env.reset(seed=3)
    live = copy.copy(env.data)
    task_obj = make_task(task, env.model, env.task.robot.name, env.task.params.model_dump())
    task_obj.target = getattr(env.task, "target", None)  # reach: the episode's random target
    session = SimSession(model=env.model, data=live)
    session.set_controller("robot", policy.controller(task_obj, env.model, live))
    for _ in range(40):
        obs, *_ = env.step(net(obs))
        session.step(env.substeps)
    np.testing.assert_allclose(live.qpos, env.data.qpos, atol=1e-9)
    assert abs(live.time - env.data.time) < 1e-9


def test_policy_methods_find_folders_and_refuse_what_does_not_fit(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    with pytest.raises(TaskError, match="no policy named 'nope'; none trained yet"):
        compare_methods("balance", ["lqr", "policy:nope"], seconds=1)
    with pytest.raises(TaskError, match="trained, not built on the spot"):
        compare_methods("balance", ["ppo"], seconds=1)
    with pytest.raises(TaskError, match="policy names use lowercase"):
        compare_methods("balance", ["policy:Bad Name"], seconds=1)

    net = _random_net(4, 1, np.random.default_rng(2))
    _fake_policy(store.policies_home(None) / "pole_net", "balance", net, robot="pole", template="cartpole")
    assert [p["name"] for p in list_policies()] == ["pole_net"]
    result = compare_methods("balance", ["lqr", "policy:pole_net"], seconds=1)
    assert [r.method for r in result.results] == ["lqr", "policy:pole_net"]
    assert result.results[1].controller["controller"] == "policy"

    arm = make_env("reach").task
    with pytest.raises(TaskError, match="trained for the balance task, not reach"):
        TrainedPolicy.load(store.policies_home(None) / "pole_net").check_fits(arm)


def test_training_explains_how_to_install_the_extra_when_it_is_missing(tmp_path, monkeypatch):
    from robotforge.control.rl import jobs

    monkeypatch.setattr(jobs, "missing_rl_modules", lambda: ["torch"])
    spec = make_spec("pole", "balance", {"name": "pole", "template": "cartpole"})
    with pytest.raises(TaskError, match=r"uv sync --extra rl.*missing: torch"):
        Jobs().start(spec, tmp_path / "pole")
    assert not (tmp_path / "pole").exists()
    assert "pip install 'robotforge[rl]'" in INSTALL_HINT


def test_job_status_notices_a_training_process_that_died(tmp_path):
    jobs = Jobs()
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    store.write_json(crashed / store.STATUS_FILE, {"state": "running", "updated": time.time()})
    (crashed / store.LOG_FILE).write_text("step 1\nSegmentation fault\n", encoding="utf-8")
    jobs.procs[crashed] = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    jobs.procs[crashed].wait()
    status = jobs.status(crashed)
    assert status["state"] == "failed" and "exited with code 3" in status["error"]
    assert status["traceback_tail"][-1] == "Segmentation fault"

    orphan = tmp_path / "orphan"  # started by an earlier server, then the computer slept
    orphan.mkdir()
    store.write_json(orphan / store.STATUS_FILE, {"state": "running", "updated": time.time() - 3600})
    assert "stopped without finishing" in jobs.status(orphan)["error"]
    assert store.read_json(orphan / store.STATUS_FILE)["state"] == "failed"  # recorded for everyone


def test_specs_take_task_defaults_and_reject_bad_settings():
    spec = make_spec("dog_walk", "walk", {"name": "dog", "template": "quadruped"}, minutes=3)
    assert spec.algo == "ppo" and spec.action_mode == "residual" and spec.push > 0 and spec.max_minutes == 3
    with pytest.raises(TaskError, match="algo is 'ppo' or 'sac'"):
        make_spec("x", "walk", {"name": "dog", "template": "quadruped"}, algo="dqn")
    with pytest.raises(TaskError, match="training settings"):
        make_spec("x", "walk", {"name": "dog", "template": "quadruped"}, timesteps=-5)
    with pytest.raises(TaskError, match="needs a cart-pole|has no 'slider'"):
        describe_env(make_spec("x", "balance", {"name": "dog", "template": "quadruped"}))
    facts = describe_env(spec)
    assert facts["base_method"] == "gait" and len(facts["actuators"]) == 12 and facts["control_dt"] == 0.02


@pytest.mark.anyio
async def test_mcp_runs_a_trained_policy_and_explains_missing_extras(tmp_path, monkeypatch):
    from robotforge.control.rl import jobs

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    world = pg.new_world("lab")
    pg.add_robot("pole", template="cartpole")
    net = _random_net(4, 1, np.random.default_rng(4))
    folder = store.policies_home(world["world_dir"]) / "pole_net"
    _fake_policy(folder, "balance", net, robot="pole", template="cartpole")
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"train_policy", "training_status", "stop_training", "list_policies"} <= names

        listed = json.loads((await client.call_tool("list_policies", {})).content[0].text)
        assert listed["policies"][0]["method"] == "policy:pole_net" and listed["policies"][0]["where"] == "world"
        algos = json.loads((await client.call_tool("list_algorithms", {})).content[0].text)
        assert "ppo" in algos["learning"] and algos["trained_policies"] == ["policy:pole_net (balance, done)"]

        ran = await client.call_tool(
            "run_controller", {"robot": "pole", "task": "balance", "method": "policy:pole_net", "seconds": 1}
        )
        out = json.loads(ran.content[0].text)
        assert out["controller"]["controller"] == "policy" and "trained neural network" in out["explain"]
        assert out["keeps_running"] is True

        wrong = {"robot": "pole", "task": "reach", "method": "policy:pole_net"}
        assert (await client.call_tool("run_controller", wrong)).is_error

        monkeypatch.setattr(jobs, "missing_rl_modules", lambda: ["stable_baselines3"])
        missing = await client.call_tool("train_policy", {"task": "balance", "robot": "pole"})
        assert missing.is_error and "uv sync --extra rl" in missing.content[0].text
        nobody = await client.call_tool("train_policy", {"task": "balance", "robot": "ghost"})
        assert nobody.is_error and "no robot named 'ghost'" in nobody.content[0].text


def test_scripts_can_hand_a_robot_to_a_trained_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    world = pg.new_world("lab")
    pg.add_robot("pole", template="cartpole")
    _fake_policy(
        store.policies_home(world["world_dir"]) / "pole_net",
        "balance",
        _random_net(4, 1, np.random.default_rng(5)),
        robot="pole",
        template="cartpole",
    )
    info = attach_policy(pg.session, "pole", "pole_net", pg.store.dir)
    assert info["policy"] == "pole_net" and info["controller"] == "policy"
    assert pg.session.controllers["pole"].task.robot.name == "pole"
    pg.run_fast(lambda: False, 0.5)  # the policy drives the cart
    assert abs(pg.session.data.ctrl[pg.session.controllers["pole"].task.actuators[0]]) > 0


# ---- with the rl extra: real training ------------------------------------------------------------


@needs_rl
def test_exported_networks_act_exactly_like_stable_baselines3():
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from robotforge.control.gym_compat import as_gymnasium
    from robotforge.control.rl.train import export_policy

    torch.manual_seed(0)
    norm = VecNormalize(DummyVecEnv([lambda: as_gymnasium(make_env("reach"))]))
    norm.obs_rms.mean[:] = np.random.default_rng(0).normal(size=norm.obs_rms.mean.shape)
    norm.obs_rms.var[:] = np.random.default_rng(1).uniform(0.2, 3.0, size=norm.obs_rms.var.shape)
    rng = np.random.default_rng(2)
    for algo in (PPO, SAC):
        model = algo("MlpPolicy", norm, policy_kwargs={"net_arch": [32, 32]}, device="cpu", seed=0)
        net = export_policy(model, norm)
        for _ in range(20):
            obs = rng.normal(0, 2, norm.observation_space.shape).astype(np.float32)
            expected, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
            np.testing.assert_allclose(net(obs), expected, atol=1e-5)
    norm.close()


@needs_rl
def test_a_tiny_training_job_runs_in_its_own_process_and_its_policy_compares(tmp_path):
    spec = make_spec("pole_tiny", "balance", {"name": "pole", "template": "cartpole"}, timesteps=4096, n_envs=2)
    folder = tmp_path / "pole_tiny"
    started = JOBS.start(spec, folder)
    assert started["actuators"] == ["cart"] and started["estimate_seconds"] > 0
    with pytest.raises(TaskError, match="still training"):
        JOBS.start(spec.model_copy(update={"name": "other"}), tmp_path / "other")
    status = JOBS.wait(folder)
    assert status["state"] == "done", status
    meta = json.loads((folder / store.POLICY_FILE).read_text(encoding="utf-8"))
    assert meta["timesteps"] >= 4096 and meta["final"]["episodes"] == 10 and "without_shoves" in meta["final"]
    assert {store.MODEL_FILE, store.NORM_FILE, store.WEIGHTS_FILE, store.LOG_FILE} <= {p.name for p in folder.iterdir()}
    assert not any(line.startswith("{") for line in (folder / store.LOG_FILE).read_text(encoding="utf-8").splitlines())
    comparison = compare_methods(
        "balance", ["lqr", "policy:pole_tiny"], seconds=2, method_params={"policy:pole_tiny": {"path": str(folder)}}
    )
    assert comparison.results[1].controller["policy"] == "pole_tiny"
    with pytest.raises(TaskError, match="already exists"):
        JOBS.start(spec, folder)


@needs_rl
@pytest.mark.anyio
async def test_mcp_trains_follows_and_stops_a_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("pole", template="cartpole")
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        before = pg.session.data.time
        started = json.loads(
            (await client.call_tool("train_policy", {"task": "balance", "robot": "pole", "timesteps": 2_000_000}))
            .content[0]
            .text
        )
        assert started["started"] == "pole_balance" and started["algo"] == "ppo" and "PPO" in started["explain"]
        assert pg.session.data.time == before  # nothing ran in the server's own simulation
        deadline = time.monotonic() + 120
        while True:
            status = json.loads(
                (await client.call_tool("training_status", {"name": "pole_balance", "wait_seconds": 1})).content[0].text
            )
            if status.get("timesteps", 0) > 0 or time.monotonic() > deadline:
                break
        assert status["state"] == "running", status
        stopped = json.loads((await client.call_tool("stop_training", {"name": "pole_balance"})).content[0].text)
        assert stopped["state"] == "stopped" and stopped["stopped_early"] == "stopped on request", stopped
        assert "run_controller" in stopped["next"]
        ran = await client.call_tool(
            "run_controller", {"robot": "pole", "task": "balance", "method": "policy:pole_balance", "seconds": 1}
        )
        assert not ran.is_error, ran.content[0].text
