"""Training metadata, playback and residual control share the registered task contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from robotforge.control.env import RobotEnv, task_world
from robotforge.control.rl import store
from robotforge.control.rl.runtime import TrainedPolicy, attach_policy
from robotforge.control.task_registry import bind_task, registered_task_name
from robotforge.control.tasks import TaskError, make_task
from robotforge.extend import REGISTRY, task
from robotforge.sim.session import SimSession
from robotforge.training import algorithms, play, trial
from robotforge.training.custom import RewardTask
from robotforge.training.envs import build_env, env_spec
from robotforge.training.policy import Shape
from robotforge.training.runs import make_run_spec, plan_for, policy_meta, run_inline
from robotforge.training.search import SearchSpec


@pytest.fixture
def plugin_tasks(monkeypatch):
    monkeypatch.setattr(REGISTRY, "items", dict(REGISTRY.items))

    @task("contract_reach", params={"target": [0.25, 0.0, 0.2]})
    def wrapped(model, robot, params):
        return make_task("reach", model, robot, params)

    @task("contract_reward")
    def reward(model, robot, params):
        return RewardTask(model, robot, params, reward=lambda ctx: 1.0)


def _save_policy(folder, meta, task):
    folder.mkdir(parents=True)
    shape = Shape(len(task.observation_labels()), len(task.actuators))
    shape.policy(shape.zeros(), np.zeros(shape.obs_dim), np.ones(shape.obs_dim)).save(folder / store.WEIGHTS_FILE)
    store.write_json(folder / store.POLICY_FILE, meta)
    return TrainedPolicy.load(folder)


@pytest.mark.parametrize("name,implementation", [("contract_reach", "reach"), ("contract_reward", "custom")])
def test_plugin_identity_survives_training_and_live_playback(plugin_tasks, tmp_path, name, implementation):
    spec = env_spec(name, robot={"name": "arm", "template": "arm"}, observation="task", episode_seconds=0.04)
    document = spec.model_dump(mode="json")
    folder = tmp_path / name
    run_inline(make_run_spec(name, document, algos=["ars"], steps=4, parallel=0, eval_episodes=1), folder)
    policy = TrainedPolicy.load(folder)
    assert policy.meta["task"] == policy.meta["env"]["task"] == name
    assert policy.meta["implementation_task"] == implementation
    env = build_env(spec)
    try:
        assert env.task.name == implementation
        assert registered_task_name(env.task) == name
        ctl = play.controller(policy, env.model, env.data, "arm")
        ctl.update(env.model, env.data)
        assert np.isfinite(env.data.ctrl).all()
        assert registered_task_name(ctl.task) == name
        if implementation == "reach":
            with pytest.raises(TaskError, match="contract_reach"):
                policy.check_fits(make_task("reach", env.model, "arm"))
    finally:
        env.close()


def test_env_task_identity_takes_precedence_over_legacy_implementation_metadata(plugin_tasks, tmp_path):
    spec = env_spec("contract_reach", robot={"name": "arm", "template": "arm"}, observation="task")
    env = build_env(spec)
    try:
        meta = policy_meta(make_run_spec("old_plugin", spec.model_dump(mode="json"), algos=["ars"]))
        meta["task"] = "reach"
        policy = _save_policy(tmp_path / "old_plugin", meta, env.task)
        policy.check_fits(env.task)
        assert registered_task_name(play.controller(policy, env.model, env.data, "arm").task) == "contract_reach"
    finally:
        env.close()


def test_attach_new_policy_merges_stored_settings_and_overrides(plugin_tasks, tmp_path, monkeypatch):
    spec = env_spec(
        "contract_reach",
        robot={"name": "arm", "template": "arm"},
        observation="task",
        task_params={"target": [0.25, 0.0, 0.2], "tolerance": 0.03},
    )
    env = build_env(spec)
    try:
        meta = policy_meta(make_run_spec("plugin_policy", spec.model_dump(mode="json"), algos=["ars"]))
        policy = _save_policy(tmp_path / "plugin_policy", meta, env.task)
        monkeypatch.setattr(store, "find_policy", lambda *args: policy.folder)
        live = SimSession(model=env.model, data=env.data)
        report = attach_policy(live, "arm", policy.name, task_params={"target": [0.2, 0.1, 0.2]})
        ctl = live.controllers["arm"]
        assert report["task"]["task"] == "contract_reach"
        assert ctl.task.params.tolerance == 0.03
        assert np.allclose(ctl.task.target, [0.2, 0.1, 0.2])
    finally:
        env.close()


def test_legacy_policy_roundtrip_and_attach_keep_saved_task_settings(tmp_path, monkeypatch):
    model, robot = task_world("reach")
    bound = make_task("reach", model, robot)
    policy = _save_policy(tmp_path / "legacy", {"task": "reach", "task_params": {"tolerance": 0.03}}, bound)
    policy.check_fits(bound)
    monkeypatch.setattr(store, "find_policy", lambda *args: policy.folder)
    live = SimSession(model=model, data=mujoco.MjData(model))
    attach_policy(live, robot, policy.name, task_params={"target": [0.2, 0.1, 0.2]})
    assert live.controllers[robot].task.params.tolerance == 0.03
    assert np.allclose(live.controllers[robot].task.target, [0.2, 0.1, 0.2])
    live.step(2)
    assert np.isfinite(live.data.ctrl).all()


@pytest.mark.parametrize("methods", [(), ("ars",), ("policy:other",)])
def test_residual_without_eligible_base_refuses_before_allocating_data(plugin_tasks, monkeypatch, methods):
    model, robot = task_world("reach")
    bound = bind_task("contract_reward", model, robot)
    bound.methods = methods
    create_data = Mock(side_effect=AssertionError("residual validation must precede allocation"))
    monkeypatch.setattr("robotforge.control.env.mujoco.MjData", create_data)
    with pytest.raises(TaskError, match="base controller"):
        RobotEnv(bound, action_mode="residual")
    create_data.assert_not_called()


@pytest.mark.parametrize("searching", [False, True])
def test_run_settings_resolve_aliases_and_refuse_unused_algorithm_keys(searching):
    search = SearchSpec(trials=2, all_settings=False) if searching else None
    spec = make_run_spec("alias", {}, algos=["ARS"], params={"ars": {"noise": 0.2}}, search=search)
    assert all(p.algo == "ars" and p.params["noise"] == 0.2 for p in plan_for(spec))
    for params in ({"cem": {"population": 8}}, {"typo": {}}, {"ars": {}, "ARS": {}}):
        with pytest.raises(TaskError):
            plan_for(spec.model_copy(update={"params": params}))


def test_search_only_fixes_explicit_settings():
    spec = make_run_spec(
        "search",
        {},
        algos=["ars"],
        params={"ARS": {"noise": 0.2}},
        search=SearchSpec(trials=3, space={"step_size": [0.3]}, all_settings=False),
    )
    assert all(p.params["noise"] == 0.2 and p.params["step_size"] == 0.3 for p in plan_for(spec))


@pytest.mark.parametrize("reason", ["params", "dependency"])
def test_segment_preflights_before_building_environment(tmp_path, monkeypatch, reason):
    monkeypatch.setattr(trial, "_load_plugins", lambda job: None)
    build = Mock(side_effect=AssertionError("must not build before preflight"))
    monkeypatch.setattr(trial, "build_env", build)
    params = {"unknown": 1} if reason == "params" else {}
    if reason == "dependency":
        monkeypatch.setattr(algorithms, "need", Mock(side_effect=TaskError("missing dependency")))
    with pytest.raises(TaskError):
        trial.run_segment({"trial_dir": str(tmp_path), "algo": "ars", "params": params, "env": {}, "budget": 1})
    build.assert_not_called()


def test_segment_closes_environment_when_context_initialization_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "_load_plugins", lambda job: None)
    env = SimpleNamespace(close=Mock())
    monkeypatch.setattr(trial, "build_env", lambda spec: env)
    monkeypatch.setattr(trial, "TrainContext", Mock(side_effect=ValueError("broken restored state")))
    with pytest.raises(ValueError, match="broken restored state"):
        trial.run_segment({"trial_dir": str(tmp_path), "algo": "ars", "env": {}, "budget": 1})
    env.close.assert_called_once_with()


def test_saved_world_policy_refuses_attachment_to_another_robot(tmp_path, monkeypatch):
    from robotforge.design.world import World

    world = World.model_validate(
        {
            "name": "two_robots",
            "robots": [
                {"name": "first", "template": "rover"},
                {"name": "second", "template": "rover", "pos": [2, 0]},
            ],
            "tasks": [{"name": "saved_goal", "kind": "go_to", "robot": "first"}],
        }
    )
    spec = env_spec("saved_goal", world=world, robot="first", observation="task")
    env = build_env(spec)
    try:
        meta = policy_meta(make_run_spec("saved", spec.model_dump(mode="json"), algos=["ars"]))
        policy = _save_policy(tmp_path / "saved", meta, env.task)
        monkeypatch.setattr(store, "find_policy", lambda *args: policy.folder)
        live = SimSession(model=env.model, data=env.data)
        with pytest.raises(TaskError, match="belongs to robot 'first'.*'second'"):
            attach_policy(live, "second", policy.name, world=world)
        assert not live.controllers
        result = attach_policy(live, "first", policy.name, world=world)
        assert result["task"]["task"] == "saved_goal"
        assert live.controllers["first"].task.robot.name == "first"
    finally:
        env.close()
