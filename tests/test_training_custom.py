"""A task a plugin writes as a reward function, trained by an algorithm from the registry on any robot."""

from __future__ import annotations

import numpy as np
import pytest

from robotforge.control import TaskError
from robotforge.extend import registry
from robotforge.extend.algorithms import task
from robotforge.training.custom import reward_task
from robotforge.training.envs import build_env, describe, env_spec
from robotforge.training.runs import make_run_spec, run_inline


def _keep_upright():
    @task("keep_upright", summary="hold the pole up", params={"limit": 0.8})
    def build(model, robot, params):
        return reward_task(
            model,
            robot,
            params,
            reward=lambda ctx: 1.0 - abs(ctx.joint("hinge")) - 0.01 * ctx.effort(),
            failed=lambda ctx: abs(ctx.joint("hinge")) > ctx.params["limit"],
            success=lambda ctx: abs(ctx.joint("hinge")) < 0.1,
            reset=lambda ctx, rng: ctx.set_joint("hinge", rng.uniform(-0.25, 0.25)),
            seconds=3,
            task_name="keep_upright",
        )


def test_a_reward_function_is_a_task_with_its_own_observation_and_documented_spaces():
    with registry.isolated():
        _keep_upright()
        env = build_env(env_spec("keep_upright", robot="cartpole", task_params={"limit": 0.5}))
        obs, _ = env.reset(seed=0)
        labels = env.task.observation_labels()
        assert len(labels) == obs.shape[0] and "hinge position" in labels
        assert describe(env)["task"] == "keep_upright"
        _, reward, terminated, _, _ = env.step(np.zeros(env.action_space.shape))
        assert np.isfinite(reward) and not terminated
        with pytest.raises(TaskError, match="planner"):
            env.task.cost(None)


def test_a_plugin_task_trains_with_a_registry_algorithm(tmp_path):
    with registry.isolated():
        _keep_upright()
        spec = env_spec("keep_upright", robot="cartpole")
        run = make_run_spec("upright", spec.model_dump(mode="json"), algos=["ars"], steps=20_000, parallel=0)
        doc = run_inline(run, tmp_path / "upright")
        final = doc["final"]
        # doing nothing lets the pole fall within a second or two; a trained policy holds it longer
        env = build_env(spec)
        env.reset(seed=10_000)
        idle = 0.0
        assert env.task.failed(env.data) is False
        for _ in range(env.max_steps):
            _, r, te, tr, _ = env.step(np.zeros(env.action_space.shape))
            idle += r
            if te or tr:
                break
        assert final["mean_return"] > idle


def test_an_unknown_task_name_lists_the_plugin_ones_too():
    with registry.isolated():
        _keep_upright()
        with pytest.raises(TaskError, match="keep_upright"):
            build_env(env_spec("keep_upwright", robot="cartpole"))
