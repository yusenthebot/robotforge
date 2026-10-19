"""A trained policy runs the way it trained: the same observation from the same sensors, on any model of the robot."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotforge.control import TaskError
from robotforge.control.rl import TrainedPolicy
from robotforge.sim.session import SimSession
from robotforge.training import play
from robotforge.training.envs import build_env, env_spec
from robotforge.training.runs import make_run_spec, run_inline

ROBOT = {"name": "arm", "template": "arm", "sensors": [{"type": "encoders"}, {"type": "imu"}]}


@pytest.fixture(scope="module")
def sensed(tmp_path_factory):
    env_dict = env_spec(
        "reach", robot=ROBOT, task_params={"target": [0.25, 0.1, 0.2]}, observation="sensors", episode_seconds=1.0
    ).model_dump(mode="json")
    folder = tmp_path_factory.mktemp("policies") / "sensed"
    run_inline(make_run_spec("sensed", env_dict, algos=["ars"], steps=3_000, parallel=0), folder)
    return env_dict, folder


def test_a_policy_trained_on_sensors_plays_exactly_as_it_trained(sensed):
    env_dict, folder = sensed
    policy = TrainedPolicy.load(folder)
    assert policy.net.obs_dim == build_env(env_dict).observation_space.shape[0]

    env = build_env(env_dict)
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs, *_ = env.step(policy.net(obs))

    twin = build_env(env_dict)  # a model of its own, as the live world's is
    data = mujoco.MjData(twin.model)
    home = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(twin.model, data, home)
    mujoco.mj_forward(twin.model, data)
    live = SimSession(model=twin.model, data=data)
    live.set_controller("arm", play.controller(policy, twin.model, data, "arm"))
    live.step(20 * env.substeps)
    assert np.allclose(data.qpos, env.data.qpos, atol=1e-6)


def test_a_policy_for_other_sensors_is_refused_with_what_it_expects(sensed):
    env_dict, folder = sensed
    policy = TrainedPolicy.load(folder)
    bare = build_env(
        env_spec("reach", robot={"name": "arm", "template": "arm"}, task_params={"target": [0.25, 0.1, 0.2]})
    )
    data = mujoco.MjData(bare.model)
    spec = play.spec_of(policy.meta)
    assert spec is not None and spec.observation == "sensors"
    with pytest.raises(TaskError, match="carries no sensor"):
        play.controller(policy, bare.model, data, "arm")
