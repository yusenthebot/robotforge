"""The training environment factory: any robot, any task, and an observation built from the robot's sensors."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robotforge.control import TaskError
from robotforge.playground import Playground
from robotforge.sim.session import SimSession
from robotforge.training import EnvSpec, build_env, describe, env_spec, make_env

ARM = {"name": "arm", "template": "arm"}


def _arm_with(*sensors: dict) -> dict:
    return {**ARM, "sensors": list(sensors)}


def test_a_robot_without_sensors_keeps_the_tasks_own_observation():
    env = make_env("reach", robot=ARM)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    layout = env.layout
    assert [b["source"] for b in layout] == ["task"]
    assert len(layout[0]["labels"]) == obs.shape[0]


def test_adding_sensors_adds_their_channels_to_the_observation():
    plain = make_env("reach", robot=ARM)
    n = plain.observation_space.shape[0]
    env = make_env("reach", robot=_arm_with({"type": "encoders"}, {"type": "imu"}))
    obs, _ = env.reset(seed=0)
    sources = {b["source"]: b for b in env.layout}
    assert set(sources) == {"task", "encoders", "imu"}
    assert obs.shape[0] == n + len(sources["encoders"]["labels"]) + len(sources["imu"]["labels"])
    assert obs.shape == env.observation_space.shape
    spans = sorted(tuple(b["slice"]) for b in env.layout)
    assert spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert spans[-1][1] == obs.shape[0]
    assert np.isfinite(obs).all()
    # the encoder block holds the arm's joint angles, which the task observation also starts with
    start = sources["encoders"]["slice"][0]
    assert np.allclose(obs[start : start + 5], obs[:5], atol=1e-3)


def test_a_lidar_is_shrunk_to_bins_of_the_range():
    env = make_env("reach", robot=_arm_with({"type": "lidar", "range": 4.0}), lidar_bins=8)
    obs, _ = env.reset(seed=0)
    block = next(b for b in env.layout if b["kind"] == "lidar")
    ranges = obs[block["slice"][0] : block["slice"][1]]
    assert len(ranges) == 8 and ranges.min() >= 0.0 and ranges.max() <= 1.0


def test_sensors_mode_is_the_goal_plus_the_sensors_only():
    env = make_env("reach", robot=_arm_with({"type": "encoders"}), observation="sensors")
    obs, _ = env.reset(seed=0)
    sources = [b["source"] for b in env.layout]
    assert sources == ["task goal", "encoders"]
    goal = env.layout[0]["labels"]
    assert goal and all(label.startswith(("target", "to target")) for label in goal)
    assert obs.shape[0] == len(goal) + len(env.layout[1]["labels"])


def test_sensors_mode_without_sensors_says_how_to_add_them():
    with pytest.raises(TaskError, match="add_sensor"):
        make_env("reach", robot=ARM, observation="sensors")


def test_describe_documents_both_spaces():
    env = make_env("reach", robot=_arm_with({"type": "imu"}))
    doc = describe(env)
    assert doc["observation"]["size"] == env.observation_space.shape[0]
    assert doc["action"]["size"] == len(doc["action"]["actuators"]) == env.action_space.shape[0]
    assert doc["action"]["range"] == [-1.0, 1.0]
    json.dumps(doc)  # plain data: it goes into status files and answers


def test_the_spec_is_plain_data_and_builds_the_same_environment():
    spec = env_spec("reach", robot=_arm_with({"type": "encoders"}))
    again = EnvSpec.model_validate(json.loads(json.dumps(spec.model_dump(mode="json"))))
    a, b = build_env(spec), build_env(again)
    assert a.observation_space.shape == b.observation_space.shape
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert np.allclose(oa, ob)


def test_the_environment_steps_with_sensors_and_is_reproducible_by_seed():
    def run() -> np.ndarray:
        env = make_env("reach", robot=_arm_with({"type": "encoders"}, {"type": "imu"}))
        env.reset(seed=5)
        rng = np.random.default_rng(1)
        obs = None
        for _ in range(20):
            obs, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
            assert np.isfinite(reward)
        return obs

    assert np.allclose(run(), run())


def test_a_robot_of_a_world_by_name_and_an_unknown_one_is_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("hand", template="arm")
    env = make_env("reach", world=pg.current_world(), robot="hand")
    assert env.task.robot.name == "hand"
    with pytest.raises(TaskError, match="ghost"):
        make_env("reach", world=pg.current_world(), robot="ghost")


def test_a_task_saved_with_the_world_is_an_environment_of_the_whole_world(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.load_scene(description="a small apartment with a kitchen, a living room and a bedroom", seed=0)
    pg.add_task("go to the bedroom")
    name = next(iter(pg.task_status()))
    robot = pg.current_world().tasks[0].robot
    pg.add_sensor(robot, {"type": "imu"})
    spec = env_spec(name, world=pg.current_world(), robot=robot)
    assert spec.scope == "world"
    env = build_env(spec)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and "imu" in {b["source"] for b in env.layout}
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isfinite(reward) and "success" in info


def test_an_unknown_task_lists_what_exists():
    with pytest.raises(TaskError, match="reach"):
        make_env("fly_to_the_moon", robot=ARM)


def test_a_reach_target_with_jitter_moves_around_its_point_each_episode():
    point = np.array([0.22, 0.08, 0.14])
    env = make_env("reach", robot=ARM, task_params={"target": point.tolist(), "jitter": 0.04})
    seen = []
    for seed in range(6):
        env.reset(seed=seed)
        offset = env.task.target - point
        assert np.all(np.abs(offset) <= 0.04 + 1e-9)
        seen.append(tuple(np.round(offset, 6)))
    assert len(set(seen)) == 6  # a different spot each episode
    env.reset(seed=0)
    again = env.task.target.copy()
    env.reset(seed=0)
    assert np.allclose(env.task.target, again)  # and the same one for the same seed
    steady = make_env("reach", robot=ARM, task_params={"target": point.tolist()})
    steady.reset(seed=3)
    assert np.allclose(steady.task.target, point)


def test_a_depth_camera_is_a_few_cells_of_the_nearest_thing_in_view(needs_gl):
    robot = _arm_with({"type": "depth_camera", "width": 64, "height": 48, "range": 3.0})
    env = make_env("reach", robot=robot, depth_size=(4, 3))
    obs, _ = env.reset(seed=0)
    block = next(b for b in env.layout if b["kind"] == "camera")
    cells = obs[block["slice"][0] : block["slice"][1]]
    assert len(cells) == 12 and cells.min() >= 0.0 and cells.max() <= 1.0
    assert cells.min() < 1.0  # something is in view: the floor
    env.close()


def test_a_run_with_a_depth_camera_measures_its_speed_and_says_it_is_slow(needs_gl):
    from robotforge.training import api
    from robotforge.training.runs import policy_meta, speed_notes

    robot = _arm_with({"type": "depth_camera", "width": 32, "height": 24, "range": 3.0})
    spec = api.build_spec("reach", name="slow", robot=robot, algo="ars", steps=50_000, parallel=0)
    meta = policy_meta(spec)
    assert meta["slow_sensors"] and meta["steps_per_second"] > 0  # measured, never assumed
    note = speed_notes(spec, meta)[0]
    assert "draws a picture at every step" in note and "steps a second" in note
    plain = api.build_spec("reach", name="fast", robot=_arm_with({"type": "encoders"}), algo="ars", parallel=0)
    assert "slow_sensors" not in policy_meta(plain)  # nothing to measure, so no delay at the start


def test_a_color_camera_is_left_out_and_said_so():
    env = make_env("reach", robot=_arm_with({"type": "encoders"}, {"type": "camera"}))
    assert [b["source"] for b in env.layout] == ["task", "encoders"]
    left_out = describe(env)["observation"]["left_out"]
    assert "camera" in {x["sensor"] for x in left_out} and all("picture" in x["why"] for x in left_out)


def test_an_object_named_as_a_target_becomes_the_point_it_stands_at():
    from robotforge.design.world import World

    world = World.model_validate(
        {
            "name": "lab",
            "robots": [{"name": "arm", "template": "arm"}],
            "objects": [{"name": "cube", "shape": "box", "size": [0.04, 0.04, 0.04], "pos": [0.25, 0.1, 0.2]}],
        }
    )
    spec = env_spec("reach", world=world, robot="arm", task_params={"target": "cube"})
    assert spec.scope == "robot"
    point = spec.task_params["target"]
    assert point == pytest.approx([0.25, 0.1, 0.2], abs=0.02)
    env = build_env(spec)
    try:
        assert env.task.target == pytest.approx(point, abs=1e-6)
    finally:
        env.close()
    # a name that is no object of the world is left for the task to explain
    unknown = env_spec("reach", world=world, robot="arm", task_params={"target": "mug"})
    assert unknown.task_params["target"] == "mug"
