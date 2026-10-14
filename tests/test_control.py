"""The algorithm layer: tasks, the Gym-style environment, classic control, sampling MPC, comparisons."""

import copy
import json
import sys
import types

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.control import TASKS, Box, TaskError, compare_methods, make_env, make_task, task_world
from robotforge.control.actuation import TorqueMap
from robotforge.control.classic import solve_dare
from robotforge.control.compare import run
from robotforge.control.gym_compat import INSTALL_HINT, as_gymnasium
from robotforge.control.mpc import interpolate
from robotforge.control.tasks import StateLayout
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


def _home(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


# ---- building blocks -----------------------------------------------------------------------


def test_state_layout_matches_mujoco_state_vectors():
    model, _ = task_world("stand")
    data = _home(model)
    data.qvel[:] = np.arange(model.nv) * 0.1
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    layout = StateLayout.of(model)
    assert layout.size == state.size
    np.testing.assert_array_equal(state[layout.qpos], data.qpos)
    np.testing.assert_array_equal(state[layout.qvel], data.qvel)


def test_box_space_behaves_like_gymnasium_box():
    box = Box(-1.0, 1.0, (3,), np.float32, seed=4)
    samples = np.array([box.sample() for _ in range(200)])
    assert samples.dtype == np.float32 and samples.shape == (200, 3)
    assert box.contains(samples[0]) and samples[0] in box
    assert not box.contains(np.array([2.0, 0, 0], np.float32))
    assert not box.contains(np.zeros(2, np.float32))
    box.seed(9)
    a = box.sample()
    box.seed(9)
    np.testing.assert_array_equal(a, box.sample())
    unbounded = Box(-np.inf, np.inf, (4,), np.float32)
    assert np.isfinite(unbounded.sample()).all()


def test_interpolation_passes_through_knots():
    knots = np.array([[[0.0], [1.0], [3.0]]])  # one sample, 3 knots over 1 s
    times = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    lin = interpolate(knots, times, 1.0, "linear")[0, :, 0]
    np.testing.assert_allclose(lin, [0.0, 0.5, 1.0, 3.0, 3.0])
    zero = interpolate(knots, times, 1.0, "zero")[0, :, 0]
    np.testing.assert_allclose(zero, [0.0, 0.0, 1.0, 3.0, 3.0])
    cubic = interpolate(knots, np.array([0.0, 0.5, 1.0]), 1.0, "cubic")[0, :, 0]
    np.testing.assert_allclose(cubic, [0.0, 1.0, 3.0])


def test_torque_map_inverts_position_servos_and_motors():
    model, robot = task_world("reach")
    data = _home(model)
    task = make_task("reach", model, robot)
    tm = TorqueMap(model, task.actuators)
    # A position servo applies no torque exactly when its target equals the current angle.
    np.testing.assert_allclose(tm.ctrl(data, np.zeros(len(task.actuators))), data.qpos[task.qadr], atol=1e-12)
    data.ctrl[task.actuators] = tm.ctrl(data, np.full(len(task.actuators), 0.5))
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force[task.actuators], 0.5, atol=1e-9)

    cp_model, cp = task_world("balance")
    cp_data = _home(cp_model)
    balance = make_task("balance", cp_model, cp)
    cp_data.ctrl[balance.actuators] = TorqueMap(cp_model, balance.actuators).ctrl(cp_data, np.array([3.0]))
    mujoco.mj_forward(cp_model, cp_data)
    assert cp_data.qfrc_actuator[balance.xd_adr] == pytest.approx(3.0)


def test_dare_solution_satisfies_the_riccati_equation():
    rng = np.random.default_rng(0)
    A = np.eye(4) + 0.01 * rng.standard_normal((4, 4))
    B = 0.01 * rng.standard_normal((4, 2))
    Q, R = np.eye(4), 0.1 * np.eye(2)
    P = solve_dare(A, B, Q, R)
    rhs = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
    np.testing.assert_allclose(P, rhs, rtol=1e-6, atol=1e-6)


def test_tasks_explain_what_does_not_fit():
    model, robot = task_world("reach")
    with pytest.raises(TaskError, match="no robot named 'cat'"):
        make_task("reach", model, "cat")
    with pytest.raises(TaskError, match="cart-pole"):
        make_task("balance", model, robot)
    with pytest.raises(TaskError, match="unknown task"):
        make_task("fly", model, robot)
    with pytest.raises(TaskError, match="tolerance"):
        make_task("reach", model, robot, {"tolerance": -1})
    with pytest.raises(TaskError, match="lqr does not fit the reach task; for reach use one of: ik, pd"):
        compare_methods("reach", ["lqr"], 0.1)


# ---- Gym-style environment contract -------------------------------------------------------------


@pytest.mark.parametrize("task", list(TASKS))
def test_env_follows_the_gym_api(task):
    env = make_env(task, episode_seconds=0.5)
    assert env.action_space.dtype == np.float32 and env.observation_space.dtype == np.float32
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    obs, info = env.reset(seed=1)
    assert isinstance(info, dict) and obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape and obs in env.observation_space
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
        assert isinstance(reward, float) and np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool) and "success" in info
        if terminated or truncated:
            break
    assert steps <= env.max_steps and (terminated or steps == env.max_steps)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(env.action_space.shape[0] + 1))
    with pytest.raises(ValueError, match="NaN"):
        env.step(np.full(env.action_space.shape, np.nan))


def _rollout(env, seed: int, actions: np.ndarray) -> tuple[np.ndarray, list[float]]:
    obs, _ = env.reset(seed=seed)
    trace, rewards = [obs], []
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        trace.append(obs)
        rewards.append(r)
        if term or trunc:
            break
    return np.array(trace), rewards


def test_env_is_deterministic_for_a_seed():
    env_a, env_b = make_env("reach"), make_env("reach")
    actions = np.random.default_rng(0).uniform(-1, 1, (30, env_a.action_space.shape[0]))
    obs_a, rew_a = _rollout(env_a, 7, actions)
    obs_b, rew_b = _rollout(env_b, 7, actions)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert rew_a == rew_b
    other, _ = _rollout(env_a, 8, actions[:1])
    assert not np.allclose(other[0], obs_a[0])  # a different seed gives a different random target


def test_balance_env_terminates_when_the_pole_falls_and_residual_mode_keeps_it_up():
    env = make_env("balance", episode_seconds=4.0)
    env.reset(seed=0)
    for _ in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(np.zeros(1))  # no push at all: the pole falls
        if terminated or truncated:
            break
    assert terminated

    residual = make_env("balance", action_mode="residual", episode_seconds=4.0)
    residual.reset(seed=0)
    total = 0.0
    for _ in range(residual.max_steps):
        _, reward, terminated, truncated, info = residual.step(np.zeros(1))  # zero correction = pure LQR
        total += reward
        if terminated or truncated:
            break
    assert truncated and not terminated and info["success"]
    assert total == pytest.approx(2.0 * 4.0, rel=0.05)  # alive bonus x seconds, almost no cost


def test_env_renders_rgb_frames():
    env = make_env("drive_to", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3 and frame.std() > 5


def test_gymnasium_wrapper_is_optional_and_thin(monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", None)
    with pytest.raises(ImportError, match=r"robotforge\[rl\]"):
        as_gymnasium(make_env("balance"))
    assert "pip install" in INSTALL_HINT

    class FakeEnv:  # the parts of gymnasium.Env the wrapper relies on
        metadata: dict = {}

        def reset(self, *, seed=None, options=None):
            self.seeded = seed

    class FakeBox:
        def __init__(self, low, high, dtype):
            self.low, self.high, self.dtype = low, high, dtype

    gym = types.ModuleType("gymnasium")
    gym.Env = FakeEnv
    gym.spaces = types.SimpleNamespace(Box=FakeBox)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", gym.spaces)
    wrapped = as_gymnasium(make_env("balance"))
    assert isinstance(wrapped, FakeEnv) and wrapped.action_space.dtype == np.float32
    obs, _ = wrapped.reset(seed=3)
    assert wrapped.seeded == 3 and obs.shape == (4,)
    assert len(wrapped.step(np.zeros(1, np.float32))) == 5


# ---- classic control ------------------------------------------------------------------------------


def test_lqr_balances_the_cartpole_and_recovers_from_a_push():
    model, robot = task_world("balance")
    data = _home(model)
    task = make_task("balance", model, robot)
    task.reset(data, np.random.default_rng(0))
    result = run(task, model, data, "lqr", 5.0, push_at=2.0, push_strength=1.5)
    assert result.controller["stable"]
    assert not result.failed and result.success
    assert abs(result.report["pole_angle_deg"]) < 2 and abs(result.report["cart_position_m"]) < 0.1


def test_arm_pd_with_gravity_compensation_beats_plain_servo_ik():
    model, robot = task_world("reach")
    start = _home(model)
    target = {"target": [0.25, 0.1, 0.15]}
    ik = run(make_task("reach", model, robot, target), model, copy.copy(start), "ik", 3.0)
    pd = run(make_task("reach", model, robot, target), model, copy.copy(start), "pd", 3.0)
    cart = run(make_task("reach", model, robot, target), model, copy.copy(start), "cartesian", 3.0)
    assert pd.report["distance_m"] < 0.005 and cart.report["distance_m"] < 0.01
    assert ik.report["distance_m"] > 2 * pd.report["distance_m"]  # servos sag under gravity


def test_rover_drives_to_a_goal():
    model, robot = task_world("drive_to")
    result = run(make_task("drive_to", model, robot, {"goal": [0.8, 0.4]}), model, _home(model), "pd", 6.0)
    assert result.success and result.report["distance_m"] < 0.05


# ---- sampling MPC ---------------------------------------------------------------------------------


def test_mpc_reaches_within_a_centimeter():
    model, robot = task_world("reach")
    task = make_task("reach", model, robot, {"target": [0.3, -0.1, 0.2]})
    result = run(task, model, _home(model), "mpc", 3.0)
    assert result.report["distance_m"] < 0.01
    assert result.first_success_at is not None and result.first_success_at < 2.0
    assert result.controller["plans"] == 150 and result.controller["plan_hz"] > 0


def test_mpc_keeps_a_quadruped_standing_through_a_shove_that_topples_holding_still():
    model, robot = task_world("stand")
    start = _home(model)
    hold = run(make_task("stand", model, robot), model, copy.copy(start), "hold", 3.5, push_at=1.5, push_strength=2.0)
    mpc = run(make_task("stand", model, robot), model, copy.copy(start), "mpc", 3.5, push_at=1.5, push_strength=2.0)
    assert hold.failed
    assert not mpc.failed and mpc.success


def test_mppi_balances_the_cartpole():
    model, robot = task_world("balance")
    data = _home(model)
    task = make_task("balance", model, robot)
    task.reset(data, np.random.default_rng(1))
    result = run(task, model, data, "mppi", 3.0)
    assert not result.failed


def test_comparison_table_lists_every_method():
    comparison = compare_methods("balance", ["lqr", "pd", "none"], seconds=2.0)
    rows = comparison.table.splitlines()
    assert rows[0].startswith("| method") and len(rows) == 2 + 3
    by_method = {r.method: r for r in comparison.results}
    assert by_method["none"].failed and not by_method["lqr"].failed
    assert by_method["lqr"].reward > by_method["none"].reward
    assert json.dumps(comparison.to_dict())


# ---- MCP tools -------------------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_tools_run_and_compare_controllers(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("pole", template="cartpole", pos=[0, 1.0])
    pg.add_robot("arm", template="arm")
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        listed = json.loads((await client.call_tool("list_algorithms", {})).content[0].text)
        assert {t["name"] for t in listed["tasks"]} == set(TASKS) and "lqr" in listed["algorithms"]

        ran = await client.call_tool(
            "run_controller", {"robot": "pole", "task": "balance", "method": "lqr", "seconds": 3, "push_at": 1.0}
        )
        out = json.loads(ran.content[0].text)
        assert out["success"] and out["failed_at"] is None and out["shown_live"] is False
        assert "shoved the pole" in out["events"][0]
        assert pg.session.controllers.keys() == {"pole"}  # keeps balancing, scoring removed

        reach = await client.call_tool(
            "run_controller",
            {
                "robot": "arm",
                "task": "reach",
                "method": "mpc",
                "seconds": 3,
                "task_params": {"target": [0.25, 0.1, 0.2]},
            },
        )
        assert json.loads(reach.content[0].text)["result"]["distance_m"] < 0.01

        before = pg.session.data.time
        args = {"robot": "arm", "task": "reach", "methods": ["ik", "pd"], "seconds": 2}
        compared = await client.call_tool("compare_controllers", {**args, "task_params": {"target": [0.2, -0.1, 0.15]}})
        table = json.loads(compared.content[0].text)
        assert table["best"] == "pd" and "| ik" in table["table"]
        assert pg.session.data.time == before  # comparisons run off-screen

        bad = await client.call_tool("run_controller", {"robot": "arm", "task": "balance", "method": "lqr"})
        assert bad.is_error and "cart-pole" in bad.content[0].text


def test_gait_preview_predicts_the_gait_without_disturbing_it():
    from robotforge.control.classic import GaitBase
    from robotforge.control.methods import _gait

    model, robot = task_world("walk")
    data = _home(model)
    task = make_task("walk", model, robot)
    base = GaitBase(task, _gait(task, model, {}))
    base.update(model, data)
    phase = base.gait._phase
    ahead = base.preview(model, data, np.array([0.0, 0.01, 0.02]))
    assert base.gait._phase == phase  # the live gait did not move on
    for i in range(1, 3):  # stepping the real gait by the same times gives the same commands
        data.time += 0.01
        base.update(model, data)
        np.testing.assert_allclose(data.ctrl[task.actuators], ahead[i], atol=1e-9)


def test_mppi_on_top_of_the_gait_walks():
    model, robot = task_world("walk")
    result = run(make_task("walk", model, robot), model, _home(model), "mppi", 3.0)
    assert not result.failed and result.report["distance_m"] > 0.6
    assert result.controller["on_top_of"]["behavior"] == "walk"


def test_ars_learns_to_balance_with_numpy_only(tmp_path):
    from robotforge.control.ars import LinearPolicy, evaluate, train_ars

    env = make_env("balance", episode_seconds=4.0)
    policy, log = train_ars(env, iterations=15, seed=2)
    returns = [evaluate(env, policy, 100 + s)[0] for s in range(3)]
    assert min(returns) > 0.95 * 2.0 * 4.0  # alive the whole episode with little cost
    assert log.iterations[-1]["mean_return"] > log.iterations[0]["mean_return"]
    policy.save(tmp_path / "pole.npz")
    again = LinearPolicy.load(tmp_path / "pole.npz")
    obs, _ = env.reset(seed=5)
    np.testing.assert_array_equal(policy(obs), again(obs))


def test_ars_method_trains_then_balances_through_a_push():
    model, robot = task_world("balance")
    data = _home(model)
    task = make_task("balance", model, robot)
    task.reset(data, np.random.default_rng(3))
    result = run(task, model, data, "ars", 4.0, {"iterations": 15, "seed": 2}, push_at=2.0)
    assert not result.failed and result.controller["iterations"] == 15
