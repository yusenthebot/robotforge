"""The algorithm layer: tasks, the Gym-style environment, classic control, sampling MPC, comparisons."""

import copy
import json
import sys
import types

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.control import (
    TASKS,
    Box,
    TaskError,
    compare_methods,
    make_controller,
    make_env,
    make_task,
    task_world,
)
from robotforge.control.actuation import TorqueMap
from robotforge.control.checks import EnvCheckError, check_env
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


_ACTUATED_CHAIN = """
<mujoco>
  <worldbody>
    <body><joint name="a" axis="0 1 0" range="-2 2"/><geom size="0.05"/>
      <body pos="0 0 -0.2"><joint name="b" axis="0 1 0"/><geom size="0.05"/>
        <body pos="0 0 -0.2"><joint name="c" axis="1 0 0"/><geom size="0.05"/></body></body></body>
  </worldbody>
  <actuator>{}</actuator>
</mujoco>
"""


@pytest.mark.parametrize(
    "actuators",
    [
        # servos only (a library robot), motors only (raw Menagerie motors), and a mix of the two
        '<position joint="a" kp="80" kv="2" ctrlrange="-2 2"/><position joint="b" kp="60" gear="2"/>'
        '<position joint="c" kp="40" ctrlrange="-1 1"/>',
        '<motor joint="a" gear="3" ctrlrange="-5 5"/><motor joint="b"/><motor joint="c" gear="0.5"/>',
        '<position joint="a" kp="80"/><motor joint="b" gear="2" ctrlrange="-5 5"/><velocity joint="c" kv="3"/>',
        # an affine gain: the general formula, no shortcut
        '<general joint="a" gaintype="affine" gainprm="50 -10 0" biastype="affine" biasprm="0 -50 -1"/>'
        '<position joint="b" kp="60"/><motor joint="c"/>',
    ],
)
def test_torque_map_shortcut_gives_the_general_formulas_numbers(actuators):
    """TorqueMap works the constant parts out once for fixed gains (learned walkers call it every physics step).
    Its ctrl must stay bit for bit what the general formula gives, or robots driven through it would move
    differently."""
    model = mujoco.MjModel.from_xml_string(_ACTUATED_CHAIN.format(actuators))
    data = mujoco.MjData(model)
    tm = TorqueMap(model, np.arange(model.nu))
    rng = np.random.default_rng(0)
    for _ in range(50):
        data.qpos[:] = rng.uniform(-1.5, 1.5, model.nq)
        data.qvel[:] = rng.normal(0.0, 3.0, model.nv)
        torque = rng.normal(0.0, 4.0, model.nu)
        length, velocity = tm.gear * data.qpos[tm.qadr], tm.gear * data.qvel[tm.dadr]
        gain = tm.gain[:, 0] + np.where(tm.affine_gain, tm.gain[:, 1] * length + tm.gain[:, 2] * velocity, 0)
        bias = np.where(tm.has_bias, tm.bias[:, 0] + tm.bias[:, 1] * length + tm.bias[:, 2] * velocity, 0.0)
        safe_gain = np.where(np.abs(gain) > 1e-9, gain, 1e-9)
        want = np.clip((torque / tm.gear - bias) / safe_gain, tm.scale.low, tm.scale.high)
        np.testing.assert_array_equal(tm.ctrl(data, torque), want)


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


@pytest.mark.parametrize("task", ["press", "wipe", "insert"])
def test_contact_tasks_have_no_gym_env_and_no_training(task):
    # Their reward needs a controller driving the plan: stepped with actions alone, an env would pay out zero for ever.
    with pytest.raises(TaskError, match="run_controller"):
        make_env(task)
    from robotforge.training import envs

    with pytest.raises(TaskError, match="run_controller"):
        envs.env_spec(task)


@pytest.mark.parametrize("mode", ["direct", "residual"])
@pytest.mark.parametrize("task", [t for t in TASKS if t not in ("press", "wipe", "insert")])
def test_env_passes_the_gym_contract_check(task, mode):
    env = make_env(task, episode_seconds=0.6, action_mode=mode)
    check_env(env, steps=80)
    env.reset(seed=1)
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert "is_success" in info
        if terminated or truncated:
            break
    assert steps <= env.max_steps and (terminated or steps == env.max_steps)


def test_check_env_catches_contract_breaks():
    env = make_env("balance")
    real_step = env.step

    def float64_obs(action):
        obs, *rest = real_step(action)
        return (obs.astype(np.float64), *rest)

    env.step = float64_obs
    with pytest.raises(EnvCheckError, match="dtype"):
        check_env(env)


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


def test_env_renders_rgb_frames(needs_gl):
    env = make_env("drive_to", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3 and frame.std() > 5


def test_gymnasium_wrapper_is_optional_and_thin(monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", None)
    with pytest.raises(ImportError, match="needs the gymnasium package, which is not installed: pip install gymnasium"):
        as_gymnasium(make_env("balance"))
    assert INSTALL_HINT.startswith("as_gymnasium needs the gymnasium package")

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
    server = build_server(pg, 0)
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
        assert not any("earlier run" in e for e in out["events"])
        again = await client.call_tool(
            "run_controller", {"robot": "pole", "task": "balance", "method": "lqr", "seconds": 1}
        )
        assert any("starts from where" in e for e in json.loads(again.content[0].text)["events"])

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


# ---- guards found in review ------------------------------------------------------------------------


def test_mpc_refuses_settings_that_would_stall_the_server():
    model, robot = task_world("stand")
    task = make_task("stand", model, robot)
    with pytest.raises(TaskError, match="physics steps per replan"):
        make_controller("mpc", task, model, _home(model), {"samples": 1024, "horizon": 2.0})
    with pytest.raises(TaskError, match="mpc settings: replan_hz"):
        make_controller("mpc", task, model, _home(model), {"replan_hz": 1000})
    with pytest.raises(TaskError, match="mpc settings: samples"):
        make_controller("mpc", task, model, _home(model), {"samples": "lots"})


@pytest.mark.parametrize("method", ["mpc", "mppi"])
def test_mpc_never_follows_a_rollout_that_blew_up(method, monkeypatch):
    from robotforge.control import mpc as mpc_module

    model, robot = task_world("balance")
    data = _home(model)
    data.qpos[1] = 0.3  # tilted: every honest future has a cost
    mujoco.mj_forward(model, data)
    task = make_task("balance", model, robot)
    controller = make_controller(method, task, model, data, {"seed": 0})
    real_pool = mpc_module._rollout_pool

    class Exploding:
        """Rollouts where every sample but the first one diverged and was reset by MuJoCo."""

        def __init__(self, pool):
            self.pool = pool

        def rollout(self, *args, **kwargs):
            state, sensordata = self.pool.rollout(*args, **kwargs)
            layout = controller.layout
            state[1:, :, layout.qpos] = 0.0  # perfectly upright: the cheapest possible future...
            state[1:, :, layout.qvel] = 0.0
            state[1:, :, layout.time] = np.arange(state.shape[1]) * controller.dt  # ...but the clock restarted
            return state, sensordata

    monkeypatch.setattr(mpc_module, "_rollout_pool", lambda n: Exploding(real_pool(n)))
    controller.update(model, data)  # first plan: nominal = hold the current command
    np.testing.assert_allclose(controller.nominal, 0.0)
    assert np.isfinite(controller.best_cost)

    class AllBad(Exploding):
        def rollout(self, *args, **kwargs):
            state, sensordata = super().rollout(*args, **kwargs)
            state[0, :, controller.layout.time] = 0.0
            return state, sensordata

    monkeypatch.setattr(mpc_module, "_rollout_pool", lambda n: AllBad(real_pool(n)))
    before = controller.nominal.copy()
    controller.plan(data)
    assert controller.best_cost == np.inf
    assert np.all(np.isfinite(controller.nominal))
    np.testing.assert_allclose(controller.nominal, before)


@pytest.fixture
def quiet_mujoco():
    """Collect MuJoCo's warnings instead of letting it append them to MUJOCO_LOG.TXT in the repo."""
    previous, seen = mujoco.get_mju_user_warning(), []
    mujoco.set_mju_user_warning(seen.append)
    yield seen
    mujoco.set_mju_user_warning(previous)


def test_env_ends_the_episode_when_the_simulation_diverges(quiet_mujoco):
    env = make_env("reach")
    env.reset(seed=0)
    target = env.task.target.copy()
    env.data.qvel[:] = 1e12  # guaranteed blow-up on the next step; MuJoCo resets the state itself
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    assert terminated and not truncated and info["unstable"]
    assert reward < 0 and np.all(np.isfinite(obs)) and "unstable" in quiet_mujoco[0]
    np.testing.assert_allclose(obs[-6:-3], target, atol=1e-6)  # the goal survived the recovery
    marker = env.model.body_mocapid[env.model.body("goal_marker").id]
    np.testing.assert_allclose(env.data.mocap_pos[marker], target)


def test_comparison_marks_a_diverged_run_as_failed(quiet_mujoco):
    model, robot = task_world("balance")
    data = _home(model)
    task = make_task("balance", model, robot)
    data.qvel[:] = 1e12
    result = run(task, model, data, "lqr", 0.5)
    assert result.failed and any("unstable" in e for e in result.events)


def test_task_and_method_params_must_be_finite_and_sensible():
    model, robot = task_world("reach")
    with pytest.raises(TaskError, match="finite"):
        make_task("reach", model, robot, {"target": [0.3, float("nan"), 0.2]})
    task = make_task("reach", model, robot, {"target": [0.25, 0.1, 0.15]})
    with pytest.raises(TaskError, match="omega"):
        make_controller("pd", task, model, _home(model), {"omega": -5})
    cp_model, cp = task_world("balance")
    balance = make_task("balance", cp_model, cp)
    with pytest.raises(TaskError, match="could not find gains"):
        make_controller("lqr", balance, cp_model, _home(cp_model), {"q_weights": [0, 0, 0, 0]})
    with pytest.raises(TaskError, match="robot settings"):
        task_world("reach", robot_params={"dof": 99})


def test_random_reach_targets_use_unlimited_joints_fully():
    model, robot = task_world("reach")
    yaw = model.joint(f"{robot}/yaw").id
    model.jnt_limited[yaw] = 0
    model.jnt_range[yaw] = [0.0, 0.0]  # what an unlimited joint looks like
    task = make_task("reach", model, robot)
    rng = np.random.default_rng(0)
    headings = [np.arctan2(*task.random_target(model.qpos0.copy(), rng)[1::-1]) for _ in range(20)]
    assert np.ptp(headings) > 2.0  # targets all around the base, not only straight ahead


def test_live_goals_without_coordinates_are_picked_near_the_robot():
    model, robot = task_world("drive_to")
    data = _home(model)
    task = make_task("drive_to", model, robot)
    data.qpos[task.qa : task.qa + 2] = [3.0, -2.0]
    task.begin(data)
    assert np.linalg.norm(task.goal - [3.0, -2.0]) < 1.25


def test_comparisons_check_every_method_and_the_push_before_running():
    with pytest.raises(TaskError, match="ik does not fit the balance task"):
        compare_methods("balance", ["lqr", "ik"], seconds=1.0)
    with pytest.raises(TaskError, match="push_at"):
        compare_methods("balance", ["lqr"], seconds=1.0, push_at=1.0)


def test_comparison_respects_a_computer_time_budget():
    comparison = compare_methods("stand", ["mpc", "hold"], seconds=30.0, wall_budget=0.3)
    assert comparison.results[0].stopped_early and comparison.results[0].sim_seconds < 30.0
    assert comparison.skipped == ["hold"]
    assert "out of computer time" in json.dumps(comparison.to_dict())


@pytest.mark.anyio
async def test_mcp_run_controller_survives_resets_and_slow_machines(tmp_path, monkeypatch):
    import asyncio

    from robotforge import mcp_server

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("pole", template="cartpole")
    server = build_server(pg, 0)
    async with Client(server) as client:
        lqr = {"robot": "pole", "task": "balance", "method": "lqr"}
        bad = await client.call_tool("run_controller", {**lqr, "seconds": 2, "push_at": 3})
        assert bad.is_error and "push_at" in bad.content[0].text

        async def reset_mid_run():
            while not any(k.startswith("__monitor__") for k in pg.session.controllers):
                await asyncio.sleep(0)
            for _ in range(3):
                await asyncio.sleep(0)
            pg.session.reset()

        resetter = asyncio.create_task(reset_mid_run())
        reset = await client.call_tool("run_controller", {**lqr, "seconds": 5})
        await resetter
        assert reset.is_error and "reset during the run" in reset.content[0].text

        monkeypatch.setattr(mcp_server, "FAST_WALL_LIMIT", 0.0)
        slow = await client.call_tool("run_controller", {**lqr, "seconds": 5})
        out = json.loads(slow.content[0].text)
        assert "computer time" in out["stopped_early"] and out["sim_seconds"] < 5
        assert out["keeps_running"] is True and out["release_with"] == "stop(robot='pole')"
        assert not any(k.startswith("__monitor__") for k in pg.session.controllers)
