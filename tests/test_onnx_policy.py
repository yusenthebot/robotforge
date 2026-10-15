"""Learned policies: the policy card, observation building, joint mapping, PD replay, the ONNX runner,
walking the library Go2 with the shipped UniLab policy, and the UniLab bridge's own logic.

Observation building and joint mapping are plain numpy and always run. Tests that execute an ONNX
model need the optional extra (`uv sync --extra policy`) and skip without it. The fixtures:

- tests/fixtures/policies/echo_joint_pos.onnx: action = joint_pos_rel + a known ramp
  (scripts/make_policy_fixture.py), so every output can be predicted exactly.
- tests/fixtures/policies/unilab_go2_trace.npz: 40 steps of the shipped Go2 policy recorded inside
  UniLab's own environment (unilab_tool.py record): states and the observations UniLab built.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import quadruped
from robotforge.behaviors.manager import BehaviorError
from robotforge.bridge import policies, unilab, unilab_tool
from robotforge.bridge.observations import JointBinding, Observer
from robotforge.bridge.onnx_policy import PolicyWalker, onnxruntime_available
from robotforge.bridge.spec import PolicySpec, SpecError, load_spec, save_spec
from robotforge.playground import Playground
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub

FIXTURES = Path(__file__).parent / "fixtures"
GO2_XML = FIXTURES / "menagerie" / "unitree_go2" / "go2.xml"
TRACE = FIXTURES / "policies" / "unilab_go2_trace.npz"
ECHO = FIXTURES / "policies" / "echo_joint_pos.onnx"
needs_onnxruntime = pytest.mark.skipif(not onnxruntime_available(), reason="needs the optional extra: policy")


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES / "menagerie"))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(unilab.ENV_VAR, raising=False)


@pytest.fixture
def card() -> PolicySpec:
    return policies.load("go2_unilab_walk").spec


@pytest.fixture
def go2_model() -> mujoco.MjModel:
    """Menagerie's Go2 on its own (no world, no name prefix): the same joints and IMU site UniLab uses."""
    return mujoco.MjModel.from_xml_path(str(GO2_XML))


@pytest.fixture
def pg() -> Playground:
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "checker")
    playground.add_robot("go2", library="go2")
    return playground


def _write_card(directory: Path, spec: PolicySpec, onnx: Path = ECHO) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, directory / spec.onnx)
    save_spec(spec, directory)
    return directory


# ---- the card -----------------------------------------------------------------------------


def test_shipped_go2_card_describes_unilabs_contract(card):
    assert card.robot == "go2" and card.policy_dt == pytest.approx(0.02)
    assert card.obs_dim == 49 and len(card.joints) == 12
    # UniLab's action order is the joint order (FL, FR, RL, RR), not its actuator order (FR, FL, RR, RL).
    assert card.joints[:4] == ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint"]
    assert [t["term"] for t in card.layout()] == [
        "base_ang_vel",
        "neg_up_axis_world",
        "joint_pos_rel",
        "joint_vel",
        "last_action",
        "velocity_command",
        "gait_phase",
    ]
    assert card.pd.kp[0] == 35 and card.pd.kd[0] == 0.5 and card.pd.joint_damping[0] == 0.5
    assert card.trained_with["framework"] == "UniLab"


def test_a_card_with_mismatched_lengths_is_rejected_with_the_field_named(card, tmp_path):
    folder = _write_card(tmp_path / "bad", card)
    raw = json.loads((folder / "policy.json").read_text(encoding="utf-8"))
    raw["pd"]["kp"] = raw["pd"]["kp"][:11]
    (folder / "policy.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SpecError, match="pd.kp has 11 values for 12 joints"):
        load_spec(folder)


def test_a_card_cannot_point_its_model_outside_its_folder(card):
    with pytest.raises(ValueError, match="file name next to policy.json"):
        PolicySpec.model_validate({**card.model_dump(), "onnx": "../elsewhere.onnx"})


# ---- joint mapping --------------------------------------------------------------------------


def test_joints_bind_by_name_whatever_the_card_order(go2_model, card):
    shuffled = list(reversed(range(12)))
    spec = card.model_copy(
        update={
            "joints": [card.joints[i] for i in shuffled],
            "default_joint_pos": [card.default_joint_pos[i] for i in shuffled],
        }
    )
    binding = JointBinding.bind(go2_model, "", spec)
    for name, act in zip(binding.names, binding.actuators, strict=True):
        joint = mujoco.mj_id2name(go2_model, mujoco.mjtObj.mjOBJ_JOINT, go2_model.actuator_trnid[act, 0])
        assert joint == name
    assert binding.names[0] == "RR_calf_joint"


def test_library_go2_binds_with_its_world_prefix(pg, card):
    binding = JointBinding.bind(pg.session.model, "go2", card)
    assert binding.names[0] == "go2/FL_hip_joint"


def test_a_missing_joint_names_the_fix(go2_model, card):
    spec = card.model_copy(update={"joints": ["FL_hip", *card.joints[1:]]})
    with pytest.raises(SpecError, match="FL_hip.*joint_map"):
        JointBinding.bind(go2_model, "", spec)
    binding = JointBinding.bind(go2_model, "", spec, joint_map={"FL_hip": "FL_hip_joint"})
    assert binding.names[0] == "FL_hip_joint"
    with pytest.raises(SpecError, match="does not use"):
        JointBinding.bind(go2_model, "", card, joint_map={"tail": "FL_hip_joint"})


def test_a_default_pose_outside_the_limits_means_a_wrong_joint_order(go2_model, card):
    # Swapping hip and knee values is the classic mix-up; the knee cannot bend to +0.8.
    pose = list(card.default_joint_pos)
    pose[1], pose[2] = pose[2], pose[1]
    with pytest.raises(SpecError, match="joint order"):
        JointBinding.bind(go2_model, "", card.model_copy(update={"default_joint_pos": pose}))


# ---- observations: robotforge rebuilds what UniLab saw -----------------------------------------


def test_observations_match_what_unilab_built_during_a_rollout(go2_model, card):
    trace = np.load(TRACE)
    phys, obs, act, cmd = trace["phys"], trace["obs"], trace["act"], trace["cmd"]
    nq = int(trace["nq"])
    data = mujoco.MjData(go2_model)
    observer = Observer(go2_model, card, JointBinding.bind(go2_model, "", card), "")
    sensor_terms = slice(0, 6)  # gyro and gravity come from MuJoCo sensors in UniLab
    gravity_errors, body_frame_errors = [], []
    for k in range(len(obs)):
        data.qpos[:] = phys[k, 1 : 1 + nq]
        data.qvel[:] = phys[k, 1 + nq :]
        mujoco.mj_forward(go2_model, data)
        last = act[k - 1] if k > 0 else np.zeros(12)
        ours = observer.build(go2_model, data, last, cmd[k])
        observer.advance_phase(moving=True)
        np.testing.assert_allclose(ours[6:], obs[k, 6:], atol=1e-5, err_msg=f"step {k}")
        if k == 0:  # right after a reset, sensors and state agree exactly
            np.testing.assert_allclose(ours[sensor_terms], obs[k, sensor_terms], atol=1e-5)
        gravity_errors.append(np.abs(ours[3:6] - obs[k, 3:6]).max())
        rot = data.site_xmat[go2_model.site("imu").id].reshape(3, 3)
        body_frame_errors.append(np.abs(rot.T @ [0, 0, -1] - obs[k, 3:6]).max())
    # Later on UniLab's sensors lag the state by one 10 ms physics step, so gravity differs a little; the
    # body-frame convention (which UniLab does NOT use here) would be off by much more.
    assert max(gravity_errors) < 0.03
    assert np.mean(gravity_errors) < 0.3 * np.mean(body_frame_errors)


def test_observation_history_stacks_oldest_first(go2_model, card):
    spec = card.model_copy(update={"observations": [card.observations[5].model_copy(update={"history": 3})]})
    observer = Observer(go2_model, spec, JointBinding.bind(go2_model, "", spec), "")
    data = mujoco.MjData(go2_model)
    first = observer.build(go2_model, data, np.zeros(12), np.array([1.0, 0, 0]))
    second = observer.build(go2_model, data, np.zeros(12), np.array([2.0, 0, 0]))
    np.testing.assert_allclose(first[::3], [1, 1, 1])  # a fresh history repeats the present
    np.testing.assert_allclose(second[::3], [1, 1, 2])


# ---- the walker: policy rate, PD replay, safety -------------------------------------------------


class ConstantPolicy:
    def __init__(self, action: np.ndarray):
        self.action = action
        self.calls: list[np.ndarray] = []

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.calls.append(obs)
        return self.action


def test_walker_runs_the_policy_at_its_rate_and_pd_every_physics_step(pg, card):
    fake = ConstantPolicy(np.linspace(-0.5, 0.5, 12))
    walker = PolicyWalker(pg.session.model, "go2", card, fake)
    pg.session.set_controller("go2", walker)
    pg.run_fast(lambda: False, 0.1)
    assert len(fake.calls) == 5  # 50 Hz for 0.1 s
    assert walker.rate_hz == pytest.approx(1 / pg.session.model.opt.timestep)
    phases = [c[-4:] for c in fake.calls]
    np.testing.assert_allclose(phases[1] - phases[0], [0.04] * 4, atol=1e-6)  # 2 Hz clock, 20 ms steps
    np.testing.assert_allclose(walker.targets, np.asarray(card.default_joint_pos) + 0.25 * fake.action)


def test_walker_replays_the_trained_pd_through_robotforges_stiffer_servos(pg, card):
    m, d = pg.session.model, pg.session.data
    walker = PolicyWalker(m, "go2", card, ConstantPolicy(np.zeros(12)))
    b = walker.binding
    assert np.all(m.actuator_gainprm[b.actuators, 0] == 100)  # the library Go2's own servo stiffness
    walker.targets = np.asarray(card.default_joint_pos) + 0.1
    d.qvel[b.dof_adr] = np.linspace(-2, 2, 12)
    walker._apply_pd(d)
    mujoco.mj_forward(m, d)
    q, qd = d.qpos[b.qpos_adr], d.qvel[b.dof_adr]
    trained = np.clip(35 * (walker.targets - q) - 0.5 * qd, -np.array(card.pd.torque_limit), card.pd.torque_limit)
    # The library joints are damped at 2.0, the training robot's at 0.5: the servo adds the difference back.
    expected = trained + (2.0 - 0.5) * qd
    np.testing.assert_allclose(d.actuator_force[b.actuators], expected, atol=1e-6)


def test_walker_holds_still_on_invalid_actions_and_when_fallen(pg, card):
    m, d = pg.session.model, pg.session.data
    walker = PolicyWalker(m, "go2", card, ConstantPolicy(np.full(12, np.nan)))
    pg.session.set_controller("go2", walker)
    pg.run_fast(lambda: False, 0.05)
    assert "invalid action" in walker.status()["paused"]
    np.testing.assert_allclose(walker.targets, card.default_joint_pos)
    walker = PolicyWalker(m, "go2", card, ConstantPolicy(np.zeros(12)))
    d.xmat[walker.binding.root_body] = np.diag([1.0, -1.0, -1.0]).ravel()  # upside down
    walker.update(m, d)
    assert "fallen" in walker.status()["paused"]


def test_commands_are_clipped_to_what_the_policy_learned(pg, card):
    walker = PolicyWalker(pg.session.model, "go2", card, ConstantPolicy(np.zeros(12)))
    walker.command(3.0, -5.0)
    assert (walker.command_speed, walker.command_turn) == (1.0, -0.8)
    walker.command(-3.0, 0.0)
    assert walker.command_speed == -0.6


# ---- choosing a walker -------------------------------------------------------------------------


@pytest.mark.skipif(onnxruntime_available(), reason="checks the message shown without the extra")
def test_rl_walker_without_onnxruntime_says_how_to_install_it(pg):
    with pytest.raises(BehaviorError, match="uv sync --extra policy"):
        pg.behaviors.move("go2", 0.5, 0.0, walker="rl")
    assert isinstance(pg.session.controllers.get("go2"), type(None))


def test_rl_walker_for_a_robot_without_a_policy_explains_the_options(pg, monkeypatch):
    monkeypatch.setattr(policies, "SHIPPED_DIR", Path("/nonexistent"))
    with pytest.raises(BehaviorError, match="walker='gait'"):
        pg.behaviors.move("go2", 0.5, 0.0, walker="rl")


@needs_onnxruntime
def test_library_go2_walks_turns_and_stops_with_the_unilab_policy(pg):
    status = pg.behaviors.move("go2", 0.5, 0.0, walker="rl")
    assert status["walker"] == "rl" and status["policy"] == "go2_unilab_walk"
    start = pg.behaviors.pose("go2")
    pg.run_fast(lambda: False, 5.0)
    walked = pg.behaviors.pose("go2")
    assert math.dist(start["position"][:2], walked["position"][:2]) > 1.8  # 0.5 m/s commanded
    assert walked["upright"]
    pg.behaviors.move("go2", 0.0, 0.8)  # walker omitted: keeps the policy, as WASD driving does
    pg.run_fast(lambda: False, 3.0)
    turned = pg.behaviors.pose("go2")["heading_deg"] - walked["heading_deg"]
    assert (turned + 180) % 360 - 180 > 70  # about 0.7 rad/s
    assert pg.behaviors.stop("go2")["stopped"]
    assert type(pg.session.controllers["go2"]).__name__ == "PolicyWalker"  # keeps balancing, at zero command
    pg.behaviors.move("go2", 0.3, 0.0, walker="gait")
    assert isinstance(pg.session.controllers["go2"], quadruped.QuadrupedGait)


def test_the_viewer_learns_which_robots_have_a_learned_walker(pg):
    assert pg.behaviors.learned_walkers() == {"go2": {"policy": "go2_unilab_walk", "max_speed": 1.0, "active": False}}
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    handlers = hub.handlers
    assert hub.scene_extras()["learned"]["go2"]["policy"] == "go2_unilab_walk"
    handlers["teleop"]({"type": "teleop", "robot": "go2", "speed": 0.2, "turn": 0.0, "walker": "gait"})
    assert isinstance(pg.session.controllers["go2"], quadruped.QuadrupedGait)


@needs_onnxruntime
def test_keyboard_driving_switches_to_and_keeps_the_learned_walker(pg):
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    handlers = hub.handlers
    handlers["teleop"]({"type": "teleop", "robot": "go2", "speed": 0.0, "turn": 0.0, "walker": "rl"})  # the toggle
    handlers["teleop"]({"type": "teleop", "robot": "go2", "speed": 0.6, "turn": 0.0})  # W held
    walker = pg.session.controllers["go2"]
    assert type(walker).__name__ == "PolicyWalker" and walker.command_speed == 0.6
    assert pg.behaviors.learned_walkers()["go2"]["active"]


@needs_onnxruntime
def test_the_echo_model_runs_end_to_end_through_onnxruntime(pg, card, tmp_path):
    from robotforge.bridge.onnx_policy import load_onnx

    spec = card.model_copy(update={"name": "echo"})
    folder = _write_card(tmp_path / "echo", spec)
    policy = load_onnx(spec, folder)
    walker = PolicyWalker(pg.session.model, "go2", spec, policy)
    d = pg.session.data
    d.qpos[walker.binding.qpos_adr[0]] += 0.2  # FL hip out by 0.2 rad
    walker.update(pg.session.model, d)
    rel = d.qpos[walker.binding.qpos_adr] - np.asarray(spec.default_joint_pos)
    ramp = (np.arange(12) - 5.5) * 0.01
    np.testing.assert_allclose(walker.last_action, rel + ramp, atol=1e-5)
    assert walker.last_action[0] == pytest.approx(0.2 + ramp[0], abs=1e-5)


@needs_onnxruntime
def test_a_model_that_does_not_match_its_card_is_refused(card, tmp_path):
    from robotforge.bridge.onnx_policy import load_onnx

    spec = card.model_copy(update={"name": "short", "observations": card.observations[:-1]})  # 45 values
    with pytest.raises(SpecError, match="reads 49 observation values but the card describes 45"):
        load_onnx(spec, _write_card(tmp_path / "short", spec))


# ---- where policies live ------------------------------------------------------------------------


def test_installed_policies_win_over_shipped_ones_and_broken_folders_are_named(card, tmp_path):
    source = _write_card(tmp_path / "src", card.model_copy(update={"name": "my_go2"}))
    installed = policies.install(source)
    assert installed.directory == policies.user_dir() / "my_go2"
    assert policies.for_robot("go2").spec.name == "my_go2"
    assert set(policies.all_policies()) >= {"my_go2", "go2_unilab_walk"}
    (policies.user_dir() / "broken").mkdir()
    (policies.user_dir() / "broken" / "policy.json").write_text("{", encoding="utf-8")
    assert "broken" not in policies.all_policies()
    with pytest.raises(SpecError, match="not valid JSON"):
        policies.load("broken")
    with pytest.raises(SpecError, match="Did you mean my_go2"):
        policies.load("my_go3")


# ---- the UniLab bridge (without UniLab) ---------------------------------------------------------


def test_unilab_not_configured_explains_the_setup():
    with pytest.raises(unilab.UniLabError, match="ROBOTFORGE_UNILAB_DIR.*uv sync --extra mujoco"):
        unilab.check()


def test_unilab_dir_must_be_a_unilab_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv(unilab.ENV_VAR, str(tmp_path))
    with pytest.raises(unilab.UniLabError, match="not a UniLab checkout"):
        unilab.checkout()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "unilab"\n', encoding="utf-8")
    with pytest.raises(unilab.UniLabError, match="has no .venv"):
        unilab.python_of(unilab.checkout())


def test_training_requests_are_validated_before_anything_starts(tmp_path, monkeypatch):
    with pytest.raises(unilab.UniLabError, match="go2_joystick_flat"):
        unilab.start_training("cartpole", 10, "x")
    with pytest.raises(unilab.UniLabError, match="iterations"):
        unilab.start_training("go2_joystick_flat", 0, "x")
    with pytest.raises(unilab.UniLabError, match="lowercase"):
        unilab.start_training("go2_joystick_flat", 10, "My Policy")
    assert not unilab.jobs_dir().exists()


def test_training_log_lines_become_progress():
    state: dict = {}
    lines = [
        "\x1b[1m                Learning iteration 41/151                \x1b[0m",
        "                            Total steps: 1032192",
        "                            Mean reward: 21.37",
        "                    Mean episode length: 993.31",
        "                           Time elapsed: 00:01:02",
        "                                    ETA: 00:01:23",
        "unrelated line",
    ]
    changed = [unilab_tool.parse_progress(line, state) for line in lines]
    assert changed == [True] * 6 + [False]
    assert state == {
        "iteration": 42,
        "iterations": 151,
        "env_steps": 1032192,
        "mean_reward": 21.37,
        "mean_episode_length": 993.31,
        "elapsed_s": 62.0,
        "eta_s": 83.0,
    }


def test_a_finished_job_installs_its_policy(card, tmp_path):
    folder = unilab.jobs_dir() / "20261014-120000-abcdef"
    exported = _write_card(folder / "policy", card.model_copy(update={"name": "go2_quick"}))
    folder.mkdir(parents=True, exist_ok=True)
    unilab_tool.write_status(folder / "status.json", {"state": "done", "policy_dir": str(exported), "pid": -1})
    status = unilab.job_status()
    assert status["state"] == "done" and status["installed"] == "go2_quick"
    assert policies.load("go2_quick").directory == policies.user_dir() / "go2_quick"
    assert unilab.job_status(folder.name)["installed"] == "go2_quick"  # remembered, not reinstalled


def test_a_job_whose_process_died_reports_failure(tmp_path):
    folder = unilab.jobs_dir() / "20261014-120000-dead00"
    folder.mkdir(parents=True)
    (folder / "tool.log").write_text("Traceback ...\nRuntimeError: boom\n", encoding="utf-8")
    unilab_tool.write_status(folder / "status.json", {"state": "training", "pid": 999_999_999})
    status = unilab.job_status(folder.name)
    assert status["state"] == "failed" and "boom" in status["log_tail"][-1]
    with pytest.raises(unilab.UniLabError, match="no UniLab job"):
        unilab.job_status("../escape")


# ---- MCP tools ---------------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_lists_policies_walks_with_one_and_explains_unilab_setup(pg):
    from mcp import Client

    from robotforge.cli import find_free_port
    from robotforge.mcp_server import build_server

    server = build_server(pg, find_free_port(8811))
    async with Client(server) as client:

        async def call(tool: str, args: dict) -> dict:
            result = await client.call_tool(tool, args)
            assert not result.is_error, result.content[0].text
            return json.loads(result.content[0].text)

        listed = await call("list_walkers", {})
        assert [p["name"] for p in listed["walkers"]] == ["go2_unilab_walk"]
        assert ("note" in listed) != onnxruntime_available()  # the install hint shows only when needed
        failed = await client.call_tool("unilab_check", {})
        assert failed.is_error and "ROBOTFORGE_UNILAB_DIR" in failed.content[0].text
        if onnxruntime_available():
            moved = await call("move", {"robot": "go2", "speed": 0.5, "seconds": 4, "walker": "rl"})
            assert moved["behavior"]["policy"] == "go2_unilab_walk"
            assert moved["distance_m"] > 1.4 and moved["to"]["upright"]


# ---- hardening (from review) -------------------------------------------------------------------


def test_policy_names_can_never_be_paths(card, tmp_path):
    folder = _write_card(tmp_path / "elsewhere", card.model_copy(update={"name": "elsewhere"}))
    for bad in (str(folder), "../elsewhere", "walk\n"):
        with pytest.raises(SpecError, match="not a policy name"):
            policies.load(bad)


def test_job_ids_can_never_be_paths():
    unilab.jobs_dir().mkdir(parents=True)
    for bad in ("..", ".", "../../policies"):
        with pytest.raises(unilab.UniLabError, match="no UniLab job"):
            unilab.job_status(bad)


def test_a_stale_pid_is_never_signalled(tmp_path):
    import os

    folder = unilab.jobs_dir() / "20261014-120000-0aa000"
    folder.mkdir(parents=True)
    # After a reboot the recorded pid may belong to anything, here this very test process.
    unilab_tool.write_status(folder / "status.json", {"state": "training", "pid": os.getpid()})
    assert unilab.cancel(folder.name)["cancelled"] is False
    assert unilab.job_status(folder.name)["state"] == "failed"


def test_a_policy_that_raises_pauses_the_walker_instead_of_the_simulation(pg, card):
    def broken(obs):
        raise RuntimeError("bad input shape")

    walker = PolicyWalker(pg.session.model, "go2", card, broken)
    pg.session.set_controller("go2", walker)
    pg.run_fast(lambda: False, 0.1)  # keeps stepping
    assert "failed to run" in walker.status()["paused"]
    np.testing.assert_allclose(walker.targets, card.default_joint_pos)


def test_a_task_of_our_own_needs_both_its_config_folder_and_its_name(tmp_path, capsys):
    args = ["train", "--task", "rf_dog", "--iterations", "3", "--log-root", str(tmp_path), "--status", "s.json"]
    with pytest.raises(SystemExit):
        unilab_tool.main([*args, "--conf-dir", str(tmp_path)])
    assert "--conf-dir and --task-name go together" in capsys.readouterr().err


def test_a_custom_task_folder_is_checked_before_anything_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(unilab, "checkout", lambda: pytest.fail("must not get as far as UniLab"))
    with pytest.raises(unilab.UniLabError, match="has no task/my_dog/mujoco.yaml"):
        unilab.start_training("my_dog", 10, "my_dog_walk", conf_dir=tmp_path)
    owner = tmp_path / "task" / "my_dog" / "mujoco.yaml"
    owner.parent.mkdir(parents=True)
    owner.write_text("# @package _global_\nenv: {}\n", encoding="utf-8")
    with pytest.raises(unilab.UniLabError, match="does not name its task"):
        unilab.start_training("my_dog", 10, "my_dog_walk", conf_dir=tmp_path)
    owner.write_text("training:\n  task_name: RobotforgeMyDog\n", encoding="utf-8")
    assert unilab._custom_task(tmp_path, "my_dog") == "RobotforgeMyDog"
    with pytest.raises(unilab.UniLabError, match="task names"):
        unilab.start_training("../my_dog", 10, "my_dog_walk", conf_dir=tmp_path)
