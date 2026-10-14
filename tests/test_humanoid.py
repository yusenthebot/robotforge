import json
import math
import random

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import Kind, humanoid
from robotforge.behaviors.biped import Foot, GaitParams, GaitPlan, PreviewControl
from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.behaviors.manager import BehaviorError
from robotforge.cli import find_free_port
from robotforge.design import templates
from robotforge.design.world import World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "wood")
    return playground


def _up(pg: Playground, robot: str = "bot") -> float:
    d, m = pg.session.data, pg.session.model
    return float(d.xmat[m.body(f"{robot}/torso").id][8])


def _run(pg: Playground, seconds: float, robot: str = "bot") -> float:
    """Simulate and return the lowest torso up-axis z seen (1 = perfectly upright)."""
    lowest = 1.0
    steps = round(0.05 / pg.session.model.opt.timestep)  # exact slices (run_fast rounds up to 0.02 s chunks)
    for _ in range(round(seconds / 0.05)):
        pg.session.step(steps)
        lowest = min(lowest, _up(pg, robot))
    return lowest


def _heading(pg: Playground) -> float:
    return pg.behaviors.pose("bot")["heading_deg"]


# ---- template ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, servos",
    [
        ({}, 20),
        ({"height": 0.3}, 20),
        ({"height": 0.6, "color": "orange"}, 20),
        ({"arms": False}, 14),
        ({"arms": False, "head": False, "servo": "xl430"}, 12),
    ],
)
def test_humanoid_variants_spawn_with_feet_on_the_floor(params, servos):
    world = World.model_validate({"robots": [{"name": "bot", "template": "humanoid", "params": params}]})
    compiled = compile_world(world)
    m = compiled.model
    assert m.nu == servos
    assert not any("warning" in n for n in compiled.notes)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    for side in ("left", "right"):
        sole = d.site_xpos[m.site(f"bot/{side}_foot").id]
        assert 0.0 < sole[2] < 0.004  # just above the floor, so it settles without a drop
    assert d.ncon == 0  # nothing intersects at spawn
    assert "bot/head_cam" in [m.camera(i).name for i in range(m.ncam)] or not params.get("head", True)


def test_weak_servos_come_with_a_warning():
    notes = templates.build("humanoid", {"servo": "sg90"}).notes
    assert any("too weak" in n for n in notes)
    assert len(templates.build("humanoid", {}).notes) == 1


def test_humanoid_is_recognized_with_and_without_its_template_name(pg):
    pg.add_robot("bot", template="humanoid")
    listed = pg.behaviors.robots()
    assert listed == [{"name": "bot", "kind": "humanoid", "max_speed": pytest.approx(0.1, abs=0.03)}]
    from robotforge.behaviors import robot_kind

    assert robot_kind(pg.session.model, "bot") == Kind.HUMANOID


# ---- balance -----------------------------------------------------------------------------


def test_humanoid_stands_ten_seconds_and_shrugs_off_pushes(pg):
    pg.add_robot("bot", template="humanoid")
    pg.behaviors.strike_pose("bot", "stand")
    assert _run(pg, 10.0) > 0.95
    torso = pg.session.model.body("bot/torso").id
    # 6 N for 0.1 s: a 0.6 N*s shove, about 0.3 m/s of sudden body speed for this 1.9 kg robot.
    for force in ([6, 0, 0], [-6, 0, 0], [0, 6, 0], [0, -6, 0]):
        pg.session.data.xfrc_applied[torso, :3] = force
        pg.run_fast(lambda: False, 0.1)
        pg.session.data.xfrc_applied[torso, :3] = 0
        assert _run(pg, 3.0) > 0.95, f"fell after a push of {force} N"
    assert not pg.behaviors.status()[0].get("fallen")


# ---- walking -----------------------------------------------------------------------------


def test_humanoid_walks_forward_at_the_commanded_speed(pg):
    pg.add_robot("bot", template="humanoid")
    start = pg.behaviors.pose("bot")["position"]
    status = pg.behaviors.move("bot", 0.1, 0.0)
    assert status["behavior"] == "walk"
    assert _run(pg, 15.0) > 0.95
    end = pg.behaviors.pose("bot")["position"]
    # 0.1 m/s for 15 s, minus about half a second to get going: ~1.45 m (measured 1.57)
    assert 1.25 < end[0] - start[0] < 1.8
    assert abs(end[1] - start[1]) < 0.1


def test_humanoid_turns_in_place_and_walks_backwards_at_the_commanded_rates(pg):
    pg.add_robot("bot", template="humanoid")
    pg.behaviors.move("bot", 0.0, 0.3)
    total, last = 0.0, _heading(pg)
    for _ in range(20):
        assert _run(pg, 0.5) > 0.95
        h = _heading(pg)
        total += (h - last + 180) % 360 - 180
        last = h
    assert 145 < total < 185  # 0.3 rad/s for 10 s is 172 degrees
    assert math.dist(pg.behaviors.pose("bot")["position"][:2], [0, 0]) < 0.1  # turned on the spot
    pg.behaviors.move("bot", -0.08, 0.0)
    before = np.array(pg.behaviors.pose("bot")["position"][:2])
    yaw = math.radians(_heading(pg))
    assert _run(pg, 6.0) > 0.95
    moved = np.array(pg.behaviors.pose("bot")["position"][:2]) - before
    assert -0.6 < moved @ [math.cos(yaw), math.sin(yaw)] < -0.35  # -0.08 m/s for 6 s: about -0.45 m


def test_stop_brings_a_walking_humanoid_to_a_still_stand(pg):
    pg.add_robot("bot", template="humanoid")
    pg.behaviors.move("bot", 0.1, 0.2)
    _run(pg, 4.0)
    pg.behaviors.stop("bot")
    assert _run(pg, 3.0) > 0.95
    ctrl = pg.session.controllers["bot"]
    assert ctrl.done and ctrl.status()["behavior"] == "stand"
    a = pg.behaviors.pose("bot")["position"]
    _run(pg, 1.0)
    assert math.dist(a, pg.behaviors.pose("bot")["position"]) < 0.005
    d, m = pg.session.data, pg.session.model
    touch = [d.sensordata[m.sensor(f"bot/{s}_foot_touch").adr[0]] for s in ("left", "right")]
    assert min(touch) > 0.25 * sum(touch)  # weight shared by both feet


@pytest.mark.parametrize("tap", [0.4, 0.8, 1.2, 2.0])
def test_tapping_the_walk_key_never_tips_it_over(pg, tap):
    # Walk, let go for half a second, walk again: restarting while the stop is still settling used to
    # swap the planned weight shift with no warning and rocked the robot over by the fourth tap.
    pg.add_robot("bot", template="humanoid")
    top = pg.behaviors.robots()[0]["max_speed"]
    lowest = 1.0
    for _ in range(8):
        pg.behaviors.move("bot", top, 0.0)
        lowest = min(lowest, _run(pg, tap))
        pg.behaviors.move("bot", 0.0, 0.0)
        lowest = min(lowest, _run(pg, 0.5))
    assert lowest > 0.9


@pytest.mark.parametrize("seed", [0, 1, 34, 39])  # 34 and 39 tipped the robot before the fix
def test_mashing_the_driving_keys_never_tips_it_over(pg, seed):
    rng = random.Random(seed)
    pg.add_robot("bot", template="humanoid")
    lowest = 1.0
    for _ in range(60):
        pg.behaviors.move("bot", rng.choice([0.0, 0.06, -0.06]), rng.choice([0.0, 1.0, -1.0]))
        lowest = min(lowest, _run(pg, rng.choice([0.05, 0.15, 0.3, 0.5])))
    assert lowest > 0.9


def test_new_commands_take_effect_within_a_step(pg):
    pg.add_robot("bot", template="humanoid")
    m, d = pg.session.model, pg.session.data
    soles = [m.site(f"bot/{s}_foot").id for s in ("left", "right")]
    pg.behaviors.move("bot", 0.1, 0.0)
    start = d.time
    pg.run_fast(lambda: max(d.site_xpos[s][2] for s in soles) > 0.01, 2.0)
    assert d.time - start < 0.8  # the first foot is in the air
    pg.run_fast(lambda: False, 3.0)
    pg.behaviors.stop("bot")
    ctrl = pg.session.controllers["bot"]
    start = d.time
    pg.run_fast(lambda: ctrl.done, 3.0)
    assert d.time - start < 1.0  # standing still again


def test_a_long_stand_keeps_the_plan_small(pg):
    pg.add_robot("bot", template="humanoid")
    pg.behaviors.move("bot", 0.0, 0.0)
    pg.run_fast(lambda: False, 60.0)
    assert len(pg.session.controllers["bot"].plan._checkpoints) < 20


# ---- poses ---------------------------------------------------------------------------------


def _play(pg: Playground, name: str) -> dict[str, tuple[float, float]]:
    """Play a pose; return the (lowest, highest) value seen of each joint, torso pitch and torso height."""
    m, d = pg.session.model, pg.session.data
    ctrl = pg.behaviors.strike_pose("bot", name)
    joints = {m.joint(j).name.split("/", 1)[1]: m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == 3}
    torso = m.body("bot/torso").id
    seen: dict[str, list[float]] = {k: [] for k in [*joints, "pitch", "height"]}
    while not ctrl.done and d.time < 20:
        pg.session.step(25)
        for k, adr in joints.items():
            seen[k].append(float(d.qpos[adr]))
        seen["pitch"].append(math.asin(-d.xmat[torso][6]))
        seen["height"].append(float(d.xpos[torso][2]))
    assert ctrl.done, f"{name} did not finish"
    return {k: (min(v), max(v)) for k, v in seen.items()}


def _reversals(pg: Playground, name: str, joint: str) -> int:
    """How many times a joint changes direction (by more than 0.1 rad) while a pose plays."""
    m, d = pg.session.model, pg.session.data
    adr = m.jnt_qposadr[m.joint(f"bot/{joint}").id]
    ctrl = pg.behaviors.strike_pose("bot", name)
    values = []
    while not ctrl.done and d.time < 20:
        pg.session.step(25)
        values.append(float(d.qpos[adr]))
    turns, anchor, direction = 0, values[0], 0
    for v in values:
        step = v - anchor
        if abs(step) > 0.1:
            if direction and np.sign(step) != direction:
                turns += 1
            direction, anchor = int(np.sign(step)), v
    return turns


def test_wave_raises_the_right_arm_and_waves_it(pg):
    pg.add_robot("bot", template="humanoid")
    seen = _play(pg, "wave")
    assert seen["right_shoulder_roll"][1] > 2.3 and seen["left_shoulder_roll"][1] < 0.4
    assert _reversals(pg, "wave", "right_shoulder_roll") >= 4  # side to side, not just up and down
    assert _run(pg, 1.0) > 0.95


def test_squat_bends_the_knees_and_lowers_the_body(pg):
    pg.add_robot("bot", template="humanoid")
    start = pg.behaviors.pose("bot")["position"][2]
    seen = _play(pg, "squat")
    assert seen["height"][0] < start - 0.045
    assert min(seen["left_knee"][1], seen["right_knee"][1]) > 1.4
    assert min(seen["left_shoulder_pitch"][1], seen["right_shoulder_pitch"][1]) > 1.1  # arms forward
    assert _run(pg, 1.0) > 0.95


def test_arms_up_raises_both_arms_overhead(pg):
    pg.add_robot("bot", template="humanoid")
    seen = _play(pg, "arms_up")
    assert min(seen["left_shoulder_roll"][1], seen["right_shoulder_roll"][1]) > 2.4
    assert _run(pg, 1.0) > 0.95


def test_bow_leans_the_torso_forward(pg):
    pg.add_robot("bot", template="humanoid")
    seen = _play(pg, "bow")
    assert seen["pitch"][1] > 0.35 and seen["neck_pitch"][1] > 0.25
    assert _run(pg, 1.0) > 0.95


def test_look_around_turns_the_head_both_ways(pg):
    pg.add_robot("bot", template="humanoid")
    seen = _play(pg, "look_around")
    assert seen["neck_yaw"][1] > 0.85 and seen["neck_yaw"][0] < -0.85
    assert _run(pg, 1.0) > 0.95


def test_stand_brings_arms_and_head_home_and_every_pose_ends_upright(pg):
    pg.add_robot("bot", template="humanoid")
    ctrl = pg.behaviors.strike_pose("bot", "arms_up")
    pg.run_fast(lambda: False, 1.5)  # arms are high now
    for name in humanoid.POSES:
        ctrl = pg.behaviors.strike_pose("bot", name)
        pg.run_fast(lambda c=ctrl: c.done, 15.0)
        assert ctrl.done and _run(pg, 0.5) > 0.95, name
    m, d = pg.session.model, pg.session.data
    for joint, home in {"left_shoulder_roll": 0.12, "right_elbow": 0.35, "neck_yaw": 0.0}.items():
        assert d.qpos[m.jnt_qposadr[m.joint(f"bot/{joint}").id]] == pytest.approx(home, abs=0.08)
    assert ctrl.status()["behavior"] == "stand"


def test_pose_errors_tell_what_to_do(pg):
    pg.add_robot("bot", template="humanoid")
    pg.add_robot("dog", template="quadruped", pos=[1, 0])
    with pytest.raises(BehaviorError, match="choose one of: stand, wave"):
        pg.behaviors.strike_pose("bot", "dance")
    with pytest.raises(BehaviorError, match="only humanoids strike poses"):
        pg.behaviors.strike_pose("dog", "wave")


@pytest.mark.parametrize("template", ["humanoid", "quadruped", "rover"])
def test_non_finite_commands_are_refused_before_they_reach_the_motors(pg, template):
    pg.add_robot("bot", template=template)
    pg.behaviors.move("bot", 0.05, 0.0)
    for speed, turn in ((math.nan, 0.2), (0.1, math.nan), (math.inf, 0.0)):
        with pytest.raises(BehaviorError, match="finite"):
            pg.behaviors.move("bot", speed, turn)
    pg.run_fast(lambda: False, 1.0)
    assert np.isfinite(pg.session.data.ctrl).all()
    assert np.isfinite(pg.session.data.qpos).all()


def test_humanoids_balance_from_the_start_and_after_a_fall_and_reset(pg, tmp_path):
    pg.add_robot("bot", template="humanoid")
    assert isinstance(pg.session.controllers.get("bot"), humanoid.HumanoidController)  # never limp
    torso = pg.session.model.body("bot/torso").id
    pg.behaviors.move("bot", 0.1, 0.0)
    pg.session.data.xfrc_applied[torso, :3] = [40, 0, 0]
    pg.run_fast(lambda: False, 0.3)
    pg.session.data.xfrc_applied[torso, :3] = 0
    pg.run_fast(lambda: False, 2.0)
    assert _up(pg) < 0.5
    status = pg.behaviors.status()[0]
    assert status["behavior"] == "fallen" and status["speed"] == 0.0
    with pytest.raises(BehaviorError, match="fallen over"):
        pg.behaviors.move("bot", 0.1, 0.0)
    pg.control("reset")
    assert pg.session.controllers["bot"].status()["behavior"] == "stand"
    pg.behaviors.move("bot", 0.1, 0.0)
    assert _run(pg, 5.0) > 0.95
    assert pg.behaviors.pose("bot")["position"][0] > 0.3
    # Opening the saved world later (as `robotforge open` does) also starts the balance.
    from robotforge.cli import load_playground

    reopened = load_playground(pg.store.dir)
    assert isinstance(reopened.session.controllers.get("bot"), humanoid.HumanoidController)


@pytest.mark.anyio
async def test_pose_and_move_tools_work_for_humanoids(pg):
    pg.add_robot("bot", template="humanoid")
    server = build_server(pg, find_free_port(8795))
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools["pose"].input_schema["properties"]["name"]["enum"]) == set(humanoid.POSES)
        posed = json.loads((await client.call_tool("pose", {"robot": "bot", "name": "arms_up"})).content[0].text)
        assert posed["upright"] is True and posed["sim_seconds"] > 2
        moved = await client.call_tool("move", {"robot": "bot", "speed": 0.1, "seconds": 8})
        report = json.loads(moved.content[0].text)
        assert report["distance_m"] > 0.3 and report["to"]["upright"]
        # It finishes its last step before measuring, so the reported spot is where it really stops.
        assert 0 < report["settle_seconds"] < 2 and pg.session.controllers["bot"].done


@pytest.mark.anyio
async def test_scripts_can_walk_a_humanoid_and_strike_poses(pg):
    pg.add_robot("kid", template="humanoid", params={"color": "teal"})
    code = (
        "from robotforge.script import connect\n"
        "sim = connect()\n"
        "start = sim.position('kid')\n"
        "sim.move('kid', speed=0.1)\n"
        "sim.step(4.0)\n"
        "sim.stop('kid')\n"
        "sim.step(1.0)\n"
        "print('walked', round(sim.position('kid')[0] - start[0], 2))\n"
        "print('waved upright', sim.pose('kid', 'wave'))\n"
        "try:\n"
        "    sim.pose('kid', 'dance')\n"
        "except Exception as exc:\n"
        "    print('refused:', exc)\n"
    )
    server = build_server(pg, find_free_port(8805))
    async with Client(server) as client:
        await client.call_tool("write_script", {"name": "kid_demo", "code": code})
        out = json.loads((await client.call_tool("run_script", {"name": "kid_demo"})).content[0].text)
    lines = out["output"]
    walked = float(lines.split("walked ")[1].split()[0])
    assert walked > 0.25, lines
    assert "waved upright True" in lines and "refused: unknown pose 'dance'" in lines


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---- building blocks -----------------------------------------------------------------------


def test_ik_can_also_match_a_full_orientation():
    world = World.model_validate({"robots": [{"name": "bot", "template": "humanoid"}]})
    m = compile_world(world).model
    joints = [m.joint(f"bot/left_{j}").id for j in humanoid.LEG]
    site = m.site("bot/left_foot").id
    chain = Chain.from_joints(m, joints, int(m.site_bodyid[site]), m.site_pos[site])
    ik = IKSolver(m, m.key_qpos[m.key("home").id])
    home = ik.point(chain, ik.base_qpos[chain.qpos_adr])
    yaw = 0.3
    want = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    target = home + [0.03, 0.01, 0.02]
    q, err = ik.solve(chain, target, ik.base_qpos[chain.qpos_adr], iterations=30, damping=1e-8, rotation=want)
    assert err < 1e-4
    assert np.allclose(ik.data.xmat[chain.body].reshape(3, 3), want, atol=1e-3)
    with pytest.raises(ValueError, match="not both"):
        ik.solve(chain, target, q, axis=(np.array([0, 0, 1.0]), np.array([0, 0, 1.0])), rotation=want)


def test_preview_control_keeps_the_zmp_on_the_planned_path():
    params = GaitParams(
        single=0.36,
        double=0.12,
        start=0.4,
        settle=0.4,
        lift=0.025,
        half_width=0.045,
        min_width=0.07,
        max_stride=0.05,
        max_turn=0.2,
        zmp_inset=0.01,
        zmp_roll=0.5,
    )
    dt = 0.01
    plan = GaitPlan(dt, Foot(0, 0.045, 0), Foot(0, -0.045, 0), params)
    preview = PreviewControl(dt, 0.23, 100)
    state, worst = np.zeros((3, 2)), 0.0
    for k in range(1200):
        command = (0.1, 0.0) if k < 700 else (0.0, 0.0)
        while len(plan) <= 100:
            plan.extend(*command)
        zmp, _, _ = plan.pop()
        state = preview.step(state, plan.zmp_ahead(100))
        worst = max(worst, float(np.linalg.norm(preview.zmp(state) - zmp)))
    assert worst < 0.02  # within the 5 cm half-length of a foot with room to spare
    left, right = plan.feet_end
    assert left.x == pytest.approx(right.x) and left.x > 0.5  # walked, then closed up side by side
    assert state[0] == pytest.approx([left.x, 0.0], abs=1e-3)  # and the body settled between the feet
