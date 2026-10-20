"""Touch in the loop: pressing and wiping with a Panda (the Menagerie fixture), controller against controller."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.control import TASKS, TaskError, compare_methods, make_controller, make_task, task_world
from robotforge.control.compare import run
from robotforge.control.impedance import ToolForce
from robotforge.control.tasks.contact import PathPlane, find_surface
from robotforge.design import end_effectors
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"
TABLE_TOP = 0.30


def _desk(height: float = TABLE_TOP, world: str = "lab") -> Playground:
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world(world, "checker")
    pg.add_robot("arm", library="panda")
    pg.update_robot("arm", end_effector="wiper")
    pg.add_object("desk", shape="table", size=[0.6, 0.6, height], pos=[0.5, 0.0])
    return pg


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """A Panda with a wiper pad in front of a 30 cm table."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    return _desk()


def _start(pg: Playground):
    m, d = pg.session.model, copy.copy(pg.session.data)
    mujoco.mj_forward(m, d)
    return m, d


def test_the_force_sense_reads_what_the_surroundings_push_back_with():
    """A 2 kg block resting on a table is pushed up by the table with its weight: the sense reads +19.6 N on z,
    and nothing at all for contact between its own parts."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.05], pos=[0, 0, -0.05])
    tool = spec.worldbody.add_body(name="tool", pos=[0, 0, 0.0505])
    tool.add_freejoint()
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], mass=2.0)
    top = tool.add_body(name="tool_top", pos=[0, 0, 0.1])  # a second part of the same tool, resting on the first
    top.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.05], mass=0.0001)
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("tool", "tool_top")}
    sense = ToolForce(model, ids, set())
    mujoco.mj_forward(model, data)
    felt = sense.read(model, data)
    assert felt[2] == pytest.approx(2.0001 * 9.81, abs=0.1) and abs(felt[0]) < 0.05 and abs(felt[1]) < 0.05
    # the same block with the table counted as part of the robot: it feels nothing
    own = ToolForce(model, ids, {0})
    assert float(np.linalg.norm(own.read(model, data))) == 0.0


def test_the_surface_under_the_tool_is_the_table_not_the_floor(desk):
    m, d = _start(desk)
    tool = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm/tcp")]
    found = find_surface(m, d, np.array([tool[0], tool[1], tool[2] + 0.02]), "arm/")
    assert found.name == "desk" and found.point[2] == pytest.approx(TABLE_TOP, abs=1e-3)
    assert found.normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    off = find_surface(m, d, np.array([-1.5, 1.5, 1.0]), "arm/")
    assert off.name == "world" and off.point[2] == pytest.approx(0.0, abs=1e-3)  # nothing else there: the floor


def test_a_surface_that_is_not_under_the_tool_is_refused_with_a_hint(desk):
    m, d = _start(desk)
    task = make_task("press", m, "arm", {"surface": "nothing_here"})
    with pytest.raises(TaskError, match="no surface named"):
        task.begin(d)
    desk.add_object("shelf", shape="table", size=[0.3, 0.3, 0.2], pos=[-0.6, 0.6])
    m, d = _start(desk)
    task = make_task("press", m, "arm", {"surface": "shelf"})
    with pytest.raises(TaskError, match="not over shelf"):
        task.begin(d)


def test_path_plane_frame_and_coordinates():
    plane = PathPlane(np.array([1.0, 2.0, 0.5]), np.array([0.0, 0.0, 1.0]))
    assert plane.u == pytest.approx([1, 0, 0]) and plane.v == pytest.approx([0, 1, 0])
    p = plane.point(0.3, -0.2, 0.05)
    assert p == pytest.approx([1.3, 1.8, 0.55]) and plane.coords(p) == pytest.approx([0.3, -0.2, 0.05])
    slope = PathPlane(np.zeros(3), np.array([-np.sin(0.2), 0.0, np.cos(0.2)]))  # tilted 0.2 rad about y
    assert abs(float(slope.u @ slope.n)) < 1e-9 and abs(float(slope.v @ slope.u)) < 1e-9


def test_a_guarded_press_stops_at_its_force_and_a_stiff_arm_does_not(desk):
    """The planner stops coming down the moment the force shows. A compliant controller then sits at the force it was
    given; the arm's own servos, stiff, are already past it by the time it stops."""
    m, d = _start(desk)
    results = {}
    for method in ("osc", "cartesian_impedance", "ik"):
        task = make_task("press", m, "arm", {"force": 6.0, "mode": "stop"})
        results[method] = run(task, m, copy.copy(d), method, 6.0)
    for method in ("osc", "cartesian_impedance"):
        r = results[method]
        assert r.success and r.report["touched_at_s"] > 0.5, (method, r.report)
        assert 6.0 <= r.report["peak_n"] <= 7.5, (method, r.report)
    assert results["ik"].report["peak_n"] > 15.0  # no force sense in its loop, and a stiff servo: way past the limit
    assert not results["ik"].success


def test_holding_a_force_needs_a_force_loop_and_osc_has_one(desk):
    m, d = _start(desk)
    out = {}
    for method in ("osc", "cartesian_impedance", "ik"):
        out[method] = run(make_task("press", m, "arm", {"force": 6.0}), m, copy.copy(d), method, 9.0).report
    assert out["osc"]["settled_n"] == pytest.approx(6.0, abs=0.4)
    assert abs(out["cartesian_impedance"]["settled_n"] - 6.0) < 2.5  # its few mm of depth, times a soft spring
    assert out["ik"]["settled_n"] > 12.0  # a few mm into a stiff servo: twice the force asked for
    assert out["osc"]["overshoot_n"] < 1.5


def test_wiping_at_a_constant_force_osc_against_a_stiff_arm(desk):
    m, d = _start(desk)
    params = {"force": 8.0, "length": 0.12, "width": 0.06, "passes": 2, "speed": 0.1}
    cmp = compare_methods("wipe", ["osc", "ik"], 14.0, task_params=params, model=m, robot="arm", data=copy.copy(d))
    by = {r.method: r for r in cmp.results}
    osc, ik = by["osc"].report, by["ik"].report
    assert by["osc"].success, osc
    assert abs(osc["force_mean_n"] - 8.0) < 1.0 and osc["force_std_n"] < 3.5
    assert osc["path_wiped_pct"] >= 90.0 and osc["lost_contact_pct"] < 10.0 and osc["track_rms_mm"] < 12.0
    assert not by["ik"].success and ik["force_mean_n"] > 3 * 8.0  # stiff servos pressing the table with the whole error


def test_a_wipe_the_arm_cannot_reach_is_refused_before_it_moves(desk):
    m, d = _start(desk)
    task = make_task("wipe", m, "arm", {"center": [1.4, 0.0], "length": 0.1})
    with pytest.raises(TaskError, match="cannot reach"):
        task.begin(d)


def test_press_and_wipe_have_a_default_world_and_list_their_methods():
    model, robot = task_world("wipe")
    assert robot == "arm" and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table") > 0
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
    mujoco.mj_forward(model, d)
    task = make_task("wipe", model, robot, {"force": 2.0, "length": 0.08, "width": 0.04, "passes": 2})
    task.begin(d)
    assert task.surface.name == "table"
    for method in ("mpc", "mppi", "lqr"):
        with pytest.raises(TaskError, match="does not fit"):
            make_controller(method, task, model, d)
    assert set(TASKS) >= {"press", "wipe"}


def test_a_wiper_pad_replaces_the_tool_point_and_holds_nothing(desk):
    m = desk.session.model
    names = [m.actuator(a).name for a in range(m.nu)]
    assert "arm/fingers" in names and "arm/gripper" not in names
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm/tcp")
    d = desk.session.data
    mujoco.mj_forward(m, d)
    pad = [g for g in range(m.ngeom) if m.geom_bodyid[g] and m.body(m.geom_bodyid[g]).name == "arm/wiper"]
    lowest = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in pad if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
    assert d.site_xpos[site][2] - lowest == pytest.approx(0.0, abs=1e-3)  # the tool point is on the pad's face
    assert end_effectors.canonical("sponge") == "wiper"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_runs_a_wipe_and_compares_how_each_controller_does(desk):
    server = build_server(desk, 0)
    params = {"force": 8.0, "length": 0.1, "width": 0.0, "passes": 1, "speed": 0.1}
    async with Client(server) as client:
        listed = json.loads((await client.call_tool("list_algorithms", {})).content[0].text)
        wipe = next(t for t in listed["tasks"] if t["name"] == "wipe")
        assert "osc" in wipe["methods"] and "force" in wipe["params"]
        compared = await client.call_tool(
            "compare_controllers",
            {"robot": "arm", "task": "wipe", "methods": ["osc", "ik"], "seconds": 9, "task_params": params},
        )
        table = json.loads(compared.content[0].text)
        assert table["best"] == "osc"
        ran = await client.call_tool(
            "run_controller", {"robot": "arm", "task": "wipe", "method": "osc", "seconds": 9, "task_params": params}
        )
        out = json.loads(ran.content[0].text)
        assert out["success"] and abs(out["result"]["force_mean_n"] - 8.0) < 1.5, out
        bad = await client.call_tool("run_controller", {"robot": "arm", "task": "wipe", "method": "lqr"})
        assert bad.is_error and "wipe" in bad.content[0].text


def test_the_default_time_of_a_wipe_counts_the_way_down_to_a_low_table(tmp_path, monkeypatch):
    """The default used to ignore how far the tool has to come down: on a 12 cm table the standard wipe was still
    on its last pass when the time ran out, and every controller seemed to fail."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    seconds = {}
    for height in (0.30, 0.12):
        m, d = _start(_desk(height, f"lab{int(height * 100)}"))
        task = make_task("wipe", m, "arm", {"force": 8.0})
        task.begin(d)
        seconds[height] = task.default_seconds
    assert seconds[0.12] > seconds[0.30] + 1.0  # a longer way down takes longer
    m, d = _start(_desk(0.12, "low"))
    cmp = compare_methods("wipe", ["osc"], None, task_params={"force": 8.0}, model=m, robot="arm", data=copy.copy(d))
    result = cmp.results[0]
    assert result.success and result.report["phase"] == "done", result.report


def _ur5e_over_a_table(top: float = 0.25) -> Playground:
    """A UR5e with a wiper pad and a table right under the pad's start (the arm's home puts it at x -0.13, y 0.49)."""
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("ur", "checker")
    pg.add_robot("arm", library="ur5e")
    pg.update_robot("arm", end_effector="wiper")
    pg.add_object("desk", shape="table", size=[0.5, 0.5, top], pos=[-0.134, 0.492])
    return pg


def test_a_ur5e_presses_and_wipes_at_a_force_and_says_which_surface(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    m, d = _start(_ur5e_over_a_table())
    held = run(make_task("press", m, "arm", {"force": 6.0, "mode": "stop"}), m, copy.copy(d), "osc", 6.0)
    assert held.success and 6.0 <= held.report["peak_n"] <= 7.5, held.report
    assert held.report["surface"] == "desk" and held.report["surface_z_m"] == pytest.approx(0.25, abs=1e-3)
    assert "note" not in held.report
    params = {"force": 8.0, "length": 0.12, "width": 0.06, "passes": 2, "speed": 0.1}
    wipe = run(make_task("wipe", m, "arm", params), m, copy.copy(d), "osc", 14.0)
    assert wipe.success, wipe.report
    assert abs(wipe.report["force_mean_n"] - 8.0) < 1.5 and wipe.report["path_wiped_pct"] >= 90.0
    assert (
        0.0 <= wipe.report["raw_dropout_pct"] <= 100.0
        and wipe.report["raw_dropout_pct"] >= wipe.report["lost_contact_pct"]
    )


def test_a_job_that_ends_up_on_the_floor_says_so(tmp_path, monkeypatch):
    """The arm was not over its table: the wipe ran on the floor beside it and only a low wiped share hinted at why."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("bare", "checker")
    pg.add_robot("arm", library="ur5e")
    pg.update_robot("arm", end_effector="wiper")
    pg.add_object("desk", shape="table", size=[0.3, 0.3, 0.25], pos=[0.9, 0.9])  # a table, but not under the tool
    m, d = _start(pg)
    task = make_task("wipe", m, "arm", {})
    task.begin(d)
    report = task.report(d)
    assert report["surface"] == "floor" and report["surface_z_m"] == pytest.approx(0.0, abs=1e-3)
    assert "not on a table" in report["note"] and "surface=" in report["note"]


def test_a_seeded_comparison_starts_the_tool_somewhere_different_each_time(desk):
    m, d = _start(desk)
    tools = []
    for seed in (0, 1, 2):
        task = make_task("wipe", m, "arm", {})
        start = copy.copy(d)
        task.reset(start, np.random.default_rng(seed))
        mujoco.mj_forward(m, start)
        tools.append(start.site_xpos[task.site].copy())
    assert min(float(np.linalg.norm(a - b)) for i, a in enumerate(tools) for b in tools[:i]) > 0.005
    assert all(float(np.linalg.norm(t - d.site_xpos[task.site])) < 0.08 for t in tools)
