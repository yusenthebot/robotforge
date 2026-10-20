"""Operating a mechanism by force: the hand pushes with at most a given force, and says when that is not enough."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import pytest
from mcp import Client

from robotforge.behaviors import operate
from robotforge.design.world import ObjectEntry
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import mechanisms
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "grass")
    playground.add_robot("arm", library="panda")
    return playground


def _operate(pg: Playground, target: str, action: str, amount: float | None, force: float | None) -> dict:
    op, found, shape = operate.start(pg.behaviors, "arm", target, action, amount, True, force)
    pg.run_fast(lambda: op.done, 90.0)
    pg.run_fast(lambda: False, 0.3)
    s = pg.session
    state = mechanisms.read(s.model, s.data, {found: shape}, s.controllers.get(mechanisms.KEY)).get(found, {})
    return operate.result(op, s.model, s.data, state)


def _friction(pg: Playground, joint: str) -> float:
    m = pg.session.model
    return float(m.dof_frictionloss[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]])


def test_a_lever_is_tipped_by_a_hand_pushing_with_a_few_newtons(pg):
    pg.add_object("lv", shape="lever", pos=[0.45, 0.0])
    out = _operate(pg, "lv", "turn", 30, 15.0)
    assert out["success"] and out["moved"] == pytest.approx(30, abs=4)
    force = out["force"]
    assert force["limit_n"] == 15.0 and 1.0 < force["pushed_n"] < 12.0 and not force["stalled"]


def test_a_drawer_is_pulled_by_force_and_the_force_it_took_is_reported(pg):
    pg.add_object("chest", shape="drawer", pos=[0.7, 0.0], yaw=-90)
    out = _operate(pg, "chest", "open", 0.2, 30.0)
    assert out["success"] and out["moved"] == pytest.approx(0.2, abs=0.02)
    assert out["force"]["pushed_n"] < 30.0 and not out["force"]["stalled"]


def test_a_stiff_mechanism_stops_a_hand_that_may_push_only_a_little(pg):
    pg.add_object("chest", shape="drawer", pos=[0.7, 0.0], yaw=-90, resist=60.0)
    assert _friction(pg, "chest/drawer") > 60.0
    out = _operate(pg, "chest", "open", 0.2, 25.0)
    assert not out["success"] and out["moved"] < 0.08
    assert out["force"]["stalled"] and out["force"]["pushed_n"] == 25.0
    assert any("limit of 25 N" in note for note in out["notes"])


def test_operating_without_a_force_is_as_it_was(pg):
    pg.add_object("chest", shape="drawer", pos=[0.7, 0.0], yaw=-90)
    out = _operate(pg, "chest", "open", 0.2, None)
    assert out["success"] and "force" not in out


def test_resist_is_a_force_at_the_handle_and_only_for_things_that_move(pg):
    pg.add_object("tap", shape="valve", size=[0.16, 0.12], pos=[0.4, -0.3], resist=10.0)
    rim = 0.16 / 2 - 0.016 / 2
    assert _friction(pg, "tap/wheel") == pytest.approx(0.04 + 10.0 * rim, rel=0.02)  # a torque: force times the lever
    with pytest.raises(ValueError, match="resist"):
        ObjectEntry(name="b", shape="box", resist=5.0)
    assert "resist" not in ObjectEntry(name="b2", shape="drawer").model_dump(mode="json", exclude_defaults=True)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_operate_takes_a_force(pg):
    pg.add_object("lv", shape="lever", pos=[0.45, 0.0])
    server = build_server(pg, 0)
    async with Client(server) as client:
        done = await client.call_tool(
            "operate", {"robot": "arm", "target": "lv", "action": "turn", "amount": 30, "force": 15}
        )
        out = json.loads(done.content[0].text)
        assert out["success"] and out["force"]["limit_n"] == 15


@pytest.mark.parametrize("force", [float("nan"), float("inf"), 0.0, -5.0, 5000.0])
def test_a_force_that_is_not_a_sensible_number_of_newtons_is_refused(pg, force):
    pg.add_object("lv", shape="lever", pos=[0.45, 0.0])
    with pytest.raises(operate.OperateError, match="newtons"):
        operate.start(pg.behaviors, "arm", "lv", "turn", 30, True, force)


def test_a_force_with_press_is_refused_instead_of_ignored(pg):
    pg.add_object("btn", shape="button", pos=[0.45, 0.0])
    with pytest.raises(operate.OperateError, match="press"):
        operate.start(pg.behaviors, "arm", "btn", "press", None, True, 5.0)
