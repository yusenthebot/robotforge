"""A behavior tree in a real script run: the tree really drives the robots, and what the viewer is shown follows it."""

import json
import textwrap

import pytest
from mcp import Client

from robotforge.flow_board import FlowBoard
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

pytestmark = pytest.mark.anyio

TREE = textwrap.dedent(
    """
    from robotforge.flow import Action, Condition, Retry, Sequence, Tree, sensor_is
    from robotforge.script import connect

    sim = connect()
    arm = sim.robot("arm")
    tries = []

    def place(ctx):
        tries.append(ctx.time)
        if len(tries) == 1:
            return False  # the first attempt does not work: Retry tries again
        return arm.arm.pick_and_place("cube", "tray")

    tree = Tree(
        Sequence(
            Condition(lambda c: arm.position is not None, name="arm is there"),
            Retry(Action(place, name="put the cube in the tray"), times=3),
            name="errand",
        ),
        name="cube errand",
    )
    result = tree.run(sim, arm)
    print("result", result)
    print("tries", len(tries))
    print("cube", [round(v, 2) for v in sim.position("cube")][:2])
    print(tree.summary())
    """
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("flowing")
    playground.add_robot("arm", template="arm")
    playground.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")
    playground.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    return playground


async def test_a_tree_with_a_retry_drives_the_arm_and_the_board_keeps_its_end_state(pg):
    server = build_server(pg, 0)
    async with Client(server) as client:
        await client.call_tool("write_script", {"name": "errand", "code": TREE})
        result = await client.call_tool("run_script", {"name": "errand"})
    out = json.loads("".join(c.text for c in result.content if getattr(c, "type", "") == "text"))
    assert out["ok"], out["output"]
    assert "result success" in out["output"]
    assert "tries 2" in out["output"]
    assert "attempt 1 of 3 failed" in out["output"]
    board = pg.flow.message()
    assert board["name"] == "cube errand"
    assert [n["name"] for n in board["nodes"]] == ["errand", "arm is there", "retry 3x", "put the cube in the tray"]
    assert board["status"] == "ssss"  # the whole job ended well
    assert board["notes"] == {"2": "attempt 2 of 3"}  # it worked on the second


def test_the_board_checks_what_a_script_sends():
    board = FlowBoard()
    nodes = [{"id": 0, "parent": -1, "kind": "sequence", "name": "job"}, {"id": 1, "parent": 0, "kind": "action"}]
    with pytest.raises(ValueError, match="needs the nodes"):
        board.update("t", None, "r", {})
    board.update("t", nodes, "rr", {"1": "attempt 2 of 3", "x": "dropped"})
    assert board.message()["notes"] == {"1": "attempt 2 of 3"}
    revision = board.rev
    board.update("t", None, "sr", {})  # the structure stays, the status moves
    assert board.rev > revision and board.message()["nodes"] == board.nodes
    with pytest.raises(ValueError, match="status"):
        board.update("t", None, "s", {})
    with pytest.raises(ValueError, match="status"):
        board.update("t", None, "sx", {})
    with pytest.raises(ValueError, match="parent"):
        board.update("t", [{"id": 0, "parent": 1, "kind": "a", "name": "a"}], "r", {})
    with pytest.raises(ValueError, match="at most"):
        board.update("t", [{"id": i, "parent": -1} for i in range(300)], "." * 300, {})
    board.clear()
    assert board.message()["nodes"] == [] and board.rev > revision
