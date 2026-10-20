"""Picking things of no simple shape with a grasp planned from the shape (behaviors/grasping.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp import Client

from robotforge.behaviors.manager import BehaviorError
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"
SPOTS = {"tee": [0.45, 0.15], "bottle": [0.62, 0.05], "stone": [0.3, 0.25]}
SIZES = {"bottle": [0.05, 0.1]}  # a small bottle: the stand-in gripper's jaws open 7 cm


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """A UR5e with a Robotiq 2F-85 beside a bin, on the floor."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot("arm", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    pg.add_object("bin", shape="bin", size=[0.25, 0.2, 0.08], pos=[0.0, -0.45], color="gray")
    return pg


def _pick(pg: Playground, obj: str) -> dict:
    arm, info = pg.behaviors.pick_and_place("arm", obj, "bin")
    pg.run_fast(lambda: arm.done, 60.0)
    pg.run_fast(lambda: False, 0.5)
    return {**pg.behaviors.placed(obj, info["region"]), "grasp": info.get("grasp")}


def test_a_t_block_is_taken_by_its_stem_not_across_its_bar(cell):
    cell.add_object("tee", shape="tee", pos=SPOTS["tee"])
    plan = cell.behaviors.plan_grasps("arm", "tee")
    best = plan["best"]
    assert plan["jaws_open_m"] == pytest.approx(0.07, abs=0.01) and best["width_m"] < 0.04
    assert best["off_center_m"] < 0.03 and len(plan["grasps"]) >= 2
    # the stem is out of the bar toward -y: the grasp is on it, at most a bar's width from the bar's middle
    assert best["at"][1] < SPOTS["tee"][1] - 0.01


@pytest.mark.parametrize("shape", ["tee", "bottle", "stone"])
def test_irregular_things_are_picked_by_the_planned_grasp_and_put_in_the_bin(cell, shape):
    cell.add_object(shape, shape=shape, pos=SPOTS[shape], **({"size": SIZES[shape]} if shape in SIZES else {}))
    cell.run_fast(lambda: False, 0.5)
    result = _pick(cell, shape)
    assert result["success"], result
    if shape == "bottle":  # a modelled object: taken by one of its own hints (behaviors/hints.py), not by the planner
        assert result["grasp"]["kind"] == "neck"
    else:
        assert result["grasp"]["width_m"] < 0.07 - 0.012 and result["grasp"]["grip"] > 0.9


def test_a_bottle_is_taken_by_its_neck_because_the_palm_is_in_the_way(cell):
    cell.add_object("bottle", shape="bottle", pos=SPOTS["bottle"], size=[0.05, 0.1])
    best = cell.behaviors.plan_grasps("arm", "bottle")["best"]
    assert best["width_m"] == pytest.approx(0.026, abs=0.006) and best["at"][2] > 0.06  # the neck, not the body


def test_a_thing_too_wide_for_the_jaws_is_refused_with_why(cell):
    cell.add_object("stone", shape="stone", pos=SPOTS["stone"], size=[0.2, 0.08])
    plan = cell.behaviors.plan_grasps("arm", "stone")
    assert plan["best"] is None and "too wide" in plan["why_none"]
    with pytest.raises(BehaviorError, match="cannot take hold of stone"):
        cell.behaviors.pick_and_place("arm", "stone", "bin")


def test_boxes_cans_and_balls_keep_their_own_grasps(cell):
    cell.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.45, 0.15])
    arm, info = cell.behaviors.pick_and_place("arm", "cube", "bin")
    assert "grasp" not in info  # a plain box is grasped square to its faces, as ever
    cell.run_fast(lambda: arm.done, 60.0)
    assert cell.behaviors.placed("cube", info["region"])["success"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_plans_a_grasp_and_reports_the_one_it_took(cell):
    cell.add_object("tee", shape="tee", pos=SPOTS["tee"])
    server = build_server(cell, 0)
    async with Client(server) as client:
        planned = json.loads((await client.call_tool("plan_grasp", {"robot": "arm", "object": "tee"})).content[0].text)
        assert planned["grasps"] and planned["best"] == planned["grasps"][0] and planned["why_none"] is None
        done = json.loads(
            (await client.call_tool("pick_and_place", {"robot": "arm", "object": "tee", "target": "bin"}))
            .content[0]
            .text
        )
        assert done["success"] and done["grasp"]["closing_deg"] == planned["best"]["closing_deg"]
        missing = await client.call_tool("plan_grasp", {"robot": "arm", "object": "nothing"})
        assert missing.is_error


def test_a_pour_keeps_the_jaws_across_the_stream_and_knows_how_high_the_finger_links_reach(cell):
    # Regression: with a finger on the lip side, or finger links round the mouth, balls caught on the hand and stayed
    # in the cup (a UR5e with a Robotiq poured 7 of 12 trials).
    import numpy as np

    from robotforge.behaviors import hints, pour

    arm = cell.behaviors._arm_motion("arm")
    assert hints.palm_of(cell.session.model, arm).links > 0.0
    arm.ik.point(arm.chain, arm.q)
    closing = arm.ik.data.xmat[arm.chain.body].reshape(3, 3) @ arm.closing
    along = np.array([closing[0], closing[1], 0.0])
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0], 0.0])
    assert not pour.jaws_across(arm, arm.q, along)
    assert pour.jaws_across(arm, arm.q, across)
