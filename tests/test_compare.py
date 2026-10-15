"""Face-offs: our template arm next to the AgileX PiPER and the Franka Panda, same tests, same physics.

The offline tests use the stand-ins in tests/fixtures/menagerie (the real models' bodies, joints, inertia
and actuators with fitted shapes instead of meshes). ROBOTFORGE_NETWORK_TESTS=1 also runs the race on
the real Menagerie models.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.design import library
from robotforge.learn import compare
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
NETWORK = os.environ.get("ROBOTFORGE_NETWORK_TESTS") == "1"


@pytest.fixture
def pg(tmp_path, monkeypatch, request):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    if not request.node.name.startswith("test_real_"):
        monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.lessons.start("arm_faceoff")
    return playground


def _play(pg: Playground, seconds: float) -> dict:
    update = None
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
    return update


def _run(pg: Playground, test: str) -> dict:
    pg.lessons.handle({"action": "press", "control": test})
    for _ in range(90):
        update = _play(pg, 1.0)
        if pg.lessons.lesson.test is None:
            return update
    raise AssertionError(f"the {test} test did not finish")


def _rows(update: dict) -> dict[str, dict]:
    return {r["label"]["en"]: r for r in update["live"]["table"]["rows"] if "cells" in r}


def _to_pick(pg: Playground) -> None:
    """Pass the steps before the race: workspace, reach test, gravity compensation, the quiz."""
    pg.lessons.handle({"action": "set", "control": "workspace", "value": 1})
    _play(pg, 0.2)
    pg.lessons.handle({"action": "next"})
    _run(pg, "reach")
    pg.lessons.handle({"action": "next"})
    pg.lessons.handle({"action": "set", "control": "gravcomp", "value": 1})
    _run(pg, "reach")
    pg.lessons.handle({"action": "next"})
    pg.lessons.handle({"action": "answer", "choice": 1})
    pg.lessons.handle({"action": "next"})


def test_piper_is_in_the_library_with_a_tool_point_and_gripper(monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    robot = library.get("piper")
    assert (robot.kind, robot.license, robot.folder) == ("arm", "MIT", "agilex_piper")
    model = library.build("piper").spec.compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper") >= 0
    panda = library.build("panda").spec.compile()  # its gripper actuator is renamed like ours
    assert mujoco.mj_name2id(panda, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper") >= 0


def test_face_offs_are_listed_apart_from_lessons_and_challenges(pg):
    kinds = {c["id"]: c["kind"] for c in pg.lessons.catalog()}
    assert kinds["arm_faceoff"] == "compare"


def test_the_spec_table_is_worked_out_from_the_models(pg):
    rows = _rows(_play(pg, 0.5))
    assert [c["en"] for c in rows["Joints"]["cells"]] == ["5 + gripper", "6 + gripper", "7 + gripper"]
    reach = [float(c.split()[0]) for c in rows["Reach"]["cells"]]
    assert reach[0] < reach[1] < reach[2] and rows["Reach"]["best"] == 2
    holds = [c["en"] for c in rows["Against gravity"]["cells"]]
    assert holds[0].startswith("P control only") and holds[1].startswith("gravity compensation")
    assert holds[2].startswith("very stiff motors")
    rig = pg.lessons.describe_active()["rig"]
    assert all(len(e["cloud"]) > 300 for e in rig["entrants"])  # workspace clouds to draw


def test_gravity_compensation_takes_away_our_arms_sag(pg):
    pg.lessons.handle({"action": "set", "control": "workspace", "value": 1})
    assert "meet" in _play(pg, 0.2)["passed"]
    pg.lessons.handle({"action": "next"})
    before = _run(pg, "reach")["results"]["reach"]
    assert before["ours"]["mean_mm"] > 10 and before["ours"]["reached"] == 0  # P-only servos sag
    assert before["piper"]["furthest"] < before["panda"]["furthest"]
    pg.lessons.handle({"action": "next"})
    pg.lessons.handle({"action": "set", "control": "gravcomp", "value": 1})
    update = _run(pg, "reach")
    after = update["results"]["reach"]
    assert after["ours"]["mean_mm"] < 2 and after["ours"]["reached"] >= 2
    assert after["panda"]["reached"] == len(compare.TARGETS)  # its last centimetre of sag is gone too
    assert after["piper"]["reached"] < after["panda"]["reached"]  # its wrist cannot point down that far out
    assert "gravity" in update["passed"]
    pg.lessons.handle({"action": "set", "control": "gravcomp", "value": 0})
    model = pg.session.model
    assert model.body_gravcomp[model.body("piper/link6").id] == 1  # off restores the PiPER's own


def test_the_race_starts_every_arm_from_home(pg):
    _to_pick(pg)
    update = _run(pg, "pick")
    race = update["results"]["pick"]
    assert race["ours"]["success"] and race["piper"]["success"]
    assert "pick" in update["passed"]
    assert {"Pick and place", "Pick-and-place time"} <= set(_rows(update))


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_arms_all_place_their_cubes(pg):
    _to_pick(pg)
    race = _run(pg, "pick")["results"]["pick"]
    assert all(race[name]["success"] for name in ("ours", "piper", "panda"))


# ---- humanoids: ours and the Unitree G1 on its shipped learned policy --------------------------------


@pytest.fixture
def humanoids(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.lessons.start("humanoid_faceoff")
    return playground


def test_the_g1_walks_straight_on_its_learned_policy(humanoids):
    pg = humanoids
    root = pg.lessons.lesson.facts["g1"]["root"]
    data = pg.session.data
    assert pg.behaviors.kind("g1").value == "humanoid"  # from the catalog: its joints are not named like ours
    _play(pg, 2.0)
    start = data.xpos[root].copy()
    pg.behaviors.move("g1", 0.5, 0.0)
    _play(pg, 6.0)
    moved = data.xpos[root] - start
    assert moved[0] > 2.5  # about 0.5 m/s
    assert abs(moved[1]) < 0.3  # the heading hold keeps "straight" straight
    assert data.xmat[root][8] > 0.9


def test_a_hand_written_walk_against_a_learned_one(humanoids):
    pg = humanoids
    assert not pg.lessons.lesson.missing  # both robots have a walking controller
    _play(pg, 1.0)
    pg.lessons.handle({"action": "next"})
    walk = _run(pg, "walk")["results"]["walk"]
    ours, g1 = walk["ours"], walk["g1"]
    assert not ours["fell"] and not g1["fell"]
    assert g1["speed"] > 4 * ours["speed"]  # much faster in metres per second...
    assert g1["heights_per_s"] < 2 * ours["heights_per_s"]  # ...far less so for its size
    assert g1["cost_of_transport"] < ours["cost_of_transport"]
    _play(pg, 0.2)
    pg.lessons.handle({"action": "next"})
    update = _run(pg, "push")
    shoves = update["results"]["push"]
    assert shoves["ours"]["levels"] >= 2 and shoves["g1"]["levels"] >= 2
    _play(pg, 0.2)
    assert {"Froude number", "Levels survived"} <= set(_rows(update))


def test_an_idle_g1_stays_where_it_stood(humanoids):
    """Regression: told to stand, the learned policy drifted about 6.5 cm/s and walked into our humanoid's lane
    within half a minute of the face-off opening."""
    pg = humanoids
    root = pg.lessons.lesson.facts["g1"]["root"]
    data = pg.session.data
    _play(pg, 1.0)
    start = data.xpos[root][:2].copy()
    worst = 0.0
    for _ in range(30):
        _play(pg, 1.0)
        worst = max(worst, float(np.linalg.norm(data.xpos[root][:2] - start)))
    assert worst < 0.35, f"drifted up to {worst:.2f} m"
    assert data.xmat[root][8] > 0.9
