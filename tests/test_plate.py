"""A robot that is not built from parts, given parts or sensors: carried under a plate with them, or replaced by one of
the same kind, and never silently turned into a box."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import pytest

from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"  # offline stand-ins for the real robots
ROVER = {
    "base": {"size": [0.3, 0.2, 0.08], "sensors": [{"type": "lidar", "at": "top"}]},
    "motor": "jga25",
    "parts": [
        {"name": "front_left_wheel", "at": [0.1, 0.12, -0.03], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
        {"name": "rear_left_wheel", "at": [-0.1, 0.12, -0.03], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
    ],
}


def _plated(error: PlaygroundError) -> dict:
    """The build a refusal offers, the robot carried on a plate."""
    found = re.search(r"build=(\{.*\})\); joints become", str(error))
    assert found, str(error)
    return json.loads(found[1])


def _build(pg: Playground, name: str) -> dict:
    return next(r for r in pg.current_world().robots if r.name == name).build


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("yard")
    return pg


def test_a_quadruped_given_only_a_lidar_rides_a_plate_with_it(world):
    """Regression: update_robot("dog", build={"base": {"sensors": ["lidar"]}}) on the quadruped template made a 1 kg
    box with no joints, kind other, and said only "now built from parts". Sensors alone now go on a plate that
    carries the dog, and it walks with the lidar on its back."""
    world.add_robot("dog", template="quadruped")
    done = world.update_robot("dog", build={"base": {"sensors": ["lidar"]}})
    note = next(n for n in done["notes"] if "plate as its base" in n)
    assert "with the sensors on the plate" in note and "still a quadruped" in note and "dog/quadruped/..." in note
    assert world.behaviors.kind("dog").value == "quadruped" and _moving(world, "dog") == 12
    world.run_fast(lambda: False, 0.5)
    start = world.behaviors.pose("dog")["position"]
    world.behaviors.move("dog", 0.3, 0.0)
    world.run_fast(lambda: False, 3.0)
    end = world.behaviors.pose("dog")
    assert end["upright"] and end["position"][0] - start[0] > 0.5
    assert world.state()["lidars"]["dog/base_lidar"]["hits"] == 0  # its own legs are not obstacles


def test_a_build_that_is_another_robot_is_refused_with_the_one_that_keeps_it(world):
    world.add_robot("dog", template="quadruped")
    box = {"base": {"size": [0.3, 0.2, 0.1], "sensors": ["lidar"]}}
    with pytest.raises(PlaygroundError, match="into a robot of kind other with 0 joints") as refused:
        world.update_robot("dog", build=box)
    assert world.current_world().robots[0].template == "quadruped"  # nothing changed
    done = world.update_robot("dog", build=_plated(refused.value))
    assert any("carrying the quadruped template as it was" in n for n in done["notes"])
    assert world.behaviors.kind("dog").value == "quadruped" and _moving(world, "dog") == 12


def test_add_part_on_a_go2_carries_it_on_a_plate_with_the_part(world):
    """The add_part refusal for a library robot used to suggest a whole rebuild, and a Go2 rebuilt with a lidar puck
    became a 1.1 kg cylinder; then it offered the plate as a build to send. In both of the reviewer's sessions the
    agent's first sensor call was add_part, a guaranteed failed call. add_part now carries the Go2 on a plate, the
    gallery's go2_arm layout (the plate 64.5 mm above the Go2's frame), with the part on the plate."""
    world.add_robot("go2", library="go2")
    puck = {"name": "puck", "at": "top", "shape": "cylinder", "size": [0.07, 0.04], "joint": "fixed"}
    puck["sensors"] = ["lidar"]
    done = world.add_part("go2", puck)
    built = _build(world, "go2")
    assert built["parts"][0] == {"name": "go2", "at": [0, 0, -0.0645], "robot": {"library": "go2"}}
    assert built["parts"][1] == puck
    note = next(n for n in done["notes"] if "plate as its base" in n)
    assert "with puck on the plate" in note and "joints are now go2/go2/..." in note, note
    assert world.behaviors.kind("go2").value == "quadruped" and _moving(world, "go2") == 12
    assert "go2/puck_lidar" in world.state()["lidars"]
    world.undo()  # one edit
    assert next(r for r in world.current_world().robots if r.name == "go2").library == "go2"
    with pytest.raises(PlaygroundError, match="the quadruped would be gone"):  # a puck alone is no Go2
        world.update_robot("go2", build={"parts": [puck]})


def test_a_template_with_params_is_carried_with_them(world):
    """The add_part refusal for a quadruped with params ran to 345 characters; the carry needs none."""
    world.add_robot("dog", template="quadruped", params={"leg_length": 0.3})
    world.add_part("dog", {"name": "eye", "at": "front", "shape": "box", "size": [0.03, 0.03, 0.02], "joint": "fixed"})
    assert _build(world, "dog")["parts"][0]["robot"] == {"template": "quadruped", "params": {"leg_length": 0.3}}


def test_a_wrong_part_for_a_template_robot_says_what_is_wrong_with_the_part(world):
    world.add_robot("dog", template="quadruped")
    with pytest.raises(ValueError, match="tail"):
        world.add_part("dog", {"name": "tail", "shape": "teapot"})
    assert world.current_world().robots[0].template == "quadruped"


def test_a_whole_robot_of_the_same_kind_still_replaces_a_template_and_the_note_says_what_it_is(world):
    world.add_robot("rover", template="rover")
    done = world.update_robot("rover", build=ROVER)
    note = next(n for n in done["notes"] if "now built from parts" in n)
    assert "still a rover, with 4 joints where it had" in note, note


def test_a_humanoid_that_cannot_ride_a_plate_is_told_so(world):
    world.add_robot("h", template="humanoid")
    with pytest.raises(PlaygroundError, match="would no longer be a humanoid") as refused:
        world.add_part("h", {"name": "puck", "at": "top", "shape": "sphere", "size": [0.05], "sensors": ["lidar"]})
    assert "a current limit: a humanoid cannot ride a plate yet" in str(refused.value)
    assert world.current_world().robots[0].template == "humanoid"


def _moving(pg: Playground, robot: str) -> int:
    m = pg.session.model
    kinds = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    return sum(int(m.jnt_type[j]) in kinds and m.joint(j).name.startswith(f"{robot}/") for j in range(m.njnt))


def test_an_arm_fixed_to_the_floor_is_carried_on_a_plate_under_it(world):
    world.add_robot("arm", template="arm")
    eye = {"name": "eye", "at": "top", "shape": "box", "size": [0.03, 0.03, 0.02], "joint": "fixed"}
    world.add_part("arm", eye)
    built = _build(world, "arm")
    assert built["base"]["fixed"] is True and built["parts"][0] == {
        "name": "arm",
        "at": "top",
        "robot": {"template": "arm"},
    }
    assert world.behaviors.kind("arm").value == "arm"


def test_the_plate_rests_on_the_rover_lid_not_on_a_dressing_meshs_box(world):
    """The box round a dressing mesh turned in the rover template's frame reached 7.1 cm, where its lid is 3.7 cm,
    and the plate floated 3 cm above the lid in the viewer. Rays down onto the body find the lid."""
    world.add_robot("rover", template="rover")
    world.add_part("rover", dict(LIDAR_PUCK))
    drop = -_build(world, "rover")["parts"][0]["at"][2]
    assert 0.037 + 0.006 < drop < 0.05  # the lid's top, half the plate and the gap; the box gave 0.0785


LIDAR_PUCK = {"name": "puck", "at": "top", "shape": "cylinder", "size": [0.06, 0.04], "joint": "fixed"}


def test_no_adapter_plate_is_drawn_through_a_robot_hanging_under_its_plate(world):
    """Regression: the aluminium plate a mounted template robot stands on was drawn for one hung under a plate too:
    a disc 0.48 m across through a carried quadruped's trunk, and one round a carried rover's chassis. An arm
    standing on its part keeps its plate."""
    world.add_robot("dog", template="quadruped", pos=[0.0, 0.6])
    world.add_robot("rover", template="rover", pos=[0.0, 0.0])
    for name in ("dog", "rover"):
        world.add_part(name, dict(LIDAR_PUCK))
    world.add_robot("picker", design="lift_picker", pos=[1.5, 0.0])
    m = world.session.model

    def drawn(body: str) -> list[float]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        return [float(max(m.geom_aabb[g][3:])) for g in range(m.ngeom) if m.geom_bodyid[g] == b]

    assert drawn("dog/quadruped") == [] and drawn("rover/rover") == []
    assert max(drawn("picker/arm")) > 0.05  # the arm on the lift still stands on its plate
