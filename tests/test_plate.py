"""A robot that is not built from parts, given parts or sensors: kept, carried under a plate, or replaced by one of the
same kind, and never silently turned into a box."""

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


def _plated(error: PlaygroundError, part: dict | None = None) -> dict:
    """The build a refusal offers, the robot carried on a plate, with `part` in add_part's place for it."""
    found = re.search(r"build=(\{.*\})\); joints become", str(error))
    assert found, str(error)
    text = found[1]
    if part is not None:
        assert text.endswith(", <the part>]}"), text
        text = text.replace("<the part>", json.dumps(part))
    return json.loads(text)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("yard")
    return pg


def test_a_quadruped_given_only_a_lidar_is_not_turned_into_a_box(world):
    """Regression: update_robot("dog", build={"base": {"sensors": ["lidar"]}}) on the quadruped template made a 1 kg
    box with no joints, kind other, and said only "now built from parts". It is refused with the build that keeps
    the dog, and that build walks it with the lidar on its back."""
    world.add_robot("dog", template="quadruped")
    with pytest.raises(PlaygroundError, match="into a robot of kind other with 0 joints") as refused:
        world.update_robot("dog", build={"base": {"sensors": ["lidar"]}})
    assert world.current_world().robots[0].template == "quadruped"  # nothing changed
    done = world.update_robot("dog", build=_plated(refused.value))
    assert any("carrying the quadruped template as it was" in n for n in done["notes"])
    assert world.behaviors.kind("dog").value == "quadruped" and _moving(world, "dog") == 12
    world.run_fast(lambda: False, 0.5)
    start = world.behaviors.pose("dog")["position"]
    world.behaviors.move("dog", 0.3, 0.0)
    world.run_fast(lambda: False, 3.0)
    end = world.behaviors.pose("dog")
    assert end["upright"] and end["position"][0] - start[0] > 0.5
    assert world.state()["lidars"]["dog/base_lidar"]["hits"] == 0  # its own legs are not obstacles


def test_add_part_on_a_go2_offers_the_plate_with_a_place_for_the_part(world):
    """The add_part refusal for a library robot used to suggest a whole rebuild, and a Go2 rebuilt with a lidar puck
    became a 1.1 kg cylinder. Now it offers the Go2 on a plate, the gallery's go2_arm layout (the plate 64.5 mm above
    the Go2's frame), with a place for the part on the plate."""
    world.add_robot("go2", library="go2")
    puck = {"name": "puck", "at": "top", "shape": "cylinder", "size": [0.07, 0.04], "joint": "fixed"}
    puck["sensors"] = ["lidar"]
    with pytest.raises(PlaygroundError, match="parts go on robots built from parts") as refused:
        world.add_part("go2", puck)
    offered = _plated(refused.value, puck)
    assert offered["parts"][0] == {"name": "go2", "at": [0, 0, -0.0645], "robot": {"library": "go2"}}
    assert offered["parts"][1] == puck and "joints become go2/go2/..." in str(refused.value)
    with pytest.raises(PlaygroundError, match="the quadruped would be gone"):  # a puck alone is no Go2
        world.update_robot("go2", build={"parts": [offered["parts"][1]]})
    world.update_robot("go2", build=offered)
    assert world.behaviors.kind("go2").value == "quadruped" and _moving(world, "go2") == 12


def test_a_whole_robot_of_the_same_kind_still_replaces_a_template_and_the_note_says_what_it_is(world):
    world.add_robot("rover", template="rover")
    done = world.update_robot("rover", build=ROVER)
    note = next(n for n in done["notes"] if "now built from parts" in n)
    assert "still a rover, with 4 joints where it had" in note, note


def test_a_humanoid_that_cannot_ride_a_plate_is_told_so(world):
    world.add_robot("h", template="humanoid")
    with pytest.raises(PlaygroundError, match="would no longer be a humanoid: build one from parts instead"):
        world.add_part("h", {"name": "puck", "at": "top", "shape": "sphere", "size": [0.05], "sensors": ["lidar"]})


def _moving(pg: Playground, robot: str) -> int:
    m = pg.session.model
    kinds = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    return sum(int(m.jnt_type[j]) in kinds and m.joint(j).name.startswith(f"{robot}/") for j in range(m.njnt))


def test_an_arm_fixed_to_the_floor_is_carried_on_a_plate_under_it(world):
    world.add_robot("arm", template="arm")
    eye = {"name": "eye", "at": "top", "shape": "box", "size": [0.03, 0.03, 0.02], "joint": "fixed"}
    with pytest.raises(PlaygroundError, match="joints become arm/arm/") as refused:
        world.add_part("arm", eye)
    offered = _plated(refused.value, eye)
    assert offered["base"]["fixed"] is True and offered["parts"][0] == {
        "name": "arm",
        "at": "top",
        "robot": {"template": "arm"},
    }
    world.update_robot("arm", build=offered)
    assert world.behaviors.kind("arm").value == "arm"
