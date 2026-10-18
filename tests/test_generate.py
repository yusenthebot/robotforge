"""Worlds generated from a sentence (design/generate): what a sentence is read as, and the placement rules the
world keeps: furniture apart, things on tops inside their edges, within reach of floor the robot gets to, and
steady once the physics runs."""

from __future__ import annotations

import math

import pytest

from robotforge.design import props
from robotforge.design.generate import draft as drafting
from robotforge.design.generate import generate, unsettled, words
from robotforge.design.generate.space import FloorMap, Footprint
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

SENTENCES = [
    "a small apartment with a kitchen, a living room and a bedroom",
    "a warehouse with three rows of racks and some pallets",
    "a park with hills, stairs and a ramp, a few trees and two benches",
    "一个有厨房、客厅和卧室的家，桌上有三个苹果",
]


@pytest.mark.parametrize(
    ("text", "kind", "rooms", "counts", "colors", "terrain"),
    [
        ("a kitchen and two bedrooms", "house", ["kitchen", "bedroom", "bedroom"], {}, {}, []),
        ("a warehouse with three rows of racks", "warehouse", [], {"rack": 3}, {}, []),
        ("a rocky hillside with many trees", "outdoor", [], {"tree": 7}, {}, ["rough", "hills"]),
        (
            "an office with four desks and three red cups",
            "house",
            ["office"],
            {"desk": 4, "cup": 3},
            {"cup": "red"},
            [],
        ),
        ("书房里有两个书架和一张书桌", "house", ["office"], {"shelf": 2, "desk": 1}, {}, []),
        ("花园里有沙发和红色的球", "outdoor", [], {"sofa": 0, "ball": 0}, {"ball": "red"}, []),
        ("把五个蓝色的方块放进篮子", "house", [], {"cube": 5, "bin": 0}, {"cube": "blue"}, []),
        ("Can you make a park with stairs and a ramp?", "outdoor", [], {}, {}, ["stairs", "ramp"]),
    ],
)
def test_a_sentence_is_read_as_a_place_its_rooms_and_what_is_in_it(text, kind, rooms, counts, colors, terrain):
    wish = words.read(text)
    assert wish.kind == kind
    assert wish.rooms == rooms
    assert wish.counts == counts
    assert wish.colors == colors
    assert sorted(wish.terrain) == sorted(terrain)


def test_one_dictionary_does_not_read_what_another_already_did():
    """书房 is a study, not a book; 沙发 a sofa, not sand on the ground; "can you" asks, it is not a can."""
    assert "book" not in words.read("一个书房").counts
    assert words.read("客厅里有一张沙发").ground is None
    assert "can" not in words.read("can you build a kitchen").counts


def test_the_same_sentence_and_seed_give_the_same_world_and_another_seed_another():
    a = generate(SENTENCES[0], seed=3, check=False).content
    b = generate(SENTENCES[0], seed=3, check=False).content
    c = generate(SENTENCES[0], seed=4, check=False).content
    assert a.objects == b.objects and a.terrain == b.terrain
    assert a.objects != c.objects


def _floor_print(o: dict) -> Footprint:
    hx, hy = props.footprint(o["shape"], o["size"])
    return Footprint(o["pos"][0], o["pos"][1], hx, hy, o.get("yaw", 0.0))


PARTS = ("_back", "_mattress", "_pillow", "_arm_1", "_arm_2", "_tv")  # parts of one piece, which touch each other


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("sentence", SENTENCES)
def test_furniture_stands_apart_and_things_stand_on_tops_in_reach(sentence, seed):
    made = generate(sentence, seed=seed, check=False)
    objects = made.content.objects
    standing = [
        o
        for o in objects
        if len(o["pos"]) == 2
        and o["shape"] not in ("marker", "wall", "tree", "rock")
        and o["name"] not in made.loose
        and not o["name"].endswith(PARTS)
    ]
    for i, a in enumerate(standing):
        for b in standing[i + 1 :]:
            racks = a["name"].startswith("rack") and b["name"].startswith("rack")  # a row of racks stands end to end
            gap = -0.01 if racks else 0.05
            assert not _floor_print(a).overlaps(_floor_print(b), gap), (a["name"], b["name"])
    walls = [_floor_print(o) for o in objects if o["shape"] == "wall"]
    start = made.content.ground_slots[0].pos
    obstacles = [_floor_print(o) for o in objects if o["shape"] not in ("marker",) and o["name"] not in made.loose]
    floor = FloorMap.of((-15, 15, -15, 15), obstacles, drafting.ROBOT_RADIUS)
    assert floor.flood(*start) > 100
    assert made.loose, "a generated world has things to pick up"
    for name in made.loose:
        o = next(o for o in objects if o["name"] == name)
        x, y = o["pos"][0], o["pos"][1]
        assert floor.reachable_within(x, y, drafting.REACH + 0.06), name
        assert not any(w.contains(x, y) for w in walls), name
        if len(o["pos"]) == 3:  # on a top: low enough for an arm, and resting on it (not floating, not sunk)
            assert drafting.LOW - 0.01 <= o["pos"][2] <= drafting.HIGH + 0.15, name
    # a robot gets to the front of every table and counter on the floor (one against a wall faces the room): a desk
    # put in a corner behind a bed could not be reached
    for o in objects:
        if o["shape"] not in ("table", "counter") or len(o["pos"]) == 3:
            continue
        fp = _floor_print(o)
        fronts = [fp.world(0.0, -fp.hy - 0.35), fp.world(0.0, fp.hy + 0.35), fp.world(fp.hx + 0.35, 0.0),
                  fp.world(-fp.hx - 0.35, 0.0)]  # fmt: skip
        assert any(floor.reachable_within(x, y, 0.15) for x, y in fronts), o["name"]
    # every room can be got to from the start
    for room, (x0, y0, x1, y1) in made.rooms.items():
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        assert floor.reachable_within(cx, cy, 1.2), room


@pytest.mark.parametrize("sentence", SENTENCES[:3])
def test_what_a_generated_world_puts_on_tops_stays_put_once_the_physics_runs(sentence):
    made = generate(sentence, seed=0)
    assert unsettled(made.content, made.loose) == []


def test_the_settling_check_finds_a_thing_put_inside_another_and_one_left_to_fall():
    made = generate("a kitchen with a table", seed=0, check=False)
    content = made.content
    first, second = (o for o in content.objects if o["name"] in made.loose[:2])
    second["pos"] = list(first["pos"])  # the same spot: inside each other
    third = next(o for o in content.objects if o["name"] == made.loose[2])
    third["pos"] = [third["pos"][0], third["pos"][1], third["pos"][2] + 0.3]  # 30 cm over its top: it falls
    bad = unsettled(content, made.loose)
    assert {first["name"], second["name"], third["name"]} <= set(bad)


def test_a_thing_that_will_not_stay_put_is_put_elsewhere_and_one_that_never_does_is_left_out(monkeypatch):
    import robotforge.design.generate as gen

    plain = generate("a kitchen with a table", seed=0, check=False)
    calls = []

    def once(content, loose, seconds=gen.SETTLE_SECONDS):
        calls.append(list(loose))
        return [plain.loose[0]] if len(calls) == 1 else []

    monkeypatch.setattr(gen, "unsettled", once)
    moved = gen.generate("a kitchen with a table", seed=0)
    assert moved.moved == 1 and plain.loose[0] in moved.loose and len(calls) == 2
    monkeypatch.setattr(gen, "unsettled", lambda content, loose, seconds=0: [plain.loose[0]])
    never = gen.generate("a kitchen with a table", seed=0)
    assert plain.loose[0] not in never.loose
    assert any("would not stay put" in n for n in never.notes)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    return Playground(session=SimSession.from_xml("<mujoco/>"))


def test_load_scene_from_a_description_opens_a_world_with_its_robot_in_the_first_room(pg):
    result = pg.load_scene(description="a flat with a kitchen and a bedroom, two red cups", seed=1)
    made = result["scene"]["generated"]
    assert made["understood"].startswith("house")
    assert set(made["rooms"]) == {"kitchen", "bedroom"}
    assert result["world"] == "home" and result["robots"] == ["picker"]
    x0, y0, x1, y1 = made["rooms"]["kitchen"]
    x, y = pg.current_world().robots[0].pos[:2]
    assert x0 < x < x1 and y0 < y < y1
    assert any(o.name == "red_cup" for o in pg.current_world().objects)
    # an ordinary world afterwards: edit it, undo it
    pg.add_object("extra", shape="box", pos=[x, y + 0.6])
    pg.undo()
    assert not any(o.name == "extra" for o in pg.current_world().objects)


def test_load_scene_takes_a_preset_or_a_description(pg):
    with pytest.raises(PlaygroundError, match="not both"):
        pg.load_scene("kitchen", description="a kitchen")
    with pytest.raises(PlaygroundError, match="give preset"):
        pg.load_scene()
    with pytest.raises(PlaygroundError, match="describe the place"):
        pg.load_scene(description="  ")


def test_a_robot_said_in_the_sentence_is_the_one_the_world_starts_with(pg):
    result = pg.load_scene(description="a park with a ramp for my robot dog", seed=0)
    assert result["robots"] == ["dog"]
    assert pg.current_world().robots[0].template == "quadruped"
    assert math.isclose(pg.current_world().robots[0].yaw, 0.0)
