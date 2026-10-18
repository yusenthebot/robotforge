"""Terrain, furniture props and scene presets: schema, generation, placement on the ground, tools."""

import json
import math
import random
import time
from collections import deque

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.design import props, scenes, terrain
from robotforge.design.world import ObjectEntry, World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession
from robotforge.viewer import scene as viewer_scene

ALL_PATCHES = [
    {"type": "hills", "pos": [5, 0]},
    {"type": "rough", "pos": [0, 5]},
    {"type": "stairs", "pos": [-4, 0]},
    {"type": "ramp", "pos": [0, -4], "angle": 15},
    {"type": "stones", "pos": [-4, -4]},
    {"type": "curb", "pos": [3, 3]},
    {"type": "platform", "pos": [-3, 4]},
]


def _world(**data) -> World:
    return World.model_validate(data)


def _settled(world: World, seconds: float = 1.0) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = compile_world(world).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return model, data


# ---- schema ------------------------------------------------------------------------------------


def test_every_terrain_type_validates_and_gets_a_unique_name():
    world = _world(terrain=ALL_PATCHES, objects=[{"name": "hills1", "pos": [9, 9]}])
    names = [p.name for p in world.terrain]
    assert names[0] == "hills2"  # hills1 is an object's name already
    assert len(set(names)) == len(names) == 7
    again = World.from_yaml(world.to_yaml())
    assert [p.name for p in again.terrain] == names
    assert world.to_yaml().index("type: hills") < world.to_yaml().index("name: hills2")  # type is listed first


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"type": "hill"}, "Did you mean 'hills'"),
        ({"type": "ramp", "angle": 60}, "angle"),
        ({"type": "hills", "size": [4]}, r"^hills size is \[x, y\]"),
        ({"type": "stairs", "steps": 0}, "steps"),
        ({"type": "stones", "stone": 1.5}, "does not fit"),
        ({"type": "curb", "colour": "red"}, "colour"),
    ],
)
def test_bad_terrain_is_explained(patch, message):
    with pytest.raises(ValueError, match=message):
        terrain.parse(patch)


def test_catalog_lists_parameters_with_defaults():
    cat = terrain.catalog()
    assert set(cat) == set(terrain.TYPES)
    assert cat["ramp"]["params"]["angle"]["default"] == 10.0
    assert "up_down" in cat["stairs"]["params"]["direction"]["enum"]
    assert "grass" in cat["hills"]["params"]["style"]["enum"]


# ---- generation --------------------------------------------------------------------------------


def test_heightfields_are_seeded_bounded_and_flush_with_the_floor():
    hills = terrain.parse({"type": "hills", "size": [4, 3], "height": 0.3, "seed": 1})
    h, cell = terrain.heights(hills)
    assert h.max() == pytest.approx(0.3) and h.min() == 0.0
    assert np.all(h[0] == 0) and np.all(h[-1] == 0) and np.all(h[:, 0] == 0) and np.all(h[:, -1] == 0)
    assert h.shape == (round(3 / cell) + 1, round(4 / cell) + 1)
    same, _ = terrain.heights(terrain.parse({"type": "hills", "size": [4, 3], "height": 0.3, "seed": 1}))
    other, _ = terrain.heights(terrain.parse({"type": "hills", "size": [4, 3], "height": 0.3, "seed": 2}))
    assert np.array_equal(h, same) and not np.allclose(h, other)
    big, _ = terrain.heights(terrain.parse({"type": "rough", "size": [60, 60]}))
    assert max(big.shape) <= terrain.MAX_GRID


def test_world_with_every_terrain_type_compiles_quickly_with_notes():
    started = time.process_time()
    compiled = compile_world(_world(terrain=ALL_PATCHES))
    assert time.process_time() - started < 1.0
    model = compiled.model
    assert model.nhfield == 2
    assert any("steepest slope" in n for n in compiled.notes) and any("4 steps of 5 cm" in n for n in compiled.notes)
    # MuJoCo keeps our grid (rows from -y to +y), so the viewer and physics see the same ground.
    patch = terrain.parse(ALL_PATCHES[0] | {"name": "h"})
    grid, _ = terrain.heights(patch)
    got = model.hfield_data[: model.hfield_nrow[0] * model.hfield_ncol[0]].reshape(grid.shape)
    assert np.allclose(got * model.hfield_size[0][2], grid, atol=1e-6)


def test_ground_height_matches_the_shapes():
    patches = [
        {"type": "stairs", "name": "s", "pos": [0, 0], "steps": 4, "step_height": 0.05, "direction": "up"},
        {"type": "ramp", "name": "r", "pos": [0, 5], "angle": 10, "length": 2.0, "landing": 1.0, "direction": "up"},
        {"type": "platform", "name": "p", "pos": [5, 0], "size": [1, 1], "height": 0.2, "yaw": 45},
    ]
    ground = terrain.Ground.build([terrain.parse(p) for p in patches], "checker")
    stairs, ramp = (terrain.parse(p) for p in patches[:2])
    x0 = -stairs.footprint()[0] / 2
    assert ground.height(x0 + 0.15, 0) == pytest.approx(0.05, abs=1e-6)  # first tread
    assert ground.height(x0 + 0.95, 0) == pytest.approx(0.2, abs=1e-6)  # landing
    rx0 = -ramp.footprint()[0] / 2
    assert ground.height(rx0 + 1.0, 5) == pytest.approx(math.tan(math.radians(10)), abs=2e-3)  # halfway up
    assert ground.height(5.6, 0) == pytest.approx(0.2, abs=1e-6)  # a corner of the turned platform
    assert ground.height(20, 20) == 0.0
    assert terrain.Ground.build([], "checker").flat


# ---- robots and objects on the ground ------------------------------------------------------------


def test_flat_worlds_place_robots_exactly_as_before():
    model = compile_world(_world(robots=[{"name": "dog", "template": "quadruped"}])).model
    from robotforge.design import templates

    assert model.body("dog/trunk").pos[2] == pytest.approx(templates.build("quadruped", {}).spawn_height)


@pytest.mark.parametrize(
    "patch",
    [
        {"type": "stairs", "pos": [0.0, 0.0], "direction": "up_down"},
        {"type": "ramp", "pos": [0.9, 0.0], "angle": 12, "direction": "up"},
        {"type": "hills", "pos": [0.5, 0.5], "height": 0.3, "seed": 3},
        {"type": "rough", "pos": [0.0, 0.0], "height": 0.04},
    ],
)
def test_a_dog_spawns_standing_on_terrain(patch):
    world = _world(terrain=[patch], robots=[{"name": "dog", "template": "quadruped"}])
    model = compile_world(world).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ground = terrain.Ground.build(world.terrain, "checker")
    gaps = []
    for side in ("front_left", "front_right", "rear_left", "rear_right"):
        foot = data.site_xpos[model.site(f"dog/{side}_foot").id]
        gaps.append(foot[2] - ground.height(foot[0], foot[1]))
    radius = 0.02  # foot sites sit at the foot ball's center; look at its bottom with some slack
    assert min(gaps) > -0.005 and min(gaps) < radius + 0.012, gaps  # the lowest foot touches, none is buried
    _, settled = _settled(world, 1.0)
    trunk = model.body("dog/trunk").id
    assert settled.xmat[trunk][8] > 0.9  # still upright after a second
    assert abs(settled.xpos[trunk][2] - data.xpos[trunk][2]) < 0.05  # no drop from the sky, no sinking


def test_objects_rest_on_terrain_and_explicit_heights_are_kept():
    world = _world(
        terrain=[{"type": "platform", "pos": [0, 0], "size": [1, 1], "height": 0.2}],
        objects=[
            {"name": "cube", "pos": [0, 0], "size": [0.1, 0.1, 0.1]},
            {"name": "table", "shape": "table", "pos": [0.1, 0.1], "size": [0.4, 0.4, 0.3]},
            {"name": "high", "pos": [3, 0, 1.0]},
        ],
    )
    model = compile_world(world).model
    assert model.body("cube").pos[2] == pytest.approx(0.25)
    assert model.body("table").pos[2] == pytest.approx(0.2)
    assert model.body("high").pos[2] == pytest.approx(1.0)


# ---- props ----------------------------------------------------------------------------------------


def test_every_shape_compiles_with_its_defaults_and_turns_with_yaw():
    objects = [{"name": f"o_{s}", "shape": s, "pos": [2.0 * i, 0.0], "yaw": 30} for i, s in enumerate(props.SHAPES)]
    world = _world(objects=objects)
    model, data = _settled(world, 0.5)
    for entry in world.objects:
        info = props.SHAPES[entry.shape]
        assert entry.size == list(info.size) and entry.fixed == info.fixed
        if entry.shape in props.PROPS:
            body = model.body(entry.name)
            assert np.allclose(body.quat, [math.cos(math.radians(15)), 0, 0, math.sin(math.radians(15))], atol=1e-6)
    marker = model.geom("o_marker").id
    assert model.geom_contype[marker] == 0 and model.geom_conaffinity[marker] == 0
    pallet = model.body("o_pallet").id
    assert model.body_subtreemass[pallet] == pytest.approx(20.0)
    assert np.isfinite(data.qpos).all()


def test_shape_sizes_are_explained():
    with pytest.raises(ValueError, match=r"wall size is \[length, thickness, height\]"):
        ObjectEntry(name="w", shape="wall", size=[1, 2])
    assert ObjectEntry(name="m", shape="marker", size=[1.0, 0.1]).size == [1.0, 0.1]


def test_shelf_boards_are_where_scenes_put_things():
    levels = props.shelf_levels([1.0, 0.4, 1.6])
    assert levels[0] == pytest.approx(props.PANEL) and levels[-1] == pytest.approx(1.6)
    world = _world(
        objects=[
            {"name": "shelf", "shape": "shelf", "size": [1.0, 0.4, 1.6]},
            {"name": "box", "pos": [0, 0, levels[1] + 0.05], "size": [0.1, 0.1, 0.1]},
        ]
    )
    model, data = _settled(world, 1.0)
    assert data.xpos[model.body("box").id][2] == pytest.approx(levels[1] + 0.05, abs=0.003)


# ---- scenes -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(scenes.SCENES))
def test_every_scene_builds_compiles_and_holds_still(name):
    content = scenes.build(name, seed=1)
    world = World.model_validate(
        {
            "floor": content.floor,
            "terrain": content.terrain,
            "objects": content.objects,
            "robots": [
                {**r, "pos": list(content.ground_slots[0].pos)} for r in content.starter if r.get("template") == "rover"
            ],
        }
    )
    started = time.process_time()
    compile_world(world)
    assert time.process_time() - started < 1.0
    model, data = _settled(world, 0.5)
    loose = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    speeds = [np.linalg.norm(data.qvel[model.jnt_dofadr[j] : model.jnt_dofadr[j] + 3]) for j in loose]
    assert max(speeds, default=0.0) < 0.2, "objects should rest where the scene puts them, not fall or fly"
    assert content.ground_slots and content.tips


def test_scenes_are_seeded():
    assert scenes.build("park", seed=4).objects == scenes.build("park", seed=4).objects
    assert scenes.build("park", seed=4).objects != scenes.build("park", seed=5).objects
    with pytest.raises(ValueError, match="Did you mean 'maze'"):
        scenes.build("mazes")
    with pytest.raises(ValueError, match="rows must be a number from 2 to 15"):
        scenes.build("maze", options={"rows": 40})
    with pytest.raises(ValueError, match="has no option 'size'"):
        scenes.build("maze", options={"size": 3})


@pytest.mark.parametrize("seed", range(6))
def test_warehouse_racks_hold_parcels_a_gripper_fits_round_clear_of_the_crates(seed):
    """The crates are 40 cm and more across: nothing on the racks fitted a gripper until the parcels. Each stands on
    a board, at its aisle edge, and the crates of the seed stay where they were without them."""
    content = scenes.build("warehouse", seed=seed)
    parcels = [o for o in content.objects if o["name"].startswith("parcel_")]
    assert len(parcels) >= 2
    boards = props.shelf_levels([2.0, 0.6, 2.0])
    crates = [o for o in content.objects if o["name"].startswith("crate_")]
    for p in parcels:
        assert max(p["size"][:2]) <= 0.05
        assert p["pos"][2] - p["size"][2] / 2 == pytest.approx(boards[2], abs=1e-3)
        assert abs(p["pos"][1] - 2.6) == pytest.approx(1.12)  # 18 cm in from the rack's aisle side
        for c in crates:
            same_board = (
                abs((c["pos"][2] - c["size"][2] / 2) - boards[2]) < 1e-3 and abs(c["pos"][1] - p["pos"][1]) < 0.4
            )
            assert not same_board or abs(c["pos"][0] - p["pos"][0]) > c["size"][0] / 2 + 0.05
    if seed == 0:  # as before there were parcels
        assert next(c for c in crates if c["name"] == "crate_24")["pos"] == [0.2834, 1.3, 0.927]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_maze_is_perfect(seed):
    rows, cols = 6, 5
    openings = scenes.carve_maze(rows, cols, random.Random(seed))
    assert len(openings) == rows * cols - 1  # a spanning tree: connected and without loops
    seen, todo = {(0, 0)}, deque([(0, 0)])
    while todo:
        r, k = todo.popleft()
        for a, b, c, d in openings:
            for here, there in (((a, b), (c, d)), ((c, d), (a, b))):
                if here == (r, k) and there not in seen:
                    seen.add(there)
                    todo.append(there)
    assert len(seen) == rows * cols


def test_maze_walls_leave_a_way_from_start_to_goal():
    """Walk the built maze with rays: a step to a neighboring cell is allowed when no wall is in the way."""
    content = scenes.build("maze", seed=5, options={"rows": 4, "cols": 4})
    model = compile_world(World.model_validate({"objects": content.objects})).model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cell, start = 1.0, content.ground_slots[0].pos
    goal = next(o["pos"] for o in content.objects if o["name"] == "goal")

    def blocked(a, b) -> bool:
        p = np.array([a[0], a[1], 0.2])
        v = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        geom = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, p, v / np.linalg.norm(v), None, 1, -1, geom)
        return 0 <= dist < np.linalg.norm(v)

    seen, todo = {tuple(start)}, deque([tuple(start)])
    while todo:
        x, y = todo.popleft()
        for dx, dy in ((cell, 0), (-cell, 0), (0, cell), (0, -cell)):
            nxt = (round(x + dx, 6), round(y + dy, 6))
            if nxt not in seen and abs(nxt[0]) < 3.5 and abs(nxt[1]) < 3.5 and not blocked((x, y), nxt):
                seen.add(nxt)
                todo.append(nxt)
    assert (round(goal[0], 6), round(goal[1], 6)) in seen


# ---- playground and MCP ---------------------------------------------------------------------------


@pytest.fixture
def playground(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    return Playground.with_demo()


def test_load_scene_opens_a_world_with_a_starter_robot(playground):
    result = playground.load_scene("desk_lab")
    assert result["world"] == "desk_lab" and result["robots"] == ["arm"]
    arm = playground.store.load().robots[0]
    assert arm.pos == [0.0, 0.32, scenes.DESK_TOP]
    assert result["scene"]["tips"]


def test_load_scene_keeps_robots_and_moves_them_to_the_scene(playground):
    playground.new_world("mine")
    playground.add_robot("dog", template="quadruped", pos=[3.0, 3.0])
    playground.add_robot("arm", template="arm", pos=[-3.0, 0.0])
    result = playground.load_scene("kitchen")
    world = playground.store.load()
    assert result["robots"] == ["dog", "arm"]
    assert world.robots[1].pos == [0.0, -0.02, scenes.COUNTER_TOP]  # the arm stands on the island
    assert world.robots[0].pos == [1.3, -1.0]
    assert playground.undo()["robots"] == ["dog", "arm"]
    assert playground.store.load().objects == []  # undo brings the empty world back

    cleared = playground.load_scene("park", keep_robots=False, starter=False)
    assert cleared["robots"] == [] and cleared["terrain"] == ["hills"]


def test_terrain_edits_through_the_playground(playground):
    playground.new_world("ground")
    playground.add_robot("dog", template="quadruped")
    result = playground.set_terrain("stairs", params={"steps": 3})
    assert result["terrain"] == ["stairs1"]
    dog_z = playground.behaviors.pose("dog")["position"][2]
    assert dog_z > 0.1 + 0.17  # standing on the 15 cm landing
    playground.add_terrain("ramp", name="slope", pos=[4, 0], params={"angle": 8})
    playground.update_terrain("slope", params={"angle": 14}, yaw=90)
    patch = playground.store.load().terrain[1]
    assert (patch.angle, patch.yaw, patch.type) == (14, 90, "ramp")
    playground.remove("slope")
    assert [p.name for p in playground.store.load().terrain] == ["stairs1"]
    assert "terrain" not in playground.set_terrain("flat")
    with pytest.raises(PlaygroundError, match="Did you mean 'stones'"):
        playground.add_terrain("stone")
    with pytest.raises(PlaygroundError, match="slope"):
        playground.update_terrain("slope", params={"angle": 5})
    with pytest.raises(PlaygroundError, match="angle"):
        playground.add_terrain("ramp", params={"angle": 80})


def test_viewer_gets_heightfields_and_ground_looks():
    model = compile_world(_world(floor={"style": "grass"}, terrain=ALL_PATCHES[:3])).model
    payload = viewer_scene.build_mesh_payload(model)
    desc = viewer_scene.describe(model, 3, payload.entries, payload.hfields)
    floats = np.frombuffer(payload.blob, dtype=np.float32)
    (hills, rough) = desc["hfields"]
    assert floats[hills["offset"] : hills["offset"] + hills["nrow"] * hills["ncol"]].max() == pytest.approx(1.0)
    geoms = {g["name"]: g for g in desc["geoms"]}
    assert geoms["hills1"]["type"] == "hfield" and geoms["hills1"]["hfield"] == 0
    assert geoms["floor"]["material"] == "ground_grass" and geoms["stairs1"]["material"] == "ground_concrete"
    assert desc["grounds"]["ground_grass"] == {"pattern": "speckle", "tile": terrain.STYLES["grass"].tile}
    json.dumps(desc)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_scene_tools_over_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_path(Playground.with_demo().session.source))
    server = build_server(playground, find_free_port(8810))
    async with Client(server) as client:
        listed = json.loads((await client.call_tool("list_scenes", {})).content[0].text)
        assert [s["name"] for s in listed["scenes"]] == list(scenes.SCENES)
        loaded = json.loads((await client.call_tool("load_scene", {"preset": "maze", "seed": 2})).content[0].text)
        assert loaded["robots"] == ["rover"] and "goal" in loaded["objects"]
        added = await client.call_tool("add_terrain", {"type": "hills", "pos": [8, 0], "params": {"height": 0.2}})
        assert json.loads(added.content[0].text)["terrain"] == ["hills1"]
        bad = await client.call_tool("add_terrain", {"type": "ramp", "params": {"angle": 99}})
        assert bad.is_error and "angle" in bad.content[0].text
        wall = await client.call_tool("add_object", {"name": "wall", "shape": "wall", "pos": [0, 6], "yaw": 90})
        assert not wall.is_error
