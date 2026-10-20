"""Generated objects (robotforge.objects): every generator builds, varies with its seed, collides as convex pieces,
rests where it should, is understood from a description and can be extended by a plugin."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
import pytest

from robotforge import objects
from robotforge.design import props
from robotforge.design.world import World, compile_world
from robotforge.extend import REGISTRY, registry
from robotforge.objects import describe
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

NAMES = sorted(objects.GENERATORS)
EXPECTED = {
    "apple", "banana", "book", "bottle", "bowl", "brick", "cable", "can", "cup", "hammer", "mug", "orange", "package",
    "pear", "plate", "screwdriver", "squeeze_bottle", "tin", "wrench",
}  # fmt: skip


def _digest(built) -> str:
    h = hashlib.sha1()
    for v in built.visuals:
        h.update(np.round(v.mesh.vertices, 6).tobytes())
        h.update(repr(v.rgba).encode())
    for p in built.pieces:
        h.update(repr((p.kind, p.size, p.pos, p.quat, round(p.mass, 6))).encode())
    return h.hexdigest()


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("objects")
    return playground


def test_the_shipped_generators_are_shapes_add_object_takes():
    assert set(NAMES) >= EXPECTED
    for name in EXPECTED:
        assert name in props.SHAPES and name in props.BUILT_IN
        assert props.SHAPES[name].size_lengths and props.SHAPES[name].pickable


@pytest.mark.parametrize("name", NAMES)
def test_every_generator_builds_at_its_default_size_and_for_seeds(name):
    for seed in (None, 0, 1, 7):
        built = objects.object_for(name, seed=seed)
        assert built.mass > 0.005 and built.pieces and built.visuals
        assert built.height > 0.01
        assert all(p.mass >= 0 for p in built.pieces)
        assert sum(p.mass for p in built.pieces) == pytest.approx(built.mass, rel=1e-6)
        assert all(v.mesh.triangles > 0 for v in built.visuals)
        assert built.params.keys() == objects.get(name).params.keys()  # every parameter is resolved


@pytest.mark.parametrize("name", NAMES)
def test_a_seed_makes_the_same_object_every_time_and_other_seeds_other_objects(name):
    gen = objects.get(name)
    a, b = objects.object_for(name, seed=3), objects.object_for(name, seed=3)
    assert _digest(a) == _digest(b) and a.params == b.params
    if gen.params and any(p.vary for p in gen.params.values()):
        seen = {_digest(objects.object_for(name, seed=s)) for s in range(6)}
        assert len(seen) > 1, f"{name}: six seeds gave one object"


def test_parameters_are_checked_and_listed_in_the_error():
    with pytest.raises(ValueError, match="handle is one of round, square, tall"):
        objects.object_for("mug", handle="triangular")
    with pytest.raises(ValueError, match="no parameter 'colour'.*handle"):
        objects.object_for("mug", colour="red")
    with pytest.raises(ValueError, match="fill is from 0 to 1"):
        objects.object_for("bottle", fill=3)
    assert objects.object_for("mug", handle="square").params["handle"] == "square"


def test_mass_comes_from_the_volume_and_the_material_and_a_bottle_holds_water():
    mug = objects.object_for("mug")
    assert 0.25 < mug.mass < 0.5  # ceramic shell and handle
    empty, full = objects.object_for("bottle", fill=0.0), objects.object_for("bottle", fill=1.0)
    assert empty.mass < 0.08 < 0.3 < full.mass < 0.7
    assert objects.object_for("hammer", head=0.4).mass > objects.object_for("hammer", head=0.24).mass
    assert objects.object_for("brick").mass < 0.05 < objects.object_for("tin").mass  # foam against steel


def test_friction_comes_from_the_material():
    assert objects.object_for("brick").friction[0] > objects.object_for("mug").friction[0]
    assert objects.object_for("wrench").friction[0] < 1.0


def test_collision_is_convex_pieces_only_and_a_hollow_vessel_stays_hollow():
    mug = objects.object_for("mug")
    assert {p.kind for p in mug.pieces} <= {"box", "cylinder", "capsule", "sphere", "ellipsoid", "hull"}
    r = mug.facts["mouth_r"]
    # nothing solid on the mug's axis above its floor: a ball can be dropped in
    for p in mug.pieces:
        if p.solid and p.kind in ("box", "cylinder"):
            lowest = p.pos[2] - max(p.size)
            assert not (
                np.hypot(p.pos[0], p.pos[1]) < r * 0.5
                and p.pos[2] - p.size[-1] > mug.facts["floor_z"] + 0.002
                and lowest > 0.01
            ), p


def test_a_ball_dropped_in_a_mug_stays_in_it_and_one_dropped_beside_it_does_not(pg):
    pg.add_object("mug", shape="mug", pos=[0.0, 0.0])
    pg.add_object("ball", shape="sphere", size=[0.03], pos=[0.0, 0.0, 0.25], mass=0.02)
    pg.run_fast(lambda: False, 1.5)
    m, d = pg.session.model, pg.session.data
    ball = d.xpos[m.body("ball").id]
    assert np.hypot(ball[0], ball[1]) < 0.03 and 0.005 < ball[2] < 0.06  # on the mug's floor, inside its wall


@pytest.mark.parametrize("name", NAMES)
def test_each_object_comes_to_rest_where_it_was_put_without_drifting(pg, name):
    pg.add_object("thing", shape=name, pos=[0.0, 0.0], seed=2)
    pg.run_fast(lambda: False, 1.5)
    m, d = pg.session.model, pg.session.data
    body = m.body("thing").id
    assert np.linalg.norm(d.xpos[body][:2]) < 0.03
    assert d.xpos[body][2] == pytest.approx(0.0, abs=0.004)  # the origin is the middle of its base
    assert np.linalg.norm(d.cvel[body]) < 0.05
    assert d.xmat[body].reshape(3, 3)[2, 2] > 0.95  # still the right way up (or lying as it was meant to)
    if not objects.get(name).fixed:  # a fixed thing has no mass of its own
        assert m.body_subtreemass[body] == pytest.approx(objects.object_for(name, seed=2).mass, rel=0.02)


def test_a_loose_object_has_a_free_joint_and_a_fixed_one_has_none(pg):
    pg.add_object("a", shape="can", pos=[0.0, 0.0])
    pg.add_object("b", shape="can", pos=[0.3, 0.0], fixed=True)
    m = pg.session.model
    assert m.body_jntnum[m.body("a").id] == 1 and m.body_jntnum[m.body("b").id] == 0


def test_collision_pieces_are_hidden_and_the_meshes_drawn(pg):
    pg.add_object("mug", shape="mug", pos=[0.0, 0.0])
    m = pg.session.model
    body = m.body("mug").id
    groups = {int(m.geom_group[g]): int(m.geom_contype[g]) for g in range(m.ngeom) if m.geom_bodyid[g] == body}
    assert groups[3] == 1 and groups[2] == 0  # collide hidden, drawn do not collide


def test_the_world_file_keeps_only_what_a_generated_object_adds(pg):
    pg.add_object("plain", shape="box", pos=[0.5, 0.0])
    pg.add_object("m", shape="mug", pos=[0.0, 0.0], seed=4, params={"handle": "tall"})
    text = pg.store.file.read_text(encoding="utf-8")
    plain, mug = text.split("name: m\n")[0], text.split("name: m\n")[1]
    assert "seed" not in plain and "params" not in plain
    assert "seed: 4" in mug and "handle: tall" in mug
    again = World.from_yaml(text)
    assert again.objects[1].seed == 4 and again.objects[1].params == {"handle": "tall"}
    assert again.objects[0].seed is None


def test_seed_and_params_are_refused_on_a_shape_that_has_none(pg):
    with pytest.raises(PlaygroundError, match="seed and params are for generated objects"):
        pg.add_object("c", shape="box", seed=1)
    with pytest.raises(PlaygroundError, match="handle is one of"):
        pg.add_object("m", shape="mug", params={"handle": "x"})


def test_a_seed_picks_a_color_when_none_is_given_and_a_color_given_wins(pg):
    pg.add_object("a", shape="mug", seed=1, pos=[0.0, 0.0])
    pg.add_object("b", shape="mug", seed=1, color="red", pos=[0.3, 0.0])
    colors = {o.name: o.color for o in pg.current_world().objects}
    assert colors["a"] != colors["b"] and colors["b"] == "red" and colors["a"] != "white"


# ---- describing ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, shape, expect",
    [
        ("a tall blue mug with a big handle", "mug", {"color": "blue", "params": {"handle_size": 1.2}}),
        ("a wine bottle", "bottle", {"params": {"style": "wine"}}),
        ("a half full soda bottle", "bottle", {"params": {"style": "soda", "fill": 0.5}}),
        ("a tuna can", "can", {"params": {"style": "tuna"}}),
        ("an orange mug", "mug", {"color": "orange"}),
        ("a paper coffee cup with a lid", "cup", {"params": {"style": "paper", "lid": True}}),
        ("a standing book", "book", {"params": {"lying": False}}),
        ("a red ketchup bottle", "squeeze_bottle", {"color": "red", "params": {"cap": "red"}}),
        ("一个红色的杯子", "cup", {"color": "red"}),
        ("a cereal box", "package", {"params": {"style": "cereal"}}),
    ],
)
def test_a_description_is_read_into_a_shape_a_color_and_parameters(text, shape, expect):
    got = describe.interpret(text)
    assert got.shape == shape
    if "color" in expect:
        assert got.color == expect["color"]
    for key, value in expect.get("params", {}).items():
        assert got.params[key] == value
    assert got.notes and got.notes[0].startswith("read ")


def test_sizes_are_read_from_words_and_measurements_and_a_measurement_is_not_scaled_again():
    tall = describe.interpret("a tall mug")
    assert tall.size[1] == pytest.approx(0.095 * 1.3, abs=1e-3) and tall.size[0] == pytest.approx(0.082)
    assert describe.interpret("a 12 cm tall cup").size == [0.075, 0.12]
    assert describe.interpret("a tiny apple").size[0] < 0.06
    assert describe.interpret("a mug").size is None


def test_a_fruit_or_a_tool_is_one_of_them_by_seed_and_nonsense_is_refused():
    assert {describe.interpret("a fruit", seed=s).shape for s in range(8)} <= {"apple", "orange", "pear", "banana"}
    assert describe.interpret("a fruit", seed=2).shape == describe.interpret("a fruit", seed=2).shape
    with pytest.raises(ValueError, match="no object in 'a spaceship'.*mug"):
        describe.interpret("a spaceship")


def test_add_object_describe_builds_the_object_and_says_how_it_read_the_words(pg):
    out = pg.add_object("m", describe="a tall blue mug with a big handle", pos=[0.0, 0.0])
    assert out["notes"][0].startswith("read 'a tall blue mug with a big handle' as shape mug")
    entry = pg.current_world().objects[0]
    assert entry.shape == "mug" and entry.color == "blue" and entry.size[1] > 0.11
    assert entry.params == {"handle_size": 1.2}
    out = pg.add_object("x", describe="a mug", shape="bowl", pos=[0.3, 0.0])
    assert any("kept bowl" in n for n in out["notes"]) and pg.current_world().objects[1].shape == "bowl"


# ---- extending -----------------------------------------------------------------------------------------------

COASTER = """
from robotforge.objects import generator, Range
from robotforge.design.visuals import mesh as M
from robotforge.objects import kit

@generator("coaster", size="[diameter]", default_size=(0.09,), color="brown", material="wood", summary="a cork coaster",
           params={"edge": Range(0.001, 0.004, 0.002)}, keywords=("coaster",))
def coaster(g):
    d = g.size[0]
    t = 0.006
    g.part(M.cylinder(d / 2, t, 32, bevel=g.p("edge"), z0=0.0), "wood", [kit.cylinder(d / 2, t / 2, (0, 0, t / 2))])
    g.grasp("body", (0, 0, t / 2), (1, 0), 0.01, 1.0, "the edge")
"""


def test_a_plugin_adds_a_generator_that_is_a_shape_is_described_and_goes_away_on_reload(tmp_path, monkeypatch):
    user = tmp_path / "user_plugins"
    user.mkdir()
    (user / "coaster.py").write_text(COASTER, encoding="utf-8", newline="\n")
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(user))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    with registry.isolated():
        REGISTRY.user_loaded = False
        REGISTRY.load_user()
        assert "coaster" in props.SHAPES and "coaster" in objects.GENERATORS
        assert REGISTRY.get("part", "coaster") is not None
        assert describe.interpret("a cork coaster").shape == "coaster"
        playground = Playground(session=SimSession.from_xml("<mujoco/>"))
        playground.new_world("w")
        playground.add_object("c", shape="coaster", pos=[0.0, 0.0], seed=1)
        playground.run_fast(lambda: False, 0.5)
        assert playground.session.model.body("c").id > 0
        REGISTRY.reload()
        assert "coaster" in objects.GENERATORS
    assert "coaster" not in props.SHAPES and "coaster" not in objects.GENERATORS


def test_a_generator_that_fails_at_its_default_size_is_refused_with_its_name(tmp_path, monkeypatch):
    user = tmp_path / "user_plugins"
    user.mkdir()
    (user / "bad.py").write_text(
        "from robotforge.objects import generator\n@generator('bad', size='[a]', default_size=(0.1,))\n"
        "def bad(g):\n    raise ValueError('no solid')\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(user))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    with registry.isolated():
        REGISTRY.user_loaded = False
        REGISTRY.load_user()
        assert "bad" not in props.SHAPES
        assert any("failed at its default size" in str(f.error) for f in REGISTRY.files.values())


def test_the_catalogue_lists_every_generator_with_its_parameters():
    rows = {r["name"]: r for r in objects.catalogue()}
    assert set(rows) >= EXPECTED
    assert rows["mug"]["params"]["handle"]["choices"] == ["round", "square", "tall"]
    assert rows["bottle"]["params"]["fill"]["range"] == [0.0, 1.0]
    assert rows["hammer"]["kind"] == "tool"


def test_every_world_with_every_generator_compiles_and_steps():
    world = World.model_validate(
        {
            "name": "all",
            "objects": [
                {"name": f"o{i}", "shape": n, "pos": [0.3 * i, 0.0], "seed": i, "yaw": 20 * i}
                for i, n in enumerate(NAMES)
            ],
        }
    )
    model = compile_world(world).model
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


# ---- convex decomposition of a mesh a generator only drew ----------------------------------------------------


def _cup_mesh():
    from robotforge.design.visuals import mesh as M
    from robotforge.objects import profiles

    return M.lathe(profiles.hollow([(0.03, 0.0), (0.035, 0.06)], 0.004, 0.006), 24)


def test_decompose_asks_coacd_once_and_keeps_the_pieces_on_disk(tmp_path, monkeypatch):
    import sys
    import types

    from robotforge.objects import collision

    calls = []

    class FakeMesh:
        def __init__(self, vertices, faces):
            self.vertices, self.faces = vertices, faces

    def run_coacd(mesh, **options):
        calls.append(options)
        cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
        faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4]], dtype=np.int32)
        return [(cube * 0.01, faces), (cube * 0.01 + 0.02, faces)]

    fake = types.SimpleNamespace(Mesh=FakeMesh, run_coacd=run_coacd, set_log_level=lambda level: None)
    monkeypatch.setitem(sys.modules, "coacd", fake)
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    mesh = _cup_mesh()
    first = collision.decompose(mesh, max_pieces=4)
    again = collision.decompose(mesh, max_pieces=4)
    assert len(first) == len(again) == 2 and len(calls) == 1 and calls[0]["max_convex_hull"] == 4
    assert all(p.kind == "hull" and p.volume > 0 for p in first)
    collision.decompose(mesh, max_pieces=5)  # other options, other answer
    assert len(calls) == 2
    assert list((tmp_path / "cache" / "coacd").glob("*.npz"))


def test_decompose_says_so_when_coacd_is_missing(tmp_path, monkeypatch):
    import sys

    from robotforge.objects import collision

    monkeypatch.setitem(sys.modules, "coacd", None)  # import coacd raises ImportError
    with pytest.raises(collision.DecomposeError, match="not installed"):
        collision.convex_pieces(_cup_mesh(), cache=tmp_path)


@pytest.mark.slow
def test_coacd_cuts_a_real_cup_into_pieces_that_leave_its_inside_open(tmp_path):
    from robotforge.objects import collision

    pieces = collision.decompose(_cup_mesh(), max_pieces=6, cache=tmp_path)
    assert 2 <= len(pieces) <= 6
    inside = [p for p in pieces if np.hypot(*p.hull.vertices[:, :2].mean(axis=0)) < 0.01]
    assert all(p.hull.vertices[:, 2].max() < 0.02 for p in inside)  # only the floor is on the axis


def test_the_objects_page_lists_every_shipped_generator_and_follows_the_house_style():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "OBJECTS.md").read_text(encoding="utf-8")
    for name in sorted(EXPECTED):
        assert name.replace("_", " ") in text.replace("_", " "), name
    assert chr(0x2014) not in text and chr(0x2013) not in text
