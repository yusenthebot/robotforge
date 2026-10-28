"""The asset engine (robotforge.assets): generators with ranges and a seed, styles, materials, collision shapes and
mass from density, the plugin hooks, the preview template and the viewer's surface table."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from robotforge import assets
from robotforge.assets import Asset, Collider, Piece
from robotforge.assets.bridge import Palette, collide, draw
from robotforge.design import templates
from robotforge.design import visuals as V
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.extend import PluginError, registry
from robotforge.extend import generator as plugin_generator
from robotforge.extend import material as plugin_material
from robotforge.extend import style as plugin_style
from robotforge.viewer import scene

STYLES = ("cobot", "lab", "hobby", "rugged")
NAMES = [g["name"] for g in assets.catalog()]
DENSITY = 1200.0


# ---- the registry ------------------------------------------------------------------------------------------


def test_the_built_in_generators_are_listed_with_their_parameters():
    found = {g["name"]: g for g in assets.catalog()}
    assert {"joint_actuator", "link_shell", "pedestal", "chassis", "wheel", "foot", "sensor_housing"} <= set(found)
    assert {"grip_pad", "gripper_body"} <= set(found)
    for g in found.values():
        assert g["summary"] and g["anchor"] and g["category"]
        for p in g["params"]:
            assert p["name"] and p["kind"] in ("float", "int", "bool", "choice")
            if p["kind"] in ("float", "int"):
                assert p["low"] <= p["default"] <= p["high"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("style", STYLES)
def test_every_generator_makes_a_sound_asset_in_every_style(name, style):
    asset = assets.generate(name, style=style)
    assert asset.name == name and asset.style == style
    assert asset.pieces and asset.colliders
    assert 100 < asset.triangles < 9000  # smooth detail, but a robot of a dozen parts must stay light
    low, high = asset.bounds()
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high)) and np.all(high > low)
    for piece in asset.pieces:
        assert piece.mesh.faces.max() < len(piece.mesh.vertices)
        assert np.allclose(np.linalg.norm(piece.mesh.normals, axis=1), 1.0, atol=1e-6)
    mass = asset.mass_properties(DENSITY)
    assert mass.mass > 0 and np.all(np.linalg.eigvalsh(mass.inertia) > 0)


@pytest.mark.parametrize("name", NAMES)
def test_the_same_arguments_make_the_same_asset(name):
    a, b = assets.generate(name, style="lab", seed=3), assets.generate(name, style="lab", seed=3)
    assert a is b or all(
        np.array_equal(p.mesh.vertices, q.mesh.vertices) for p, q in zip(a.pieces, b.pieces, strict=True)
    )


def test_a_style_changes_how_a_part_looks_and_never_how_big_it_is():
    cobot = assets.generate("joint_actuator", radius=0.04, length=0.08, style="cobot")
    lab = assets.generate("joint_actuator", radius=0.04, length=0.08, style="lab")
    rugged = assets.generate("joint_actuator", radius=0.04, length=0.08, style="rugged")
    assert {p.material for p in cobot.pieces} != {p.material for p in lab.pieces}
    assert "brushed" in {p.material for p in lab.pieces} and "gloss" in {p.material for p in cobot.pieces}
    assert rugged.triangles > cobot.triangles  # ribs and bolts
    for other in (lab, rugged):  # the physics shape is the same
        assert other.colliders == cobot.colliders
        assert other.mass_properties(DENSITY).mass == cobot.mass_properties(DENSITY).mass


def test_a_wrong_value_names_the_range_and_a_wrong_name_lists_what_exists():
    with pytest.raises(assets.AssetError, match=r"radius is between 0.004 and 0.3 m, got 5"):
        assets.generate("joint_actuator", radius=5)
    with pytest.raises(assets.AssetError, match="whole number"):
        assets.generate("joint_actuator", bolts=2.5)
    with pytest.raises(assets.AssetError, match="one of ball, pad"):
        assets.generate("foot", shape="cube")
    with pytest.raises(assets.AssetError, match="unknown parameter colour; it takes radius"):
        assets.generate("joint_actuator", colour="red")
    with pytest.raises(assets.AssetError, match="the generators are .*joint_actuator"):
        assets.generate("gearbox")
    with pytest.raises(ValueError, match="the styles are .*cobot"):
        assets.generate("wheel", style="shiny")
    with pytest.raises(assets.AssetError, match="true or false"):
        assets.generate("joint_actuator", gland="yes")
    with pytest.raises(assets.AssetError, match="seed is a whole number"):
        assets.generate("wheel", seed=1.5)
    with pytest.raises(assets.AssetError, match="seed is a whole number from 0 up"):
        assets.generate("wheel", seed=-1)


@pytest.mark.parametrize("name", ["rng", "values", "style", "seed", "_hidden"])
def test_a_parameter_cannot_take_the_name_of_something_the_generator_receives(name):
    """A parameter called rng would be replaced by the random generator without a word."""
    gen = assets.Generator("probe", "a probe", (assets.Param(name, 0.1, 0.0, 1.0, "x"),), lambda a: None)
    with pytest.raises(assets.AssetError, match="taken or reserved"):
        assets.register_generator(gen)


def test_a_material_changes_the_shell_a_part_is_made_of():
    plain = assets.for_material(assets.get_style("cobot"), "pla")
    metal = assets.for_material(assets.get_style("cobot"), "aluminum")
    carbon = assets.for_material(assets.get_style("cobot"), "carbon")
    assert plain is assets.get_style("cobot")
    assert metal.body_material == "aluminum" and carbon.body_material == "carbon"
    asset = assets.generate("link_shell", style=metal)
    assert "aluminum" in {p.material for p in asset.pieces}


# ---- mass from density -----------------------------------------------------------------------------------------


def _mesh_cylinder(r, half, n=160):
    return M.cylinder(r, 2 * half, segments=n)


def test_the_closed_mesh_integrals_agree_with_the_analytic_solids():
    box = M.box(0.03, 0.02, 0.05)
    v, com, inertia = assets.mesh_solid(box)
    assert math.isclose(v, 8 * 0.03 * 0.02 * 0.05, rel_tol=1e-9)
    assert np.allclose(com, 0, atol=1e-12)
    analytic = np.diag([0.02**2 + 0.05**2, 0.03**2 + 0.05**2, 0.03**2 + 0.02**2]) * v / 3
    assert np.allclose(inertia, analytic, rtol=1e-9)
    # a cylinder mesh (160 sides) is within a fraction of a percent of the exact one
    cyl = Collider("cylinder", (0.02, 0.03))
    v2, _, i2 = assets.mesh_solid(_mesh_cylinder(0.02, 0.03))
    exact = assets.asset._solid(cyl)
    assert math.isclose(v2, exact[0], rel_tol=2e-3)
    assert np.allclose(i2, exact[2], rtol=5e-3)
    # faces turned inward still give a positive solid
    flipped = M.Mesh(box.vertices, box.normals, box.faces[:, ::-1].copy(), box.keys)
    assert math.isclose(assets.mesh_solid(flipped)[0], v, rel_tol=1e-9)


def test_an_open_mesh_is_refused_with_a_reason():
    flat = M.Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]),
        np.tile([0, 0, 1.0], (4, 1)),
        np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        np.arange(4),
    )
    with pytest.raises(ValueError, match="encloses no volume"):
        assets.mesh_solid(flat)


def test_colliders_add_up_with_the_parallel_axis_rule():
    one = Asset("a", (), (Collider("box", (0.05, 0.02, 0.01)),)).mass_properties(DENSITY)
    two = Asset(
        "b",
        (),
        (Collider("box", (0.025, 0.02, 0.01), (-0.025, 0, 0)), Collider("box", (0.025, 0.02, 0.01), (0.025, 0, 0))),
    ).mass_properties(DENSITY)
    assert math.isclose(one.mass, two.mass, rel_tol=1e-12)
    assert np.allclose(one.inertia, two.inertia, rtol=1e-9)
    assert np.allclose(two.com, 0, atol=1e-12)
    with pytest.raises(ValueError, match="no colliders"):
        Asset("c", (), ()).mass_properties(DENSITY)
    with pytest.raises(ValueError, match="density must be positive"):
        one_asset = Asset("d", (), (Collider("sphere", (0.1,)),))
        one_asset.mass_properties(0)


def _compiled_body(asset: Asset, **kw) -> tuple[mujoco.MjModel, int]:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="b", pos=[0, 0, 1])
    body.add_freejoint()
    collide(spec, body, asset, DENSITY, **kw)
    model = spec.compile()
    return model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")


@pytest.mark.parametrize(
    "collider",
    [
        Collider("box", (0.05, 0.03, 0.02)),
        Collider("cylinder", (0.03, 0.04)),
        Collider("capsule", (0.02, 0.05)),
        Collider("sphere", (0.04,)),
    ],
    ids=lambda c: c.kind,
)
def test_mass_and_inertia_equal_what_mujoco_computes_for_the_same_shape(collider):
    asset = Asset("x", (), (collider,))
    model, b = _compiled_body(asset)
    mine = asset.mass_properties(DENSITY)
    assert math.isclose(model.body_mass[b], mine.mass, rel_tol=1e-6)
    assert np.allclose(model.body_inertia[b], np.sort(np.linalg.eigvalsh(mine.inertia)), rtol=1e-4) or np.allclose(
        np.sort(model.body_inertia[b]), np.sort(np.linalg.eigvalsh(mine.inertia)), rtol=1e-4
    )


def test_a_placed_pair_of_colliders_weighs_what_the_asset_says():
    asset = Asset(
        "pair",
        (),
        (
            Collider("box", (0.03, 0.03, 0.03), (0.05, 0, 0)),
            Collider("cylinder", (0.02, 0.05), (-0.04, 0.01, 0), tuple(map(tuple, M.rot_x(math.pi / 2)))),
        ),
    )
    model, b = _compiled_body(asset)
    mine = asset.mass_properties(DENSITY)
    assert math.isclose(model.body_mass[b], mine.mass, rel_tol=1e-6)
    assert np.allclose(model.body_ipos[b], mine.com, atol=1e-6)
    assert np.allclose(np.sort(model.body_inertia[b]), np.sort(np.linalg.eigvalsh(mine.inertia)), rtol=1e-4)


def test_a_mesh_collider_weighs_what_mujoco_says_for_its_hull():
    hull = M.rounded_box(0.04, 0.03, 0.02, 0.006, segments=3)
    asset = Asset("hull", (), (Collider("mesh", mesh=hull),))
    model, b = _compiled_body(asset)
    mine = asset.mass_properties(DENSITY)
    assert math.isclose(model.body_mass[b], mine.mass, rel_tol=2e-3)


def test_an_asset_can_be_moved_and_its_colliders_move_with_it():
    asset = assets.generate("wheel", radius=0.05, width=0.03)
    moved = asset.moved((0.1, 0, 0), M.rot_x(math.pi / 2))
    assert math.isclose(moved.mass_properties(DENSITY).mass, asset.mass_properties(DENSITY).mass, rel_tol=1e-12)
    assert np.allclose(moved.mass_properties(DENSITY).com, [0.1, 0, 0], atol=1e-12)
    # the wheel's axle was z; turned by 90 degrees about x it lies along -y
    axle = np.array(moved.colliders[0].matrix()) @ np.array([0, 0, 1.0])
    assert np.allclose(axle, [0, -1, 0], atol=1e-12)


@pytest.mark.parametrize("name", ["chassis", "gripper_body"])
def test_a_generator_can_collide_as_convex_pieces_cut_from_its_drawn_shell(name):
    """collider="hulls" gives mesh colliders that follow the shell; MuJoCo weighs them as this module does, and the
    mass stays near the primitive's (the shell's rim makes the chassis a little bigger, the hull leaves out the
    gripper's trim and screws)."""
    pytest.importorskip("coacd")
    plain = assets.generate(name)
    hulls = assets.generate(name, collider="hulls")
    assert hulls.colliders and all(c.kind == "mesh" for c in hulls.colliders)
    assert hulls.params["collider"] == "hulls" and plain.params["collider"] == "primitive"
    model, b = _compiled_body(hulls)
    mine = hulls.mass_properties(DENSITY)
    assert math.isclose(model.body_mass[b], mine.mass, rel_tol=2e-3)
    assert 0.75 * plain.mass_properties(DENSITY).mass < mine.mass <= 1.1 * plain.mass_properties(DENSITY).mass
    with pytest.raises(assets.AssetError, match="one of primitive, hulls"):
        assets.generate(name, collider="voxels")


@pytest.mark.slow
def test_a_concave_mesh_splits_into_convex_pieces():
    pytest.importorskip("coacd")
    ring = M.torus(0.04, 0.01, segments=24, sides=8)
    pieces = assets.hulls_of(ring, max_pieces=6)
    assert 2 <= len(pieces) <= 6
    total = sum(assets.mesh_solid(p)[0] for p in pieces)
    assert total > assets.mesh_solid(ring)[0] * 0.9


# ---- drawing --------------------------------------------------------------------------------------------------


def test_drawing_an_asset_adds_meshes_and_no_mass_and_no_contacts():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="b", pos=[0, 0, 1])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], density=500)
    vis = Visuals(spec)
    draw(vis, body, assets.generate("joint_actuator", style="rugged"), Palette.of("rugged"))
    draw(vis, body, assets.generate("grip_pad"), Palette.of("rugged", body=[1, 0, 0]), (0.1, 0, 0), M.rot_z(1.0))
    vis.finish()
    model = spec.compile()
    bare = mujoco.MjSpec()
    b2 = bare.worldbody.add_body(name="b", pos=[0, 0, 1])
    b2.add_freejoint()
    b2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], density=500)
    assert model.nmesh >= 4
    assert model.body_mass[1] == bare.compile().body_mass[1]
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
            assert model.geom_group[g] == V.VISUAL_GROUP


def test_a_piece_may_carry_its_own_color_and_the_palette_gives_the_rest():
    asset = assets.generate("joint_actuator", style="cobot")
    palette = Palette.of("cobot", body=[0.1, 0.2, 0.3, 1.0])
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="b")
    vis = Visuals(spec)
    draw(vis, body, asset, palette)
    vis.finish()
    model = spec.compile()
    colors = {tuple(round(float(c), 2) for c in model.geom_rgba[g]) for g in range(model.ngeom)}
    assert (0.1, 0.2, 0.3, 1.0) in colors  # the shell
    assert (0.45, 0.9, 1.0, 1.0) in colors  # the light ring keeps its own
    with pytest.raises(ValueError, match="role"):
        Piece("shiny", "gloss", M.box(0.01, 0.01, 0.01))


# ---- materials -------------------------------------------------------------------------------------------------


def test_the_builtin_materials_are_presets_too_and_reach_the_viewer():
    for name in ("gloss", "soft_touch", "brushed", "anodized", "cast", "powder", "tread", "knurled"):
        assert name in assets.MATERIALS and name in V.PRESETS
        assert V.material_name(name) == f"rf_{name}"
    s = assets.surface_of("rf_brushed")
    assert s["metalness"] == 1.0 and s["texture"] == "brushed" and s["textureStrength"] == 1.0
    assert assets.surface_of("robot/rf_gloss")["clearcoat"] > 0  # a robot's prefix is ignored
    assert assets.surface_of("rf_shell") is None  # the viewer knows its own presets by name
    assert assets.surface_of("ground_grass") is None


def test_a_bad_material_says_what_is_wrong():
    with pytest.raises(ValueError, match="texture must be one of"):
        assets.Material("odd", texture="marble")
    with pytest.raises(ValueError, match="roughness is between 0 and 1"):
        assets.Material("odd", roughness=1.5)
    with pytest.raises(ValueError, match="already exists"):
        assets.register_material(assets.Material("brushed"))
    with pytest.raises(ValueError, match="material name"):
        assets.Material("Not Ok")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_environment_multiplier_rejects_non_finite_and_negative_values(value):
    with pytest.raises(ValueError, match="env must be finite and non-negative"):
        assets.Material("probe", env=value)


@pytest.mark.parametrize("value", [0.0, 1.0, 1.45, 10.0])
def test_environment_multiplier_keeps_valid_authored_values(value):
    material = assets.Material("probe", env=value)
    assert material.env == value
    assert material.surface().get("env", 1.0) == value


def test_the_scene_description_carries_the_surfaces_of_the_materials_a_model_uses():
    asset = assets.generate("joint_actuator", style="lab")
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="b")
    vis = Visuals(spec)
    draw(vis, body, asset, Palette.of("lab"))
    vis.finish()
    desc = scene.describe(spec.compile(), 1, [])
    assert desc["surfaces"]["rf_brushed"]["texture"] == "brushed"
    assert "rf_metal" not in desc["surfaces"]  # known to the viewer by name
    assert all(name.startswith("rf_") for name in desc["surfaces"])


# ---- the preview template and the arm -------------------------------------------------------------------------


def test_the_asset_template_puts_one_part_on_a_turntable():
    built = templates.build("asset", {"generator": "wheel", "args": {"radius": 0.07}, "style": "rugged", "seed": 2})
    model = built.spec.compile()
    assert model.nu == 1 and model.njnt == 1
    assert model.nmesh > 3
    assert "wheel" in built.notes[0] and "rugged" in built.notes[0]
    assert all(model.geom_contype[g] == 0 for g in range(model.ngeom))  # nothing bumps into it
    with pytest.raises(ValueError, match="radius is between"):
        templates.build("asset", {"generator": "wheel", "args": {"radius": 3.0}})
    with pytest.raises(ValueError, match="the generators are"):
        templates.build("asset", {"generator": "nothing"})


@pytest.mark.parametrize("style", STYLES)
def test_an_arm_in_any_style_is_the_same_machine(style):
    base = templates.build("arm", {"dof": 3}).spec.compile()
    styled = templates.build("arm", {"dof": 3, "style": style}).spec.compile()
    for field in ("body_mass", "body_inertia", "body_pos", "jnt_range", "actuator_gainprm"):
        np.testing.assert_array_equal(getattr(base, field), getattr(styled, field), err_msg=field)
    solid = lambda m: [g for g in range(m.ngeom) if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]  # noqa: E731
    assert len(solid(base)) == len(solid(styled))
    np.testing.assert_array_equal(base.geom_size[solid(base)], styled.geom_size[solid(styled)])
    assert styled.nmesh >= 8  # a housing, a shell, a hand: drawn, not boxed


def test_an_arm_asked_for_an_unknown_style_says_which_exist():
    with pytest.raises(ValueError, match="the styles are .*cobot"):
        templates.build("arm", {"style": "steampunk"})


@pytest.mark.parametrize("dof", [3, 4, 5, 6])
def test_every_arm_stays_under_the_triangle_budget(dof):
    model = templates.build("arm", {"dof": dof}).spec.compile()
    triangles = sum(
        int(model.mesh_facenum[model.geom_dataid[g]])
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    )
    assert triangles < 30_000


# ---- plugins ---------------------------------------------------------------------------------------------------

HEX = """
from robotforge.extend import generator, Param, Asset, Piece, Collider
from robotforge.design.visuals import mesh as M

@generator("hex_standoff", summary="a hexagonal standoff", params=[
    Param("height", 0.02, 0.005, 0.1, "its height"), Param("across", 0.008, 0.003, 0.03, "across flats")])
def hex_standoff(a):
    body = M.cylinder(a.across / 1.732, a.height, segments=6, z0=0.0)
    return Asset("hex_standoff", (Piece("trim", "metal", body),),
                 (Collider("cylinder", (a.across / 2, a.height / 2), (0, 0, a.height / 2)),))
"""


def test_a_plugin_registers_a_generator_a_material_and_a_style(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(tmp_path / "p"))
    with registry.isolated() as reg:
        reg.user_loaded = False
        (tmp_path / "p").mkdir()
        (tmp_path / "p" / "standoff.py").write_text(HEX, encoding="utf-8", newline="\n")
        reg.load_user()
        assert reg.problems() == {}
        assert "hex_standoff" in [g["name"] for g in assets.catalog()]
        asset = assets.generate("hex_standoff", height=0.03, style="lab")
        assert asset.mass_properties(7800).mass > 0 and asset.params["height"] == 0.03
        with pytest.raises(assets.AssetError, match="height is between 0.005 and 0.1 m"):
            assets.generate("hex_standoff", height=1.0)
        report = reg.report()
        assert report["generators"][0]["name"] == "hex_standoff"
        assert report["generators"][0]["params"][0]["name"] == "height"
        built = templates.build("asset", {"generator": "hex_standoff", "args": {"across": 0.01}})
        assert built.spec.compile().nmesh >= 1
        mat = plugin_material("velvet", roughness=0.95, texture="rubber", texture_scale=0.5)
        assert assets.surface_of("rf_velvet")["texture"] == "rubber" and mat.name == "velvet"
        made = plugin_style("sleek", based_on="cobot", fillet=1.8, light_ring=False)
        assert assets.get_style("sleek").fillet == 1.8 and made.body_material == "gloss"
    assert "hex_standoff" not in [g["name"] for g in assets.catalog()]  # taken out with its plugin
    assert "velvet" not in assets.MATERIALS and "sleek" not in assets.STYLES and "velvet" not in V.PRESETS


def test_a_plugin_that_cannot_make_its_asset_is_refused_with_its_name():
    with registry.isolated():

        def broken(a):
            raise RuntimeError("no mesh today")

        with pytest.raises(PluginError, match="generator 'broken' does not make an asset.*no mesh today"):
            plugin_generator("broken", summary="x", params=[])(broken)
        assert "broken" not in [g["name"] for g in assets.catalog()]

        def not_an_asset(a):
            return [1, 2]

        with pytest.raises(PluginError, match="must return an Asset"):
            plugin_generator("lists", summary="x")(not_an_asset)
        with pytest.raises(PluginError, match="already exists"):
            plugin_generator("wheel", summary="x")(not_an_asset)
        with pytest.raises(PluginError, match="lowercase"):
            plugin_generator("Bad Name", summary="x")(not_an_asset)


def test_dict_shorthand_params_and_bad_materials_and_styles_in_plugins():
    with registry.isolated():

        @plugin_generator("pin", summary="a pin", params={"length": (0.03, 0.01, 0.1, "its length")})
        def pin(a):
            return Asset(
                "pin", (Piece("metal", "metal", M.cylinder(0.002, a.length, z0=0.0)),), (Collider("sphere", (0.01,)),)
            )

        assert assets.generate("pin", length=0.05).params == {"length": 0.05}
        with pytest.raises(PluginError, match="texture must be one of"):
            plugin_material("bad", texture="marble")
        with pytest.raises(PluginError, match="fillet is between"):
            plugin_style("blobby", fillet=9.0)
        with pytest.raises(PluginError, match="unknown style"):
            plugin_style("x", based_on="nothing")
        assert "bad" not in assets.MATERIALS and "blobby" not in assets.STYLES


# ---- the page and the agent's view --------------------------------------------------------------------------


def test_the_page_names_every_generator_style_and_material():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "ASSETS.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"`{name}`" in text, name
    for name in STYLES:
        assert f"`{name}`" in text, name
    for name in ("gloss", "soft_touch", "brushed", "anodized", "cast", "powder", "tread", "knurled"):
        assert f"`{name}`" in text, name
    assert chr(0x2014) not in text and chr(0x2013) not in text  # no em or en dashes
    assert all(ord(c) < 0x2190 for c in text)  # no emoji


def test_the_overview_an_agent_reads_lists_generators_styles_and_materials():
    view = assets.overview()
    assert {g["name"] for g in view["generators"]} == set(NAMES)
    assert set(view["styles"]) >= set(STYLES) and "brushed" in view["materials"]
    assert "template='asset'" in view["how"]


def test_list_templates_carries_the_asset_listing():
    """By default a sentence per generator (an agent pays for every token); detail="assets" adds the ranges."""
    import asyncio
    import json

    from mcp import Client

    from robotforge.mcp_server import build_server
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    async def run():
        server = build_server(Playground(session=SimSession.from_xml("<mujoco/>")), 8899)
        async with Client(server) as client:
            return [(await client.call_tool("list_templates", a)).content[0].text for a in ({}, {"detail": "assets"})]

    short_text, full_text = asyncio.run(run())
    short = json.loads(short_text)
    assert "asset" in {t["name"] for t in short["templates"]}
    assert set(short["assets"]["generators"]) >= {"joint_actuator", "link_shell"}
    assert all(isinstance(v, str) for v in short["assets"]["generators"].values())
    assert "detail='assets'" in short["assets"]["more"]
    full = json.loads(full_text)
    assert {g["name"] for g in full["assets"]["generators"]} >= {"joint_actuator", "link_shell"}
    assert all(g["params"] for g in full["assets"]["generators"] if g["name"] == "wheel")
    assert len(full_text) - len(short_text) > 5000  # the parameter tables are what the short listing leaves out
