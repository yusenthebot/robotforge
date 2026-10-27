"""Authored basic-object materials share looks without changing geometry or contact physics."""

import mujoco
import numpy as np
import pytest
from test_visuals import assert_same_physics

from robotforge.design import visuals
from robotforge.design.world import ObjectEntry, RobotEntry, World, compile_world


def visible(model):
    return [g for g in range(model.ngeom) if model.geom_group[g] == visuals.VISUAL_GROUP]


def materials(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g])) for g in visible(model)]


@pytest.mark.parametrize("shape,size", [("box", [0.18, 0.18, 0.18]), ("sphere", [0.18]), ("cylinder", [0.1, 0.24])])
@pytest.mark.parametrize("fixed", [False, True])
def test_authored_material_binds_each_basic_shape_without_polluting_cached_default(shape, size, fixed):
    entry = ObjectEntry(name="sample", shape=shape, size=size, color="#5689ab", fixed=fixed, mass=0.7)
    default = compile_world(World(objects=[entry])).model
    assert materials(default) == ["rf_shell"]
    # Same name, dimensions and color force reuse of the same cached geometry across worlds.
    for material, preset in [("steel", "metal"), ("wood", "wood"), ("plastic", "shell"), (None, "shell")]:
        authored = entry.model_copy(update={"material": material})
        actual = compile_world(World(objects=[authored])).model
        assert materials(actual) == [f"rf_{preset}"]
        for field in ("vert", "face", "normal"):
            np.testing.assert_array_equal(getattr(actual, f"mesh_{field}"), getattr(default, f"mesh_{field}"))
        np.testing.assert_array_equal(actual.geom_rgba, default.geom_rgba)
        with visuals.disabled():
            bare = compile_world(World(objects=[authored])).model
        assert_same_physics(bare, actual, steps=0)
        # Explicit masses and all friction remain unchanged; steel retains its existing affinity.
        np.testing.assert_array_equal(actual.body_mass, default.body_mass)
        np.testing.assert_array_equal(actual.geom_friction, default.geom_friction)
    assert materials(compile_world(World(objects=[entry])).model) == ["rf_shell"]


@pytest.mark.parametrize(
    "name,shape,size,default,accents",
    [
        ("crate", "box", [0.3, 0.3, 0.3], "wood", set()),
        ("fridge", "box", [0.5, 0.5, 1], "paint", {"dark"}),
        ("apple", "sphere", [0.1], "shell", {"bark", "foliage"}),
        ("jar", "cylinder", [0.1, 0.2], "ceramic", {"wood"}),
        ("can", "cylinder", [0.1, 0.2], "shell", {"metal"}),
        ("ball", "sphere", [0.18], "shell", set()),
    ],
)
@pytest.mark.parametrize("material,preset", [("plastic", "shell"), ("wood", "wood"), ("steel", "metal")])
def test_named_geometry_and_decorative_accents_survive_surface_material(
    name, shape, size, default, accents, material, preset
):
    entry = ObjectEntry(name=name, shape=shape, size=size, color="#5689ab", mass=0.7)
    original = compile_world(World(objects=[entry])).model
    authored = entry.model_copy(update={"material": material})
    actual = compile_world(World(objects=[authored])).model
    assert set(materials(original)) == {f"rf_{p}" for p in {default, *accents}}
    assert set(materials(actual)) == {f"rf_{p}" for p in {preset, *accents}}
    for field in ("vert", "face", "normal"):
        np.testing.assert_array_equal(getattr(actual, f"mesh_{field}"), getattr(original, f"mesh_{field}"))
    np.testing.assert_array_equal(actual.geom_rgba, original.geom_rgba)
    with visuals.disabled():
        bare = compile_world(World(objects=[authored])).model
    assert_same_physics(bare, actual, steps=0)


def test_default_mass_still_uses_authored_density_and_steel_affinity():
    from robotforge.design.world import DENSITY

    for material in ("plastic", "wood", "steel"):
        entry = ObjectEntry(name="sample", size=[0.1, 0.2, 0.3], material=material)
        model = compile_world(World(objects=[entry])).model
        assert model.body("sample").mass[0] == pytest.approx(0.006 * DENSITY[material])
        with visuals.disabled():
            bare = compile_world(World(objects=[entry])).model
        assert_same_physics(bare, model, steps=0)


def test_authored_object_material_does_not_change_imported_native_material(tmp_path):
    path = tmp_path / "native.xml"
    path.write_text(
        '<mujoco><asset><material name="printed" rgba=".2 .4 .6 1" specular=".17"/></asset>'
        '<worldbody><body name="base"><geom name="panel" type="box" size=".1 .1 .1" '
        'material="printed"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    robot = RobotEntry(name="native", file=path.name, pos=[-2, 0])
    baseline = compile_world(World(robots=[robot]), tmp_path).model
    actual = compile_world(
        World(robots=[robot], objects=[ObjectEntry(name="sample", material="steel", pos=[2, 0])]), tmp_path
    ).model
    old = baseline.geom("native/panel").matid[0]
    new = actual.geom("native/panel").matid[0]
    assert mujoco.mj_id2name(actual, mujoco.mjtObj.mjOBJ_MATERIAL, int(new)) == "native/printed"
    np.testing.assert_array_equal(actual.mat_rgba[new], baseline.mat_rgba[old])
    assert actual.mat_specular[new] == baseline.mat_specular[old]
