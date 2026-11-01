"""Surface finishes never change a primitive's physical material or decorative geometry."""

import numpy as np
import pytest
from test_prop_materials import materials
from test_visuals import assert_same_physics

from robotforge.design.world import ObjectEntry, World, compile_world


@pytest.mark.parametrize("shape,size", [("box", [0.1, 0.2, 0.3]), ("sphere", [0.18]), ("cylinder", [0.1, 0.24])])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("physical", [None, "plastic", "wood", "steel"])
def test_finish_matrix_keeps_physical_arrays_and_geometry(shape, size, fixed, physical):
    entry = ObjectEntry(name="sample", shape=shape, size=size, fixed=fixed, material=physical)
    original = compile_world(World(objects=[entry])).model
    for finish, preset in [("plastic", "shell"), ("wood", "wood"), ("steel", "metal")]:
        changed = ObjectEntry.model_validate({**entry.model_dump(), "surface_material": finish})
        actual = compile_world(World(objects=[changed])).model
        assert changed.material == physical and materials(actual) == [f"rf_{preset}"]
        for field in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "geom_type",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_contype",
            "geom_conaffinity",
            "geom_friction",
            "geom_condim",
            "geom_solref",
            "geom_solimp",
            "mesh_vert",
            "mesh_face",
            "mesh_normal",
            "geom_rgba",
        ):
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field), err_msg=field)
    assert materials(compile_world(World(objects=[entry])).model) == materials(original)


@pytest.mark.parametrize(
    "name,shape,size,accents",
    [
        ("apple", "sphere", [0.1], {"rf_bark", "rf_foliage"}),
        ("jar", "cylinder", [0.1, 0.2], {"rf_wood"}),
        ("can", "cylinder", [0.1, 0.2], {"rf_metal"}),
    ],
)
def test_named_accents_and_explicit_mass_survive_finish(name, shape, size, accents):
    entry = ObjectEntry(name=name, shape=shape, size=size, mass=0.731, material="steel")
    before = compile_world(World(objects=[entry])).model
    changed = ObjectEntry.model_validate({**entry.model_dump(), "surface_material": "plastic"})
    after = compile_world(World(objects=[changed])).model
    assert set(materials(after)) == {"rf_shell", *accents}
    for field in ("body_mass", "body_inertia", "geom_contype", "geom_conaffinity", "mesh_vert", "mesh_face"):
        np.testing.assert_array_equal(getattr(after, field), getattr(before, field))


def test_default_serialization_stays_compatible_and_override_roundtrips():
    entry = ObjectEntry(name="sample")
    assert "surface_material" not in entry.model_dump() and "surface_material" not in entry.model_dump_json()
    world = World(objects=[entry])
    original = world.to_yaml()
    assert "surface_material" not in original
    world.objects[0] = ObjectEntry.model_validate({**entry.model_dump(), "surface_material": "wood"})
    assert World.from_yaml(world.to_yaml()) == world
    world.objects[0] = ObjectEntry.model_validate({**world.objects[0].model_dump(), "surface_material": None})
    assert world.to_yaml() == original


@pytest.mark.parametrize("value", ["gold", True, 1, {}, []])
def test_unknown_finish_is_not_coerced(value):
    with pytest.raises(ValueError):
        ObjectEntry(name="sample", surface_material=value)


@pytest.mark.parametrize("shape", ["wall", "drawer", "shelf"])
def test_surface_override_is_limited_to_supported_rigid_props(shape):
    with pytest.raises(ValueError, match="surface_material"):
        ObjectEntry(name="unsupported", shape=shape, surface_material="wood")


@pytest.mark.parametrize("shape", ["table", "bin"])
def test_furniture_finish_does_not_expand_physical_material_eligibility(shape):
    with pytest.raises(ValueError, match="material is for loose"):
        ObjectEntry(name="sample", shape=shape, material="steel", surface_material="wood")


@pytest.mark.parametrize("shape,size", [("table", [1, 0.6, 0.7]), ("bin", [0.4, 0.3, 0.2])])
@pytest.mark.parametrize("fixed,yaw", [(False, 0), (False, 37), (True, 0), (True, 37)])
def test_furniture_finish_keeps_all_geometry_colors_physics_and_transforms(shape, size, fixed, yaw):
    entry = ObjectEntry(
        name="sample", shape=shape, size=size, fixed=fixed, yaw=yaw, pos=[1, -2, 0.4], color="#5689ab", mass=0.731
    )
    original = compile_world(World(objects=[entry])).model
    for finish, preset in [("plastic", "shell"), ("wood", "wood"), ("steel", "metal")]:
        changed = ObjectEntry.model_validate({**entry.model_dump(), "surface_material": finish})
        actual = compile_world(World(objects=[changed])).model
        expected = [f"rf_{preset}"] * (2 if shape == "table" else 1)
        if shape == "table":
            expected.append("rf_rubber")
        assert materials(actual) == expected
        assert changed.material is None
        assert_same_physics(original, actual, steps=12)
        for field in (
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_bodyid",
            "geom_rgba",
            "mesh_vert",
            "mesh_face",
            "mesh_normal",
        ):
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field), err_msg=field)
    assert materials(compile_world(World(objects=[entry])).model) == materials(original)


@pytest.mark.parametrize(
    "name,shape,color,expected",
    [
        ("desk", "table", "#9c6b3f", ["rf_wood", "rf_wood", "rf_rubber"]),
        ("desk", "table", "#5689ab", ["rf_paint", "rf_paint", "rf_rubber"]),
        ("tray", "bin", "#5689ab", ["rf_shell"]),
        ("kitchen_sink", "bin", "#5689ab", ["rf_metal"]),
        ("fruit_bowl", "bin", "#5689ab", ["rf_ceramic"]),
        ("serving_dish", "bin", "#5689ab", ["rf_ceramic"]),
    ],
)
def test_original_furniture_serialization_named_inference_and_cached_defaults(name, shape, color, expected):
    original = World(objects=[ObjectEntry(name=name, shape=shape, color=color)])
    raw = original.to_yaml()
    baseline = compile_world(original).model
    assert materials(baseline) == expected and "surface_material" not in raw
    for finish in ("steel", "wood", "plastic"):
        changed = World.model_validate(original.model_dump())
        changed.objects[0].surface_material = finish
        assert World.from_yaml(changed.to_yaml()) == changed
        compile_world(changed)
    cleared = World.from_yaml(changed.to_yaml())
    cleared.objects[0].surface_material = None
    assert cleared.to_yaml() == raw
    restored = compile_world(cleared).model
    assert materials(restored) == expected
    for field in ("geom_rgba", "mesh_vert", "mesh_face", "mesh_normal"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(baseline, field))
