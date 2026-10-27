"""Surface finishes never change a primitive's physical material or decorative geometry."""

import numpy as np
import pytest
from test_prop_materials import materials

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


def test_surface_override_is_limited_to_supported_primitives():
    with pytest.raises(ValueError, match="surface_material"):
        ObjectEntry(name="desk", shape="table", size=[1, 0.6, 0.7], surface_material="wood")
