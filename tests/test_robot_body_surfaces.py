"""Body finishes isolate materials while preserving geometry, ray visibility and source colours."""

import mujoco
import numpy as np
import pytest

from robotforge.design import body_surfaces
from robotforge.design.world import RobotEntry, World, compile_world
from robotforge.viewer.scene import build_mesh_payload, describe

XML = """<mujoco><asset><material name="shared" rgba=".2 .3 .4 1"/>
<material name="rays" rgba="1 1 1 0"/></asset><worldbody>
<body name="base"><geom type="box" size=".1 .2 .3" material="shared"/>
<geom type="sphere" size=".05" pos=".3 0 0" material="rays" rgba=".7 .6 .2 1"/>
<body name="child"><geom type="box" size=".1 .1 .1" material="shared"/></body>
</body></worldbody></mujoco>"""


def colours(model):
    payload = build_mesh_payload(model)
    return [g["rgba"] for g in describe(model, 1, payload.entries, payload.hfields, payload.textures)["geoms"]]


def test_shared_material_inherited_colours_and_ray_visibility_are_isolated():
    spec = mujoco.MjSpec.from_string(XML)
    old = spec.compile()
    body_surfaces.apply(spec, {"base": "steel"})
    new = spec.compile()
    assert colours(old) == colours(new)
    for field in [
        "geom_rgba",
        "geom_size",
        "geom_pos",
        "geom_contype",
        "geom_conaffinity",
        "body_mass",
        "body_inertia",
    ]:
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    assert new.geom_matid[2] == old.geom_matid[2]
    assert new.mat_rgba[new.geom_matid[1], 3] == old.mat_rgba[old.geom_matid[1], 3] == 0
    assert new.mat_metallic[new.geom_matid[0]] == 1 and old.mat_metallic[old.geom_matid[0]] < 0
    assert build_mesh_payload(old).blob == build_mesh_payload(new).blob
    for model in [old, new]:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        hit = np.zeros(1, dtype=np.int32)
        # Isolated ray through the decorative sphere must remain transparent to MuJoCo rays.
        assert mujoco.mj_ray(model, data, np.array([0.3, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, hit) == -1


@pytest.mark.parametrize("finish", ["plastic", "steel", "wood"])
def test_no_material_default_grey_retains_its_exact_colour(finish):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="base"><geom size=".1"/></body></worldbody></mujoco>'
    )
    before = spec.compile()
    body_surfaces.apply(spec, {"base": finish})
    assert colours(before) == colours(spec.compile())


@pytest.mark.parametrize("body", ["absent", "world", ""])
def test_unknown_or_world_body_is_not_a_target(body):
    with pytest.raises(ValueError, match="choose an opaque"):
        body_surfaces.apply(mujoco.MjSpec.from_string(XML), {body: "steel"})


@pytest.mark.parametrize(
    "extra",
    [
        '<geom type="sphere" size=".1" rgba="1 1 1 .5"/>',
        '<geom type="sphere" size=".1" material="image"/>',
    ],
)
def test_transparent_or_textured_part_is_rejected_atomically(extra):
    xml = (
        '<mujoco><asset><texture name="t" type="2d" builtin="checker" width="2" height="2"/>'
        '<material name="image" texture="t"/></asset><worldbody><body name="base">'
        '<geom type="sphere" size=".1"/>' + extra + "</body></worldbody></mujoco>"
    )
    spec = mujoco.MjSpec.from_string(xml)
    before = spec.to_xml()
    with pytest.raises(ValueError, match="opaque, untextured"):
        body_surfaces.apply(spec, {"base": "steel"})
    assert spec.to_xml() == before


def test_empty_mapping_keeps_legacy_yaml_and_compiled_arrays():
    world = World(name="style", robots=[RobotEntry(name="bot", template="rover")])
    assert "body_surfaces" not in world.to_yaml()
    explicit = World.model_validate(
        {**world.model_dump(), "robots": [{**world.robots[0].model_dump(), "body_surfaces": {}}]}
    )
    assert explicit.to_yaml() == world.to_yaml()
    first, second = compile_world(world).model, compile_world(explicit).model
    for field in ["geom_rgba", "geom_matid", "body_mass", "geom_size", "mat_rgba"]:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


@pytest.mark.parametrize(
    "mapping", [{"base": "gold"}, {"base": True}, {"": "steel"}, {"x" * 161: "wood"}, {True: "steel"}]
)
def test_mapping_schema_refuses_unbounded_or_unknown_values(mapping):
    with pytest.raises(ValueError):
        RobotEntry(name="bot", template="rover", body_surfaces=mapping)
