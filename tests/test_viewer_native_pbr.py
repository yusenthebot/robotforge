"""Preserve native PBR values without changing material names or legacy scene fields."""

from __future__ import annotations

import json

import mujoco
import pytest

from robotforge.viewer.scene import describe


def model(fields: str = "") -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(f"""<mujoco><asset>
      <material name="robot/white_metal" {fields}/>
      <material name="unused"/>
      </asset><worldbody>
      <geom name="one" type="sphere" size=".1" material="robot/white_metal"/>
      <geom name="two" type="box" size=".1 .1 .1" pos="1 0 0" material="robot/white_metal"/>
      <geom name="plain" type="sphere" size=".1" pos="2 0 0"/>
      </worldbody></mujoco>""")


def test_unset_native_fields_leave_legacy_description_unchanged():
    desc = describe(model(), 1, [])
    assert "native_materials" not in desc
    assert all("material_id" not in geom for geom in desc["geoms"])


@pytest.mark.parametrize(
    "fields,expected",
    [
        ('roughness="0" metallic="1"', {"roughness": 0, "metalness": 1}),
        ('roughness="1" metallic="0"', {"roughness": 1, "metalness": 0}),
        ('roughness=".2"', {"roughness": 0.2}),
        ('metallic=".7"', {"metalness": 0.7}),
    ],
)
def test_shared_material_id_keeps_zero_partial_and_prefixed_author_values(fields, expected):
    desc = describe(model(fields), 1, [])
    assert set(desc["native_materials"]) == {0}
    assert desc["native_materials"][0] == pytest.approx(expected)
    one, two, plain = desc["geoms"]
    assert one["material_id"] == two["material_id"] == 0
    assert one["material"] == two["material"] == "robot/white_metal"
    assert "material_id" not in plain
    wire = json.loads(json.dumps(desc, allow_nan=False))
    assert wire["native_materials"][str(one["material_id"])] == desc["native_materials"][0]


@pytest.mark.parametrize("invalid", [-1, -0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_values_are_omitted_independently_and_never_serialize_nan(invalid):
    m = model('roughness=".5" metallic=".8"')
    m.mat_roughness[0] = invalid
    desc = describe(m, 1, [])
    assert desc["native_materials"][0] == pytest.approx({"metalness": 0.8})
    json.dumps(desc, allow_nan=False)
    m.mat_metallic[0] = invalid
    assert "native_materials" not in describe(m, 1, [])


def test_material_id_supports_anonymous_and_runtime_aliased_material_names():
    m = mujoco.MjModel.from_xml_string("""<mujoco><asset>
      <material name="a" roughness=".1" metallic=".9"/>
      <material name="b" roughness=".8" metallic=".2"/>
      </asset><worldbody><geom type="sphere" size=".1" material="a"/>
      <geom type="sphere" size=".1" pos="1 0 0" material="b"/></worldbody></mujoco>""")
    # Runtime MjModel metadata can alias names even though XML forbids duplicate material names.
    m.name_matadr[1] = m.name_matadr[0]
    desc = describe(m, 1, [])
    assert desc["geoms"][0]["material"] == desc["geoms"][1]["material"] == "a"
    assert [g["material_id"] for g in desc["geoms"]] == [0, 1]
    assert desc["native_materials"][0] != desc["native_materials"][1]
    m.name_matadr[0] = m.names.index(b"\0")
    anonymous = describe(m, 1, [])
    assert anonymous["geoms"][0]["material"] == ""
    assert anonymous["geoms"][0]["material_id"] == 0
    assert anonymous["native_materials"] == desc["native_materials"]
