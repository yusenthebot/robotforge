"""Native mesh images preserve interior detail, with legacy vertex colours as an additive fallback."""

import mujoco
import numpy as np
import pytest

from robotforge.viewer import scene
from robotforge.viewer import textures as model_textures

# A tetrahedron on a 2x2 checker texture: two of its corners on a red square of it, two on a blue one.
XML = """
<mujoco>
  <asset>
    <texture name="halves" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 0 1" width="2" height="2"/>
    <material name="painted" texture="halves"/>
    <mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"
          texcoord="0.1 0.1  0.1 0.1  0.9 0.1  0.9 0.1"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="tetra" material="painted"/>
    <geom type="box" size=".1 .1 .1" pos="2 0 0"/>
  </worldbody>
</mujoco>
"""


def test_a_textured_mesh_is_sent_with_its_texture_colors():
    model = mujoco.MjModel.from_xml_string(XML)
    payload = scene.build_mesh_payload(model)
    (entry,) = payload.entries
    assert "colors" in entry
    floats = np.frombuffer(payload.blob, dtype=np.float32)
    colors = floats[entry["colors"] : entry["colors"] + 3 * entry["count"]].reshape(-1, 3)
    assert colors.shape == (12, 3)
    # The texture's colors, in linear light: each vertex is pure red or pure blue, and both appear.
    assert np.all(np.isclose(colors.max(axis=1), 1.0, atol=1e-3)) and np.all(colors[:, 1] < 1e-3)
    assert {tuple(np.round(c)) for c in colors} == {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}


def test_an_untextured_mesh_has_no_colors():
    model = mujoco.MjModel.from_xml_string(XML.replace(' material="painted"', ""))
    (entry,) = scene.build_mesh_payload(model).entries
    assert "colors" not in entry


def test_a_mesh_colored_per_vertex_keeps_its_material_color():
    """A textured mesh sent with its colors per vertex is not also tinted with the texture's mean color (the way a
    textured shell without them is shown, see test_scene): three.js multiplies the two, which darkened it twice."""
    model = mujoco.MjModel.from_xml_string(XML)
    payload = scene.build_mesh_payload(model)
    (mesh, _) = scene.describe(model, 1, payload.entries)["geoms"]
    assert mesh["rgba"] == [1, 1, 1, 1]


def test_native_images_and_uvs_retain_the_legacy_vertex_fallback():
    model = mujoco.MjModel.from_xml_string(XML)
    payload = scene.build_mesh_payload(model)
    (entry,) = payload.entries
    floats = np.frombuffer(payload.blob, dtype=np.float32)
    uv = floats[entry["uv"] : entry["uv"] + 2 * entry["count"]].reshape(-1, 2)
    np.testing.assert_array_equal(uv, scene._mesh_uvs(model, 0))
    (image,) = payload.textures
    assert (image["width"], image["height"], image["length"]) == (2, 2, 16)
    pixels = np.frombuffer(payload.texture_blob, dtype=np.uint8).reshape(2, 2, 4)
    np.testing.assert_array_equal(pixels[:, :, :3], model.tex_data.reshape(2, 2, 3))
    assert np.all(pixels[:, :, 3] == 255) and not image["alpha"]
    assert "colors" in entry
    old = scene.describe(model, 1, payload.entries)
    new = scene.describe(model, 1, payload.entries, payload.hfields, payload.textures)
    assert "textures" not in old and new["textures"] == payload.textures
    assert new["geoms"][0]["texture"] == {"id": image["id"], "repeat": [1.0, 1.0]}
    assert not new["geoms"][0]["vertex_colors"]
    assert old["geoms"][0]["rgba"] == new["geoms"][0]["rgba"]


def test_shared_geometry_uses_each_geoms_own_image_and_repeat():
    xml = XML.replace(
        '<material name="painted" texture="halves"/>',
        '<texture name="other" type="2d" builtin="flat" rgb1="0 1 0" width="2" height="2"/>'
        '<material name="painted" texture="halves"/>'
        '<material name="green" texture="other" texrepeat="2 3"/>',
    ).replace(
        '<geom type="mesh" mesh="tetra" material="painted"/>',
        '<geom name="first" type="mesh" mesh="tetra" material="painted"/>'
        '<geom name="second" type="mesh" mesh="tetra" material="green" pos="2 0 0"/>'
        '<geom name="plain" type="mesh" mesh="tetra" rgba="1 1 0 1" pos="4 0 0"/>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    payload = scene.build_mesh_payload(model)
    desc = scene.describe(model, 1, payload.entries, payload.hfields, payload.textures)
    first, second, plain = desc["geoms"][:3]
    assert first["mesh"] == second["mesh"] == plain["mesh"]
    assert first["texture"]["id"] != second["texture"]["id"]
    assert second["texture"]["repeat"] == [2.0, 3.0]
    assert "texture" not in plain and not plain["vertex_colors"]
    assert plain["rgba"] == [1, 1, 0, 1]


@pytest.mark.parametrize("uniform", [False, True])
def test_native_repeat_matches_mujoco_explicit_uv_scaling(uniform):
    xml = XML.replace(
        '<material name="painted" texture="halves"/>',
        f'<material name="painted" texture="halves" texrepeat="0 -1" texuniform="{str(uniform).lower()}"/>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    payload = scene.build_mesh_payload(model)
    geom = scene.describe(model, 1, payload.entries, payload.hfields, payload.textures)["geoms"][0]
    expected = model.geom_size[0, :2] if uniform else [1, 1]
    np.testing.assert_allclose(geom["texture"]["repeat"], expected)


@pytest.mark.parametrize("invalid", [-1, 4])
def test_invalid_face_uv_indices_use_the_existing_fallback(invalid):
    model = mujoco.MjModel.from_xml_string(XML)
    model.mesh_facetexcoord[0, 0] = invalid
    payload = scene.build_mesh_payload(model)
    assert not payload.textures and not payload.texture_blob
    assert "uv" not in payload.entries[0] and "colors" not in payload.entries[0]


def test_nonfinite_uv_uses_the_existing_fallback():
    model = mujoco.MjModel.from_xml_string(XML)
    model.mesh_texcoord[0, 0] = np.nan
    assert not scene.build_mesh_payload(model).textures


def test_non_rgb_images_do_not_create_malformed_vertex_attributes():
    model = mujoco.MjModel.from_xml_string(XML)
    model.tex_nchannel[0] = 1  # encoded formats are not one RGB triplet per pixel
    payload = scene.build_mesh_payload(model)
    assert not payload.textures and "colors" not in payload.entries[0]
    assert scene.describe(model, 1, payload.entries)["geoms"][0]["rgba"] == [1, 1, 1, 1]


def test_untextured_scenes_keep_their_original_payload():
    model = mujoco.MjModel.from_xml_string(XML.replace(' material="painted"', ""))
    payload = scene.build_mesh_payload(model)
    assert not payload.textures and payload.texture_blob == b""
    assert "uv" not in payload.entries[0]
    assert "textures" not in scene.describe(model, 1, payload.entries, payload.hfields, payload.textures)


def test_duplicate_images_share_one_budget_and_data_block():
    xml = XML.replace(
        '<material name="painted" texture="halves"/>',
        '<texture name="same" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 0 1" width="2" height="2"/>'
        '<material name="painted" texture="halves"/>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    payload = model_textures.build(model, {0, 1})
    assert len(payload.entries) == 1
    assert payload.entries[0]["aliases"] == [0, 1] and len(payload.blob) == 16


def test_native_images_keep_the_legacy_srgb_convention_and_source_provenance():
    xml = XML.replace(
        '<material name="painted" texture="halves"/>',
        '<texture name="linear_copy" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 0 1" '
        'width="2" height="2" colorspace="linear"/>'
        '<material name="painted" texture="halves"/>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    model.tex_colorspace[0] = int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
    payload = model_textures.build(model, {0, 1})
    assert len(payload.entries) == 1
    assert payload.entries[0]["color_space"] == "srgb"
    assert payload.entries[0]["source_color_spaces"] == [1, 2]


def test_rgba_images_retain_alpha_in_the_bounded_bundle(tmp_path):
    from PIL import Image

    Image.new("RGBA", (3, 2), (128, 64, 192, 64)).save(tmp_path / "glass.png")
    xml = tmp_path / "glass.xml"
    xml.write_text(
        '<mujoco><asset><texture type="2d" nchannel="4" file="glass.png"/></asset></mujoco>', encoding="utf-8"
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    payload = model_textures.build(model, {0})
    assert payload.entries[0]["alpha"]
    np.testing.assert_array_equal(np.frombuffer(payload.blob, dtype=np.uint8).reshape(-1, 4), [[128, 64, 192, 64]] * 6)


def test_many_large_images_obey_a_total_pixel_budget(monkeypatch):
    xml = (
        "<mujoco><asset>"
        + "".join(
            f'<texture name="t{i}" type="2d" builtin="flat" rgb1="{i / 7} 0 0" width="64" height="16"/>'
            for i in range(8)
        )
        + "</asset></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    monkeypatch.setattr(model_textures, "MAX_EDGE", 32)
    monkeypatch.setattr(model_textures, "MAX_PIXELS", 128)
    first = model_textures.build(model, set(range(8)))
    second = model_textures.build(model, set(reversed(range(8))))
    assert first == second
    assert sum(e["width"] * e["height"] for e in first.entries) <= 128
    assert len(first.blob) <= 4 * 128
    assert all(e["width"] == 4 * e["height"] for e in first.entries)


def test_image_count_limit_falls_back_without_exceeding_the_budget(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        XML.replace(
            '<material name="painted" texture="halves"/>',
            '<texture name="other" type="2d" builtin="flat" rgb1="0 1 0" width="2" height="2"/>'
            '<material name="painted" texture="halves"/>',
        )
    )
    monkeypatch.setattr(model_textures, "MAX_IMAGES", 1)
    payload = model_textures.build(model, {0, 1})
    assert len(payload.entries) == 1 and payload.entries[0]["aliases"] == [0]
