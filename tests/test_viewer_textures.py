"""The viewer draws no textures, so a textured mesh is sent with its texture's colors per vertex.

Before, the Skydio X2 (one mesh whose texture holds the body, the arms and the spinning-propeller discs) showed
in the browser as a single flat white shape."""

import mujoco
import numpy as np

from robotforge.viewer import scene

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
