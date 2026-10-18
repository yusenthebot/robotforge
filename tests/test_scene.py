import mujoco
import numpy as np

from robotforge.viewer import scene

MESH_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <material name="m" rgba="0.1 0.2 0.3 1"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 1"/>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom name="mesh_geom" type="mesh" mesh="tet" material="m"/>
      <geom name="red_box" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1" material="m"/>
      <geom name="hidden" type="sphere" size="0.1" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MESH_XML)


def test_mesh_payload_is_non_indexed_triangles_with_normals():
    model = _model()
    payload = scene.build_mesh_payload(model)
    floats = np.frombuffer(payload.blob, dtype=np.float32)
    (entry,) = payload.entries
    assert entry["count"] == 3 * model.mesh_facenum[0]
    assert floats.size == 6 * entry["count"]
    normals = floats[3 * entry["count"] :].reshape(-1, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)


def test_describe_lists_drawable_geoms_with_colors():
    model = _model()
    desc = scene.describe(model, version=7, mesh_entries=[])
    by_name = {g["name"]: g for g in desc["geoms"]}
    assert desc["version"] == 7
    assert by_name["mesh_geom"]["type"] == "mesh"
    assert by_name["mesh_geom"]["mesh"] == 0
    # default rgba + material: material color wins; explicit rgba overrides material
    assert by_name["mesh_geom"]["rgba"][:3] == [0.1, 0.2, 0.3]
    assert by_name["red_box"]["rgba"][:3] == [1.0, 0.0, 0.0]
    assert by_name["hidden"]["group"] == 3


def test_a_textured_material_shows_the_mean_color_of_its_picture():
    """The viewer draws no textures: ANYmal C's red shell (a texture on a white material) showed plain white."""
    xml = """
    <mujoco>
      <asset>
        <texture name="half" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 0 1" width="64" height="64"/>
        <material name="shell" texture="half" rgba="1 1 1 1"/>
      </asset>
      <worldbody>
        <geom name="ground" type="plane" size="1 1 0.1" material="shell"/>
        <body name="b"><freejoint/><geom name="shell" type="box" size="0.1 0.1 0.1" material="shell"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    geoms = {g["name"]: g for g in scene.describe(model, 1, [])["geoms"]}
    np.testing.assert_allclose(geoms["shell"]["rgba"], [0.5, 0, 0.5, 1], atol=0.02)  # half red, half blue
    assert geoms["ground"]["rgba"] == [1, 1, 1, 1]  # the ground draws its own textures


def test_describe_marks_moving_bodies():
    desc = scene.describe(_model(), version=1, mesh_entries=[])
    bodies = {b["name"]: b for b in desc["bodies"]}
    assert bodies["b"]["dynamic"] is True
    assert bodies["world"]["dynamic"] is False
