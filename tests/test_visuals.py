"""Visual meshes are looks only: masses, inertias, contacts and motion must not change."""

import mujoco
import numpy as np
import pytest

from robotforge.design import templates, visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.world import World, compile_world
from robotforge.sim.geometry import collides

GEOM_FIELDS = ("type", "size", "pos", "quat", "contype", "conaffinity", "friction", "condim", "solref", "solimp")
GEOM_FIELDS += ("margin", "gap", "priority", "solmix")
BODY_FIELDS = ("mass", "inertia", "ipos", "iquat", "pos", "quat", "subtreemass")
TRIANGLE_BUDGET_ROBOT = 30_000
TRIANGLE_BUDGET_SCENE = 200_000


def _names(model: mujoco.MjModel, obj: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj, i) or "" for i in range(count)]


def assert_same_physics(bare: mujoco.MjModel, dressed: mujoco.MjModel, steps: int = 200) -> None:
    assert _names(bare, mujoco.mjtObj.mjOBJ_BODY, bare.nbody) == _names(
        dressed, mujoco.mjtObj.mjOBJ_BODY, dressed.nbody
    )
    assert _names(bare, mujoco.mjtObj.mjOBJ_JOINT, bare.njnt) == _names(
        dressed, mujoco.mjtObj.mjOBJ_JOINT, dressed.njnt
    )
    assert _names(bare, mujoco.mjtObj.mjOBJ_ACTUATOR, bare.nu) == _names(
        dressed, mujoco.mjtObj.mjOBJ_ACTUATOR, dressed.nu
    )
    assert _names(bare, mujoco.mjtObj.mjOBJ_SENSOR, bare.nsensor) == _names(
        dressed, mujoco.mjtObj.mjOBJ_SENSOR, dressed.nsensor
    )
    assert _names(bare, mujoco.mjtObj.mjOBJ_SITE, bare.nsite) == _names(
        dressed, mujoco.mjtObj.mjOBJ_SITE, dressed.nsite
    )
    for field in BODY_FIELDS:
        np.testing.assert_array_equal(getattr(bare, f"body_{field}"), getattr(dressed, f"body_{field}"), err_msg=field)
    for field in ("dof_armature", "dof_damping", "dof_frictionloss", "actuator_gainprm", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(bare, field), getattr(dressed, field), err_msg=field)
    # Every geom that collides or weighs something is the same, in the same order and on the same body.
    old = [g for g in range(bare.ngeom) if collides(bare, g)]
    new = [g for g in range(dressed.ngeom) if collides(dressed, g)]
    assert len(old) == len(new)
    for field in GEOM_FIELDS:
        a, b = getattr(bare, f"geom_{field}")[old], getattr(dressed, f"geom_{field}")[new]
        np.testing.assert_array_equal(a, b, err_msg=f"geom_{field}")
    np.testing.assert_array_equal(bare.geom_bodyid[old], dressed.geom_bodyid[new])
    # Added geoms are visual only: no contacts and no mass.
    added = [g for g in range(dressed.ngeom) if dressed.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    for g in added:
        assert dressed.geom_contype[g] == 0 and dressed.geom_conaffinity[g] == 0
    # And the robot moves exactly as before.
    da, db = mujoco.MjData(bare), mujoco.MjData(dressed)
    if bare.nkey:
        mujoco.mj_resetDataKeyframe(bare, da, 0)
        mujoco.mj_resetDataKeyframe(dressed, db, 0)
    for _ in range(steps):
        mujoco.mj_step(bare, da)
        mujoco.mj_step(dressed, db)
    np.testing.assert_allclose(da.qpos, db.qpos, rtol=0, atol=1e-9)


def _world(robot: dict) -> World:
    return World.model_validate({"robots": [{"name": "r", **robot}]})


@pytest.mark.parametrize("name", list(templates.TEMPLATES))
def test_template_looks_leave_physics_unchanged(name):
    with visuals.disabled():
        bare = compile_world(_world({"template": name})).model
    dressed = compile_world(_world({"template": name})).model
    assert bare.nmesh == 0 and dressed.nmesh > 0
    assert_same_physics(bare, dressed, steps=300)


PARAM_SWEEP = [
    ("quadruped", {"body_length": 0.12, "body_width": 0.06, "body_height": 0.03, "leg_length": 0.08}),
    ("quadruped", {"body_length": 1.0, "body_width": 0.5, "body_height": 0.3, "leg_length": 0.8, "face": False}),
    ("quadruped", {"servo": "bldc_joint", "material": "aluminum", "color": [0.2, 0.2, 0.2]}),
    ("arm", {"dof": 3, "reach": 0.15, "base_height": 0.02, "gripper": False}),
    ("arm", {"dof": 4, "reach": 1.2, "base_height": 0.5, "gripper_opening": 0.2}),
    ("arm", {"dof": 6, "servo": "xl430", "gripper_opening": 0.02}),
    ("rover", {"length": 0.08, "width": 0.06, "wheel_radius": 0.015}),
    ("rover", {"length": 1.0, "width": 1.0, "wheel_radius": 0.3, "motor": "jga25"}),
    ("humanoid", {"height": 0.3, "arms": False, "head": False}),
    ("humanoid", {"height": 1.0, "servo": "bldc_joint"}),
    ("cartpole", {"rail_length": 0.4, "rail_height": 0.1, "pole_length": 0.1}),
    ("gripper", {"opening": 0.2}),
]


@pytest.mark.parametrize(("name", "params"), PARAM_SWEEP)
def test_template_looks_hold_across_parameters(name, params):
    with visuals.disabled():
        bare = templates.build(name, params).spec.compile()
    dressed = templates.build(name, params).spec.compile()
    assert_same_physics(bare, dressed, steps=50)


@pytest.mark.parametrize("name", list(templates.TEMPLATES))
def test_template_looks_fit_the_triangle_budget_and_material_contract(name):
    model = templates.build(name, {}).spec.compile()
    triangles = sum(
        int(model.mesh_facenum[model.geom_dataid[g]]) for g in range(model.ngeom) if model.geom_dataid[g] >= 0
    )
    assert 0 < triangles <= TRIANGLE_BUDGET_ROBOT
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        material = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g]))
        assert material is not None and material.startswith("rf_") and material[3:] in visuals.PRESETS
        assert model.geom_group[g] == visuals.VISUAL_GROUP


def test_hidden_primitives_are_covered_and_physics_geoms_stay_put():
    """Only primitives that a mesh replaces move to the hidden group; nothing else changes group."""
    with visuals.disabled():
        bare = templates.build("quadruped", {}).spec.compile()
    dressed = templates.build("quadruped", {}).spec.compile()
    shown_before = sum(1 for g in range(bare.ngeom) if bare.geom_group[g] < 3)
    hidden_after = sum(1 for g in range(dressed.ngeom) if dressed.geom_group[g] == visuals.HIDDEN_GROUP)
    assert hidden_after == shown_before  # every quadruped primitive has a mesh over it


def test_meshes_are_closed_and_face_outward():
    shapes = {
        "rounded_box": M.rounded_box(0.1, 0.05, 0.02, 0.01),
        "cylinder": M.cylinder(0.05, 0.1, bevel=0.01),
        "sphere": M.sphere(0.05),
        "link": M.link(0.12, (0.02, 0.015), (0.012, 0.01)),
        "torus": M.torus(0.05, 0.01),
        "revolve": M.revolve([(0.02, 0), (0.03, 0), (0.03, 0.01), (0.02, 0.01)]),
        "blob": M.blob((0.5, 0.5, 0.4), seed=3),
    }
    for name, mesh in shapes.items():
        v, f = mesh.vertices, mesh.faces
        volume = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6
        assert volume > 0, name
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        big = np.linalg.norm(fn, axis=1) > 1e-10
        agree = np.einsum("ij,ij->i", fn[big], mesh.normals[f[big, 0]])
        assert (agree > 0).all(), name
    box = M.rounded_box(0.1, 0.05, 0.02, 0.01)
    assert np.allclose(box.vertices.min(axis=0), [-0.1, -0.05, -0.02])
    assert np.allclose(box.vertices.max(axis=0), [0.1, 0.05, 0.02])


def test_unknown_material_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown material preset"):
        visuals.material_name("chrome")
