"""Robots built from parts are dressed like machines, and the dressing is looks only.

Physics, sensor readings and contacts must be exactly what the primitives give; every added geom is
visual (no contacts, no mass, transparent to rangefinder rays); every primitive is covered; and the
look stays within a triangle budget the viewer can draw.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest
from built_robots import GALLERY, SCOUT
from test_visuals import assert_same_physics

from robotforge.design import visuals
from robotforge.design.world import World, compile_world
from robotforge.sim.geometry import collides

# Mesh triangles a built robot may add (the templates' budget is 30k; the hexapod alone has 18 servos,
# each a case and a horn). Measured 2026-10-17: snake 35k (16 wheels), centaur 33k, spider 32k.
TRIANGLE_BUDGET_BUILT = 40_000
MESH = mujoco.mjtGeom.mjGEOM_MESH


def _compile(build: dict) -> mujoco.MjModel:
    return compile_world(World.model_validate({"robots": [{"name": "r", "build": build}]})).model


def _home(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _rounded(rgba) -> tuple[float, ...]:
    return tuple(round(float(c), 3) for c in rgba)


def _triangles(model: mujoco.MjModel) -> int:
    return sum(int(model.mesh_facenum[model.geom_dataid[g]]) for g in range(model.ngeom) if model.geom_type[g] == MESH)


@pytest.mark.parametrize("name", list(GALLERY))
def test_dressing_leaves_physics_and_sensors_unchanged(name):
    build = GALLERY[name]
    with visuals.disabled():
        bare = _compile(build)
    dressed = _compile(build)
    assert dressed.nmesh > bare.nmesh
    assert_same_physics(bare, dressed, steps=500)
    # Rangefinders, lidars, IMUs and touch sensors read the same at home and after a second of motion.
    a, b = _home(bare), _home(dressed)
    np.testing.assert_array_equal(a.sensordata, b.sensordata)
    for _ in range(500):
        mujoco.mj_step(bare, a)
        mujoco.mj_step(dressed, b)
    np.testing.assert_array_equal(a.sensordata, b.sensordata)


@pytest.mark.parametrize("name", list(GALLERY))
def test_dressing_is_visual_only_and_covers_every_primitive(name):
    model = _compile(GALLERY[name])
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if not body.startswith("r/") or "/" in body[2:]:
            continue  # the floor, and mounted templates (dressed and tested on their own)
        if model.geom_type[g] == MESH:
            assert not collides(model, g)
            assert model.geom_group[g] == visuals.VISUAL_GROUP
            material = int(model.geom_matid[g])
            name_ = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material)
            assert name_.startswith("r/rf_") and name_[5:] in visuals.PRESETS
            assert model.mat_rgba[material][3] == 0  # rays look through it
            assert model.geom_rgba[g][3] == 1  # but it is drawn in its own color
        elif model.geom_rgba[g][3] > 0:
            # Nothing of the robot shows as a bare primitive: each one is under a mesh (or invisible).
            assert model.geom_group[g] == visuals.HIDDEN_GROUP, (name, mujoco.mj_id2name(model, 5, g), body)
    # Visual geoms weigh nothing: every body weighs what the bare build says.
    with visuals.disabled():
        bare = _compile(GALLERY[name])
    np.testing.assert_array_equal(bare.body_mass, model.body_mass)


@pytest.mark.parametrize("name", list(GALLERY))
def test_built_robots_fit_the_triangle_budget(name):
    assert _triangles(_compile(GALLERY[name])) <= TRIANGLE_BUDGET_BUILT


def test_rays_pass_through_the_dressing():
    """A ray aimed straight at any piece of a built robot's dressing goes through it: it hits a collision
    shape or nothing, never a mesh."""
    model = _compile(SCOUT)
    data = _home(model)
    meshes = [g for g in range(model.ngeom) if model.geom_type[g] == MESH]
    assert meshes
    hit = np.array([-1], dtype=np.int32)
    for g in meshes:
        target = data.geom_xpos[g]
        start = target + np.array([0.0, 0.0, 1.0])
        mujoco.mj_ray(model, data, start, np.array([0.0, 0.0, -1.0]), None, 1, -1, hit)
        assert hit[0] < 0 or model.geom_type[hit[0]] != MESH


def test_powered_hinges_get_a_servo_case_on_the_parent_and_a_horn_on_the_part():
    model = _compile(GALLERY["dog"])
    meshes = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) for i in range(model.nmesh)}
    for side in ("left", "right"):
        for i in (1, 2):
            # The thigh's servo sits on the hip and its horn turns with the thigh; the knee's on the thigh.
            assert f"r/rf_{side}_hip_{i}_dark" in meshes and f"r/rf_{side}_thigh_{i}_metal" in meshes
            assert f"r/rf_{side}_thigh_{i}_dark" in meshes and f"r/rf_{side}_shin_{i}_metal" in meshes
            assert f"r/rf_{side}_shin_{i}_rubber" in meshes  # the foot


MATERIAL_PROBE = {
    "color": "teal",
    "base": {"shape": "box", "size": [0.16, 0.12, 0.05], "fixed": True},
    "parts": [
        {"name": "post", "at": "top", "joint": "fixed", "shape": "cylinder", "size": [0.05, 0.06], "direction": "up"},
        {
            "name": "arm",
            "parent": "post",
            "at": "end",
            "shift": [0, 0, 0.02],
            "joint": {"axis": "y", "range": [-90, 90], "home": 30},
            "size": [0.03, 0.12],
            "direction": "up",
            "color": "#ff8800",
        },
    ],
}


@pytest.mark.parametrize(
    ("material", "surface", "bare_post"),
    [
        ("pla", "rf_shell", [0.9, 0.91, 0.93, 1.0]),
        ("aluminum", "rf_aluminum", [0.76, 0.78, 0.8, 1.0]),
        ("carbon", "rf_carbon", [0.15, 0.155, 0.165, 1.0]),
        ("rubber", "rf_rubber", [0.9, 0.91, 0.93, 1.0]),
    ],
)
def test_material_picks_the_surface_and_a_color_stays_a_color(material, surface, bare_post):
    build = copy.deepcopy(MATERIAL_PROBE)
    build["material"] = material
    model = _compile(build)

    def looks(body: str) -> set[tuple[str, tuple[float, ...]]]:
        b = model.body(f"r/{body}").id
        return {
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g])),
                _rounded(model.geom_rgba[g]),
            )
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == b and model.geom_type[g] == MESH
        }

    teal = _rounded([0.2, 0.7, 0.7, 1.0])
    orange = _rounded([1.0, 0x88 / 255, 0.0, 1.0])
    assert (f"r/{surface}", teal) in looks("base")  # the robot's color, in the material's surface
    assert (f"r/{surface}", orange) in looks("arm")  # a part's own color is kept exactly
    # A part with no color shows white plastic or rubber, bare aluminium or black carbon fibre.
    assert (f"r/{surface}", _rounded(bare_post)) in looks("post")


def test_a_robot_camera_never_sees_its_own_housing(needs_gl):
    """The board camera drawn round a camera point stands in front of it; a closed lens ring showed up as a
    dark circle in the robot's own camera image. Even at a 150 degree field of view nothing of the camera's
    own body may be in the picture."""
    build = copy.deepcopy(SCOUT)
    build["parts"][-1]["sensors"] = [{"type": "camera", "at": "front", "fov": 150}]
    model = _compile(build)
    data = _home(model)
    cam = model.camera("r/tilt_cam").id
    renderer = mujoco.Renderer(model, 120, 160)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
    finally:
        renderer.close()
    geoms = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    own = [g for g in np.unique(seg[..., 0][geoms]) if g >= 0 and model.geom_bodyid[g] == model.cam_bodyid[cam]]
    assert own == []
