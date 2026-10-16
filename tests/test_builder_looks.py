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
from built_robots import CENTIPEDE, GALLERY, MATERIAL_PROBE, SCOUT
from test_visuals import assert_same_physics

from robotforge.design import components, visuals
from robotforge.design.builder import looks
from robotforge.design.templates.base import SHELL
from robotforge.design.visuals import parts as P
from robotforge.design.world import World, compile_world
from robotforge.sim.geometry import collides

# Mesh triangles a built robot may draw: its base with the base's sensors, plus so many per part. A part
# with its servo case and horn measured 1.4k to 1.7k (2026-10-17), a wheel with its gear motor about 3k.
TRIANGLES_BASE, TRIANGLES_PER_PART = 14_000, 1_600
# The gallery's robots stay under 40k each (the templates' budget is 30k): snake 35k, centaur 33k, spider 32k.
TRIANGLE_BUDGET_GALLERY = 40_000
MESH, BOX, CAPSULE = mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE


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


def _own(model: mujoco.MjModel, body: int) -> bool:
    """A body of the built robot itself (not the floor, not a mounted template, which has its own budget)."""
    name = model.body(body).name
    return name.startswith("r/") and "/" not in name[2:]


def _triangles(model: mujoco.MjModel, own: bool = False) -> int:
    return sum(
        int(model.mesh_facenum[model.geom_dataid[g]])
        for g in range(model.ngeom)
        if model.geom_type[g] == MESH and (not own or _own(model, int(model.geom_bodyid[g])))
    )


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


@pytest.mark.parametrize("name", [*GALLERY, "centipede"])
def test_built_robots_fit_the_triangle_budget(name):
    build = CENTIPEDE if name == "centipede" else GALLERY[name]
    model = _compile(build)
    parts = sum(1 for b in range(model.nbody) if _own(model, b)) - 1  # all but the base
    assert _triangles(model, own=True) <= TRIANGLES_BASE + TRIANGLES_PER_PART * parts
    if name in GALLERY:
        assert _triangles(model) <= TRIANGLE_BUDGET_GALLERY


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

    def surfaces(body: str) -> set[tuple[str, tuple[float, ...]]]:
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
    assert (f"r/{surface}", teal) in surfaces("base")  # the robot's color, in the material's surface
    assert (f"r/{surface}", orange) in surfaces("arm")  # a part's own color is kept exactly
    # A part with no color shows white plastic or rubber, bare aluminium or black carbon fibre.
    assert (f"r/{surface}", _rounded(bare_post)) in surfaces("post")


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


def _own_pixels(model: mujoco.MjModel, cam: int, height: int = 120, width: int = 160) -> int:
    """Pixels of the camera's own body in its picture at the home pose."""
    data = _home(model)
    renderer = mujoco.Renderer(model, height, width)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
    finally:
        renderer.close()
    geoms = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = seg[..., 0][geoms]
    return int(
        np.sum([np.sum(ids == g) for g in np.unique(ids) if g >= 0 and model.geom_bodyid[g] == model.cam_bodyid[cam]])
    )


ON_TOP = {"at": "top", "shift": [0, 0, 0.05], "joint": {"axis": "z", "range": [-90, 90]}, "color": "white"}
INSIDE = {  # a part with a camera at its middle (the default "camera" shorthand)
    "box base": {"base": {"shape": "box", "size": [0.24, 0.16, 0.06]}},
    "cylinder base": {"base": {"shape": "cylinder", "size": [0.26, 0.05]}},
    "box part": {"parts": [{"name": "head", **ON_TOP, "shape": "box", "size": [0.08, 0.1, 0.05]}]},
    "sphere part": {"parts": [{"name": "head", **ON_TOP, "shape": "sphere", "size": [0.08]}]},
}


@pytest.mark.parametrize("direction", ["forward", "down", "back"])
@pytest.mark.parametrize("shape", list(INSIDE))
def test_a_camera_inside_its_part_sees_out_of_it(needs_gl, shape, direction):
    """The default camera sits in the middle of its part. The dressing drew faces that look into the shell (a
    dark pan's lid, the core behind a parting line, a hemisphere's rim, the backs of a vent, a screw's and an
    LED's undersides) and the camera filmed them: up to all of its picture was the robot's dark insides. A
    camera sees out through its own shell, as through the primitive it had before. Screw heads and vent slats
    outside the shell may show through it, a few pixels of them."""
    build = {"color": "blue", "base": {"shape": "box", "size": [0.2, 0.16, 0.06], "fixed": True}}
    extra = copy.deepcopy(INSIDE[shape])
    build["base"].update(extra.get("base", {}))
    build["parts"] = extra.get("parts", [])
    holder = build["parts"][0] if build["parts"] else build["base"]
    holder["sensors"] = [{"type": "camera", "direction": direction, "fov": 90}]
    model = _compile(build)
    cam = next(c for c in range(model.ncam) if model.camera(c).name.startswith("r/"))
    assert _own_pixels(model, cam) <= 0.015 * 120 * 160


# ---- where details sit -------------------------------------------------------------------------------------

FACES = {"top": (0, 0, 1), "bottom": (0, 0, -1), "front": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0)}
FACES["right"] = (0, -1, 0)


def _mesh_points(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> np.ndarray:
    """A mesh geom's vertices in its body's frame."""
    m = int(model.geom_dataid[g])
    start = int(model.mesh_vertadr[m])
    local = model.mesh_vert[start : start + int(model.mesh_vertnum[m])]
    world = data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T
    b = int(model.geom_bodyid[g])
    return (world - data.xpos[b]) @ data.xmat[b].reshape(3, 3)


def _mesh_triangles(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> np.ndarray:
    m = int(model.geom_dataid[g])
    faces = model.mesh_face[int(model.mesh_faceadr[m]) : int(model.mesh_faceadr[m]) + int(model.mesh_facenum[m])]
    return _mesh_points(model, data, g)[faces]


def _colored(model: mujoco.MjModel, body: str, rgba) -> list[int]:
    b = model.body(f"r/{body}").id
    want = _rounded(rgba)
    return [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == b and model.geom_type[g] == MESH and _rounded(model.geom_rgba[g]) == want
    ]


def _ray(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> float:
    """How far along `direction` from `origin` the first of the triangles is (Moller-Trumbore)."""
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-15
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - tris[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return float(t[hit].min())


@pytest.mark.parametrize("face", list(FACES))
def test_an_imu_on_a_face_lies_flat_on_it(face):
    """An IMU board on a side face stood out of it like a shelf, half buried: its normal was always up."""
    half = np.array([0.1, 0.07, 0.03])
    base = {"shape": "box", "size": (2 * half).tolist(), "fixed": True, "sensors": [{"type": "imu", "at": face}]}
    model = _compile({"base": base})
    data = _home(model)
    n = np.array(FACES[face], float)
    (board,) = _colored(model, "base", P.PCB)
    height = (_mesh_points(model, data, board) - n * float(np.abs(n) @ half)) @ n  # above the face
    assert -0.001 <= height.min() <= 0.0005 and height.max() <= 0.003  # flat on the face, not into it


@pytest.mark.parametrize("face", ["front", "back", "left", "right", "top"])
def test_an_imu_on_a_link_lies_on_the_link_as_drawn(face):
    """A hinged capsule is drawn tapered and flattened, not as its capsule: a board on its side sits on the
    surface you see, neither floating off it nor sunk into it."""
    r, length = 0.02, 0.14
    build = {
        "color": "orange",
        "base": {"shape": "box", "size": [0.2, 0.14, 0.06], "fixed": True},
        "parts": [
            {
                "name": "arm",
                "at": "top",
                "shift": [0, 0, 0.025],
                "joint": {"axis": "y", "range": [-60, 60]},
                "size": [2 * r, length],
                "direction": "up",
                "sensors": [{"type": "imu", "at": face}],
            }
        ],
    }
    model = _compile(build)
    data = _home(model)
    n = np.array(FACES[face], float)
    at = np.array([0.0, 0.0, length + r]) if face == "top" else np.array([0.0, 0.0, length / 2]) + n * r
    (link,) = _colored(model, "arm", SHELL)
    surface = 0.05 - _ray(at + n * 0.05, -n, _mesh_triangles(model, data, link))  # the drawn surface, above `at`
    (board,) = _colored(model, "arm", P.PCB)
    height = (_mesh_points(model, data, board) - at) @ n - surface
    assert -0.001 <= height.min() <= 0.0005 and height.max() <= 0.003


def _presets(model: mujoco.MjModel, body: str) -> set[str]:
    b = model.body(f"r/{body}").id
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g]))
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == b and model.geom_type[g] == MESH
    }


def test_without_servo_caps_a_powered_hinge_shows_no_hardware():
    """servo_caps: false asks for no joint hardware; it drew a dark knuckle drum at every powered hinge."""
    build = {
        "color": "teal",
        "base": {"shape": "capsule", "size": [0.1, 0.2], "axis": "x", "fixed": True},
        "parts": [
            {
                "name": "arm",
                "at": "top",
                "shift": [0, 0, 0.025],
                "joint": {"axis": "y", "range": [-60, 60]},
                "size": [0.03, 0.14],
                "direction": "up",
            }
        ],
    }
    with_caps = _compile(build)
    assert "r/rf_dark" in _presets(with_caps, "base") and "r/rf_metal" in _presets(with_caps, "arm")
    without = _compile({**build, "servo_caps": False})
    assert _presets(without, "base") == _presets(without, "arm") == {"r/rf_shell"}  # just the two shells


def test_a_tall_base_keeps_its_color_down_to_a_thin_dark_pan():
    """A base's dark lower pan took 36 % of its height: the balancer's red body had an 8 cm black band."""
    model = _compile(GALLERY["balancer"])
    data = _home(model)
    base = model.body("r/base").id
    (box,) = [g for g in range(model.ngeom) if model.geom_bodyid[g] == base and model.geom_type[g] == BOX]
    bottom = float(model.geom_pos[box][2] - model.geom_size[box][2])
    (shell,) = _colored(model, "base", components.color(GALLERY["balancer"]["color"]))
    assert _mesh_points(model, data, shell)[:, 2].min() <= bottom + 0.016  # 15 mm of pan and a parting line


def _ring_points(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    """Points round the ring of a mesh lofted along a -> b that is nearest the middle."""
    d = (b - a) / np.linalg.norm(b - a)
    s = np.round((points - a) @ d, 5)
    stations, counts = np.unique(s, return_counts=True)
    ring = stations[counts >= 8][np.argmin(np.abs(stations[counts >= 8] - float(np.linalg.norm(b - a)) / 2))]
    return len(np.unique(np.round(points[s == ring], 5), axis=0))


def test_a_thick_capsule_is_smooth_and_round_where_it_does_not_swing():
    """Every capsule was a 16-sided rounded square with 4 rings per end, however big: a 30 cm capsule body
    looked lumpy and polygonal. A thick one now gets as many points round as a cylinder, and one with no hinge
    across it (a base) is drawn as the capsule it is."""
    build = {
        "color": "yellow",
        "base": {"shape": "capsule", "size": [0.3, 0.4], "axis": "x", "fixed": True},
        "parts": [
            {
                "name": "boom",
                "at": "top",
                "shift": [-0.1, 0, 0.065],
                "joint": {"axis": "y", "range": [-60, 60], "home": 30},
                "size": [0.12, 0.3],
                "direction": "up",
                "color": "teal",
            }
        ],
    }
    model = _compile(build)
    data = _home(model)
    for body, color, r in (("base", "yellow", 0.15), ("boom", "teal", 0.06)):
        b = model.body(f"r/{body}").id
        (cap,) = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_type[g] == CAPSULE]
        axis, half = model.geom_quat[cap], float(model.geom_size[cap][1])
        z = np.zeros(9)
        mujoco.mju_quat2Mat(z, axis)
        a = model.geom_pos[cap] - z.reshape(3, 3)[:, 2] * half
        end = model.geom_pos[cap] + z.reshape(3, 3)[:, 2] * half
        (mesh,) = _colored(model, body, components.color(color))
        points = _mesh_points(model, data, mesh)
        assert _ring_points(points, a, end) >= min(60, 2 * np.pi * r / looks.CURVE_STEP)
        if body == "base":  # round and untapered: every point on the collision capsule's surface
            t = np.clip((points - a) @ (end - a) / float((end - a) @ (end - a)), 0.0, 1.0)
            off = np.linalg.norm(points - (a + np.outer(t, end - a)), axis=1) - r
            assert np.abs(off).max() < 0.0003


def test_a_camera_on_top_stands_on_it_and_never_sees_inside_its_robot(needs_gl):
    """A camera on a face it does not look out of (on top, looking forward) was halfway into the shell: its
    picture showed the dark insides of its own robot. It now stands on the face and sees the robot's top."""
    build = {"color": "blue", "base": {"shape": "box", "size": [0.24, 0.16, 0.06], "fixed": True}}
    build["base"]["sensors"] = [{"type": "camera", "at": "top"}]
    model = _compile(build)
    data = _home(model)
    cam = model.camera("r/base_cam").id
    assert model.cam_pos[cam][2] == pytest.approx(0.03 + looks.CAMERA_HALF - looks.CAMERA_SEAT)
    renderer = mujoco.Renderer(model, 120, 160)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
    finally:
        renderer.close()
    geoms = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    seen = {int(g) for g in np.unique(seg[..., 0][geoms]) if g >= 0 and model.geom_bodyid[g] == model.cam_bodyid[cam]}
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g])) for g in seen}
    assert "r/rf_dark" not in names  # the pan, the parting line's core and the camera's own housing


PASSIVE = {  # parts on unpowered hinges: a ball turning about its middle, a disc about its axis, a hanging bob
    "color": "teal",
    "base": {"shape": "box", "size": [0.4, 0.2, 0.06], "fixed": True},
    "parts": [
        {"name": "ball", "at": "top", "shift": [-0.14, 0, 0.05], "shape": "sphere", "size": [0.06], "color": "red"},
        {"name": "disc", "at": "top", "shift": [-0.03, 0, 0.07], "shape": "cylinder", "size": [0.12, 0.02]},
        {"name": "bob", "at": "top", "shift": [0.1, 0, 0], "shape": "sphere", "size": [0.05], "direction": "up"},
    ],
}
PASSIVE["parts"][1].update(direction="left", color="yellow")
PASSIVE["parts"][2]["color"] = "orange"
for _part in PASSIVE["parts"]:
    _part.update(joint={"axis": "y", "range": [-90, 90]}, actuator="none")


@pytest.mark.parametrize("name", ["ball", "disc", "bob"])
def test_a_part_on_an_unpowered_hinge_keeps_its_color(needs_gl, name):
    """The cover on an unpowered hinge was sized by the part's whole radius: a ball turning about its middle
    and a disc spinning about its axis vanished into dark drums, and a bob hung under a dark wheel. A ball is
    held by a neck, a hinge deep inside its part or along a disc's axis needs no cover: seen from anywhere
    round it, the part is mostly its own color."""
    model = _compile(PASSIVE)
    data = _home(model)
    body = model.body(f"r/{name}").id
    color = _rounded(components.color(next(p["color"] for p in PASSIVE["parts"] if p["name"] == name)))
    renderer = mujoco.Renderer(model, 120, 160)
    own = colored = 0
    try:
        renderer.enable_segmentation_rendering()
        for azimuth in (0, 90, 180, 270):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = data.xipos[body]
            cam.distance, cam.azimuth, cam.elevation = 0.3, azimuth, -20
            renderer.update_scene(data, camera=cam)
            seg = renderer.render()
            geoms = seg[..., 0][seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
            for g in np.unique(geoms):
                if g >= 0 and model.geom_bodyid[g] == body:
                    count = int(np.sum(geoms == g))
                    own += count
                    colored += count if _rounded(model.geom_rgba[g]) == color else 0
    finally:
        renderer.close()
    assert own > 0 and colored >= 0.85 * own


def test_a_lead_screw_with_nothing_near_its_far_end_ends_free():
    """A lifter rising off a base's top had the bearing at the far end of its lead screw drawn as a stub in
    mid-air, 9 cm above the base, holding nothing. With nothing of the parent near to hold a bearing, the
    screw ends where the nut stops and no bracket is drawn there."""
    build = {"color": "teal", "linear": "l16", "base": {"shape": "box", "size": [0.25, 0.18, 0.06], "fixed": True}}
    lifter = {"name": "lifter", "at": "top", "shift": [0.05, 0, 0], "shape": "box", "size": [0.06, 0.06, 0.02]}
    lifter["direction"] = "up"
    lifter["joint"] = {"type": "slide", "axis": "z", "range": [0, 0.08]}
    build["parts"] = [lifter]
    model = _compile(build)
    data = _home(model)
    top = 0.03  # the base's top face, in its frame
    base = model.body("r/base").id
    meshes = [g for g in range(model.ngeom) if model.geom_bodyid[g] == base and model.geom_type[g] == MESH]
    highest = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g])): float(
            _mesh_points(model, data, g)[:, 2].max()
        )
        for g in meshes
    }
    assert highest["r/rf_dark"] < top + 0.005  # the pan, the vent, the motor: no bracket up in the air
    assert highest["r/rf_metal"] < top + 0.08 + 0.0056 + 0.001  # the screw ends level with the nut at full stroke
