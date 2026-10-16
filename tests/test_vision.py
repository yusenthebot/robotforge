"""The color finder: masks, blobs and centroids on pictures drawn by hand, and back-projection on real renders."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotforge.design.world import World, compile_world
from robotforge.sim import vision
from robotforge.sim.render import SnapshotRenderer

RED, GREEN, BLUE = (230, 40, 30), (40, 200, 70), (40, 90, 230)


def _picture(h: int = 60, w: int = 80, fill=(200, 200, 200)) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ---- hue, saturation, value and the mask ---------------------------------------------------------------------


def test_hsv_of_known_colors():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [255, 255, 0], [0, 0, 0]]], np.uint8)
    hsv = vision.to_hsv(rgb)[0]
    assert hsv[:, 0][:3] == pytest.approx([0, 120, 240])
    assert hsv[4, 0] == pytest.approx(60)
    assert hsv[3, 1] == 0 and hsv[3, 2] == pytest.approx(128 / 255)  # grey: no saturation
    assert hsv[5].tolist() == [0, 0, 0]  # black: no division by zero


def test_the_hue_window_wraps_around_red():
    rgb = np.array([[[255, 0, 40], [255, 40, 0], [255, 120, 0]]], np.uint8)  # hues about 351, 9 and 28
    mask = vision.threshold(vision.to_hsv(rgb), vision.ColorRange(0.0, 15.0))
    assert mask.tolist() == [[True, True, False]]


def test_saturation_and_value_floors_keep_grey_and_dark_out():
    rgb = np.array([[[200, 170, 170], [90, 10, 10], [40, 4, 4]]], np.uint8)  # pale red, dark red, nearly black
    hsv = vision.to_hsv(rgb)
    assert vision.threshold(hsv, vision.ColorRange(0, 15, sat=0.3, val=0.2)).tolist() == [[False, True, False]]


def test_clean_removes_specks_and_keeps_a_solid_blob():
    mask = np.zeros((30, 30), bool)
    mask[10:16, 10:16] = True
    mask[2, 25] = mask[27, 3] = True  # lone noisy pixels
    cleaned = vision.clean(mask)
    assert not cleaned[2, 25] and not cleaned[27, 3]
    assert cleaned[10:16, 10:16].all() and cleaned.sum() == 36


def test_blur_shrinks_noise_and_white_balance_takes_out_a_tint():
    rng = np.random.default_rng(0)
    flat = _picture(fill=(120, 120, 120))
    noisy = np.clip(flat + rng.normal(0, 9, flat.shape), 0, 255).astype(np.uint8)
    assert np.std(vision.blur(noisy).astype(float) - 120) < np.std(noisy.astype(float) - 120) / 2.5
    warm = _picture(fill=(180, 120, 60))
    balanced = vision.white_balance(warm).reshape(-1, 3).mean(axis=0)
    assert np.ptp(balanced) < 2  # the average is grey again


# ---- blobs and centroids ------------------------------------------------------------------------------------


def test_blobs_are_counted_measured_and_sorted_by_size():
    mask = np.zeros((40, 60), bool)
    mask[5:15, 10:30] = True  # 10 x 20
    mask[25:30, 40:45] = True  # 5 x 5
    mask[30, 45] = True  # touches the small square only at a corner: still the same blob
    mask[0, 59] = True  # too small to count
    found = vision.blobs(mask, min_area=4)
    assert [b.area for b in found] == [200, 26]
    big = found[0]
    assert (big.u, big.v) == pytest.approx((20.0, 10.0))  # pixel x covers x..x+1, so the middle of 10..29 is 20
    assert big.box == (10, 5, 29, 14)
    ys, xs = big.pixels()
    assert len(ys) == 200 and mask[ys, xs].all()


def test_blobs_join_runs_that_meet_further_down():
    mask = np.zeros((10, 12), bool)
    mask[1:6, 1] = True  # a U: two arms joined only along the bottom row
    mask[1:6, 9] = True
    mask[5, 1:10] = True
    assert len(vision.blobs(mask, min_area=1)) == 1


def test_interior_leaves_out_the_edge():
    mask = np.zeros((20, 20), bool)
    mask[5:11, 5:11] = True
    blob = vision.blobs(mask)[0]
    ys, xs = blob.interior()
    assert len(ys) == 16 and ys.min() == 6 and ys.max() == 9


def test_a_drawn_red_square_is_found_at_its_middle():
    picture = _picture()
    picture[20:30, 40:56] = RED
    picture[5:9, 5:9] = GREEN
    found = vision.blobs(vision.threshold(vision.to_hsv(picture), vision.NAMED["red"]))
    assert len(found) == 1 and (found[0].u, found[0].v) == pytest.approx((48.0, 25.0))


def test_named_colors_do_not_overlap_and_unknown_or_grey_colors_are_refused():
    windows = sorted(((r.hue - r.width) % 360, (r.hue + r.width) % 360) for r in vision.NAMED.values())
    hues = np.arange(0, 360, 0.5)
    inside = sum((np.abs((hues - r.hue + 180) % 360 - 180) <= r.width).astype(int) for r in vision.NAMED.values())
    assert inside.max() == 1, windows
    assert vision.color_range("#2b7de9").hue == pytest.approx(214, abs=1)  # any color: a window round it
    with pytest.raises(ValueError, match="unknown color"):
        vision.color_range("chartreuse")
    with pytest.raises(ValueError, match="too grey"):
        vision.color_range("gray")


# ---- a pinhole camera -------------------------------------------------------------------------------------


def _pinhole() -> vision.Pinhole:
    forward = np.array([0.0, np.sin(np.radians(40)), -np.cos(np.radians(40))])
    right = np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    return vision.Pinhole(np.array([0.2, -0.4, 0.5]), right, up, forward, 300.0, 300.0, 160.0, 120.0, 320, 240)


def test_a_point_projected_and_back_projected_lands_where_it_was():
    cam = _pinhole()
    for point in ([0.25, -0.05, 0.0], [0.1, 0.1, 0.02], [0.3, -0.12, 0.04]):
        u, v = cam.project(np.array(point))
        assert cam.on_plane(u, v, point[2]) == pytest.approx(point, abs=1e-9)
        depth = float((np.array(point) - cam.pos) @ cam.forward)
        assert cam.pos + depth * cam.ray(u, v) == pytest.approx(point, abs=1e-9)


def test_the_wrong_plane_height_lands_behind_the_object():
    cam = _pinhole()
    middle = np.array([0.2, 0.0, 0.02])
    u, v = cam.project(middle)
    on_table = cam.on_plane(u, v, 0.0)
    assert on_table[1] > middle[1] + 0.01  # further from a camera that looks along +y
    assert cam.on_plane(u, v, 0.6) is None  # a plane above the camera is never met looking down


# ---- real renders -----------------------------------------------------------------------------------------

PALETTE = ("red", "orange", "yellow", "green", "teal", "blue", "purple", "pink")


def _palette_world() -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, np.ndarray]]:
    spots = [(0.15 + 0.1 * (k % 3), -0.12 + 0.12 * (k // 3)) for k in range(len(PALETTE))]
    objects = [
        {"name": f"{c}_cube", "shape": "box", "size": [0.04] * 3, "pos": list(spot), "color": c}
        for c, spot in zip(PALETTE, spots, strict=True)
    ]
    model = compile_world(World.model_validate({"name": "palette", "objects": objects})).model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, {c: data.body(f"{c}_cube").xpos.copy() for c in PALETTE}


@pytest.mark.parametrize("view", ["top", "default", "front", "overview"])
def test_every_named_color_finds_its_own_cube_and_no_other(needs_gl, view):
    model, data, truth = _palette_world()
    renderer = SnapshotRenderer()
    sun = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    model.light_diffuse[sun] *= vision.EXPOSURE  # what render.layers(exposure=...) does for observe(find=...)
    model.vis.headlight.ambient[:] *= vision.EXPOSURE
    model.vis.headlight.diffuse[:] *= vision.EXPOSURE
    layers = renderer.layers_of(model, data, view, 640, 480, 0)
    renderer.close()
    for color in PALETTE:
        found = vision.find(layers.rgb, layers.depth, layers.camera, color)
        assert found, f"{color} not found from {view}"
        assert np.linalg.norm(found[0].position - truth[color]) < 0.02, (color, view)
        others = [f for f in found if np.linalg.norm(f.position - truth[color]) > 0.04]
        assert not others, f"{color} from {view} also found at {[f.position.round(3) for f in others]}"


def test_one_renderer_follows_a_change_of_model(needs_gl):
    """layers_of is given models that are not a session's: a renderer made for one must not draw another."""
    renderer = SnapshotRenderer()
    first = _palette_world()[0]  # dozens of geoms
    second = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="0.1"/></worldbody></mujoco>')
    for model in (first, second):  # the same version number and size for both
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        layers = renderer.layers_of(model, data, "default", 64, 48, 0, geoms=True)
        assert 0 <= layers.geoms.max() < model.ngeom
    renderer.close()


def test_exposure_is_for_one_picture_only(needs_gl):
    from robotforge.sim.session import SimSession

    session = SimSession.from_xml(compile_world(World.model_validate({"name": "w"})).spec.to_xml())
    before = (session.model.light_diffuse.copy(), session.model.vis.headlight.diffuse.copy())
    renderer = SnapshotRenderer()
    dim = renderer.layers(session, "top", 64, 48, exposure=0.5).rgb.mean()
    full = renderer.layers(session, "top", 64, 48).rgb.mean()
    renderer.close()
    assert dim < full * 0.8
    assert np.array_equal(session.model.light_diffuse, before[0])
    assert np.array_equal(session.model.vis.headlight.diffuse, before[1])


@pytest.mark.parametrize("template", ["arm", "gripper", "rover", "quadruped", "humanoid"])
def test_a_template_robots_camera_sees_out_past_its_own_lens(needs_gl, template):
    """The arm's wrist camera sat behind its lens and saw the bezel as a black ring round every picture (15 % of
    it); what a robot's camera may see of itself is the tips of its fingers (5 %)."""
    world = World.model_validate({"name": "w", "robots": [{"name": "bot", "template": template}]})
    model = compile_world(world).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    renderer = SnapshotRenderer()
    for cam in range(model.ncam):
        layers = renderer.layers_of(model, data, model.camera(cam).name, 320, 240, 0, geoms=True)
        shown = layers.geoms[layers.geoms >= 0]
        own = sum((model.body(int(model.geom_bodyid[g])).name or "").startswith("bot/") for g in shown.tolist())
        assert own / layers.geoms.size < 0.08, model.camera(cam).name
    renderer.close()
