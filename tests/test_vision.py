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
    hue, sat = np.meshgrid(np.arange(0, 360, 0.5), np.arange(0, 1.001, 0.01))
    hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)  # every color at full brightness
    inside = sum(vision.threshold(hsv, r).astype(int) for r in vision.NAMED.values())
    assert inside.max() == 1  # orange and brown share hues, not saturations
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


# ---- locating from depth, and the doubts about a find --------------------------------------------------------


def _looking_down() -> vision.Pinhole:
    """A camera 1 m above the floor looking straight down, 100 px across, a pixel 1 cm wide on the floor (and
    0.9 cm on a 10 cm box's top)."""
    down, right, up = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    return vision.Pinhole(np.zeros(3) + [0, 0, 1.0], right, up, down, 100.0, 100.0, 50.0, 50.0, 100, 100)


def _box_seen_from_above(top: float = 0.1) -> tuple[np.ndarray, vision.Blob]:
    """Depth of a 10 cm box (its top `top` m high) standing on the floor, pixels 60..69 across, 45..54 down."""
    depth = np.ones((100, 100))
    depth[45:55, 60:70] = 1.0 - top
    mask = np.zeros((100, 100), bool)
    mask[45:55, 60:70] = True
    return depth, vision.blobs(mask)[0]


def test_edge_pixels_of_something_in_front_or_behind_do_not_move_a_find():
    """A hand over a box's edge lent the box's blob a few pixels 10 cm above its top (observe(find=) put the box
    7.7 cm off); a bench far behind a ball lent it pixels at the ball's height (77 cm off)."""
    cam = _looking_down()
    depth, blob = _box_seen_from_above()
    clean, _ = vision.locate(cam, depth, blob)
    depth[45:55, 60] = 0.5  # the left column blends in a hand half a metre above the floor
    depth[45, 61:70] = 2.0  # the top row blends in something a metre further away
    assert len(vision.own_points(cam, depth, blob)) == 100 - 10 - 9
    position, _ = vision.locate(cam, depth, blob)
    assert position == pytest.approx(clean, abs=0.005)  # half a pixel: the box's left column is gone (2 cm before)
    assert clean == pytest.approx([0.135, 0.0, 0.05], abs=1e-6)  # pixel 65's ray 0.9 m down; half way up


def test_an_object_stands_on_the_flat_surface_seen_most_around_it():
    """A cereal box on a kitchen island sat 15 cm low: around it the island's side ran down to the floor, and the
    median height of everything round it was 0.6 m though the island is 0.9 m high."""
    cam = _looking_down()
    depth, blob = _box_seen_from_above(top=0.1)
    ring = np.zeros((100, 100), bool)
    ring[41:59, 56:74] = True
    ring[45:55, 60:70] = False
    ys, xs = np.nonzero(ring)
    slope = np.linspace(0.0, 0.04, len(ys))  # most of the ring: a slope from the floor up to 4 cm
    depth[ys, xs] = 1.0 - slope
    depth[ys[::3], xs[::3]] = 1.0 - 0.05  # a third of it: a flat shelf 5 cm high the box stands on
    position, size = vision.locate(cam, depth, blob)
    assert position[2] == pytest.approx(0.075, abs=0.002) and size[2] == pytest.approx(0.05, abs=0.003)


def test_a_find_says_when_it_is_cut_off_or_hidden():
    depth, blob = _box_seen_from_above()
    assert not vision.cut_off(blob, depth.shape) and not vision.hidden(depth, blob)
    behind = depth.copy()
    behind[40:60, 55:60] = 3.0  # something further away next to it hides nothing
    assert not vision.hidden(behind, blob)
    covered = depth.copy()
    covered[44:56, 55:61] = 0.5  # an arm in front covers its left edge
    assert vision.hidden(covered, blob)
    mask = np.zeros((100, 100), bool)
    mask[0:10, 60:70] = True  # runs into the picture's top edge
    assert vision.cut_off(vision.blobs(mask)[0], depth.shape)


def test_a_box_under_a_shelf_seen_from_the_side_is_side_on(needs_gl):
    """With its top hidden the camera sees one face, whose middle is half the box's depth in front of the box's."""
    seen = {}
    for shelf in (False, True):
        objects = [{"name": "cube", "shape": "box", "size": [0.06] * 3, "pos": [0.3, 0.0], "color": "red"}]
        if shelf:  # a grey board 1.5 cm over the cube's top, wider than it
            board = {"size": [0.3, 0.3, 0.01], "pos": [0.3, 0.0, 0.075], "color": "gray", "fixed": True}
            objects.append({"name": "shelf", "shape": "box", **board})
        model = compile_world(World.model_validate({"name": "w", "objects": objects})).model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = SnapshotRenderer()
        layers = renderer.layers_of(model, data, "front", 640, 480, 0)
        renderer.close()
        seen[shelf] = vision.find(layers.rgb, layers.depth, layers.camera, "red")[0]
    assert not seen[False].side_on and seen[True].side_on
    truth = np.array([0.3, 0.0, 0.03])
    assert np.linalg.norm(seen[False].position - truth) < 0.01
    assert seen[True].position[0] == pytest.approx(0.3 + 0.03, abs=0.01)  # the face toward the camera (+x)


# ---- real renders -----------------------------------------------------------------------------------------

PALETTE = ("red", "orange", "brown", "yellow", "green", "teal", "blue", "purple", "pink")


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


@pytest.mark.parametrize("view", ["top", "default", "front", "overview", "left"])
def test_every_named_color_finds_its_own_cube_and_no_other(needs_gl, view):
    """Through the finder's own camera (its exposure scales every light, the shine on tops too)."""
    from robotforge.sim.session import SimSession

    model, data, truth = _palette_world()
    renderer = SnapshotRenderer()
    layers = renderer.layers(
        SimSession(model=model, data=data, source="palette"), view, 640, 480, exposure=vision.EXPOSURE
    )
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


def test_depth_leaves_no_mujoco_log_behind(needs_gl, tmp_path, monkeypatch):
    """On a Mac every new renderer's first depth picture made MuJoCo append a warning to MUJOCO_LOG.TXT in the
    folder robotforge was started from."""
    monkeypatch.chdir(tmp_path)
    model, data, _ = _palette_world()
    renderer = SnapshotRenderer()
    renderer.layers_of(model, data, "top", 64, 48, 0)
    renderer.close()
    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()
    assert mujoco.get_mju_user_warning() is None  # MuJoCo's own handling is back for everything else


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


def test_a_camera_picture_shows_no_marker_sites(needs_gl):
    """The arm template's tool-centre marker (an orange site between the fingers) filled the top of the wrist
    camera's picture, 1808 px of it, and observe(find="orange") reported it as a thing."""
    world = World.model_validate({"name": "w", "robots": [{"name": "arm", "template": "arm"}]})
    model = compile_world(world).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    renderer = SnapshotRenderer()
    layers = renderer.layers_of(model, data, "arm/wrist_cam", 640, 480, 0, geoms=True)
    renderer.close()
    drawn = layers.depth < layers.camera.far * vision.SKY
    assert not (drawn & (layers.geoms < 0)).any()  # every pixel drawn is a geom's
    assert not vision.find(layers.rgb, layers.depth, layers.camera, "orange")
