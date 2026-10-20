"""Choosing a grasp from a shape: the hull, and what the planner makes of boxes, balls and things made of several."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import grasping as G

OPENING = 0.08  # a Panda's jaws


def _box(hx, hy, hz, centre=(0.0, 0.0, 0.0), yaw=0.0):
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return corners @ rot.T + np.array(centre)


def _ball(radius, centre):
    return G._sphere_points() * radius + np.array(centre)


def _across(grasp: G.Grasp) -> float:
    return math.degrees(math.atan2(grasp.closing[1], grasp.closing[0])) % 180.0


def test_the_hull_of_a_box_has_its_volume_and_centre():
    verts, faces = G.convex_hull(_box(0.02, 0.03, 0.04, (0.1, -0.2, 0.04)))
    volume, centre = G.hull_volume_centroid(verts, faces)
    assert len(verts) == 8 and len(faces) == 12
    assert volume == pytest.approx(0.04 * 0.06 * 0.08, rel=1e-9)
    assert centre == pytest.approx([0.1, -0.2, 0.04], abs=1e-9)
    normals, areas = G.face_normals(verts, faces)
    assert areas.sum() == pytest.approx(2 * (0.04 * 0.06 + 0.04 * 0.08 + 0.06 * 0.08))
    inside = verts.mean(axis=0)
    assert all(n @ (verts[f[0]] - inside) > 0 for n, f in zip(normals, faces, strict=True))  # outward


def test_a_hull_of_a_ball_is_close_to_the_ball():
    verts, faces = G.convex_hull(_ball(0.03, (0, 0, 0.03)))
    volume, centre = G.hull_volume_centroid(verts, faces)
    assert volume == pytest.approx(4 / 3 * math.pi * 0.03**3, rel=0.05) and centre[2] == pytest.approx(0.03, abs=1e-6)


def test_points_in_a_plane_or_a_line_have_no_hull():
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]])
    with pytest.raises(ValueError, match="plane"):
        G.convex_hull(flat)
    with pytest.raises(ValueError, match="line"):
        G.convex_hull(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0.0]]))


def test_a_box_is_taken_across_its_narrow_side_squarely():
    best = G.plan(_box(0.02, 0.03, 0.04, (0, 0, 0.04)), OPENING)[0]
    assert _across(best) == pytest.approx(0.0, abs=1.0) and best.width == pytest.approx(0.04, abs=1e-4)
    assert best.grip > 0.99 and best.offset < 0.002
    assert best.center[2] == pytest.approx(0.036, abs=0.02) and best.below == pytest.approx(best.center[2])
    turned = G.plan(_box(0.02, 0.03, 0.04, (0.1, 0.05, 0.04), math.radians(30)), OPENING)[0]
    assert _across(turned) == pytest.approx(30.0, abs=1.0)  # the jaws turn with the box
    assert turned.center[:2] == pytest.approx([0.1, 0.05], abs=1e-3)


def test_a_thing_wider_than_the_jaws_cannot_be_taken_and_a_flat_one_cannot_either():
    assert G.plan(_box(0.05, 0.05, 0.03, (0, 0, 0.03)), OPENING) == []
    assert G.plan(_box(0.03, 0.03, 0.004, (0, 0, 0.004)), OPENING) == []  # 8 mm: fingers cannot get under it


def test_a_ball_is_taken_through_its_middle_whatever_way_the_jaws_close():
    best = G.plan(_ball(0.03, (0.2, 0.1, 0.03)), OPENING)[0]
    assert best.center[:2] == pytest.approx([0.2, 0.1], abs=2e-3) and best.width == pytest.approx(0.06, abs=2e-3)
    assert best.center[2] == pytest.approx(0.03, abs=0.006)


def test_a_thing_that_gets_wider_above_the_grasp_is_taken_where_the_fingers_can_come_down():
    """A post with a plate on it. The jaws open 8 cm. A 10 x 10 cm plate leaves no way to bring the fingers down beside
    the post, so the only grasps are on the plate (not at all across its 10 cm: it is across the plate or none); a
    10 x 6 cm plate lets the jaws straddle its 6 cm, so the post can be taken with the jaws closing along that side."""
    post = _box(0.015, 0.015, 0.03, (0, 0, 0.03))
    square = G.plan([post, _box(0.05, 0.05, 0.01, (0, 0, 0.07))], OPENING)
    assert square == []
    plate = G.plan([post, _box(0.05, 0.03, 0.01, (0, 0, 0.07))], OPENING)
    low = [g for g in plate if g.center[2] < 0.06]
    assert low and all(_across(g) == pytest.approx(90.0, abs=1.0) for g in low)  # closing along the plate's short side
    assert all(g.width <= OPENING - 2 * G.SPARE for g in plate)


def test_the_planner_keeps_a_grasp_near_the_centre_of_mass():
    """An L: a 4 x 4 x 4 cm block with a long bar out the side. Taking the block squeezes far from the mass when the
    mass is out along the bar, so the bar's root wins."""
    block = _box(0.02, 0.02, 0.02, (0, 0, 0.02))
    bar = _box(0.04, 0.0125, 0.0175, (0.06, 0, 0.0175))
    points = [block, bar]
    grasps = G.plan(points, OPENING, centre_of_mass=np.array([0.04, 0.0, 0.02]))
    assert grasps and grasps[0].offset < 0.02
    far = G.plan(points, OPENING, centre_of_mass=np.array([0.2, 0.2, 0.2]))
    assert far[0].score < grasps[0].score


def test_points_come_from_every_shape_a_body_is_made_of():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="thing", pos=[0, 0, 0.05])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.1)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.01, 0.03, 0], pos=[0, 0, 0.05], mass=0.1)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], pos=[0.05, 0, 0], mass=0.1, contype=0, conaffinity=0
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    parts = G.body_parts(model, data, 1)
    assert len(parts) == 2  # the box and the can; the ball collides with nothing
    points = np.vstack(parts)
    assert points[:, 2].min() == pytest.approx(0.03, abs=1e-6) and points[:, 2].max() == pytest.approx(0.13, abs=1e-6)
    assert points[:, 0].max() == pytest.approx(0.02, abs=1e-6)  # the ball that collides with nothing is not part of it


def test_a_mesh_is_taken_by_the_hull_of_its_vertices():
    verts = np.array([[0.03, 0, 0], [-0.03, 0, 0], [0, 0.015, 0], [0, -0.015, 0], [0, 0, 0.025], [0, 0, -0.025]])
    spec = mujoco.MjSpec()
    spec.add_mesh(name="gem", uservert=verts.flatten().tolist())
    body = spec.worldbody.add_body(name="gem", pos=[0, 0, 0.05])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="gem", mass=0.05)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    parts = G.body_parts(model, data, 1)
    assert len(parts) == 1 and parts[0].shape == (6, 3)
    grasps = G.plan(parts, OPENING, data.xipos[1].copy())
    assert grasps and grasps[0].width < 0.03 and grasps[0].center[:2] == pytest.approx([0.0, 0.0], abs=2e-3)
    assert grasps[0].grip < 0.95  # an octahedron's faces are slanted to any squeeze


def _dense_ellipsoid(count, radii=(0.03, 0.02, 0.025), centre=(0.0, 0.0, 0.03)):
    return G._sphere_points(count) * np.array(radii) + np.array(centre)


def test_a_dense_mesh_is_cut_down_to_its_extremes_and_planned_in_bounded_time():
    # A scanned mesh has thousands of vertices; the hull used to be cubic in them (minutes at 2500 points).
    import time

    points = _dense_ellipsoid(2500)
    assert len(G.extremes(points)) <= 2 * G.HULL_DIRECTIONS + 6
    started = time.monotonic()
    grasps = G.plan([points], OPENING, np.array([0.0, 0.0, 0.03]))
    assert time.monotonic() - started < 60.0  # generous: the point is minutes, not seconds
    assert grasps and grasps[0].width == pytest.approx(0.04, abs=0.004)  # across the 2 cm half axis, to a few mm


def test_cutting_a_mesh_down_keeps_its_hull_to_a_millimetre_or_two():
    points = _dense_ellipsoid(2500)
    full_verts, full_faces = G.convex_hull(points[::5])  # 500 points: affordable to hull whole now
    vol_full, _ = G.hull_volume_centroid(full_verts, full_faces)
    cut_verts, cut_faces = G.convex_hull(G.extremes(points))
    vol_cut, _ = G.hull_volume_centroid(cut_verts, cut_faces)
    assert cut_verts[:, 0].max() == pytest.approx(0.03, abs=0.0015)
    assert vol_cut == pytest.approx(4 / 3 * math.pi * 0.03 * 0.02 * 0.025, rel=0.04)
    assert vol_full > 0


def test_a_repeated_plan_is_remembered():
    parts = [_box(0.02, 0.015, 0.03, centre=(0.0, 0.0, 0.03))]
    first = G.plan_cached(parts, OPENING, np.array([0.0, 0.0, 0.03]))
    again = G.plan_cached([p.copy() for p in parts], OPENING, np.array([0.0, 0.0, 0.03]))
    assert again is first
    moved = G.plan_cached([parts[0] + [0.1, 0.0, 0.0]], OPENING, np.array([0.1, 0.0, 0.03]))
    assert moved is not first


@pytest.mark.parametrize("shape", ["cloud", "grid", "cylinder"])
def test_the_hull_is_a_closed_surface_with_every_point_inside(shape):
    rng = np.random.default_rng(3)
    if shape == "cloud":
        points = rng.normal(size=(300, 3))
    elif shape == "grid":  # many coplanar and collinear points: the awkward case for a hull
        points = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(3)], float)
    else:
        ring = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        disc = np.column_stack([np.cos(ring), np.sin(ring), np.zeros(40)])
        points = np.vstack([disc, disc + [0, 0, 2]])
    verts, faces = G.convex_hull(points)
    edges = {}
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            edges[e] = edges.get(e, 0) + 1
    assert all(n == 1 and (e[1], e[0]) in edges for e, n in edges.items())  # each edge once, its twin the other way
    assert len(verts) - len(edges) // 2 + len(faces) == 2  # Euler: a sphere
    normals, _ = G.face_normals(verts, faces)
    base = verts[faces[:, 0]]
    assert ((points @ normals.T - np.einsum("ij,ij->i", normals, base)) <= 1e-7).all()
