"""Planar faces and tangent joins stay flat regardless of bevel triangulation."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.viewer.scene import build_mesh_payload

SHAPES = [
    (0.6, 0.4, 0.0125, 0.008, 3, 0.03),  # table
    (0.8, 0.045, 0.015, 0.006, 2, 0.008),  # bench slat
    (0.13, 0.07, 0.03, 0.01, 3, None),  # robot housing
    (0.12, 0.08, 0.035, 0.02, 4, 0.004),  # small corner
    (0.1, 0.08, 0.01, 0.01, 3, None),  # coincident middle rings
    (0.1, 0.08, 0.01, 0.005, 1, None),  # coarse bevel
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rounded_box_flat_faces_and_bevel_tangents_have_axial_normals(shape, axis):
    mesh = M.rounded_box(*shape)
    on_face = np.isclose(np.abs(mesh.vertices[:, axis]), shape[axis], rtol=0, atol=1e-12)
    assert on_face.any()
    expected = np.zeros((on_face.sum(), 3))
    expected[:, axis] = np.sign(mesh.vertices[on_face, axis])
    np.testing.assert_allclose(mesh.normals[on_face], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "shape",
    [*SHAPES, (0.01, 0.01, 0.01, 0.01, 4, None), (2e-5, 3e-5, 1e-5, 1e-5, 16, 1e-5)],
)
def test_rounded_box_normals_are_finite_unit_and_consistent_per_smoothing_key(shape):
    mesh = M.rounded_box(*shape)
    assert np.isfinite(mesh.normals).all()
    np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1, rtol=0, atol=1e-12)
    for key in np.unique(mesh.keys):
        values = mesh.normals[mesh.keys == key]
        np.testing.assert_allclose(values, np.broadcast_to(values[0], values.shape), rtol=0, atol=1e-12)
    assert not mesh.normals.flags.writeable


@pytest.mark.parametrize("corner", [0.004, 0.03])
def test_rounded_box_bevel_normals_are_orthogonal_to_surface_tangents(corner):
    hx, hy, hz, radius, segments = 0.12, 0.08, 0.035, 0.02, 8
    mesh = M.rounded_box(hx, hy, hz, radius, segments, corner)
    per_ring = 4 * (segments + 1)

    def point(a, phi):
        inset = radius * (1 - math.sin(a))
        r = max(corner - inset, 1e-5)
        return np.array(
            [
                max(hx - inset - r, 0) + r * math.cos(phi),
                max(hy - inset - r, 0) + r * math.sin(phi),
                -hz + radius - radius * math.cos(a),
            ]
        )

    for ring in range(1, segments):
        a = ring * math.pi / (2 * segments)
        for vertex in range(1, segments):
            phi = vertex * math.pi / (2 * segments)
            normal = mesh.normals[ring * per_ring + vertex]
            # The clamped corner is only 10 micrometers wide. A 1e-7 radian
            # step subtracts nearly equal positions and measures cancellation.
            for da, dp in [(1e-5, 0), (0, 1e-4)]:
                tangent = point(a + da, phi + dp) - point(a - da, phi - dp)
                tangent /= np.linalg.norm(tangent)
                assert abs(normal @ tangent) < 1e-7


def test_authored_rounded_box_normals_reach_the_compiled_viewer_mesh():
    spec = mujoco.MjSpec()
    mesh = M.rounded_box(0.6, 0.4, 0.0125, 0.008, 3, 0.03)
    visuals = Visuals(spec)
    visuals.add(spec.worldbody, "wood", [0.5, 0.4, 0.3, 1], mesh)
    visuals.finish()
    model = spec.compile()
    # MuJoCo stores the mesh in a principal-axis frame; geom rotation restores it.
    geom = next(g for g in range(model.ngeom) if model.geom_dataid[g] == 0)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    rotation = rotation.reshape(3, 3)
    payload = build_mesh_payload(model)
    entry = payload.entries[0]
    data = np.frombuffer(payload.blob, dtype=np.float32)
    offset, count = entry["offset"], entry["count"] * 3
    positions = data[offset : offset + count].reshape(-1, 3) @ rotation.T + model.geom_pos[geom]
    expanded_normals = data[offset + count : offset + 2 * count].reshape(-1, 3)
    np.testing.assert_array_equal(expanded_normals, model.mesh_normal[model.mesh_facenormal].reshape(-1, 3))
    normals = expanded_normals @ rotation.T
    top = np.isclose(positions[:, 2], 0.0125, rtol=0, atol=1e-7)
    assert top.sum() > 20
    np.testing.assert_allclose(normals[top], np.tile([0, 0, 1], (top.sum(), 1)), rtol=0, atol=1e-6)


@pytest.mark.parametrize("scale", [(1, 1, 1), (2, 0.5, 3), (-2, 0.5, 3)])
def test_chamfered_box_keeps_flat_normals_after_rotation_scale_and_mirroring(scale):
    mesh = M.box(0.1, 0.08, 0.02, chamfer=0.005)
    rotation = M.rot_x(0.7) @ M.rot_z(0.3)
    moved = mesh.moved((0.4, -0.2, 0.6), rotation, scale)
    top = np.isclose(mesh.vertices[:, 2], 0.02, rtol=0, atol=1e-12)
    expected = np.array([0, 0, 1]) @ rotation.T
    np.testing.assert_allclose(moved.normals[top], np.tile(expected, (top.sum(), 1)), rtol=0, atol=1e-12)
    face_normals = np.cross(
        moved.vertices[moved.faces[:, 1]] - moved.vertices[moved.faces[:, 0]],
        moved.vertices[moved.faces[:, 2]] - moved.vertices[moved.faces[:, 0]],
    )
    assert np.all(np.einsum("ij,ij->i", face_normals, moved.normals[moved.faces[:, 0]]) > 0)
