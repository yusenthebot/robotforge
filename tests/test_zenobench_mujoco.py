"""ZenoBench on MuJoCo: USD physics conversion, PhysX-style drives, the base anchor and the Isaac stand-ins.

Small stages are built in memory with usd-core, so these tests need the `zenobench` extra but no
ZenoBench checkout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
pytest.importorskip("scipy")

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from robotforge.zenobench import hulls, usd_mjcf  # noqa: E402
from robotforge.zenobench.world import World  # noqa: E402


def _cube(stage, path, pos, size, body=False, mass=None, purpose="default"):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    c.CreatePurposeAttr(purpose)
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if body:
        UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    if mass is not None:
        UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(mass)
    return c.GetPrim()


def _scene():
    """Ground, a free box, a cabinet door on a driven hinge, and a base held by a fixed anchor."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    _cube(stage, "/World/ground", (0, 0, -0.05), (10, 10, 0.1))
    _cube(stage, "/World/box", (1.0, 0, 0.1), (0.2, 0.2, 0.2), body=True, mass=2.0)
    # cabinet: carcass welded to the world, door hinged at x = 2 (z axis), drive in degrees
    carcass = _cube(stage, "/World/cab/carcass", (2.3, 1.0, 0.5), (0.6, 0.6, 1.0), body=True, mass=10.0)
    door = _cube(stage, "/World/cab/door", (2.3, 0.68, 0.5), (0.6, 0.02, 0.9), body=True, mass=1.0)
    root = UsdPhysics.FixedJoint.Define(stage, "/World/cab/root_joint")
    root.CreateBody1Rel().SetTargets([carcass.GetPath()])
    root.CreateLocalPos0Attr(Gf.Vec3f(2.3, 1.0, 0.5))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", pxr.Sdf.ValueTypeNames.Bool).Set(False)
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/cab/joints/hinge")
    hinge.CreateBody0Rel().SetTargets([carcass.GetPath()])
    hinge.CreateBody1Rel().SetTargets([door.GetPath()])
    # carcass frame is scaled (0.6, 0.6, 1.0): local -0.5 x -> world x 2.0
    hinge.CreateLocalPos0Attr(Gf.Vec3f(-0.5, -0.5333, 0.0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(-0.5, 0.0, 0.0))
    hinge.CreateAxisAttr("Z")
    hinge.CreateLowerLimitAttr(-90.0)
    hinge.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("acceleration")
    drive.CreateStiffnessAttr(10.0)  # per degree
    drive.CreateDampingAttr(1.0)
    drive.CreateTargetPositionAttr(0.0)
    # base held by an anchor that the test moves through the stage
    base = _cube(stage, "/World/robot/base", (-1.0, 0.0, 0.3), (0.4, 0.4, 0.2), body=True, mass=20.0)
    anchor = UsdPhysics.FixedJoint.Define(stage, "/World/robot/base_anchor")
    anchor.CreateBody1Rel().SetTargets([base.GetPath()])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(-1.0, 0.0, 0.3))
    anchor.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    return stage


OPTIONS = usd_mjcf.Options(planar_anchors=("/World/robot/base_anchor",), visuals=False)


def test_conversion_keeps_joint_frames_limits_and_drive_units():
    conv = usd_mjcf.convert(_scene(), OPTIONS)
    m = conv.spec.compile()
    j = m.joint("/World/cab/joints/hinge")
    assert j.type[0] == 3  # hinge
    assert np.allclose(m.jnt_range[j.id], np.radians([-90, 0]))
    info = conv.joints["/World/cab/joints/hinge"]
    assert info.drive.acceleration
    assert info.drive.stiffness == pytest.approx(10.0 * 180 / math.pi)
    d = __import__("mujoco").MjData(m)
    __import__("mujoco").mj_forward(m, d)
    # the joint anchor sits on the door's hinge edge in the world
    assert np.allclose(d.xanchor[j.id], (2.0, 0.68, 0.5), atol=2e-3)
    assert np.allclose(d.xaxis[j.id], (0, 0, 1), atol=1e-6)
    # the welded carcass has no joint and stays where USD put it
    assert np.allclose(d.xpos[m.body("/World/cab/carcass").id], (2.3, 1.0, 0.5))
    # the free box got a free joint, the base three planar ones
    assert m.body_jntnum[m.body("/World/box").id] == 1
    assert m.body_jntnum[m.body("/World/robot/base").id] == 3
    # the articulation switched self collision off: carcass and door never touch
    assert conv.stats["excludes"] >= 1


def test_mass_only_bodies_get_their_mass():
    m = usd_mjcf.convert(_scene(), OPTIONS).spec.compile()
    assert m.body_mass[m.body("/World/box").id] == pytest.approx(2.0, rel=1e-3)


def test_drive_tracks_target_and_anchor_drags_base():
    stage = _scene()
    w = World(stage, dt=1 / 120, substeps=4, options=OPTIONS)
    w.build()
    w.set_target("/World/cab/joints/hinge", math.radians(-60))
    w.step(240)
    assert w.joint_q("/World/cab/joints/hinge") == pytest.approx(math.radians(-60), abs=0.02)
    anchor = stage.GetPrimAtPath("/World/robot/base_anchor")
    for i in range(1, 121):  # 0.5 m in 1 s while turning 90 degrees, the way ZenoBench drives
        u = i / 120
        anchor.GetAttribute("physics:localPos0").Set(Gf.Vec3f(-1.0 + 0.5 * u, 0.0, 0.3))
        yaw = math.pi / 2 * u
        anchor.GetAttribute("physics:localRot0").Set(Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        w.step(1)
    w.step(30)
    p, q = w.body_pose("/World/robot/base")
    assert np.allclose(p, (-0.5, 0.0, 0.3), atol=3e-3)
    assert 2 * math.atan2(q[3], q[0]) == pytest.approx(math.pi / 2, abs=0.01)
    box, _ = w.body_pose("/World/box")
    assert box[2] == pytest.approx(0.1, abs=5e-3)  # resting on the ground, not sinking


def test_sleeping_world_wakes_for_new_targets():
    stage = _scene()
    w = World(stage, options=OPTIONS)
    w.build()
    w.step(1200)  # ten idle seconds: every island falls asleep
    assert int(w.data.body_awake.sum()) == 0
    w.set_target("/World/cab/joints/hinge", math.radians(-45))
    w.step(240)
    assert w.joint_q("/World/cab/joints/hinge") == pytest.approx(math.radians(-45), abs=0.02)
    stage.GetPrimAtPath("/World/robot/base_anchor").GetAttribute("physics:localPos0").Set(Gf.Vec3f(-0.5, 0.0, 0.3))
    w.step(240)
    assert np.allclose(w.body_pose("/World/robot/base")[0], (-0.5, 0.0, 0.3), atol=3e-3)


def test_prisms_reproduce_a_concave_room_exactly():
    # an L-shaped floor with one wall: concave, so a hull would fill the corner
    pts = np.array([[0, 0, 0], [4, 0, 0], [4, 1, 0], [1, 1, 0], [1, 4, 0], [0, 4, 0], [0, 0, 2], [4, 0, 2]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 6, 7], [0, 7, 1]])
    pieces = hulls.prisms(pts, faces, depth=0.03)
    assert all(len(p) >= 6 for p in pieces)
    # nothing above the floor in the empty corner (x, y in (1, 4)), so a box there would not collide
    corner = np.array([2.5, 2.5, 0.01])
    for p in pieces:
        h = hulls.hull(p)
        if h is None:
            continue
        inside = np.all(h.equations[:, :3] @ corner + h.equations[:, 3] <= 1e-9)
        assert not inside


def test_wall_with_a_doorway_merges_into_three_slabs():
    v = np.array([[0, 0], [4, 0], [4, 3], [0, 3], [1.5, 0], [2.5, 0], [2.5, 2], [1.5, 2]], float)
    faces = np.array([[0, 4, 7], [0, 7, 3], [7, 6, 3], [6, 2, 3], [6, 5, 2], [5, 1, 2]])
    polys = hulls.convex_polygons(faces, v[:, 0], v[:, 1])
    assert len(polys) == 3
    covered = {i for p in polys for i in p}
    assert covered == set(range(8))


def test_open_room_is_not_convex():
    # floor and two walls: every vertex lies on the hull, but the hull would fill the room
    pts = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0], [0, 0, 2], [3, 0, 2], [0, 3, 2]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4], [0, 3, 6], [0, 6, 4]])
    assert hulls.inside_depth(pts, hulls.hull(pts)) == pytest.approx(0.0, abs=1e-9)
    assert not hulls.is_convex(pts, faces, hulls.hull(pts))
    box = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    h = hulls.hull(box)
    assert hulls.is_convex(box, h.simplices, h)


def test_split_parts_and_convexity():
    box = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    h = hulls.hull(box)
    assert hulls.inside_depth(box, h) == pytest.approx(0.0, abs=1e-9)
    faces = np.array([[0, 1, 2], [4, 5, 6]])
    assert len(hulls.split_parts(box, faces)) == 2


@pytest.fixture
def isolated_shim(monkeypatch):
    """Install the stand-ins for one test, then put back every module and patch they replaced, so tests
    that run later in the same process see the real torch (or none) and unpatched pxr and NumPy."""
    import sys

    import pxr

    from robotforge.zenobench import shim

    modules = dict(sys.modules)
    monkeypatch.setattr(np, "cross", np.cross)
    monkeypatch.setattr(Usd.Prim, "RemoveAPI", Usd.Prim.RemoveAPI)
    monkeypatch.setattr(Usd.Prim, "HasAPI", Usd.Prim.HasAPI)
    had_physx = hasattr(pxr, "PhysxSchema")
    old_physx = getattr(pxr, "PhysxSchema", None)
    for field in ("stage", "world", "options", "substeps"):
        monkeypatch.setattr(shim.STATE, field, getattr(shim.STATE, field))
    yield shim
    for name in [n for n in sys.modules if n not in modules]:
        del sys.modules[name]
    sys.modules.update({n: m for n, m in modules.items() if sys.modules.get(n) is not m})
    if had_physx:
        pxr.PhysxSchema = old_physx
    elif hasattr(pxr, "PhysxSchema"):
        del pxr.PhysxSchema


def test_shim_leaves_no_trace_after_its_test(isolated_shim):
    import sys

    isolated_shim.install(OPTIONS)
    assert "isaacsim.core.prims" in sys.modules


def test_stand_ins_are_gone_after_an_isolated_test():
    import sys

    assert "isaacsim.core.prims" not in sys.modules
    assert not getattr(np.cross, "__robotforge_compat__", False)
    from robotforge.zenobench import shim

    assert getattr(sys.modules.get("torch"), "Tensor", None) is not shim.Tensor


def test_numpy_cross_accepts_2d_vectors_like_numpy_1(monkeypatch):
    from robotforge.zenobench import shim

    monkeypatch.setattr(np, "cross", np.cross)  # restored after the test
    shim._numpy_compat()
    assert np.cross([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    tris = np.array([[[0, 0], [2, 0], [0, 3]]], float)
    assert np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]) == pytest.approx([6.0])
    assert np.allclose(np.cross([0.0, 0.0, 1.0], [1.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(np.cross([1.0, 0, 0], [0, 1.0, 0]), [0, 0, 1.0])


def test_shim_runs_isaac_style_calls(isolated_shim):
    shim = isolated_shim
    stage = _scene()
    shim.install(OPTIONS)
    shim.STATE.stage = stage
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import PhysxSchema

    box = stage.GetPrimAtPath("/World/box")
    PhysxSchema.PhysxRigidBodyAPI.Apply(box).CreateEnableCCDAttr().Set(True)
    assert box.GetAttribute("physxRigidBody:enableCCD").Get() is True
    assert box.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
    box.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    assert not box.HasAPI(PhysxSchema.PhysxRigidBodyAPI)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    cam = Camera(CameraCfg(prim_path="/World/Cam", height=48, width=64, spawn=sim_utils.PinholeCameraCfg(13.0)))
    cab = SingleArticulation("/World/cab")
    sim.reset()
    cab.initialize()
    assert cab.dof_names == ["hinge"]
    cab.apply_action(ArticulationAction(joint_positions=np.array([-1.0]), joint_indices=np.array([0])))
    for _ in range(240):
        sim.step()
    assert float(cab.get_joint_positions()[0]) == pytest.approx(-1.0, abs=0.02)
    p, q = SingleRigidPrim("/World/box").get_world_pose()
    assert p.cpu().numpy().shape == (3,) and q.cpu().numpy().shape == (4,)
    cam.set_world_poses_from_view(np.array([[0.0, -3.0, 2.0]]), np.array([[0.0, 0.0, 0.0]]))
    try:
        cam.update(0.0)
    except Exception as exc:  # no OpenGL (headless CI): rendering is optional here
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    assert cam.data.output["rgb"][0, ..., :3].cpu().numpy().shape == (48, 64, 3)
