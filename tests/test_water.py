"""Water: buoyancy per shape against Archimedes, sinking and floating, drag, waves, thrusters, worlds with water."""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np
import pytest

from robotforge.design.water import WaterEntry
from robotforge.design.world import World, compile_world
from robotforge.sim import water
from robotforge.sim.session import SimSession

G = mujoco.mjtGeom
LYING = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # a cylinder's axis turned from z to -y
LEVEL = 1.0


def _pool(shape, size, density, quat=(1.0, 0.0, 0.0, 0.0), z=None, waves=0.0, mesh=None) -> SimSession:
    """One free body in still water 1 m deep, dropped 5 cm above where it should float (or at z)."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(type=G.mjGEOM_PLANE, size=[0, 0, 1])
    body = spec.worldbody.add_body(name="thing", pos=[0, 0, LEVEL + 0.05 if z is None else z], quat=list(quat))
    body.add_freejoint()
    if mesh is not None:
        spec.add_mesh(name="hull", uservert=np.asarray(mesh, float).ravel().tolist())
        body.add_geom(type=G.mjGEOM_MESH, meshname="hull", density=density)
    else:
        body.add_geom(type=shape, size=size, density=density)
    water.encode(spec, water.Water(level=LEVEL, waves=waves))
    model = spec.compile()
    session = SimSession(model=model, data=mujoco.MjData(model))
    mujoco.mj_forward(model, session.data)
    return session


def _rest_height(session: SimSession, seconds: float = 8.0) -> float:
    session.step(round(seconds / session.model.opt.timestep))
    heights = []
    for _ in range(250):
        session.step(2)
        heights.append(session.data.xpos[1][2])
    return float(np.mean(heights))


def _cap_depth(share: float) -> float:
    """How far below the waterline the bottom of a ball sits when `share` of it is under, in radii."""
    lo, hi = 0.0, 2.0
    for _ in range(60):
        d = (lo + hi) / 2
        lo, hi = (d, hi) if d * d * (3 - d) / 4 < share else (lo, d)
    return (lo + hi) / 2


# ---- Archimedes, shape by shape ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "shape", "size", "density", "quat", "draft"),
    [
        # A box floats with its density's share of its height under water (wide enough to float flat).
        ("box 500", G.mjGEOM_BOX, [0.3, 0.2, 0.1], 500, (1, 0, 0, 0), 0.5 * 0.2),
        ("box 300", G.mjGEOM_BOX, [0.4, 0.3, 0.08], 300, (1, 0, 0, 0), 0.3 * 0.16),
        ("ball 500", G.mjGEOM_SPHERE, [0.15], 500, (1, 0, 0, 0), 0.15),
        ("ball 250", G.mjGEOM_SPHERE, [0.15], 250, (1, 0, 0, 0), _cap_depth(0.25) * 0.15),
        ("upright can", G.mjGEOM_CYLINDER, [0.12, 0.2], 400, (1, 0, 0, 0), 0.4 * 0.4),
        ("lying log", G.mjGEOM_CYLINDER, [0.1, 0.4], 500, LYING, 0.1),
        ("lying capsule", G.mjGEOM_CAPSULE, [0.08, 0.3], 500, LYING, 0.08),
        ("ellipsoid", G.mjGEOM_ELLIPSOID, [0.3, 0.2, 0.1], 500, (1, 0, 0, 0), 0.1),
    ],
)
def test_floats_at_the_draft_archimedes_predicts(label, shape, size, density, quat, draft):
    session = _pool(shape, list(size) + [0.0] * (3 - len(size)), density, quat)
    z = _rest_height(session)
    bottom = z - _half_height(quat, shape, size)
    measured = LEVEL - bottom
    assert measured == pytest.approx(draft, rel=0.05), f"{label}: draft {measured:.4f} m, Archimedes {draft:.4f} m"
    assert np.abs(session.data.qvel).max() < 0.02, f"{label} still moving: {session.data.qvel}"


def _half_height(quat, shape, size) -> float:
    """From the center to the bottom of the shape as it floats."""
    if shape == G.mjGEOM_SPHERE:
        return size[0]
    if shape in (G.mjGEOM_BOX, G.mjGEOM_ELLIPSOID):
        return size[2]
    return size[0] if quat == LYING else size[1]  # cylinders and capsules: lying on their side, or upright


def test_a_mesh_floats_by_its_bounding_box():
    corners = [[x, y, z] for x in (-0.3, 0.3) for y in (-0.2, 0.2) for z in (-0.1, 0.1)]
    session = _pool(None, None, 400, mesh=corners)
    bottom = _rest_height(session) - 0.1
    assert LEVEL - bottom == pytest.approx(0.4 * 0.2, rel=0.05)


def test_steel_sinks_and_wood_floats_in_a_world():
    world = World.from_yaml(
        """
        name: lake
        water: 1.0
        objects:
          - {name: crate, shape: box, size: [0.6, 0.4, 0.2], pos: [0, 0], mass: 24.0}
          - {name: steel_block, shape: box, size: [0.2, 0.2, 0.2], pos: [1, 0], mass: 62.4}
        """.replace("\n        ", "\n")
    )
    model = compile_world(world).model
    session = SimSession(model=model, data=mujoco.MjData(model))
    session.reset()
    crate, steel = model.body("crate").id, model.body("steel_block").id
    # It starts where it floats: half of a 500 kg/m^3 crate under, the steel on the bottom.
    assert session.data.xpos[crate][2] == pytest.approx(LEVEL, abs=0.003)
    assert session.data.xpos[steel][2] == pytest.approx(0.1, abs=0.003)
    session.step(1500)
    assert session.data.xpos[crate][2] == pytest.approx(LEVEL, abs=0.003)
    assert session.data.xpos[steel][2] == pytest.approx(0.1, abs=0.003)
    found = {r["name"]: r for r in session.water.report(session.data)}
    assert found["crate"]["under_water"] == pytest.approx(0.5, abs=0.02)
    assert found["steel_block"]["under_water"] == pytest.approx(1.0)
    assert found["steel_block"]["depth_m"] == pytest.approx(0.9, abs=0.01)


def test_steel_dropped_in_the_water_sinks_to_the_bottom():
    session = _pool(G.mjGEOM_BOX, [0.1, 0.1, 0.1], 7800, z=1.5)
    session.step(2000)
    assert session.data.xpos[1][2] == pytest.approx(0.1, abs=0.003)


def test_a_light_ball_pushed_under_comes_back_up():
    session = _pool(G.mjGEOM_SPHERE, [0.1, 0, 0], 200, z=0.4)
    session.step(250)
    assert session.data.qvel[2] > 0.5  # rising fast
    assert _rest_height(session, 6.0) == pytest.approx(LEVEL + 0.1 - _cap_depth(0.2) * 0.1, abs=0.003)


# ---- drag ---------------------------------------------------------------------------------------------------


def test_drag_slows_a_floating_crate_as_the_drag_law_says():
    """A 24 kg crate, half under, pushed to 1 m/s: m dv/dt = -(c1 v + c2 v^2) on its wet front, solved exactly."""
    session = _pool(G.mjGEOM_BOX, [0.3, 0.2, 0.1], 500, z=LEVEL)
    session.data.qvel[0] = 1.0
    speeds = []
    for _ in range(60):
        session.step(25)  # 0.05 s
        speeds.append(float(session.data.qvel[0]))
    assert all(b <= a + 1e-6 for a, b in zip(speeds, speeds[1:], strict=False)), "the crate sped up"
    assert min(speeds) > -1e-3, "drag pushed it backwards"
    mass, front = 24.0, 0.4 * 0.1  # kg; m^2 of its front under water
    c1 = 1000 * front * water.LINEAR_DRAG * water.HORIZONTAL_SHARE
    c2 = 0.5 * 1000 * water.DRAG_COEFFICIENT[int(G.mjGEOM_BOX)] * front
    for t, v in ((1.0, speeds[19]), (3.0, speeds[59])):
        exact = c1 / ((c1 + c2) * math.exp(c1 * t / mass) - c2)
        assert v == pytest.approx(exact, rel=0.05), f"at {t} s: {v:.3f} m/s, the drag law says {exact:.3f}"


def test_heave_settles_a_dropped_crate_within_seconds():
    session = _pool(G.mjGEOM_BOX, [0.2, 0.2, 0.1], 500, z=LEVEL + 0.12)
    session.step(round(4.0 / 0.002))
    zs = []
    for _ in range(200):
        session.step(5)
        zs.append(session.data.xpos[1][2])
    assert np.ptp(zs) < 0.004, f"still bobbing {np.ptp(zs):.4f} m after 4 s"
    assert np.mean(zs) == pytest.approx(LEVEL, abs=0.003)


def test_the_world_frame_force_matches_mj_apply_ft():
    """The free-body shortcut gives the same generalized force as MuJoCo's own mj_applyFT."""
    session = _pool(G.mjGEOM_BOX, [0.3, 0.2, 0.1], 500, quat=(0.97, 0.15, 0.1, 0.05), z=LEVEL - 0.02)
    m, d = session.model, session.data
    d.qvel[:] = [0.3, -0.2, 0.1, 0.5, -0.4, 0.2]
    mujoco.mj_forward(m, d)
    d.qfrc_applied[:] = 0
    session.water.apply(m, d)
    fast = d.qfrc_applied.copy()
    force, moment = session.water.wrenches(d)
    com = d.xipos[1]
    reference = np.zeros(m.nv)
    mujoco.mj_applyFT(m, d, force[0], moment[0] - np.cross(com, force[0]), com, 1, reference)
    assert fast == pytest.approx(reference, abs=1e-9)


# ---- waves and thrusters ------------------------------------------------------------------------------------


def test_waves_rock_a_crate_up_and_down():
    session = _pool(G.mjGEOM_BOX, [0.2, 0.2, 0.1], 500, z=LEVEL, waves=0.05)
    zs = []
    for _ in range(400):
        session.step(5)
        zs.append(session.data.xpos[1][2])
    assert 0.03 < np.ptp(zs[100:]) < 0.25  # it rides the swell
    assert np.mean(zs[100:]) == pytest.approx(LEVEL, abs=0.02)


def _thruster_boat(z: float) -> SimSession:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.worldbody.add_geom(type=G.mjGEOM_PLANE, size=[0, 0, 1])
    body = spec.worldbody.add_body(name="raft", pos=[0, 0, z])
    body.add_freejoint()
    body.add_geom(type=G.mjGEOM_BOX, size=[0.4, 0.3, 0.05], density=400)
    body.add_site(name="stern_thruster", pos=[-0.4, 0, -0.1], size=[0.03, 0, 0])
    act = spec.add_actuator(name="push", target="stern_thruster", trntype=mujoco.mjtTrn.mjTRN_SITE)
    act.gear = [1, 0, 0, 0, 0, 0]
    act.ctrlrange = [-1, 1]
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.gainprm[0] = 20.0
    water.encode(spec, water.Water(level=LEVEL))
    model = spec.compile()
    session = SimSession(model=model, data=mujoco.MjData(model))
    mujoco.mj_forward(model, session.data)
    return session


def test_a_thruster_pushes_in_the_water_and_not_out_of_it():
    afloat = _thruster_boat(LEVEL)
    assert afloat.water.thruster_actuators.tolist() == [0]
    afloat.data.ctrl[0] = 1.0
    afloat.step(1000)
    assert afloat.data.qvel[0] > 0.3
    ashore = _thruster_boat(0.05)  # on the floor of a dry world part: its propeller is out of the water
    ashore.water = water.Hydro(ashore.model, water.Water(level=0.01))
    ashore.data.ctrl[0] = 1.0
    ashore.step(1000)
    assert abs(ashore.data.qvel[0]) < 0.01


# ---- worlds -------------------------------------------------------------------------------------------------


def test_worlds_without_water_are_unchanged():
    text = "name: plain\nobjects:\n  - {name: cube, shape: box, pos: [0.3, 0]}\n"
    world = World.from_yaml(text)
    assert "water" not in world.to_yaml()  # saved as before
    model = compile_world(world).model
    assert water.decode(model) is None
    assert water.surface_geom(model) < 0
    session = SimSession(model=model, data=mujoco.MjData(model))
    assert session.water is None


def test_water_in_world_yaml_round_trips_and_takes_a_bare_level():
    world = World.from_yaml("name: pool\nwater: 1.4\n")
    assert world.water.level == 1.4
    again = World.from_yaml(world.to_yaml())
    assert again.water == world.water
    model = compile_world(again).model
    decoded = water.decode(model)
    assert decoded.level == pytest.approx(1.4) and decoded.area is None
    assert water.surface_geom(model) >= 0


@pytest.mark.parametrize(
    "bad",
    [{"level": 0}, {"level": 1, "current": [5, 0]}, {"level": 1, "area": [2, 0, 1, 1]}, {"level": 1, "wave": 1}],
)
def test_water_is_checked(bad):
    with pytest.raises(ValueError):
        WaterEntry.model_validate(bad)


def test_area_keeps_the_water_inside():
    entry = WaterEntry(level=1.0, area=[-1, -1, 1, 1])
    hydro = water.Hydro(_pool(G.mjGEOM_SPHERE, [0.1, 0, 0], 500).model, entry.physics())
    heights = hydro.surface(np.array([0.0, 3.0]), np.array([0.0, 0.0]), 0.0)
    assert heights[0] == pytest.approx(1.0) and heights[1] < -1000


def test_water_costs_little_per_step():
    """A harbor of twenty floating things: the water forces stay a fixed handful of numpy operations."""
    yaml_objects = "\n".join(
        f"  - {{name: crate_{i}, shape: box, size: [0.4, 0.3, 0.25], pos: [{i % 5 - 2}, {i // 5 - 2}], mass: 12}}"
        for i in range(20)
    )
    world = World.from_yaml(f"name: harbor\nwater: {{level: 1.0, waves: 0.03}}\nobjects:\n{yaml_objects}\n")
    model = compile_world(world).model
    session = SimSession(model=model, data=mujoco.MjData(model))
    session.reset()
    session.step(50)
    start = time.perf_counter()
    session.step(500)
    per_step = (time.perf_counter() - start) / 500
    assert per_step < 2e-3, f"{per_step * 1e6:.0f} us per step"
