from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.sim.session import ModelLoadError, SimSession

DEMO = Path(__file__).parents[1] / "src" / "robotforge" / "assets" / "demo.xml"


@pytest.fixture
def session() -> SimSession:
    return SimSession.from_path(DEMO)


def test_advance_tracks_wall_time_scaled_by_speed(session):
    session.speed = 0.5
    session.advance(0.04)
    assert session.data.time == pytest.approx(0.02, abs=session.model.opt.timestep)


def test_advance_caps_long_stalls(session):
    session.advance(5.0)  # a stalled tab must not cause a burst of simulated time
    assert session.data.time <= 0.1 + session.model.opt.timestep


def test_paused_session_does_not_move(session):
    session.paused = True
    assert session.advance(0.05) == 0
    assert session.data.time == 0.0


def test_set_ctrl_clamps_to_ctrlrange(session):
    session.set_ctrl(1, 99.0)
    assert session.data.ctrl[1] == pytest.approx(session.model.actuator_ctrlrange[1][1])
    with pytest.raises(IndexError):
        session.set_ctrl(session.model.nu, 0.0)


def test_drag_pulls_body_toward_target(session):
    cube = session.model.body("red_cube").id
    start = session.data.xpos[cube].copy()
    target = start + np.array([0.0, 0.0, 0.15])
    session.start_drag(cube, start)
    session.move_drag(target)
    session.step(500)
    assert session.data.xpos[cube][2] == pytest.approx(target[2], abs=0.01)
    session.end_drag()
    session.step(1)
    assert not session.data.qfrc_applied.any()


def test_world_body_is_not_draggable(session):
    with pytest.raises(IndexError):
        session.start_drag(0, np.zeros(3))


def test_reload_bumps_version_and_resets_state(session):
    session.step(10)
    v = session.version
    session.load_path(DEMO)
    assert session.version == v + 1
    assert session.data.time == 0.0


def test_body_poses_layout(session):
    poses = session.body_poses()
    assert poses.dtype == np.float32
    assert poses.shape == (1 + 7 * session.model.nbody,)


def test_bad_model_reports_compiler_error(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<mujoco><worldbody><geom type='nonsense'/></worldbody></mujoco>", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="bad.xml"):
        SimSession.from_path(bad)
    with pytest.raises(ModelLoadError, match="not found"):
        SimSession.from_path(tmp_path / "missing.xml")


def test_snapshots_frame_bodies_where_they_are_now():
    from robotforge.sim.render import _framing

    session = SimSession.from_xml(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
        '<body name="ball" pos="0 0 0.1"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    session.data.qpos[:3] = [5.0, -2.0, 0.1]  # the ball rolled far away from where the model put it
    session.step(1)
    center, _extent = _framing(session.model, session.data)
    assert np.allclose(center[:2], [5.0, -2.0], atol=0.01)


def test_dragging_a_standing_robot_pushes_it_and_a_loose_object_is_carried():
    """Holding the weight of a robot's whole body floated it off its feet while its legs kept pushing, and a G1
    or our humanoid toppled at a gentle tug. A robot that stands is pushed; a cube or an arm link is carried."""
    from robotforge.sim.session import _stands_on_its_own

    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body name="bot/trunk" pos="0 0 0.3">
          <freejoint/>
          <geom type="box" size="0.1 0.05 0.03"/>
          <body name="bot/leg">
            <joint name="bot/hip" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02"/>
          </body>
        </body>
        <body name="cube" pos="0.5 0 0.05"><freejoint/><geom type="box" size="0.03 0.03 0.03"/></body>
        <body name="arm/link" pos="-0.5 0 0.1">
          <joint name="arm/j" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        </body>
      </worldbody>
      <actuator><position joint="bot/hip" kp="10"/><position joint="arm/j" kp="10"/></actuator>
    </mujoco>
    """
    session = SimSession.from_xml(xml)
    m = session.model
    body = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)  # noqa: E731
    assert _stands_on_its_own(m, body("bot/trunk")) and _stands_on_its_own(m, body("bot/leg"))
    assert not _stands_on_its_own(m, body("cube"))
    assert not _stands_on_its_own(m, body("arm/link"))
    session.start_drag(body("cube"), session.data.xpos[body("cube")].copy())
    assert session.drag.hold_weight
    session.start_drag(body("bot/trunk"), session.data.xpos[body("bot/trunk")].copy())
    assert not session.drag.hold_weight
