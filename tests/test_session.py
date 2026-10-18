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
    from robotforge.sim.render import _camera_for

    session = SimSession.from_xml(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
        '<body name="ball" pos="0 0 0.1"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    session.data.qpos[:3] = [5.0, -2.0, 0.1]  # the ball rolled far away from where the model put it
    session.step(1)
    camera = _camera_for(session.model, "default", session.data)
    assert np.allclose(camera.lookat[:2], [5.0, -2.0], atol=0.01)


def test_a_robots_name_is_a_view_framed_on_that_robot():
    """An agent asked to show a Go2 with a lidar called observe(view="go2") first, and it was refused."""
    from robotforge.sim.render import VIEWS, _camera_for

    def robot(name: str, x: float) -> str:
        return f'<body name="{name}/base" pos="{x} 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>'

    session = SimSession.from_xml(
        '<mujoco><worldbody><geom type="plane" size="9 9 .1"/>'
        f"{robot('rover', -3.0)}{robot('dog', 3.0)}"
        '<body name="crate" pos="0 2 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>'
        "</worldbody></mujoco>"
    )
    whole = _camera_for(session.model, "default", session.data)
    for name, x in (("dog", 3.0), ("Rover", -3.0)):  # any case, like the presets
        cam = _camera_for(session.model, name, session.data)
        assert np.allclose(cam.lookat[:2], [x, 0.0], atol=0.01), name
        assert (cam.azimuth, cam.elevation) == VIEWS["default"][:2]
        assert cam.distance < whole.distance / 3  # one robot, not the three things that move
    # The crate stands between the default camera and nothing; a wall put between it and the dog is drawn faded.
    from robotforge.sim.render import occluders, robot_geoms

    assert not occluders(session.model, session.data, cam, robot_geoms(session.model, "rover"))
    walled = SimSession.from_xml(
        '<mujoco><worldbody><geom type="plane" size="9 9 .1"/>'
        f"{robot('dog', 3.0)}"
        '<body name="wall" pos="3.6 -0.45 0.3"><geom name="wall" type="box" size="0.6 0.03 0.3" euler="0 0 38"/>'
        "</body></worldbody></mujoco>"
    )
    mujoco.mj_forward(walled.model, walled.data)
    cam = _camera_for(walled.model, "dog", walled.data)
    wall = mujoco.mj_name2id(walled.model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    assert occluders(walled.model, walled.data, cam, robot_geoms(walled.model, "dog")) == {wall}
    with pytest.raises(ValueError, match="Views: .*rover, dog") as refused:
        _camera_for(session.model, "crate", session.data)  # a loose object is not a robot
    assert "crate" not in str(refused.value).split("Views:")[1]


def test_views_step_back_just_far_enough_for_what_they_show():
    """A fixed multiple of the scene size left a small arm and a cube a few pixels wide in a top view."""
    from robotforge.sim.render import VIEWS, _camera_for

    def scene(spread: float) -> SimSession:
        bodies = "".join(
            f'<body pos="{x} {y} 0.05"><freejoint/><geom type="box" size="0.05 0.05 0.05"/></body>'
            for x, y in ((-spread, -spread), (spread, spread))
        )
        return SimSession.from_xml(
            f'<mujoco><worldbody><geom type="plane" size="9 9 .1"/>{bodies}</worldbody></mujoco>'
        )

    near, far = scene(0.5), scene(2.0)
    for view in VIEWS:
        a = _camera_for(near.model, view, near.data).distance
        b = _camera_for(far.model, view, far.data).distance
        assert 2.5 < b / a < 5.0, view  # four times the spread, about four times as far
    top = _camera_for(far.model, "top", far.data, aspect=4 / 3)
    # the boxes' corners, 4.1 m apart along y, span about three quarters of the image height (45 degree fov)
    half_height = top.distance * np.tan(np.radians(far.model.vis.global_.fovy) / 2)
    assert 0.65 < 2.05 / half_height < 0.85


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
