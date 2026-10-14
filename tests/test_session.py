from pathlib import Path

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
    bad.write_text("<mujoco><worldbody><geom type='nonsense'/></worldbody></mujoco>")
    with pytest.raises(ModelLoadError, match="bad.xml"):
        SimSession.from_path(bad)
    with pytest.raises(ModelLoadError, match="not found"):
        SimSession.from_path(tmp_path / "missing.xml")
