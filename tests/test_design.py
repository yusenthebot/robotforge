import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from robotforge.design import templates
from robotforge.design.store import WorldStore
from robotforge.design.world import World, compile_world
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession


def _settle(model: mujoco.MjModel, seconds: float = 2.0) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    return data


@pytest.mark.parametrize("name", list(templates.TEMPLATES))
def test_every_template_compiles_with_defaults(name):
    built = templates.build(name, {})
    model = built.spec.compile()
    assert model.nu > 0


def test_quadruped_stands_on_its_own():
    world = World.model_validate({"robots": [{"name": "dog", "template": "quadruped"}]})
    model = compile_world(world).model
    data = _settle(model, 3.0)
    trunk = model.body("dog/trunk").id
    assert data.xpos[trunk][2] > 0.12  # still standing, not collapsed
    assert data.xmat[trunk].reshape(3, 3)[2, 2] > 0.95  # upright


def test_arm_holds_its_ready_pose_against_gravity():
    world = World.model_validate({"robots": [{"name": "arm", "template": "arm"}]})
    model = compile_world(world).model
    data = _settle(model, 2.0)
    shoulder = model.jnt_qposadr[model.joint("arm/shoulder").id]
    assert data.qpos[shoulder] == pytest.approx(0.35, abs=0.08)


def test_rover_drives_forward_when_both_wheels_spin():
    world = World.model_validate({"robots": [{"name": "rover", "template": "rover"}]})
    model = compile_world(world).model
    data = _settle(model, 0.5)
    start = data.xpos[model.body("rover/chassis").id].copy()
    data.ctrl[model.actuator("rover/left_wheel").id] = 10
    data.ctrl[model.actuator("rover/right_wheel").id] = 10
    for _ in range(1000):
        mujoco.mj_step(model, data)
    moved = data.xpos[model.body("rover/chassis").id] - start
    assert moved[0] > 0.2 and abs(moved[1]) < 0.05


def test_robots_and_objects_are_prefixed_and_placed():
    world = World.model_validate(
        {
            "robots": [
                {"name": "a", "template": "arm", "pos": [0, 0]},
                {"name": "b", "template": "arm", "pos": [1, 0], "yaw": 90},
            ],
            "objects": [{"name": "cube", "pos": [0.3, 0.1]}],
        }
    )
    model = compile_world(world).model
    assert model.actuator("a/shoulder") and model.actuator("b/shoulder")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.xpos[model.body("b/base").id][:2] == pytest.approx([1, 0])
    assert data.xpos[model.body("cube").id][2] == pytest.approx(0.025)


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"robots": [{"name": "Dog", "template": "quadruped"}]}, "lowercase"),
        ({"robots": [{"name": "dog"}]}, "exactly one of"),
        ({"objects": [{"name": "ball", "shape": "sphere", "size": [0.1, 0.1]}]}, "[diameter]"),
        ({"objects": [{"name": "x", "color": "chartreuse"}]}, "unknown color"),
        ({"robots": [{"name": "x", "template": "arm"}], "objects": [{"name": "x"}]}, "unique"),
    ],
)
def test_validation_messages_tell_the_agent_what_to_fix(bad, message):
    with pytest.raises(ValidationError, match=message):
        World.model_validate(bad)


def test_bad_template_params_are_reported_per_robot():
    world = World.model_validate({"robots": [{"name": "dog", "template": "quadruped", "params": {"legs": 6}}]})
    with pytest.raises(ValueError, match="robot 'dog'"):
        compile_world(world)


def test_store_edits_are_atomic_and_undoable(tmp_path):
    store = WorldStore.create("Test World!", parent=tmp_path)
    assert store.dir.name == "test_world"
    store.apply(lambda w: w.robots.append(_entry("dog", "quadruped")))
    before = store.file.read_text()
    with pytest.raises(ValueError):
        store.apply(lambda w: w.robots.append(_entry("cat", "spider")))
    assert store.file.read_text() == before  # failed edit left the world untouched
    assert (store.build_dir / "world.xml").exists()
    store.undo()
    assert store.load().robots == []
    with pytest.raises(ValueError, match="nothing to undo"):
        store.undo()


def test_playground_world_flow_updates_the_live_session(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    v0 = pg.session.version
    pg.new_world("garden", "grass")
    out = pg.add_robot("dog", template="quadruped", params={"color": "orange"})
    assert pg.session.version > v0 + 1
    assert "dog/front_left_knee" in [a["name"] for a in out["actuators"]]
    pg.add_object("ball", shape="sphere", size=[0.08], pos=[0.5, 0], color="yellow")
    pg.update_robot("dog", params={"leg_length": 0.3})
    assert pg.store.load().robots[0].params == {"color": "orange", "leg_length": 0.3}
    with pytest.raises(PlaygroundError, match="no robot named 'cat'"):
        pg.update_robot("cat", pos=[1, 1])
    pg.remove("ball")
    assert pg.store.load().objects == []
    pg.undo()
    assert [o.name for o in pg.store.load().objects] == ["ball"]


def _entry(name, template):
    from robotforge.design.world import RobotEntry

    return RobotEntry(name=name, template=template)
