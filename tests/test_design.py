import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from robotforge.design import templates
from robotforge.design.store import WorldStore, slug
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
    before = store.file.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        store.apply(lambda w: w.robots.append(_entry("cat", "spider")))
    assert store.file.read_text(encoding="utf-8") == before  # failed edit left the world untouched
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


def test_undo_steps_back_one_edit_at_a_time_even_if_the_clock_stands_still(tmp_path, monkeypatch):
    # Windows reads the same time for edits less than ~15 ms apart.
    monkeypatch.setattr("robotforge.design.store.time.time_ns", lambda: 1_000)
    store = WorldStore.create("Frozen Clock", parent=tmp_path)
    for name in ("a", "b", "c"):
        store.apply(lambda w, name=name: w.robots.append(_entry(name, "rover")))
    seen = []
    for _ in range(3):
        seen.append([r.name for r in store.undo().robots])
    assert seen == [["a", "b"], ["a"], []]


def test_worlds_named_in_another_script_get_a_folder_each(tmp_path):
    """Both names cleaned to nothing, so the second new_world reopened the first world and said it created it."""
    first = WorldStore.create("机器人乐园", parent=tmp_path)
    second = WorldStore.create("我的世界", parent=tmp_path)
    assert first.dir != second.dir and not first.reopened and not second.reopened
    assert first.dir.name.startswith("world_") and first.dir.name == slug("机器人乐园")  # stable: open finds it
    assert slug("机器人乐园 ") == slug("机器人乐园") and slug("robot 机器人").startswith("robot_")
    assert WorldStore.create("机器人乐园", parent=tmp_path).reopened
    assert slug("My Cool World") == "my_cool_world" and slug("???") == "world"  # Latin names stay as they were


def test_new_world_on_a_name_that_exists_reopens_it_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    assert pg.new_world("garden")["created"] is True
    pg.add_object("ball", shape="sphere", size=[0.08], pos=[0.5, 0])
    again = pg.new_world("Garden", floor="grass")
    assert again["created"] is False and again["objects"] == ["ball"]
    assert "already existed" in again["notes"][0] and "floor and contents unchanged" in again["notes"][0]


def test_a_world_yaml_with_a_mistake_is_explained_by_entry_not_by_position(tmp_path):
    text = "name: w\nrobots:\n- template: rover\nobjects:\n- name: ball\n  colour: red\n"
    with pytest.raises(ValueError) as caught:
        World.from_yaml(text)
    said = str(caught.value)
    assert "robot number 1: name is required" in said
    assert "object ball: 'colour' is not a known field; did you mean color?" in said
    assert "pydantic" not in said and "robots.0" not in said and "objects.0" not in said
    with pytest.raises(ValueError, match=r"world.yaml is not valid YAML at line 3: expected ',' or '\]'"):
        World.from_yaml("name: w\nfloor: [1\n  style: checker\n")


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("name: w\nterrain:\n- type: ramp\n  angle: 80\n", "terrain patch number 1: angle must be at most 45"),
        (
            "name: w\nterrain:\n- type: mountain\n",
            "terrain patch number 1: type 'mountain' is not one of: hills, rough",
        ),
        ("name: w\nterrain:\n- angle: 3\n", "terrain patch number 1: type null is not one of: hills, rough"),
        ("name: w\nrobots:\n- 7\n", "robot number 1: must be an object, got 7"),
        ("name: w\nobjects:\n- {name: b, shape: pyramid}\n", "object b: shape 'pyramid' is not one of: box, sphere"),
        ("name: w\nrobots: 5\n", "robots must be a list, got 5"),
    ],
)
def test_other_mistakes_in_a_world_yaml_are_worded_by_entry_too(text, says):
    with pytest.raises(ValueError) as caught:
        World.from_yaml(text)
    said = str(caught.value)
    assert said.startswith("world.yaml is not valid, fix it in the file. ") and says in said, said
    assert "tag" not in said and "pydantic" not in said


def test_world_files_are_utf8_with_lf_endings(tmp_path):
    store = WorldStore.create("Plain", parent=tmp_path)
    store.apply(lambda w: w.robots.append(_entry("dog", "quadruped")))
    for path in (store.file, store.build_dir / "world.xml", *store.history.glob("*.yaml")):
        assert b"\r" not in path.read_bytes(), path


def test_show_world_with_no_world_open_says_so_and_lists_the_saved_ones(tmp_path, monkeypatch):
    """Agents call show_world first to see where things stand; with no world open it was a failed call (both
    go2_arm_kitchen sessions on the evals branch began with one)."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    empty = pg.show_world()
    assert empty["world"] is None and empty["saved_worlds"] == [] and "new_world makes one" in empty["notes"][0]
    pg.new_world("garden")
    assert "name: garden" in pg.show_world()["world_yaml"]
    assert Playground(session=SimSession.from_xml("<mujoco/>")).show_world()["saved_worlds"] == ["garden"]
