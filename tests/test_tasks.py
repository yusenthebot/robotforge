"""Tasks saved with a world (design/tasks): a sentence read as a task, the success check judged from where things are,
tasks in world.yaml, add_task on a generated world, and a task done for real with the robot's own behaviors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robotforge.behaviors import operate as operating
from robotforge.design.tasks import check, read
from robotforge.design.tasks.check import State, judge
from robotforge.design.tasks.spec import TaskEntry
from robotforge.design.world import World
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession


@pytest.mark.parametrize(
    ("sentence", "kind", "thing", "place"),
    [
        ("Bring the red cup from the kitchen to the bedroom desk", "bring", ("cup", "red", "kitchen"), ("desk", None,
         "bedroom")),
        ("put the cube on the table into the bin", "put", ("cube", None, None), ("bin", None, None)),
        ("go to the bedroom", "go_to", None, ("room", None, "bedroom")),
        ("open the drawer in the kitchen", "open", ("drawer", None, "kitchen"), None),
        ("pick up the blue book", "pick_up", ("book", "blue", None), None),
        ("push the button", "press", ("button", None, None), None),
        ("turn on the switch", "switch_on", ("switch", None, None), None),
        ("push the ball onto the goal", "push", ("ball", None, None), ("goal", None, None)),
        ("把厨房的苹果拿到卧室的桌子上", "bring", ("apple", None, "kitchen"), ("table", None, "bedroom")),
        ("把红色的方块放进篮子里", "put", ("cube", "red", None), ("bin", None, None)),
        ("打开厨房的抽屉", "open", ("drawer", None, "kitchen"), None),
        ("去客厅", "go_to", None, ("room", None, "living_room")),
    ],
)  # fmt: skip
def test_a_task_sentence_is_read_as_a_kind_a_thing_and_a_place(sentence, kind, thing, place):
    asked = read.read(sentence)
    assert asked.kind == kind
    for ref, want in ((asked.thing, thing), (asked.place, place)):
        if want is None:
            assert ref is None or ref.concept is None
        else:
            assert (ref.concept, ref.color, ref.room) == want


def test_a_sentence_with_nothing_to_do_says_what_it_can_do():
    with pytest.raises(ValueError, match="say what to do"):
        read.read("a lovely kitchen")


WORLD = World.model_validate(
    {
        "objects": [
            {"name": "kitchen", "shape": "marker", "pos": [0, 0], "size": [4, 3]},
            {"name": "goal", "shape": "marker", "pos": [5, 0], "size": [0.6]},
            {"name": "desk", "shape": "table", "pos": [1, 1], "size": [1.0, 0.6, 0.75]},
            {"name": "tote", "shape": "bin", "pos": [-1, 1, 0.75], "size": [0.4, 0.3, 0.12], "fixed": True},
            {"name": "chest", "shape": "drawer", "pos": [-1, -1, 0.9], "size": [0.3, 0.3, 0.14]},
            {"name": "cup", "shape": "cylinder", "pos": [1, 1, 0.79], "size": [0.06, 0.085]},
        ]
    }
)


def _task(**fields) -> TaskEntry:
    return TaskEntry(name="t", robot="bot", **fields)


@pytest.mark.parametrize(
    ("task", "state", "done"),
    [
        # on the desk's top, and 40 cm above it (falling: not on it), and beside it on the floor
        (_task(kind="put", object="cup", target="desk"), State(things={"cup": (1.2, 1.1, 0.795)}), True),
        (_task(kind="put", object="cup", target="desk"), State(things={"cup": (1.2, 1.1, 1.2)}), False),
        (_task(kind="put", object="cup", target="desk"), State(things={"cup": (1.7, 1.1, 0.04)}), False),
        # inside the tote (its floor at 0.75), and resting on its rim from outside
        (_task(kind="put", object="cup", target="tote"), State(things={"cup": (-1.05, 1.0, 0.8)}), True),
        (_task(kind="put", object="cup", target="tote"), State(things={"cup": (-1.25, 1.0, 0.8)}), False),
        # anywhere in the room, at any height
        (_task(kind="bring", object="cup", target="kitchen"), State(things={"cup": (1.5, -1.2, 0.0)}), True),
        (_task(kind="bring", object="cup", target="kitchen"), State(things={"cup": (2.5, 0, 0.8)}), False),
        # a robot within radius of the desk, and one in the room but not 15 cm inside its edge
        (_task(kind="go_to", target="desk", radius=0.5), State(robot=(1.9, 1.0, 0.3)), True),
        (_task(kind="go_to", target="desk", radius=0.5), State(robot=(2.1, 1.0, 0.3)), False),
        (_task(kind="go_to", target="kitchen"), State(robot=(1.0, 0.5, 0.3)), True),
        (_task(kind="go_to", target="kitchen"), State(robot=(1.95, 0.5, 0.3)), False),
        (_task(kind="go_to", target="kitchen"), State(robot=(1.0, 0.5, 0.1), upright=False), False),
        # a mark on the floor
        (_task(kind="push", object="cup", target="goal"), State(things={"cup": (5.1, 0.2, 0.03)}), True),
        # mechanisms as observe reads them
        (_task(kind="open", object="chest"), State(mechanisms={"chest": {"open_m": 0.15, "of_m": 0.23}}), True),
        (_task(kind="open", object="chest"), State(mechanisms={"chest": {"open_m": 0.1, "of_m": 0.23}}), False),
        (_task(kind="close", object="chest"), State(mechanisms={"chest": {"open_m": 0.005, "of_m": 0.23}}), True),
    ],
)
def test_the_success_check_reads_where_things_are(task, state, done):
    verdict = judge(task, WORLD, state)
    assert verdict.done is done, verdict.why
    assert verdict.known
    assert verdict.reward == pytest.approx(-verdict.distance + (check.SUCCESS_REWARD if done else 0.0))
    assert (verdict.distance == 0.0) is done


def test_picked_up_counts_from_where_the_attempt_began_and_unseen_things_are_not_judged():
    task = _task(kind="pick_up", object="cup")
    start = State(things={"cup": (1, 1, 0.79)})
    assert not judge(task, WORLD, State(things={"cup": (1, 1, 0.82)}), start).done
    assert judge(task, WORLD, State(things={"cup": (1, 1, 0.85)}), start).done
    unseen = judge(task, WORLD, State(), start)
    assert not unseen.done and not unseen.known
    pressed = _task(kind="press", object="chest")
    assert not judge(
        pressed, WORLD, State(mechanisms={"chest": {"presses": 2}}), State(mechanisms={"chest": {"presses": 2}})
    ).done
    assert judge(
        pressed, WORLD, State(mechanisms={"chest": {"presses": 3}}), State(mechanisms={"chest": {"presses": 2}})
    ).done


def test_the_reward_grows_as_the_thing_comes_closer():
    task = _task(kind="put", object="cup", target="desk")
    far = judge(task, WORLD, State(robot=(4, 0, 0.3), things={"cup": (3, 0, 0.79)}))
    near = judge(task, WORLD, State(robot=(2, 0.5, 0.3), things={"cup": (2, 0.8, 0.79)}))
    assert near.reward > far.reward


def test_a_world_without_tasks_saves_as_it_did_and_one_with_them_keeps_them():
    plain = World.model_validate({"objects": [{"name": "cube", "pos": [0.3, 0]}]})
    assert "tasks" not in plain.to_yaml()
    with_task = plain.model_copy(update={"tasks": [_task(kind="pick_up", object="cube")]})
    again = World.from_yaml(with_task.to_yaml())
    assert again.tasks == with_task.tasks
    with pytest.raises(ValueError, match="task names must be unique"):
        World.model_validate({"tasks": [_task(kind="pick_up", object="a"), _task(kind="pick_up", object="b")]})


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.load_scene(description="a small apartment with a kitchen, a living room and a bedroom", seed=0)
    return pg


def test_add_task_finds_the_thing_and_the_place_by_kind_color_and_room(home):
    result = home.add_task("bring the apple from the kitchen to the coffee table in the living room")
    task = result["task"]
    want = ("bring", "picker", "apple", "living_room_coffee_table")
    assert (task["kind"], task["robot"], task["object"], task["target"]) == want
    assert task["status"]["done"] is False and task["status"]["distance_m"] > 3
    saved = home.current_world().tasks[0]
    assert saved.start.robot is not None and saved.start.objects["apple"]
    status = home.task_status()["bring_apple"]
    assert status["done"] is False and status["goal"] == "apple in or on living_room_coffee_table"
    home.remove("bring_apple")
    assert home.current_world().tasks == []


def test_add_task_adds_the_thing_the_world_lacks_where_the_robot_reaches(home):
    result = home.add_task("pick up the purple cube in the living room")
    name = result["task"]["object"]
    assert name == "purple_cube" and any("added purple_cube" in n for n in result["notes"])
    cube = next(o for o in home.current_world().objects if o.name == name)
    assert len(cube.pos) == 3 and 0.3 < cube.pos[2] < 1.1  # on a top
    with pytest.raises(PlaygroundError, match="no 'the piano'"):
        home.add_task("bring the cup to the piano")


def test_a_drawer_task_gets_an_arm_beside_it_that_really_opens_it(home):
    result = home.add_task("open the drawer in the kitchen")
    task = result["task"]
    assert task["robot"] == "arm" and task["object"] == "kitchen_drawer"
    assert home.task_status()["open_kitchen_drawer"]["done"] is False
    op, _, _ = operating.start(home.behaviors, "arm", "kitchen_drawer", "open", None)
    home.run_fast(lambda: op.done, 60)
    home.run_fast(lambda: False, 0.5)
    status = home.task_status()["open_kitchen_drawer"]
    assert status["done"] is True, status
    home.control("reset")  # a reset starts the attempt again: closed, not done
    assert home.task_status()["open_kitchen_drawer"]["done"] is False


def test_a_task_saved_with_a_world_is_a_learning_environment(home):
    from robotforge.control.tasks.world_task import world_env

    home.add_task("go to the bedroom")
    env = world_env(home.store.dir, "go_to_bedroom")
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape and info["success"] is False
    start = env.task.entry.start.robot
    placed = env.data.xpos[env.task.probe.root][:2]
    assert math.dist(placed, start[:2]) <= env.task.entry.randomize.robot_xy * math.sqrt(2) + 0.05
    starts = set()
    for seed in range(3):
        env.reset(seed=seed)
        starts.add(tuple(np.round(env.data.xpos[env.task.probe.root][:2], 3)))
    assert len(starts) == 3  # each seed starts elsewhere
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isfinite(reward) and reward < 0 and not info["success"]
    # the reward is the check's: minus the distance still to go, per second
    verdict = env.task.verdict(env.data)
    assert reward == pytest.approx(-verdict.distance * env.control_dt, rel=1e-6)


def test_the_thing_taken_is_one_not_already_where_it_has_to_go(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("shelf")
    pg.add_object("desk", shape="table", size=[1.0, 0.6, 0.75], pos=[1.0, 0.0])
    pg.add_object("cup_a", shape="cylinder", size=[0.06, 0.085], pos=[1.0, 0.0, 0.7925])  # on the desk already
    pg.add_object("cup_b", shape="cylinder", size=[0.06, 0.085], pos=[-1.0, 0.0])
    pg.add_robot("picker", design="lift_picker", pos=[0.0, -1.0])
    result = pg.add_task("put a cup on the desk")
    assert result["task"]["object"] == "cup_b"


def test_chinese_names_the_thing_last_and_english_first():
    assert read.read("打开厨房台面上的柜子").thing.concept == "cabinet"
    assert read.read("open the cabinet on the kitchen counter").thing.concept == "cabinet"


def test_a_lift_picker_can_put_things_into_a_loose_bowl(tmp_path, monkeypatch):
    """A loose bin (a bowl) stood on its own floor, so a fetch called it full of itself."""
    from robotforge.behaviors import fetch

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("bowl")
    pg.add_object("bowl", shape="bin", size=[0.26, 0.26, 0.07], pos=[1.0, 0.0])
    pg.add_object("cube", shape="box", size=[0.045, 0.045, 0.045], pos=[-1.0, 0.0])
    m, d = pg.session.model, pg.session.data
    place = fetch.place_for(m, d, "bowl", "bin", m.body("cube").id, 0.03)
    assert place.spots


def test_going_to_the_room_the_robot_is_in_starts_it_in_another(home):
    """The robot starts in the first room; "go to the kitchen" from the kitchen would be done before it began."""
    result = home.add_task("go to the kitchen")
    assert result["task"]["status"]["done"] is False
    assert any("away from kitchen" in n for n in result["notes"])
    start = home.current_world().tasks[0].start.robot
    robot = home.current_world().robots[0]
    assert [round(v, 3) for v in robot.pos[:2]] == [round(v, 3) for v in start[:2]]
