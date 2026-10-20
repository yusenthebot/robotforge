"""The harder tasks (design/tasks/hard.py, hard_check.py, hard_bench.py): sentences read into tasks, success judged from
how things end up, and the scripted expert doing each one, offline with the Menagerie stand-ins."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotforge.design.tasks import hard, hard_bench
from robotforge.design.tasks.check import State, judge
from robotforge.design.tasks.spec import HARD_KINDS, TaskEntry
from robotforge.design.world import World
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
Z = 0.4


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))


def _world(*objects: dict, robot: bool = True) -> World:
    return World.model_validate(
        {
            "name": "t",
            "robots": [{"name": "arm", "library": "franka_emika_panda"}] if robot else [],
            "objects": list(objects),
        }
    )


def _task(kind: str, **fields) -> TaskEntry:
    return TaskEntry(name=kind, kind=kind, robot="arm", **fields)


def _state(things: dict, tilts=None, speeds=None, **extra) -> State:
    return State(things={k: tuple(v) for k, v in things.items()}, tilts=tilts or {}, speeds=speeds or {}, **extra)


SHELF = {"name": "shelf", "shape": "shelf", "size": [0.4, 0.2, 0.2], "pos": [0.5, 0.3, Z]}
MUG = {"name": "mug", "shape": "mug", "pos": [0.4, -0.2, Z]}


# ---- reading a sentence -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("put the mug on the shelf upright", "upright"),
        ("stand the mug up on the shelf", "upright"),
        ("stack the brick on the tin on the book", "stack"),
        ("insert the bottle into the rack", "insert"),
        ("sort the things in the bin into the trays by kind", "sort"),
        ("pour the balls from the cup into the bowl", "pour"),
        ("hand the hammer to the other arm", "hand_over"),
        ("把杯子竖着放在架子上", "upright"),
        ("bring the red cup to the table", None),
        ("put the cube in the bin", None),
    ],
)
def test_a_sentence_is_recognized_as_a_hard_task_or_left_to_the_old_kinds(sentence, kind):
    assert hard.recognizes(sentence) == kind


def test_a_stack_sentence_puts_the_last_thing_named_at_the_bottom_when_it_says_on():
    world = _world(
        {"name": "book", "shape": "book", "pos": [0.5, 0.0, Z]},
        {"name": "tin", "shape": "tin", "pos": [0.3, -0.2, Z]},
        {"name": "brick", "shape": "brick", "pos": [0.6, -0.2, Z]},
    )
    arms = [SimpleNamespace(name="arm", kind="arm", arms=[])]
    task = hard.plan(world, "stack the brick on the tin on the book", arms, None, None, 0, "stack").task
    assert task.items == ["book", "tin", "brick"] and task.kind == "stack"
    assert task.describe() == "brick on tin on book"
    listed = hard.plan(world, "stack the book, the tin and the brick", arms, None, None, 0, "stack").task
    assert listed.items == ["book", "tin", "brick"]


def test_a_sentence_names_things_by_what_they_are_and_their_color_and_a_missing_one_says_what_to_add():
    world = _world(
        {"name": "m1", "shape": "mug", "pos": [0.4, 0.0, Z], "color": "red"},
        {"name": "m2", "shape": "mug", "pos": [0.4, 0.2, Z], "color": "blue"},
        SHELF,
    )
    arms = [SimpleNamespace(name="arm", kind="arm", arms=[])]
    task = hard.plan(world, "put the blue mug on the shelf upright", arms, None, None, 0, "upright").task
    assert task.object == "m2" and task.target == "shelf"
    with pytest.raises(hard.TaskProblem, match='no rack in this world.*add_object\\(name, shape="rack"\\)'):
        hard.plan(world, "insert the mug into the rack", arms, None, None, 0, "insert")


def test_an_insert_sentence_can_name_the_socket_by_number_or_by_ordinal():
    world = _world(
        {"name": "bottle", "shape": "bottle", "pos": [0.4, 0.0, Z]},
        {"name": "rack", "shape": "rack", "pos": [0.5, 0.3, Z]},
    )
    arms = [SimpleNamespace(name="arm", kind="arm", arms=[])]
    assert hard.plan(world, "insert the bottle into the rack", arms, None, None, 0, "insert").task.target == "rack"
    assert (
        hard.plan(world, "insert the bottle into socket 3 of the rack", arms, None, None, 0, "insert").task.target
        == "rack/3"
    )
    assert (
        hard.plan(world, "insert the bottle in the second slot of the rack", arms, None, None, 0, "insert").task.target
        == "rack/2"
    )


def test_a_sort_sentence_sends_each_thing_to_the_tray_named_for_its_kind():
    world = _world(
        {"name": "fruit_tray", "shape": "tray", "pos": [0.3, 0.4, Z], "params": {"tag": "fruit"}},
        {"name": "cans_tray", "shape": "tray", "pos": [0.5, 0.4, Z], "params": {"tag": "cans"}},
        {"name": "bin", "shape": "tray", "size": [0.34, 0.26, 0.06], "pos": [0.45, -0.3, Z], "params": {"tag": "none"}},
        {"name": "apple_1", "shape": "apple", "pos": [0.4, -0.3, Z]},
        {"name": "can_1", "shape": "can", "pos": [0.5, -0.3, Z]},
        {"name": "mug_1", "shape": "mug", "pos": [0.9, 0.9, Z]},  # not in the bin
    )
    arms = [SimpleNamespace(name="arm", kind="arm", arms=[])]
    task = hard.plan(world, "sort the things in the bin into the trays", arms, None, None, 0, "sort").task
    assert task.into == {"apple_1": "fruit_tray", "can_1": "cans_tray"} and "mug_1" not in task.items


def test_add_task_makes_a_hard_task_from_a_sentence_and_saves_it_only_where_it_is_used():
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("tasks")
    plain = pg.add_object("cube", pos=[0.4, 0.0])
    del plain
    hard_bench.bench(pg, "panda")
    pg.add_object("shelf", shape="shelf", size=[0.4, 0.2, 0.2], pos=[0.5, 0.3, Z])
    pg.add_object("mug", shape="mug", pos=[0.4, -0.2, Z])
    out = pg.add_task("put the mug on the shelf upright", robot="arm")
    assert out["task"]["kind"] == "upright" and out["task"]["goal"] == "mug upright on shelf"
    text = pg.store.file.read_text(encoding="utf-8")
    assert "kind: upright" in text and "items" not in text and "into" not in text and "tilt" not in text
    with pytest.raises(PlaygroundError, match="no rack in this world"):
        pg.add_task("insert the mug into the rack", robot="arm")


# ---- judging how things ended up -----------------------------------------------------------------------------------


def test_upright_wants_the_thing_on_a_board_of_the_shelf_level_and_at_rest():
    world = _world(SHELF, MUG)
    task = _task("upright", object="mug", target="shelf")
    top = Z + 0.2
    ok = judge(task, world, _state({"mug": (0.5, 0.3, top)}))
    assert ok.done and "stands on shelf" in ok.why
    tipped = judge(task, world, _state({"mug": (0.5, 0.3, top)}, tilts={"mug": 40.0}))
    assert not tipped.done and "tipped 40 degrees" in tipped.why
    moving = judge(task, world, _state({"mug": (0.5, 0.3, top)}, speeds={"mug": 0.5}))
    assert not moving.done and "still moving" in moving.why
    beside = judge(task, world, _state({"mug": (0.9, 0.3, top)}))
    assert not beside.done and beside.distance > 0.2
    floor = judge(task, world, _state({"mug": (0.5, 0.3, Z)}))  # the lower board is not the top, but it is a board
    assert floor.known and floor.done is True or "from standing" in floor.why
    assert judge(task, world, _state({})).known is False


def test_a_tilt_the_task_allows_is_used_and_the_default_is_fifteen_degrees():
    world = _world(SHELF, MUG)
    state = _state({"mug": (0.5, 0.3, Z + 0.2)}, tilts={"mug": 20.0})
    assert not judge(_task("upright", object="mug", target="shelf"), world, state).done
    assert judge(_task("upright", object="mug", target="shelf", tilt=25.0), world, state).done


def test_a_stack_is_judged_bottom_to_top_and_a_lean_or_an_overhang_breaks_it():
    world = _world(
        {"name": "book", "shape": "book", "pos": [0.5, 0.0, Z]},
        {"name": "tin", "shape": "tin", "pos": [0.3, -0.2, Z]},
        {"name": "brick", "shape": "brick", "pos": [0.6, -0.2, Z]},
    )
    task = _task("stack", object="brick", target="book", items=["book", "tin", "brick"])
    book_top, tin_top = Z + 0.028, Z + 0.028 + 0.082
    good = _state({"book": (0.5, 0.0, Z), "tin": (0.5, 0.01, book_top), "brick": (0.5, 0.0, tin_top)})
    assert judge(task, world, good).done
    off = _state({"book": (0.5, 0.0, Z), "tin": (0.5, 0.01, book_top), "brick": (0.58, 0.0, tin_top)})
    assert not judge(task, world, off).done
    low = _state({"book": (0.5, 0.0, Z), "tin": (0.5, 0.0, book_top), "brick": (0.5, 0.0, book_top)})
    assert not judge(task, world, low).done  # the brick is beside the tin's top, level with it
    lean = _state(
        {"book": (0.5, 0.0, Z), "tin": (0.5, 0.0, book_top), "brick": (0.5, 0.0, tin_top)}, tilts={"brick": 30.0}
    )
    assert not judge(task, world, lean).done and "tipped" in judge(task, world, lean).why
    assert judge(_task("stack", object="tin", target="tin", items=["tin"]), world, good).known is False


def test_an_insert_is_judged_by_the_socket_it_stands_in_down_to_its_floor():
    rack = {"name": "rack", "shape": "rack", "pos": [0.5, 0.3, Z]}
    world = _world(rack, {"name": "bottle", "shape": "bottle", "pos": [0.4, 0.0, Z]})
    task = _task("insert", object="bottle", target="rack/2")
    height, depth = 0.10, 0.085  # the rack's defaults
    floor = Z + height - depth
    inside = judge(task, world, _state({"bottle": (0.5, 0.3, floor)}))
    assert inside.done and "socket 2" in inside.why
    resting_on_top = judge(task, world, _state({"bottle": (0.5, 0.3, Z + height)}))
    assert not resting_on_top.done and "over the bottom of the socket" in resting_on_top.why
    next_socket = judge(task, world, _state({"bottle": (0.6, 0.3, floor)}))
    assert not next_socket.done and "from the middle" in next_socket.why
    crooked = judge(task, world, _state({"bottle": (0.5, 0.3, floor)}, tilts={"bottle": 20.0}))
    assert not crooked.done


def test_a_sort_is_done_when_every_thing_is_in_its_own_tray():
    world = _world(
        {"name": "fruit_tray", "shape": "tray", "pos": [0.3, 0.4, Z], "params": {"tag": "fruit"}},
        {"name": "cans_tray", "shape": "tray", "pos": [0.6, 0.4, Z], "params": {"tag": "cans"}},
        {"name": "apple_1", "shape": "apple", "pos": [0.4, -0.3, Z]},
        {"name": "can_1", "shape": "can", "pos": [0.5, -0.3, Z]},
    )
    task = _task("sort", object="apple_1", target="fruit_tray", items=["apple_1", "can_1"],
                 into={"apple_1": "fruit_tray", "can_1": "cans_tray"})  # fmt: skip
    right = judge(task, world, _state({"apple_1": (0.3, 0.4, Z + 0.006), "can_1": (0.6, 0.4, Z + 0.006)}))
    assert right.done and "all 2" in right.why
    swapped = judge(task, world, _state({"apple_1": (0.6, 0.4, Z + 0.006), "can_1": (0.3, 0.4, Z + 0.006)}))
    assert not swapped.done and "2 of 2 not in their tray" in swapped.why
    one = judge(task, world, _state({"apple_1": (0.3, 0.4, Z + 0.006), "can_1": (0.5, -0.3, Z)}))
    assert not one.done and one.distance > 0.5 and "can_1 is not in cans_tray" in one.why


def test_a_pour_counts_the_balls_that_end_up_in_the_cup_and_needs_three_quarters():
    world = _world(
        {"name": "src", "shape": "cup", "pos": [0.4, -0.2, Z]},
        {"name": "dst", "shape": "bowl", "pos": [0.5, 0.2, Z]},
    )
    balls = [f"b{i}" for i in range(8)]
    task = _task("pour", object="src", target="dst", items=balls)
    inside = {f"b{i}": (0.5, 0.2, Z + 0.02) for i in range(6)}
    outside = {f"b{i}": (0.4, -0.2, Z + 0.02) for i in range(6, 8)}
    assert judge(task, world, _state({**inside, **outside})).done
    half = {f"b{i}": (0.5, 0.2, Z + 0.02) for i in range(4)} | {f"b{i}": (0.9, 0.9, Z) for i in range(4, 8)}
    short = judge(task, world, _state(half))
    assert not short.done and "4 of 8 balls" in short.why and "(6 needed)" in short.why


def test_a_hand_over_is_done_when_the_receiver_holds_it_up_and_the_giver_has_let_go():
    world = _world({"name": "hammer", "shape": "hammer", "pos": [0.4, 0.0, Z]})
    task = _task("hand_over", object="hammer", target="other")
    start = _state({"hammer": (0.4, 0.0, Z)})
    held = _state({"hammer": (0.5, 0.0, Z + 0.25)}, touching={"hammer": {"other"}})
    assert judge(task, world, held, start).done
    both = _state({"hammer": (0.5, 0.0, Z + 0.25)}, touching={"hammer": {"other", "arm"}})
    assert "hand-over is under way" in judge(task, world, both, start).why
    low = _state({"hammer": (0.5, 0.0, Z)}, touching={"hammer": {"other"}})
    assert not judge(task, world, low, start).done


def test_every_hard_kind_has_a_check_and_the_old_kinds_are_judged_as_they_were():
    assert set(HARD_KINDS) == {"upright", "stack", "insert", "sort", "pour", "hand_over"}
    world = _world({"name": "box", "shape": "box", "pos": [0.4, 0.0, Z]})
    done = judge(
        _task("pick_up", object="box"), world, _state({"box": (0.4, 0.0, Z + 0.1)}), _state({"box": (0.4, 0.0, Z)})
    )
    assert done.done


# ---- the bench and the expert -------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["upright", "insert", "stack", "pour", "hand_over"])
def test_the_scripted_expert_does_the_task_and_the_tasks_own_check_says_so(kind):
    report = hard_bench.run(kind, 1, "panda")
    assert report["done"], report.get("cause") or report["why"]
    assert all(step.get("success") for step in report["steps"]) and not any(s.get("knocked") for s in report["steps"])
    assert report["sentence"]


def test_a_failure_is_given_a_cause_from_its_steps():
    refused = hard_bench.cause([{"thing": "mug", "target": "shelf", "refused": "arm cannot take hold of it: ..."}], "")
    assert refused.startswith("refused: arm cannot take hold")
    never = hard_bench.cause(
        [{"thing": "mug", "target": "shelf", "success": False, "lifted_cm": 0.4, "moved_cm": 0.2, "knocked": []}], ""
    )
    assert "never lifted" in never
    slipped = hard_bench.cause(
        [{"thing": "mug", "target": "shelf", "success": False, "lifted_cm": 20.0, "moved_cm": 3.0, "knocked": []}], ""
    )
    assert "slipped out of the fingers" in slipped
    knocked = hard_bench.cause(
        [{"thing": "can_1", "target": "t", "success": True, "knocked": ["apple_1"], "tilt_deg": 0}], ""
    )
    assert knocked == "knocked over apple_1 while handling can_1"


def test_a_failed_pour_says_how_many_balls_got_across():
    spilled = hard_bench.cause(
        [{"thing": "cup", "target": "bowl", "success": False, "balls_in_target": 2, "balls": 7}], ""
    )
    assert spilled == "only 2 of 7 balls reached bowl (the rest spilled)"


def test_a_pour_sentence_takes_the_balls_in_the_cup_and_a_cup_without_balls_is_refused():
    inside = [
        {"name": f"b{i}", "shape": "sphere", "size": [0.012], "pos": [0.4 + 0.01 * i, -0.2, Z + 0.02]} for i in range(3)
    ]
    far = {"name": "far", "shape": "sphere", "size": [0.012], "pos": [0.9, 0.5, Z + 0.02]}
    cup = {"name": "cup", "shape": "cup", "pos": [0.4, -0.2, Z]}
    bowl = {"name": "bowl", "shape": "bowl", "pos": [0.5, 0.2, Z]}
    arms = [SimpleNamespace(name="arm", kind="arm", arms=[])]
    sentence = "pour the balls from the cup into the bowl"
    task = hard.plan(_world(cup, bowl, far, *inside), sentence, arms, None, None, 0, "pour").task
    assert (task.object, task.target, task.items) == ("cup", "bowl", ["b0", "b1", "b2"])
    with pytest.raises(hard.TaskProblem, match="nothing in cup to pour"):
        hard.plan(_world(cup, bowl, far), sentence, arms, None, None, 0, "pour")


def test_a_cup_with_a_lid_or_held_by_a_handle_only_is_not_poured_before_the_arm_moves():
    pg = hard_bench.fresh()
    hard_bench.bench(pg, "panda")
    pg.add_object("bowl", shape="bowl", pos=[0.5, 0.25, Z])
    pg.add_object("lidded", shape="cup", params={"style": "tumbler", "lid": True}, pos=[0.45, -0.22, Z])
    pg.add_object("mug", shape="mug", pos=[0.3, -0.3, Z])
    pg.add_object("ball", shape="sphere", size=[0.012], pos=[0.3, -0.3, Z + 0.03])
    pg.run_fast(lambda: False, 0.5)
    arm = pg.behaviors.resolve("arm", arm=True)
    with pytest.raises(Exception, match="no open mouth"):
        pg.behaviors.pick_and_place(arm, "lidded", "bowl", True, None, True)
    with pytest.raises(Exception, match="no grip on its own axis"):
        pg.behaviors.pick_and_place(arm, "mug", "bowl", True, None, True)


def test_a_hand_over_goes_through_the_table_and_the_second_arm_ends_up_holding_the_tool_up():
    pg = hard_bench.fresh()
    made = hard_bench.build(pg, "hand_over", 0, "panda")
    assert made["sentence"] == "hand the hammer to the other arm"
    task = pg.current_world().tasks[0]
    assert (task.kind, task.robot, task.target) == ("hand_over", "arm", "partner")
    report = hard_bench.solve(pg)
    assert report["done"], report.get("cause") or report["why"]
    assert [s["kind"] for s in report["steps"]] == ["put down", "take and hold"]
    held = pg.behaviors.resolve("partner", arm=True)
    pg.control("reset")
    del held


def test_target_hold_keeps_the_thing_lifted_in_the_fingers_and_a_second_pick_can_put_it_down():
    pg = hard_bench.fresh()
    hard_bench.bench(pg, "panda")
    pg.add_object("hammer", shape="hammer", pos=[0.4, -0.2, Z])
    pg.run_fast(lambda: False, 0.5)
    arm = pg.behaviors.resolve("arm", arm=True)
    motion, info = pg.behaviors.pick_and_place(arm, "hammer", "hold")
    pg.run_fast(lambda: bool(motion.done), 40)
    pg.run_fast(lambda: False, 1.5)
    result = pg.behaviors.held(info)
    assert result["success"] and result["held"] and result["lifted_cm"] >= 4.0
    pg.control("reset")  # a reset puts back what the firm hold changed
    assert pg.session.model.opt.noslip_iterations == 0 or not pg.session.reset_hooks


def test_a_suite_of_hard_tasks_has_a_world_on_disk_for_each_kind(tmp_path):
    from robotforge.design.tasks.suite import load_suite, make_hard_suite

    cases = make_hard_suite(6, seed=1, out=tmp_path / "suite")
    assert [c.kind for c in cases] == list(HARD_KINDS)
    assert [c.name for c in load_suite(tmp_path / "suite")] == [c.name for c in cases]
    for case in cases:
        text = (tmp_path / "suite" / "worlds" / case.name / "world.yaml").read_text(encoding="utf-8")
        assert f"kind: {case.kind}" in text and case.task
