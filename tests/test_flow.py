"""Behavior trees and state machines (robotforge.flow): the nodes' logic with a clock and no physics."""

import pytest

from robotforge.flow import (
    FAILURE,
    RUNNING,
    SUCCESS,
    Action,
    Condition,
    Fallback,
    FlowError,
    Inverter,
    Parallel,
    Repeat,
    Retry,
    Sequence,
    State,
    StateMachine,
    Timeout,
    Tree,
    Wait,
    WaitFor,
    sensor_is,
)
from robotforge.script import ScriptError, ScriptTimeLimit


class FakeSim:
    """Time that passes when stepped, and a record of what the viewer would have been shown."""

    def __init__(self):
        self.time = 0.0
        self.shown = []

    def step(self, seconds):
        self.time = round(self.time + seconds, 6)
        return self.time

    def show_flow(self, name="tree", nodes=None, status="", notes=None):
        self.shown.append({"name": name, "nodes": nodes, "status": status, "notes": notes or {}})


class FakeSensors:
    def __init__(self, readings):
        self.readings = readings

    def get(self, name, default=None):
        return self.readings.get(name, default)

    def __getitem__(self, name):
        return self.readings[name]


class FakeRobot:
    def __init__(self, **readings):
        self.sensors = FakeSensors(readings)


def run(root, **kwargs):
    sim = FakeSim()
    tree = Tree(root, name="t")
    return tree.run(sim, kwargs.pop("robot", None), **kwargs), tree, sim


def test_sequence_runs_children_in_order_and_stops_at_the_first_failure():
    seen = []
    ok = lambda name: Action(lambda c: seen.append(name), name=name)  # noqa: E731
    status, tree, _ = run(Sequence(ok("a"), ok("b"), Action(lambda c: False, name="c"), ok("d")))
    assert status is FAILURE
    assert seen == ["a", "b"]
    assert tree.statuses() == "fssf."


def test_fallback_takes_the_first_child_that_works():
    seen = []
    status, tree, _ = run(
        Fallback(
            Action(lambda c: False, name="first"),
            Action(lambda c: seen.append("second"), name="second"),
            Action(lambda c: seen.append("third"), name="third"),
        )
    )
    assert status is SUCCESS and seen == ["second"]
    assert tree.statuses() == "sfs."


def test_action_returns_none_true_false_or_a_status():
    assert run(Action(lambda c: None))[0] is SUCCESS
    assert run(Action(lambda c: True))[0] is SUCCESS
    assert run(Action(lambda c: False))[0] is FAILURE
    assert run(Action(lambda c: FAILURE))[0] is FAILURE


def test_a_refused_call_is_a_failure_with_its_reason_and_a_time_limit_is_not_caught():
    def refused(c):
        raise ScriptError("franka cannot reach [3.0, 0.0, 0.2]")

    status, tree, _ = run(Action(refused, name="reach"))
    assert status is FAILURE
    assert "cannot reach" in tree.notes()["0"]
    assert "cannot reach" in tree.summary()

    def out_of_time(c):
        raise ScriptTimeLimit("time limit of 60 simulated seconds reached")

    with pytest.raises(ScriptTimeLimit):
        run(Action(out_of_time))


def test_retry_tries_again_up_to_times_and_counts_the_attempts():
    calls = []

    def flaky(c):
        calls.append(c.time)
        return len(calls) == 3  # the third attempt works

    status, tree, sim = run(Retry(Action(flaky, name="grasp"), times=4, wait=0.5))
    assert status is SUCCESS and len(calls) == 3
    assert calls[1] - calls[0] >= 0.5  # waited between attempts
    assert any("attempt 1 of 4 failed" in line for line in tree.trace)
    assert any(s["notes"].get("0") == "attempt 2 of 4" for s in sim.shown)  # the viewer saw the second attempt


def test_retry_gives_up_after_the_last_attempt():
    calls = []
    status, _, _ = run(Retry(Action(lambda c: (calls.append(1), False)[1]), times=3))
    assert status is FAILURE and len(calls) == 3


def test_a_retried_sequence_starts_again_from_its_first_child():
    log = []
    attempts = []

    def go(c):
        log.append("go")

    def grasp(c):
        attempts.append(1)
        log.append("grasp")
        return len(attempts) > 1

    status, _, _ = run(Retry(Sequence(Action(go, name="go"), Action(grasp, name="grasp")), times=2))
    assert status is SUCCESS
    assert log == ["go", "grasp", "go", "grasp"]


def test_a_sequence_carries_on_from_the_child_that_was_running():
    entered = []
    status, _, sim = run(
        Sequence(Action(lambda c: entered.append("a")), Wait(1.0), Action(lambda c: entered.append("b")))
    )
    assert status is SUCCESS and entered == ["a", "b"]  # a was not run again while the wait ran
    assert sim.time >= 1.0


def test_a_reactive_sequence_checks_its_condition_every_tick():
    robot = FakeRobot(bump={"hit": False})
    sim = FakeSim()
    state = {"n": 0}

    def work(c):
        state["n"] += 1
        if state["n"] == 3:
            robot.sensors.readings["bump"]["hit"] = True  # something got in the way
        return RUNNING

    tree = Tree(Sequence(sensor_is("bump", "hit", "==", False), Action(work, name="work"), reactive=True))
    assert tree.run(sim, robot, max_seconds=5.0) is FAILURE
    assert state["n"] == 3


def test_parallel_all_any_and_a_number():
    def slow(seconds):
        return Wait(seconds)

    status, _, sim = run(Parallel(slow(0.3), slow(1.0), success="all"))
    assert status is SUCCESS and sim.time >= 1.0
    status, _, sim = run(Parallel(slow(0.3), slow(1.0), success="any"))
    assert status is SUCCESS and sim.time < 0.6
    status, _, _ = run(Parallel(slow(0.2), Action(lambda c: False), slow(0.4), success=2))
    assert status is SUCCESS
    status, _, _ = run(Parallel(Action(lambda c: False), slow(0.4), success="all"))
    assert status is FAILURE
    with pytest.raises(FlowError):
        Parallel(slow(1), success=3)


def test_parallel_stops_the_children_still_running_when_it_is_decided():
    long = Wait(10.0)
    status, tree, _ = run(Parallel(Wait(0.2), long, success="any"))
    assert status is SUCCESS
    assert tree.statuses() == "sr"[0] + "s" + "."  # the long wait was stopped, not left running


def test_repeat_inverter_and_timeout():
    count = []
    status, _, _ = run(Repeat(Action(lambda c: count.append(1)), times=3))
    assert status is SUCCESS and len(count) == 3
    assert run(Inverter(Action(lambda c: False)))[0] is SUCCESS
    assert run(Inverter(Action(lambda c: True)))[0] is FAILURE
    status, tree, sim = run(Timeout(Wait(5.0), seconds=1.0))
    assert status is FAILURE and 1.0 <= sim.time < 1.3
    assert tree.notes() == {"0": "over 1 s"}


def test_waitfor_succeeds_when_the_test_holds_and_fails_at_its_timeout():
    sim = FakeSim()
    tree = Tree(WaitFor(lambda c: c.time >= 0.5, timeout=2.0))
    assert tree.run(sim) is SUCCESS
    tree = Tree(WaitFor(lambda c: False, timeout=0.5))
    assert tree.run(FakeSim()) is FAILURE


def test_a_condition_on_a_sensor_reads_down_the_reading():
    robot = FakeRobot(lidar={"ahead": {"distance_m": 0.8}, "sectors": [1.0, 2.0]})
    assert run(sensor_is("lidar", "ahead.distance_m", ">", 0.5), robot=robot)[0] is SUCCESS
    assert run(sensor_is("lidar", "ahead.distance_m", "<", 0.5), robot=robot)[0] is FAILURE
    assert run(sensor_is("lidar", "sectors.1", ">=", 2.0), robot=robot)[0] is SUCCESS
    assert run(sensor_is("lidar", "nothing.here", ">", 0), robot=robot)[0] is FAILURE  # not there: failed, no error
    assert run(sensor_is("sonar", "range", ">", 0), robot=robot)[0] is FAILURE
    with pytest.raises(FlowError):
        sensor_is("lidar", "x", "~", 1)


def test_the_tree_is_shown_once_and_then_only_what_changed():
    status, tree, sim = run(Sequence(Condition(lambda c: True, name="ready"), Wait(0.3), name="job"))
    assert status is SUCCESS
    first = sim.shown[0]
    assert first["name"] == "t"
    assert [n["name"] for n in first["nodes"]] == ["job", "ready", "wait 0.3 s"]
    assert [n["parent"] for n in first["nodes"]] == [-1, 0, 0]
    assert all(s["nodes"] is None for s in sim.shown[1:])  # the structure goes once
    assert sim.shown[-1]["status"] == "sss"
    assert any(s["status"] == "rsr" for s in sim.shown)  # the wait showed as running
    assert len(sim.shown) < 8  # a change is sent, not every tick


def test_a_tree_runs_without_a_viewer_hook_and_gives_up_at_its_time_limit():
    class Bare:
        time = 0.0

        def step(self, seconds):
            self.time += seconds

    tree = Tree(Wait(100.0))
    assert tree.run(Bare(), max_seconds=1.0) is FAILURE
    assert "stopped after 1 s" in tree.summary()


def test_a_node_cannot_be_in_a_tree_twice():
    shared = Action(lambda c: True)
    with pytest.raises(FlowError):
        Tree(Sequence(shared, shared))
    with pytest.raises(FlowError):
        Tree(lambda c: True)


def _patrol():
    fsm = StateMachine("patrol", start="go")
    fsm.add(State("go", do=lambda c: c.bb.__setitem__("steps", c.bb.get("steps", 0) + 1)))
    fsm.add(State("turn", do=lambda c: c.bb.__setitem__("turned", c.bb.get("turned", 0) + 1)))
    fsm.add(State("done", result=SUCCESS))
    fsm.transition("go", "turn", when=lambda c: c.bb["steps"] % 3 == 0)
    fsm.transition("turn", "go", when=lambda c: c.bb["turned"] >= 2 and c.bb["steps"] < 7)
    fsm.transition("turn", "done", when=lambda c: c.bb["steps"] >= 7)
    return fsm


def test_a_state_machine_moves_between_states_and_ends_in_a_result_state():
    fsm = _patrol()
    status, tree, sim = run(fsm)
    assert status is SUCCESS
    entered = [line.split("patrol: ")[1] for line in tree.trace if "patrol: " in line]
    assert [name for name in entered if name not in ("success", "failure")] == [
        "go",
        "turn",
        "go",
        "turn",
        "go",
        "turn",
        "done",
    ]
    assert tree.statuses()[-1] == "s"  # the final state shows as done
    assert sim.shown[0]["nodes"][1]["kind"] == "state"


def test_state_enter_and_exit_run_once_and_a_state_can_hold_a_node():
    log = []
    fsm = StateMachine("m", start="a")
    fsm.add(State("a", do=Sequence(Action(lambda c: log.append("work")), Wait(0.2)), enter=lambda c: log.append("in")))
    fsm.add(State("b", result=SUCCESS))
    fsm.transition("a", "b", when=SUCCESS)  # the node in a finished well
    assert run(fsm)[0] is SUCCESS
    assert log == ["in", "work"]

    failing = StateMachine("m", start="a")
    failing.add(State("a", do=Action(lambda c: False), exit=lambda c: log.append("out")))
    failing.add(State("lost", result=FAILURE))
    failing.transition("a", "lost", when=FAILURE)
    assert run(failing)[0] is FAILURE
    assert log[-1] == "out"


def test_a_state_machine_in_a_tree_and_its_mistakes():
    fsm = StateMachine("m", start="only")
    fsm.add(State("only", do=lambda c: None))
    with pytest.raises(FlowError, match="no state 'nowhere'"):
        fsm.transition("only", "nowhere", when=lambda c: True)
    with pytest.raises(FlowError, match="already has"):
        fsm.add(State("only"))
    with pytest.raises(FlowError, match="starts in"):
        run(StateMachine("m", start="missing"))
    end = StateMachine("n", start="a", states=[State("a", result=SUCCESS)])
    with pytest.raises(FlowError, match="ends the machine"):
        end.transition("a", "a", when=lambda c: True)


# ---- a reactive composite stops what its earlier children preempt; blocking calls; failures of a condition ----


def test_a_reactive_sequence_halts_the_child_that_was_running_when_an_earlier_one_fails():
    sim = FakeSim()
    tree = Tree(
        Sequence(Condition(lambda c: c.time < 0.5, name="early"), Wait(10.0, name="long wait"), reactive=True),
        name="t",
    )
    assert tree.run(sim, None, max_seconds=5.0) is FAILURE
    assert tree.statuses() == "ff."  # the wait is not left marked as running under a finished tree
    assert "r" not in sim.shown[-1]["status"]
    # a node inside the preempted branch starts again from its beginning next time
    retry = Retry(Action(lambda c: False, name="try"), times=5, wait=1.0)
    seq = Sequence(Condition(lambda c: c.time < 0.35, name="early"), retry, reactive=True)
    assert Tree(seq).run(FakeSim(), None, max_seconds=5.0) is FAILURE
    assert retry.attempt == 1 and retry.status is None


def test_a_reactive_fallback_halts_the_later_child_when_an_earlier_one_succeeds():
    sim = FakeSim()
    tree = Tree(Fallback(Condition(lambda c: c.time >= 0.5, name="ready"), Wait(10.0, name="long wait"), reactive=True))
    assert tree.run(sim, None, max_seconds=5.0) is SUCCESS
    assert tree.statuses() == "ss."
    assert "r" not in sim.shown[-1]["status"]


def test_a_state_machine_in_a_preempted_branch_runs_its_exit():
    exits = []
    machine = StateMachine("m", start="a")
    machine.add(State("a", do=lambda c: None, exit=lambda c: exits.append(c.time)))
    machine.add(State("b", result=SUCCESS))
    machine.transition("a", "b", when=lambda c: False)
    tree = Tree(Sequence(Condition(lambda c: c.time < 0.25), machine, reactive=True))
    assert tree.run(FakeSim(), None, max_seconds=5.0) is FAILURE
    assert machine.current is None  # halted: it will start from "a" again
    assert len(exits) == 1 and exits[0] >= 0.25  # and its state left through its exit


def test_a_timeout_fails_a_blocking_child_that_overran_and_notes_how_long():
    sim = FakeSim()
    tree = Tree(Timeout(Action(lambda c: c.sim.step(10.0), name="block"), seconds=2.0))
    assert tree.run(sim, None) is FAILURE
    assert sim.time == pytest.approx(10.0, abs=0.2)  # a call that blocks cannot be cut short
    assert "took 10.0 s, over 2 s" in tree.notes()["0"]
    ok = Tree(Timeout(Action(lambda c: c.sim.step(1.0), name="block"), seconds=2.0))
    assert ok.run(FakeSim(), None) is SUCCESS
    waited = Tree(Timeout(Wait(1.0), seconds=1.0))  # the clock between ticks is not a blocking call
    assert waited.run(FakeSim(), None) is SUCCESS


def test_parallel_blocking_actions_run_one_after_the_other():
    sim = FakeSim()
    tree = Tree(Parallel(Action(lambda c: c.sim.step(5.0)), Action(lambda c: c.sim.step(5.0))))
    assert tree.run(sim, None) is SUCCESS
    assert sim.time == pytest.approx(10.0, abs=0.2)  # as the docs say: blocking calls do not overlap


def test_a_condition_that_the_playground_refuses_or_cannot_read_is_a_failed_test():
    def lidar(c):
        return c.robot.sensors["lidar"]["ahead"] > 0.5  # a robot without a lidar: KeyError

    def refused(c):
        raise ScriptError("no sensor called lidar")

    robot = FakeRobot()
    for fn in (lidar, refused):
        tree = Tree(Fallback(Condition(fn, name="path clear"), Action(lambda c: None, name="go round")))
        sim = FakeSim()
        assert tree.run(sim, robot) is SUCCESS
        assert tree.statuses() == "sfs"
        assert "path clear" in tree.summary()
    with pytest.raises(ScriptTimeLimit):
        Tree(Condition(lambda c: (_ for _ in ()).throw(ScriptTimeLimit("out of time")))).run(FakeSim(), None)


def test_an_error_in_a_node_leaves_no_node_marked_running_in_the_panel():
    sim = FakeSim()

    def boom(c):
        raise ZeroDivisionError("bug in the script")

    tree = Tree(Sequence(Action(lambda c: None), Action(boom, name="boom")))
    with pytest.raises(ZeroDivisionError):
        tree.run(sim, None)
    assert "r" not in sim.shown[-1]["status"]
    assert sim.shown[-1]["status"][1:] == "sf"  # the one that ran, and the one that was running when it broke


def test_a_tree_stopped_at_its_time_limit_shows_that_in_the_panel():
    sim = FakeSim()
    tree = Tree(Wait(100.0))
    assert tree.run(sim, None, max_seconds=1.0) is FAILURE
    assert sim.shown[-1]["status"] == "."


def test_a_big_tree_is_shown_cut_short_instead_of_failing_the_run():
    from robotforge.flow_board import MAX_NODES

    sim = FakeSim()
    tree = Tree(Sequence(*[Action(lambda c: None, name=f"step {i}") for i in range(MAX_NODES + 50)]))
    assert tree.run(sim, None) is SUCCESS
    first = sim.shown[0]
    assert len(first["nodes"]) == MAX_NODES and len(first["status"]) == MAX_NODES
    assert str(MAX_NODES) in first["name"]
