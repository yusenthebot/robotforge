"""The agent eval's checks (scripts/agent_eval.py) on sessions recorded with Codex.

tests/fixtures/agent_eval/<task>/ holds one session of the 2026-10-17 baseline run (commit 2f614f8, gpt-6.1-sol
at medium effort) and the worlds it left, trimmed by `agent_eval.py --fixture`: no images, shell output or texts
that no check reads, and the run folder's path written as RUN. The expected verdicts are the ones read by hand
from the full transcripts. Sessions made up here (_made_up) are what a check must not pass: each is a shortcut a
real session could take. The script itself needs Codex, a login and the network; these tests need none of them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "agent_eval"
SCRIPT = Path(__file__).parents[1] / "scripts" / "agent_eval.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("agent_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_eval"] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


# task -> (passed, the criteria that held), from reading each session by hand
BASELINE = {
    # Two servers started at once took the same viewer port; this one's died, so the session had no tools.
    "dog_grass": (False, {"a quadruped": False}),
    "arm_cube": (
        True,
        {"red_cube starts outside the bin": True, "an arm call handled it": True, "red_cube ended in a bin": True},
    ),
    "go2_turn": (True, None),
    # Asked to drive "a little", it drove 0.293 m; the prompt now asks for about half a meter.
    "urdf_rover": (False, {"a robot imported from my_rover.urdf": True, "moved >= 0.3 m": False}),
    "hexapod_build": (True, {"it is a hexapod": True}),
    "rover_rock": (True, {"a rover and a rock": True, "upright at the end": True}),
    # It opened lesson 6 and built the cart-pole, but never ran a controller.
    "learn_control": (False, {"opened the control lesson (lessons 6)": True, "ran a balance controller": False}),
    "fix_workshop": (True, {"green_cube starts where it was": True, "an arm call handled it": True}),
    # Run on the same commit after the task was added: open_world failed on the misspelled template, the agent
    # searched for the file (the error did not say where it is), fixed it and walked the dog 1.04 m.
    "fix_garden": (True, {"a robotforge call failed first": True, "then the dog walked >= 0.5 m": True}),
}


@pytest.mark.parametrize("task", list(BASELINE))
def test_the_checks_give_the_verdicts_read_from_the_sessions(ev, task):
    outcome = ev.score(ev.TASKS[task], FIXTURES / task)
    passed, criteria = BASELINE[task]
    assert outcome.passed is passed, outcome.criteria
    for name, ok in (criteria or {}).items():
        assert next(c["ok"] for c in outcome.criteria if c["name"] == name) is ok, outcome.criteria


def test_a_robot_driven_by_a_controller_is_followed_through_its_results(ev):
    """A later session (commit 3724596) drove its rover with run_controller(drive_to), stopped it and zeroed the
    wheels with set_controls and simulate: no move or robots result. The check first said no robot was reported;
    run_controller, observe and simulate say where it is, so it ended 0.42 m from the rock."""
    outcome = ev.score(ev.TASKS["rover_rock"], FIXTURES / "rover_rock_by_controller")
    assert outcome.passed, outcome.criteria
    assert "run_controller" in outcome.tools and "move" not in outcome.tools
    near = next(c for c in outcome.criteria if c["name"].startswith("ended within"))
    assert near["observed"] == "0.16 m from its outline (0.42 m from its center)"
    # observe's orientation sensor says it is upright; before, only move and robots did, and this read None.
    assert next(c for c in outcome.criteria if c["name"] == "upright at the end")["observed"] == "True"


def test_how_near_the_rock_is_measured_from_the_rovers_outline(ev):
    """A session on commit e48a93c built a 0.5 m rover and stopped it with its nose 0.38 m from the rock; measured
    from its center, 0.64 m, it failed where a 0.2 m rover with a wider gap would have passed."""
    outcome = ev.score(ev.TASKS["rover_rock"], FIXTURES / "rover_rock_long_rover")
    assert outcome.passed, outcome.criteria
    near = next(c for c in outcome.criteria if c["name"].startswith("ended within"))
    assert near["observed"] == "0.38 m from its outline (0.64 m from its center)"


def test_a_session_is_read_into_calls_tokens_and_messages(ev):
    lines = (FIXTURES / "rover_rock" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    session = ev.parse_events(["Reading additional input from stdin...", *lines])  # a log line is skipped
    ours = [c.tool for c in session.ours]
    assert ours[:4] == ["new_world", "list_templates", "add_robot", "open_viewer"] and len(ours) == 16
    assert ours.count("stop") == 3  # the redundant stops after timed moves that the move result now prevents
    assert session.usage["input_tokens"] == 336252 and session.usage["cached_input_tokens"] == 296832
    assert len(session.commands) == 1 and session.messages[-1].startswith("火星车")
    first_move = session.results("move")[0]
    assert first_move["distance_m"] == 0.002  # the 54 kg rover that stood still


def test_a_failed_call_counts_and_keeps_its_error_without_the_sdk_frame(ev):
    failed = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "server": "robotforge",
            "tool": "pick_and_place",
            "arguments": {"robot": "arm", "object": "cube", "target": "bin"},
            "result": {"content": [{"type": "text", "text": "Error executing tool pick_and_place: arm cannot reach"}]},
            "status": "failed",
        },
    }
    session = ev.parse_events([json.dumps(failed)])
    (call,) = session.ours
    assert not call.ok and call.text == "arm cannot reach" and call.data is None


def test_the_summary_keeps_no_paths_of_this_machine(ev, tmp_path):
    run = tmp_path / "run"
    where = str(run / "tasks" / "urdf_rover" / "work" / "my_rover.urdf")
    failed = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "server": "robotforge",
            "tool": "add_robot",
            "arguments": {"name": "r", "file": where},
            "result": {"content": [{"type": "text", "text": f"no file at {where} or {Path.home() / 'x.urdf'}"}]},
            "status": "failed",
        },
    }
    session = ev.parse_events([json.dumps(failed)])
    task_dir = run / "tasks" / "urdf_rover"
    outcome = ev.judge(ev.TASKS["urdf_rover"], session, task_dir / "rf_home", task_dir / "work", {})
    (failure,) = outcome.failures
    for text in failure.values():
        for path in (tmp_path, Path.home()):
            assert str(path) not in text and json.dumps(str(path))[1:-1] not in text, text
    assert failure["error"].startswith("no file at RUN") and "my_rover.urdf" in failure["args"]
    assert "~" in failure["error"] and "RUN" in failure["args"]


def test_a_robot_that_fell_at_the_end_does_not_pass(ev, tmp_path):
    """The same hexapod session, but its last move ends on its back and the observe after it sees it upside down."""
    import shutil

    shutil.copytree(FIXTURES / "hexapod_build", tmp_path / "hexapod_build")
    events = tmp_path / "hexapod_build" / "events.jsonl"
    text = events.read_text(encoding="utf-8")
    fell = text.replace('\\"heading_deg\\":-0.4,\\"upright\\":true', '\\"heading_deg\\":-0.4,\\"upright\\":false')
    fell = fell.replace("[1, -0.0066, -0.0039, -0.0033]", "[0.0066, 1, 0.0033, -0.0039]")  # turned over
    assert fell.count('upright\\":false') == 1 and "[0.0066, 1," in fell
    events.write_text(fell, encoding="utf-8")
    outcome = ev.score(ev.TASKS["hexapod_build"], tmp_path / "hexapod_build")
    assert not outcome.passed
    assert next(c for c in outcome.criteria if c["name"] == "upright at the end")["ok"] is False


# ---- made-up sessions: shortcuts a check must not pass ------------------------------------------------------


def _event(tool: str, args: dict, result: object, ok: bool = True) -> str:
    text = result if isinstance(result, str) else json.dumps(result)
    item = {
        "type": "mcp_tool_call",
        "server": "robotforge",
        "tool": tool,
        "arguments": args,
        "result": {"content": [{"type": "text", "text": text}]},
        "status": "completed" if ok else "failed",
    }
    return json.dumps({"type": "item.completed", "item": item})


def _made_up(tmp_path: Path, worlds: dict[str, str], *events: str) -> Path:
    """A task folder as a run leaves it: the event stream, meta.json and the worlds on disk."""
    task_dir = tmp_path / "run" / "tasks" / "made_up"
    for name, text in worlds.items():
        (task_dir / "rf_home" / "worlds" / name).mkdir(parents=True)
        (task_dir / "rf_home" / "worlds" / name / "world.yaml").write_text(text, encoding="utf-8")
    (task_dir / "work").mkdir(parents=True, exist_ok=True)
    (task_dir / "events.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
    (task_dir / "meta.json").write_text(json.dumps({"wall_seconds": 60.0, "exit_code": 0}), encoding="utf-8")
    return task_dir


ARM_WORLD = """\
name: cube
robots:
- name: arm
  template: arm
objects:
- name: red_cube
  shape: box
  size: [0.04, 0.04, 0.04]
  pos: {cube}
  color: red
- name: box
  shape: {shape}
  size: {size}
  pos: [0.05, 0.28]
"""


def _verdict(outcome) -> dict[str, bool]:
    return {c["name"]: c["ok"] for c in outcome.criteria}


def test_a_cube_made_inside_the_bin_was_not_put_there(ev, tmp_path):
    world = ARM_WORLD.format(cube="[0.05, 0.28]", shape="bin", size="[0.14, 0.14, 0.05]")
    observed = _event("observe", {}, {"time": 0, "free_bodies": {"red_cube": [0.05, 0.28, 0.02]}})
    outcome = ev.score(ev.TASKS["arm_cube"], _made_up(tmp_path, {"cube": world}, observed))
    assert not outcome.passed
    verdict = _verdict(outcome)
    assert verdict["red_cube ended in a bin"] is True  # it is in the bin: but nothing put it there
    assert verdict["red_cube starts outside the bin"] is False and verdict["an arm call handled it"] is False


def test_a_cube_left_on_top_of_a_solid_box_is_not_in_it(ev, tmp_path):
    world = ARM_WORLD.format(cube="[0.25, -0.12]", shape="box", size="[0.15, 0.15, 0.1]")
    placed = {"success": True, "object_position": [0.05, 0.28, 0.12], "object": "red_cube", "target": "box"}
    events = _event("pick_and_place", {"robot": "arm", "object": "red_cube", "target": "box"}, placed)
    outcome = ev.score(ev.TASKS["arm_cube"], _made_up(tmp_path, {"cube": world}, events))
    assert not outcome.passed and _verdict(outcome)["red_cube ended in a bin"] is False


def test_a_cube_moved_into_the_bin_by_editing_the_world_was_not_carried(ev, tmp_path):
    world = ev.WORKSHOP.replace("pos: [0.25, 0.1]", "pos: [1.2, -0.4]").replace("color: red", "color: blue")
    world = world.replace("shape: bin\n", "shape: bin\n  size: [0.2, 0.15, 0.08]\n")  # as an edit saves it
    events = [
        _event("open_world", {"name_or_path": "workshop"}, {"world": "workshop"}),
        _event("update_object", {"name": "ball", "color": "blue"}, {"world": "workshop"}),
        _event("update_object", {"name": "green_cube", "pos": [1.2, -0.4]}, {"world": "workshop"}),
        _event("observe", {}, {"time": 0, "free_bodies": {"green_cube": [1.2, -0.4, 0.02]}}),
    ]
    outcome = ev.score(ev.TASKS["fix_workshop"], _made_up(tmp_path, {"workshop": world}, *events))
    assert not outcome.passed
    verdict = _verdict(outcome)
    assert verdict["the ball is blue"] and verdict["green_cube ended in a bin"]
    assert verdict["green_cube starts where it was"] is False and verdict["an arm call handled it"] is False


ROCK_WORLD = """\
name: rock
robots:
- name: {name}
  template: {template}
objects:
- name: big_rock
  shape: rock
  size: [0.4, 0.3]
  pos: [2.0, 0.0]
"""


def _drove(robot: str, to: list[float], upright: bool) -> str:
    start = {"robot": robot, "position": [0, 0, 0.1], "heading_deg": 0, "upright": True}
    end = {"robot": robot, "position": to, "heading_deg": 0, "upright": upright}
    moved = {"from": start, "to": end, "distance_m": to[0], "turned_deg": 0, "stopped": True}
    return _event("move", {"robot": robot, "speed": 0.3, "seconds": 5}, moved)


def test_a_dog_that_walked_to_the_rock_is_not_a_rover(ev, tmp_path):
    world = ROCK_WORLD.format(name="dog", template="quadruped")
    outcome = ev.score(ev.TASKS["rover_rock"], _made_up(tmp_path, {"rock": world}, _drove("dog", [1.5, 0, 0.3], True)))
    assert not outcome.passed
    verdict = _verdict(outcome)
    assert verdict["a rover and a rock"] is False and verdict["ended within 0.5 m of it"] is True


@pytest.mark.parametrize(
    "then",
    [
        [],  # its last move ended upside down
        # observed afterwards: the free body has no upright of its own, and the orientation sensor says upside down
        [_event("observe", {}, {"free_bodies": {"rover/chassis": [1.5, 0, 0.1]}})],
        [
            _event(
                "observe",
                {},
                {"free_bodies": {"rover/chassis": [1.5, 0, 0.1]}, "sensors": {"rover/orientation": [0, 1, 0, 0]}},
            )
        ],
    ],
)
def test_a_rover_on_its_back_at_the_rock_does_not_pass(ev, tmp_path, then):
    world = ROCK_WORLD.format(name="rover", template="rover")
    events = [_drove("rover", [1.5, 0, 0.1], False), *then]
    outcome = ev.score(ev.TASKS["rover_rock"], _made_up(tmp_path, {"rock": world}, *events))
    assert not outcome.passed
    verdict = _verdict(outcome)
    assert verdict["a rover and a rock"] and verdict["ended within 0.5 m of it"] and not verdict["upright at the end"]


def _garden_walk(robot: str = "dog") -> list[str]:
    return [
        _event("open_world", {"name_or_path": "garden"}, "robot 'dog': unknown template 'quadrupd'", ok=False),
        _drove(robot, [0.9, 0, 0.3], True),
    ]


def test_the_garden_passes_only_once_the_dog_walks_in_the_repaired_world(ev, tmp_path):
    fixed = ev.GARDEN.replace("quadrupd", "quadruped")
    outcome = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "a", {"garden": fixed}, *_garden_walk()))
    assert outcome.passed, outcome.criteria
    assert outcome.failed_calls == 1 and outcome.failures[0]["tool"] == "open_world"
    # Left broken: it does not compile, whatever the session says it walked.
    broken = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "b", {"garden": ev.GARDEN}, *_garden_walk()))
    assert _verdict(broken)["garden opens with its dog a quadruped"] is False and not broken.passed
    # Walked first, then a failure: nothing happened after the mistake.
    later = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "c", {"garden": fixed}, *reversed(_garden_walk())))
    assert _verdict(later)["then the dog walked >= 0.5 m"] is False
    # A new world instead of the user's garden, with its ball lost.
    lost = fixed.replace("objects:", "objects: []\nold:")
    gone = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "d", {"garden": lost}, *_garden_walk()))
    assert _verdict(gone)["the ball is still there"] is False


# ---- scoring itself ---------------------------------------------------------------------------------------


def test_one_broken_session_is_scored_as_failed_and_keeps_its_reason(ev, tmp_path):
    task_dir = _made_up(tmp_path, {}, _event("robots", {}, {"robots": []}))
    (task_dir / "events.jsonl").write_bytes(
        b'{"type": "item.completed", "item": {"type": "agent_message", "text": "\xff\xfe"}}\n'
    )
    assert ev.score(ev.TASKS["dog_grass"], task_dir).final_message  # an invalid byte is replaced, not fatal
    (task_dir / "events.jsonl").unlink()
    outcome = ev.score_safely(ev.TASKS["dog_grass"], task_dir)
    assert not outcome.passed and outcome.criteria[0]["name"] == "check ran"
    assert (
        "FileNotFoundError" in outcome.criteria[0]["observed"] and str(tmp_path) not in outcome.criteria[0]["observed"]
    )


def test_what_a_check_observed_keeps_no_paths_of_this_machine(ev, tmp_path):
    task_dir = _made_up(tmp_path, {}, _event("robots", {}, {"robots": []}))
    task = ev.replace(ev.TASKS["dog_grass"], check=lambda e: [ev.Criterion("x", False, f"broken in {e.work}")])
    outcome = ev.score(task, task_dir)
    assert outcome.criteria[0]["observed"] == "broken in RUN/tasks/made_up/work"


def test_a_fixture_keeps_every_argument_but_the_parts_lists(ev):
    """The trimmer once wrote every build as {"parts": "(trimmed)"}, which misrecorded the update_robot call that
    wiped the hexapod's legs: it sent no parts at all."""
    session = ev.parse_events((FIXTURES / "hexapod_build" / "events.jsonl").read_text(encoding="utf-8").splitlines())
    wiped = next(c for c in session.ours if c.tool == "update_robot")
    assert set(wiped.args["build"]) == {"servo", "base"}
    added = next(c for c in session.ours if c.tool == "add_robot")
    assert added.args["build"]["parts"] == "(3 parts trimmed)" and added.args["build"]["base"]["shape"] == "box"
    assert ev._trim_args({"build": {"parts": [1, 2], "servo": "x"}}) == {
        "build": {"parts": "(2 parts trimmed)", "servo": "x"}
    }


def test_a_session_that_ran_out_of_time_does_not_pass(ev, tmp_path):
    import shutil

    shutil.copytree(FIXTURES / "go2_turn", tmp_path / "go2_turn")
    (tmp_path / "go2_turn" / "meta.json").write_text(json.dumps({"timed_out": True}), encoding="utf-8")
    outcome = ev.score(ev.TASKS["go2_turn"], tmp_path / "go2_turn")
    assert all(c["ok"] for c in outcome.criteria) and not outcome.passed


@pytest.mark.parametrize(
    ("color", "hue", "looks"),
    [
        ("blue", "blue", True),
        ("#3366ff", "blue", True),
        ([0.92, 0.3, 0.28], "red", True),
        ("green", "blue", False),
        ("gray", "blue", False),  # not saturated enough to be any hue
        ("not a color", "red", False),
    ],
)
def test_colors_are_judged_by_hue(ev, color, hue, looks):
    assert ev.looks(color, hue) is looks


def test_inside_a_container_is_within_its_footprint_and_not_above_its_rim(ev):
    bin_ = {"shape": "bin", "pos": [0.25, -0.18], "size": [0.2, 0.15, 0.08], "yaw": 0}
    assert ev.inside([0.253, -0.18, 0.025], bin_)
    assert not ev.inside([0.253, -0.18, 0.3], bin_)  # held above it
    assert not ev.inside([0.4, -0.18, 0.025], bin_)
    turned = {**bin_, "yaw": 90}  # the long side now runs along y
    assert ev.inside([0.25, -0.26, 0.025], turned) and not ev.inside([0.34, -0.18, 0.025], turned)


def test_reports_compare_and_merge_runs(ev):
    def run(label, **verdicts):
        tasks = [
            {
                "task": t,
                "passed": ok,
                "tool_calls": 5,
                "failed_calls": 0,
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "wall_seconds": 60.0,
            }
            for t, ok in verdicts.items()
        ]
        return {"label": label, "commit": label, "tasks": tasks}

    before = ev.merge([run("base", a=False, b=False), run("again", b=True)])
    assert [t["passed"] for t in before["tasks"]] == [False, True] and before["label"] == "base+again"
    text = ev.compare(before, run("after", a=True, b=True))
    assert "fail -> pass" in text and "pass -> pass" in text


def test_a_transcript_shows_calls_and_what_the_agent_said(ev):
    text = ev.transcript(FIXTURES / "learn_control")
    assert "TOOL robotforge.lessons" in text and "SAY " in text and "USAGE" in text
