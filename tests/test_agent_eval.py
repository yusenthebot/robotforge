"""The agent eval's checks (scripts/agent_eval.py) on sessions recorded with Codex.

tests/fixtures/agent_eval/<task>/ holds one session of the 2026-10-17 baseline run (commit 2f614f8, gpt-6.1-sol
at medium effort) and the worlds it left, trimmed by `agent_eval.py --fixture`: no images, shell output or texts
that no check reads, and the run folder's path written as RUN. The other folders are sessions on later commits,
named where they are used. The expected verdicts are the ones read by hand
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
MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"  # offline stand-ins for the Go2 and the PiPER
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
    "urdf_rover": (False, {"a robot imported from my_rover.urdf": True, "went >= 0.3 m the way it was sent": False}),
    "hexapod_build": (True, {"it is a hexapod": True}),
    "rover_rock": (True, {"a rover and a rock": True, "upright at the end": True}),
    # It opened lesson 6 and built the cart-pole, but never ran a controller.
    "learn_control": (False, {"opened the control lesson (lessons 6)": True, "ran a balance controller live": False}),
    "fix_workshop": (True, {"green_cube starts where it was": True, "an arm call handled it": True}),
    # Run on the same commit after the task was added: open_world failed on the misspelled template, the agent
    # searched for the file (the error did not say where it is), fixed it and walked the dog 1.04 m.
    "fix_garden": (True, {"the dog walked >= 0.5 m forward in the garden": True}),
    # On 15fddbb: lesson 6, then only compare_controllers, which runs off-screen; the observe after it shows time 0.
    "learn_control_table_only": (False, {"ran a balance controller live": False}),
    # On f281b69: compare_controllers, then run_controller(pd) live with a push at 3 s.
    "learn_control_live": (True, {"ran a balance controller live": True}),
    # On ae450e3: a hexapod on xl430 servos sent forwards at 0.12 m/s for 5 s went 0.21 m backwards; counted as
    # distance it passed, and the agent told the user it had walked.
    "hexapod_backwards": (False, {"it is a hexapod": True, "went >= 0.2 m the way it was sent": False}),
    # On main at c2e37cd, the tasks for going somewhere, the gallery and the camera, all four done:
    # the starter rover from [-5.5, 2.6] to a waypoint at the ramp's foot, then up onto the dock;
    "warehouse_dock": (True, {"started >= 3.0 m from the dock's top": True, "ended on the dock's top": True}),
    # the centaur design in the park, 2.8 m to bench_1;
    "centaur_bench": (True, {"a centaur: four legs that walk and two arms": True}),
    # the wrist camera saw no red (it looks down beside the base), the top view found block_2, then the pick;
    "find_red": (True, {"found before an arm handled it": True, "the other cubes are still out of the bin": True}),
    # the go2_arm design put at [-2, 0] by the agent, 0.72 m to the island.
    "go2_arm_kitchen": (True, {"a Go2 carrying an arm": True, "not next to the island at the start": True}),
    # On 9bf99c0, the tasks that are not lessons. A 64-ray lidar car driven 0.87 m into the maze: observe listed 40
    # sensors, 37 of them its rays, and the script the agent wrote to read them all got the same 37. Its answer,
    # 0.43 m, happens to be right (all 64 rays recomputed at that pose: ray 16, 0.43 m), but nothing it read could
    # say so.
    "lidar_maze": (
        False,
        {
            "a robot carrying a lidar, in a maze": True,
            "started outside the maze (at its entrance)": True,
            "read its whole lidar inside the maze, a wall in sight": False,
        },
    ),
}
TASK_OF = {
    "learn_control_table_only": "learn_control",
    "learn_control_live": "learn_control",
    "hexapod_backwards": "hexapod_build",
}


@pytest.fixture(autouse=True)
def offline_menagerie(monkeypatch):
    """The Go2 and the PiPER a world mounts compile from the offline stand-ins, never a download."""
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))


@pytest.mark.parametrize("task", list(BASELINE))
def test_the_checks_give_the_verdicts_read_from_the_sessions(ev, task):
    outcome = ev.score(ev.TASKS[TASK_OF.get(task, task)], FIXTURES / task)
    passed, criteria = BASELINE[task]
    assert outcome.passed is passed, outcome.criteria
    for name, ok in (criteria or {}).items():
        assert next(c["ok"] for c in outcome.criteria if c["name"] == name) is ok, outcome.criteria


def test_a_live_balance_run_says_whether_it_held(ev):
    outcome = ev.score(ev.TASKS["learn_control"], FIXTURES / "learn_control_live")
    ran = next(c for c in outcome.criteria if c["name"] == "ran a balance controller live")
    assert ran["observed"] == "1 run_controller (success [True]), 1 compare_controllers (off-screen)"


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


def _opened(world: str) -> str:
    """What open_world (or new_world, or any edit) answers: the world now open. Where things went is read from the
    calls made while their world was open."""
    return _event("open_world", {"name_or_path": world}, {"world": world, "world_dir": f"RUN/rf_home/worlds/{world}"})


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
    outcome = ev.score(ev.TASKS["arm_cube"], _made_up(tmp_path, {"cube": world}, _opened("cube"), observed))
    assert not outcome.passed
    verdict = _verdict(outcome)
    assert verdict["red_cube ended in a bin"] is True  # it is in the bin: but nothing put it there
    assert verdict["red_cube starts outside the bin"] is False and verdict["an arm call handled it"] is False


def test_a_cube_left_on_top_of_a_solid_box_is_not_in_it(ev, tmp_path):
    world = ARM_WORLD.format(cube="[0.25, -0.12]", shape="box", size="[0.15, 0.15, 0.1]")
    placed = {"success": True, "object_position": [0.05, 0.28, 0.12], "object": "red_cube", "target": "box"}
    events = _event("pick_and_place", {"robot": "arm", "object": "red_cube", "target": "box"}, placed)
    outcome = ev.score(ev.TASKS["arm_cube"], _made_up(tmp_path, {"cube": world}, _opened("cube"), events))
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


def test_progress_is_measured_the_way_a_move_sent_the_robot(ev):
    def moved(speed, turn, to, heading=0.0):
        start = {"robot": "dog", "position": [0, 0, 0.3], "heading_deg": heading, "upright": True}
        end = {"robot": "dog", "position": to, "heading_deg": heading, "upright": True}
        result = {"from": start, "to": end, "distance_m": abs(to[0]), "turned_deg": 0, "stopped": True}
        return _event("move", {"robot": "dog", "speed": speed, "turn": turn, "seconds": 4}, result)

    def went(*events):
        return ev.progress(ev.parse_events(list(events)), "dog")

    assert went(moved(0.3, 0, [1.0, 0, 0.3])) == pytest.approx(1.0)
    assert went(moved(0.3, 0, [-0.4, 0, 0.3])) == pytest.approx(-0.4)  # sent forwards, went backwards
    assert went(moved(-0.3, 0, [-1.0, 0, 0.3])) == pytest.approx(1.0)  # sent backwards, and went
    assert went(moved(0.3, 0, [0, 1.0, 0.3], heading=90)) == pytest.approx(1.0)  # facing +y
    # A quarter arc to the left: the chord points 45 degrees left, and the end lies along it.
    arc = 0.3 / (3.14159 / 8) * 2**0.5
    assert went(moved(0.3, 3.14159 / 8, [arc / 2**0.5, arc / 2**0.5, 0.3])) == pytest.approx(arc, rel=1e-3)


def test_a_dog_that_walked_to_the_rock_is_not_a_rover(ev, tmp_path):
    world = ROCK_WORLD.format(name="dog", template="quadruped")
    events = [_opened("rock"), _drove("dog", [1.5, 0, 0.3], True)]
    outcome = ev.score(ev.TASKS["rover_rock"], _made_up(tmp_path, {"rock": world}, *events))
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
    events = [_opened("rock"), _drove("rover", [1.5, 0, 0.1], False), *then]
    outcome = ev.score(ev.TASKS["rover_rock"], _made_up(tmp_path, {"rock": world}, *events))
    assert not outcome.passed
    verdict = _verdict(outcome)
    assert verdict["a rover and a rock"] and verdict["ended within 0.5 m of it"] and not verdict["upright at the end"]


def _garden_walk(robot: str = "dog") -> list[str]:
    """open_world fails on the misspelled template, the file is fixed in the shell, the garden opens, the dog walks."""
    return [
        _event("open_world", {"name_or_path": "garden"}, "robot 'dog': unknown template 'quadrupd'", ok=False),
        _opened("garden"),
        _drove(robot, [0.9, 0, 0.3], True),
    ]


def test_the_garden_passes_once_its_dog_walks_in_the_repaired_world(ev, tmp_path):
    fixed = ev.GARDEN.replace("quadrupd", "quadruped")
    outcome = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "a", {"garden": fixed}, *_garden_walk()))
    assert outcome.passed, outcome.criteria
    assert outcome.failed_calls == 1 and outcome.failures[0]["tool"] == "open_world"
    # Left broken: it does not compile, whatever the session says it walked.
    broken = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "b", {"garden": ev.GARDEN}, *_garden_walk()))
    assert _verdict(broken)["garden opens with its dog a quadruped"] is False and not broken.passed
    # Walked before the garden opened: in whatever world was open then, not the garden.
    walk = _garden_walk()
    early = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "c", {"garden": fixed}, walk[0], walk[2], walk[1]))
    assert _verdict(early)["the dog walked >= 0.5 m forward in the garden"] is False
    # A new world instead of the user's garden, with its ball lost.
    lost = fixed.replace("objects:", "objects: []\nold:")
    gone = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path / "d", {"garden": lost}, *_garden_walk()))
    assert _verdict(gone)["the ball is still there"] is False


def test_the_garden_fixed_before_it_ever_failed_passes(ev, tmp_path):
    """The best session reads the file, fixes the template and then opens the garden: no call fails. A failed
    call was a pass condition, so this one failed on three criteria (review)."""
    fixed = ev.GARDEN.replace("quadrupd", "quadruped")
    shell = json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "cat world.yaml"}})
    events = [shell, _opened("garden"), _drove("dog", [1.0, 0, 0.3], True)]
    outcome = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path, {"garden": fixed}, *events))
    assert outcome.passed and outcome.failed_calls == 0, outcome.criteria


def test_a_dog_walked_in_another_world_is_not_the_gardens_dog(ev, tmp_path):
    """open_world fails, a dog of the same name walks 1 m in a new world, and the garden file is fixed in the shell
    but never opened: it passed every criterion, matched by the robot's name alone (review)."""
    fixed = ev.GARDEN.replace("quadrupd", "quadruped")
    park = "name: park\nfloor:\n  style: checker\nrobots:\n- name: dog\n  template: quadruped\n"
    events = [_garden_walk()[0], _opened("park"), _drove("dog", [1.0, 0, 0.3], True)]
    outcome = ev.score(ev.TASKS["fix_garden"], _made_up(tmp_path, {"garden": fixed, "park": park}, *events))
    assert not outcome.passed and _verdict(outcome)["the dog walked >= 0.5 m forward in the garden"] is False


def test_a_dog_on_grass_must_walk_on_the_grass(ev, tmp_path):
    """A dog on a grass world that never moved, and a dog of the same name that walked 3 m in a checker world:
    dog_grass passed on the name alone (review)."""
    lawn = "name: lawn\nfloor:\n  style: grass\nrobots:\n- name: dog\n  template: quadruped\n"
    lab = "name: lab\nfloor:\n  style: checker\nrobots:\n- name: dog\n  template: quadruped\n"
    events = [_opened("lab"), _drove("dog", [3.0, 0, 0.3], True), _opened("lawn")]
    outcome = ev.score(ev.TASKS["dog_grass"], _made_up(tmp_path, {"lawn": lawn, "lab": lab}, *events))
    assert not outcome.passed, outcome.criteria


def test_each_call_is_in_the_world_open_when_it_ran(ev):
    started = _event("new_world", {"name": "My Park"}, {"world": "my_park", "world_dir": "C:\\rf\\worlds\\my_park"})
    session = ev.parse_events(
        [
            _event("robots", {}, {"robots": []}),  # the demo, before any world
            started,
            _event("move", {"robot": "dog"}, {"distance_m": 1.0}),
            _event("open_world", {"name_or_path": "garden"}, "unknown template", ok=False),  # still my_park
            _event("observe", {}, {"time": 1.0}),
            _event("open_model", {"path": "x.urdf"}, {"model": "x.urdf"}),  # a model, not a saved world
            _event("simulate", {"seconds": 1}, {"moved": {}}),
            _event("add_object", {"name": "box"}, {"world": "lab"}),  # an edit names its world too
        ]
    )
    assert session.worlds == (None, "my_park", "my_park", "my_park", "my_park", None, None, "lab")
    assert [c.tool for c in session.in_world("my_park").calls] == ["new_world", "move", "open_world", "observe"]


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


def test_a_results_file_is_never_written_over(ev, tmp_path):
    """A rerun of two tasks on the same commit and day replaced the full run's results file (review)."""
    full = {"started": "20261017-031624", "label": "head", "tasks": [{"task": t} for t in ev.TASKS]}
    some = {"started": "20261017-032953", "label": "head", "tasks": [{"task": "arm_cube"}, {"task": "fix_workshop"}]}
    paths = [ev.save(full, tmp_path), ev.save(some, tmp_path), ev.save(full, tmp_path), ev.save(full, tmp_path)]
    assert [p.name for p in paths] == [
        "20261017-head.json",
        "20261017-head-arm_cube+fix_workshop.json",
        "20261017-head-031624.json",
        "20261017-head-031624-2.json",
    ]
    assert len(json.loads(paths[0].read_text(encoding="utf-8"))["tasks"]) == len(ev.TASKS)
    named = ev.save(some, tmp_path / "20261017-head.json")  # a file named on purpose is written over
    assert named == paths[0] and len(json.loads(named.read_text(encoding="utf-8"))["tasks"]) == 2


def test_compare_says_when_a_task_was_asked_differently(ev):
    def run(label, prompt):
        task = {"task": "urdf_rover", "passed": True, "tool_calls": 5, "failed_calls": 0, "input_tokens": 1000}
        task.update(cached_input_tokens=500, wall_seconds=60.0, prompt=prompt)
        return {"label": label, "commit": label, "tasks": [task]}

    same = ev.compare(run("a", "drive it about half a meter"), run("b", "drive it about half a meter"))
    assert "prompts differ" not in same
    text = ev.compare(run("a", "drive it forward a little"), run("b", "drive it about half a meter"))
    assert "urdf_rover: the prompts differ" in text and "a little" in text and "half a meter" in text
    assert "(not recorded)" in ev.compare(run("a", None), run("b", "drive it about half a meter"))


def test_codex_errors_on_stderr_count_as_problems(ev, tmp_path):
    task_dir = _made_up(tmp_path, {}, _event("robots", {}, {"robots": []}))
    (task_dir / "stderr.txt").write_text(
        "Reading additional input from stdin...\n2026-10-16T19:16:43Z ERROR codex_skills_extension::host_service: "
        "failed to install system skills: io error\n",
        encoding="utf-8",
    )
    outcome = ev.score(ev.TASKS["dog_grass"], task_dir)
    assert outcome.problems == ["codex_skills_extension::host_service: failed to install system skills: io error"]
    assert ev.summarize([outcome])["sessions_with_problems"] == 1
    # Listed, but not counted: Codex keeps the configured model when its model list does not refresh.
    (task_dir / "stderr.txt").write_text(
        "2026-10-16T17:48:39Z ERROR codex_models_manager::manager: failed to refresh available models: request "
        "timed out\n",
        encoding="utf-8",
    )
    outcome = ev.score(ev.TASKS["dog_grass"], task_dir)
    assert len(outcome.problems) == 1 and ev.summarize([outcome])["sessions_with_problems"] == 0


def test_a_fixture_keeps_which_world_each_result_is_in(ev):
    long = json.dumps({"world": "garden", "world_dir": "RUN/w/garden", "robots": ["dog"], "actuators": ["x"] * 99})
    assert json.loads(ev._trim_result("open_world", long)) == {
        "world": "garden",
        "world_dir": "RUN/w/garden",
        "robots": ["dog"],
    }
    assert json.loads(ev._trim_result("open_model", json.dumps({"model": "a.urdf", "joints": [1] * 99}))) == {
        "model": "a.urdf"
    }


# ---- going somewhere, the gallery and the camera (main at c2e37cd) -------------------------------------------


def _copy(tmp_path: Path, fixture: str) -> Path:
    import shutil

    return Path(shutil.copytree(FIXTURES / fixture, tmp_path / "run" / "tasks" / fixture))


def _edit_world(task_dir: Path, world: str, change) -> None:
    """Change a recorded world as a session might have left it instead (change takes and edits its YAML data)."""
    import yaml

    path = task_dir / "rf_home" / "worlds" / world / "world.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _robot(data: dict, name: str) -> dict:
    return next(r for r in data["robots"] if r["name"] == name)


def test_a_ramps_landing_and_the_gaps_seen_from_above(ev):
    dock = {"type": "ramp", "pos": [5.0, -1.4], "yaw": 0, "angle": 8, "length": 2.1, "landing": 2.0, "width": 2.4}
    xs, ys = zip(*ev.landing({**dock, "direction": "up"}), strict=True)
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((5.05, 7.05, -2.6, -0.2))
    xs, _ = zip(*ev.landing({**dock, "direction": "up_down"}), strict=True)  # the landing between two slopes
    assert (min(xs), max(xs)) == pytest.approx((5.0 - 3.1 + 2.1, 5.0 - 3.1 + 4.1))
    xs, _ = zip(*ev.landing({**dock, "direction": "down", "yaw": 180}), strict=True)  # first, then turned round
    assert (min(xs), max(xs)) == pytest.approx((5.05, 7.05))
    assert ev.landing({"type": "stairs", "pos": [0, 0]}) is None
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert ev.polygon_gap(square, [(2, 0), (3, 0), (3, 1), (2, 1)]) == pytest.approx(1.0)
    assert ev.polygon_gap(square, [(1.5, 1.5)]) == pytest.approx(0.5**0.5)  # corner to corner
    assert ev.polygon_gap(square, [(0.5, 0.5)]) == 0.0  # inside
    turned = [(1.5, 0.5), (2.0, 1.0), (1.5, 1.5), (1.0, 1.0)]  # a diamond touching the square's side
    assert ev.polygon_gap(square, turned) == 0.0
    # A robot 0.4 m long facing a counter's side from 1.0 m away (its middle), its nose 0.2 m off it.
    counter = {"shape": "counter", "pos": [0.0, 0.0], "size": [1.4, 0.8, 0.9]}
    assert ev._gap_to(-1.0, 0.0, 0.0, (-0.3, 0.1, -0.15, 0.15), counter) == pytest.approx(0.2)


def test_a_trip_counts_as_sent_toward_its_destination(ev):
    """move(to=...) plans its own way, turns and all: the way it was sent is toward the destination, not the way it
    faced when it set off (a dog facing +x sent to a point behind it went the right way)."""
    start = {"robot": "dog", "position": [0, 0, 0.3], "heading_deg": 0, "upright": True}
    end = {"robot": "dog", "position": [-1.0, 0.0, 0.3], "heading_deg": 180, "upright": True}
    result = {"arrived": True, "destination": {"point": [-1.0, 0.0]}, "distance_m": 1.1, "from": start, "to": end}
    session = ev.parse_events([_event("move", {"robot": "dog", "to": [-1, 0]}, result)])
    assert ev.progress(session, "dog") == pytest.approx(1.0)


def test_the_warehouse_dock_is_measured_from_the_saved_start_to_the_top(ev, tmp_path):
    outcome = ev.score(ev.TASKS["warehouse_dock"], FIXTURES / "warehouse_dock")
    assert _verdict(outcome)["ended on the dock's top"] and outcome.tools.count("move") == 2
    far = next(c for c in outcome.criteria if c["name"].startswith("started >="))
    assert far["observed"] == "10.92 m"  # from [-5.5, 2.6] to the landing's corner at [5.05, -0.2]
    # Put on the dock by hand (its saved place moved there): no trip across the warehouse.
    teleported = _copy(tmp_path / "a", "warehouse_dock")
    _edit_world(teleported, "warehouse", lambda d: _robot(d, "rover").update(pos=[6.0, -1.4]))
    assert _verdict(ev.score(ev.TASKS["warehouse_dock"], teleported))["started >= 3.0 m from the dock's top"] is False
    # Stopped on the ramp, short of the top (as the trip and the observe after it report it).
    short = _copy(tmp_path / "b", "warehouse_dock")
    events = short / "events.jsonl"
    text = events.read_text(encoding="utf-8")
    assert text.count("5.977,") == 1 and text.count("5.9768,") == 1
    events.write_text(text.replace("5.977,", "4.2,").replace("5.9768,", "4.2,"), encoding="utf-8")
    verdict = _verdict(ev.score(ev.TASKS["warehouse_dock"], short))
    assert verdict["ended on the dock's top"] is False and verdict["started >= 3.0 m from the dock's top"]


def test_a_centaur_must_be_one_and_must_walk_to_the_bench(ev, tmp_path):
    outcome = ev.score(ev.TASKS["centaur_bench"], FIXTURES / "centaur_bench")
    assert "centaur_park/centaur (quadruped, arms ['centaur/left_hand', 'centaur/right_hand'])" in str(outcome.criteria)
    # Saved beside the bench it ended at: it walked, but not to the bench.
    beside = _copy(tmp_path / "a", "centaur_bench")
    _edit_world(beside, "centaur_park", lambda d: _robot(d, "centaur").update(pos=[-2.8, -0.4], yaw=180))
    verdict = _verdict(ev.score(ev.TASKS["centaur_bench"], beside))
    assert verdict["not next to a bench at the start"] is False and verdict["ended within 0.5 m of a bench"]
    # A dog from parts with no arms: not a centaur, though it walked there.
    armless = _copy(tmp_path / "b", "centaur_bench")

    def no_hands(data):
        build = _robot(data, "centaur")["build"]
        build["parts"] = [p for p in build["parts"] if not str(p.get("name", "")).endswith("hand")]

    _edit_world(armless, "centaur_park", no_hands)
    verdict = _verdict(ev.score(ev.TASKS["centaur_bench"], armless))
    assert verdict["a centaur: four legs that walk and two arms"] is False and verdict["ended within 0.5 m of a bench"]


def test_the_red_cube_must_be_found_by_the_camera_before_it_is_picked(ev, tmp_path):
    outcome = ev.score(ev.TASKS["find_red"], FIXTURES / "find_red")
    found = next(c for c in outcome.criteria if c["name"].startswith("the camera found"))
    assert found["observed"] == "3 observe(find=...): red, red, red; saw ['block_2']"
    lines = (FIXTURES / "find_red" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    looks = [i for i, line in enumerate(lines) if '"find": "red"' in line and '"item.completed"' in line]
    picked = next(i for i, line in enumerate(lines) if '"tool": "pick_and_place"' in line)
    # Picked without looking: the world file says which cube is red, but the user asked for the camera.
    blind = _copy(tmp_path / "a", "find_red")
    (blind / "events.jsonl").write_text("\n".join(x for i, x in enumerate(lines) if i not in looks), encoding="utf-8")
    verdict = _verdict(ev.score(ev.TASKS["find_red"], blind))
    assert verdict["the camera found block_2, the red cube"] is False and verdict["block_2 ended in a bin"]
    # Looked only after picking it (the last observe saw it in the bin).
    late = _copy(tmp_path / "b", "find_red")
    kept = [x for i, x in enumerate(lines) if i not in looks or i > picked]
    (late / "events.jsonl").write_text("\n".join(kept), encoding="utf-8")
    verdict = _verdict(ev.score(ev.TASKS["find_red"], late))
    assert verdict["the camera found block_2, the red cube"] and verdict["found before an arm handled it"] is False


def test_the_wrong_cube_in_the_bin_does_not_pass(ev, tmp_path):
    seen = {"found": {"view": "top", "color": "red", "objects": [{"n": 1, "object": "block_2"}]}, "time": 0}
    placed = {"success": True, "object_position": [-0.05, 0.3, 0.025], "object": "block_1", "target": "bin"}
    events = [
        _opened("desk"),
        _event("observe", {"view": "top", "find": "red"}, seen),
        _event("pick_and_place", {"robot": "arm", "object": "block_1", "target": "bin"}, placed),
    ]
    outcome = ev.score(ev.TASKS["find_red"], _made_up(tmp_path, {"desk": ev.DESK}, *events))
    verdict = _verdict(outcome)
    assert not outcome.passed and verdict["the camera found block_2, the red cube"]
    assert verdict["block_2 ended in a bin"] is False and verdict["the other cubes are still out of the bin"] is False


def test_a_go2_saved_inside_the_island_made_no_trip_to_it(ev, tmp_path):
    """Added without a place after load_scene, a robot stands at [0, 0], inside the kitchen's island: pushed out,
    it is next to the island at once."""
    inside = _copy(tmp_path / "a", "go2_arm_kitchen")
    _edit_world(inside, "go2_arm_kitchen", lambda d: _robot(d, "go2_arm").update(pos=[0.0, 0.0]))
    verdict = _verdict(ev.score(ev.TASKS["go2_arm_kitchen"], inside))
    assert verdict["not next to the island at the start"] is False and verdict["a Go2 carrying an arm"]
    # The plain Go2 from the library carries no arm.
    plain = _copy(tmp_path / "b", "go2_arm_kitchen")
    _edit_world(plain, "go2_arm_kitchen", lambda d: _robot(d, "go2_arm").update(build=None, library="go2"))
    assert _verdict(ev.score(ev.TASKS["go2_arm_kitchen"], plain))["a Go2 carrying an arm"] is False


def test_a_fixture_keeps_what_the_camera_found(ev):
    found = {"view": "top", "color": "red", "how": "x" * 400, "objects": [{"n": 1, "object": "block_2", "pixels": 9}]}
    trimmed = json.loads(ev._trim_result("observe", json.dumps({"time": 1, "joints": {"a": 1}, "found": found})))
    assert trimmed == {"time": 1, "found": {"view": "top", "color": "red", "objects": [{"n": 1, "object": "block_2"}]}}


# ---- the tasks that are not lessons: a lidar in the maze, a video of a robot from parts (9bf99c0) ---------------


def _said(text: str) -> str:
    return json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}})


MAZE_WORLD = FIXTURES / "lidar_maze" / "rf_home" / "worlds" / "lidar_maze" / "world.yaml"


def _lidar_seen(where: list[float], nearest: float, rays: int = 64) -> str:
    """An observe as robotforge answers it now: the whole lidar under "lidars"."""
    ranges = [round(nearest + 0.01 * abs(i - 16), 3) for i in range(rays)]
    whole = {"rays": rays, "range_m": 5.0, "hits": rays, "nearest_m": nearest, "nearest_side": "left"}
    state = {
        "time": 6,
        "free_bodies": {"lidar_car/base": [*where, 0.06]},
        "lidars": {"lidar_car/base_lidar": {**whole, "ranges_m": ranges}},
    }
    return _event("observe", {"view": "top"}, state)


def _maze_session(tmp_path: Path, *events: str, world: str | None = None) -> Path:
    text = world if world is not None else MAZE_WORLD.read_text(encoding="utf-8")
    return _made_up(tmp_path, {"lidar_maze": text}, _opened("lidar_maze"), *events)


def test_the_answer_is_held_to_the_whole_lidar_read_inside_the_maze(ev, tmp_path):
    inside = [-2.0, -2.13]
    good = _maze_session(tmp_path / "a", _lidar_seen(inside, 0.43), _said("最近的墙在左边，约 43 厘米。"))
    assert ev.score(ev.TASKS["lidar_maze"], good).passed
    wrong = _maze_session(tmp_path / "b", _lidar_seen(inside, 0.43), _said("最近的墙大约 0.9 米远。"))
    verdict = _verdict(ev.score(ev.TASKS["lidar_maze"], wrong))
    assert verdict["read its whole lidar inside the maze, a wall in sight"]
    assert verdict["the answer says how far the nearest wall is"] is False
    # Read at the start mark, outside the maze: the car never drove in.
    outside = _maze_session(tmp_path / "c", _lidar_seen([-2.0, -3.0], 0.5), _said("最近的墙 0.5 米。"))
    assert (
        _verdict(ev.score(ev.TASKS["lidar_maze"], outside))["read its whole lidar inside the maze, a wall in sight"]
        is False
    )


def test_a_lidar_car_saved_inside_the_maze_did_not_start_at_its_entrance(ev, tmp_path):
    import yaml

    data = yaml.safe_load(MAZE_WORLD.read_text(encoding="utf-8"))
    _robot(data, "lidar_car").update(pos=[-2.0, -2.13])
    placed = _maze_session(tmp_path, _lidar_seen([-2.0, -2.13], 0.43), _said("0.43 m"), world=yaml.safe_dump(data))
    verdict = _verdict(ev.score(ev.TASKS["lidar_maze"], placed))
    assert verdict["started outside the maze (at its entrance)"] is False
    assert verdict["the answer says how far the nearest wall is"]


def test_the_maze_starter_rover_carries_no_lidar(ev, tmp_path):
    import yaml

    data = yaml.safe_load(MAZE_WORLD.read_text(encoding="utf-8"))
    data["robots"] = [{"name": "lidar_car", "template": "rover", "pos": [-2.0, -3.0], "yaw": 90}]
    plain = _maze_session(tmp_path, _lidar_seen([-2.0, -2.13], 0.43), _said("0.43 m"), world=yaml.safe_dump(data))
    verdict = _verdict(ev.score(ev.TASKS["lidar_maze"], plain))
    assert verdict["a robot carrying a lidar, in a maze"] is False


def test_distances_are_read_in_meters_from_an_answer(ev):
    said = ev.distances_said("左边约 0.43 米，前方 1.2m，速度 0.2 m/s，另一面 43 厘米，then 50 cm and 0.5 meters")
    assert said == [0.43, 1.2, 0.43, 0.5, 0.5]
    assert ev.said_near([0.4], 0.43) and not ev.said_near([0.3], 0.43) and ev.said_near([2.2], 2.5)


def _gif(path: Path, frames: int = 20, moving: bool = True, ms: int = 100) -> Path:
    from PIL import Image, ImageDraw

    images = []
    for i in range(frames):
        im = Image.new("RGB", (160, 120), (200, 200, 200))
        x = 5 + 6 * i if moving else 5
        ImageDraw.Draw(im).rectangle([x, 40, x + 20, 60], fill=(30, 160, 150))
        images.append(im)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=ms, loop=0)
    return path


def test_a_video_of_the_robot_from_parts_must_be_there_and_move(ev, tmp_path):
    """The recorded session left MP4s in its working folder; a fixture keeps no video, so each case puts one there."""
    task = ev.TASKS["parts_video"]
    none = _copy(tmp_path / "a", "parts_video")
    verdict = _verdict(ev.score(task, none))
    assert verdict["a robot built from parts"] and verdict["it moved >= 0.2 m"]
    assert verdict["a video >= 1 s in which something moves"] is False
    good = _copy(tmp_path / "b", "parts_video")
    _gif(good / "work" / "pip.gif")
    outcome = ev.score(task, good)
    assert outcome.passed, outcome.criteria
    assert "pip.gif: 20 frames, 2.0 s" in outcome.criteria[-1]["observed"]
    still = _copy(tmp_path / "c", "parts_video")
    _gif(still / "rf_home" / "worlds" / "pip_rover" / "videos" / "pip.gif", moving=False)
    assert _verdict(ev.score(task, still))["a video >= 1 s in which something moves"] is False
    short = _copy(tmp_path / "d", "parts_video")
    _gif(short / "work" / "pip.gif", frames=5)
    assert _verdict(ev.score(task, short))["a video >= 1 s in which something moves"] is False


def test_a_video_saved_elsewhere_counts_when_the_answer_names_it_and_it_is_new(ev, tmp_path):
    import os

    task = ev.TASKS["parts_video"]
    named = _copy(tmp_path / "a", "parts_video")
    elsewhere = _gif(tmp_path / "Desktop" / "pip.gif")
    with (named / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(_said(f"Here is the video: [pip.gif]({elsewhere})") + "\n")
    assert ev.score(task, named).passed
    started = json.loads((named / "meta.json").read_text(encoding="utf-8"))["started"]
    os.utime(elsewhere, (started - 3600, started - 3600))  # there before the session: not its video
    assert _verdict(ev.score(task, named))["a video >= 1 s in which something moves"] is False


def test_a_robot_that_never_moved_is_not_a_video_of_it_moving(ev, tmp_path):
    world = (FIXTURES / "parts_video" / "rf_home" / "worlds" / "pip_rover" / "world.yaml").read_text(encoding="utf-8")
    still = _made_up(tmp_path, {"pip_rover": world}, _opened("pip_rover"))
    _gif(still / "work" / "pip.gif")
    verdict = _verdict(ev.score(ev.TASKS["parts_video"], still))
    assert verdict["a robot built from parts"] and verdict["it moved >= 0.2 m"] is False


def test_a_fixture_keeps_each_lidar_whole(ev):
    state = {
        "time": 1,
        "joints": {"a": 1},
        "sensors": {"c/imu": [0, 0, 1], "c/l_0": 0.5},
        "lidars": {"c/l": {"rays": 3}},
    }
    kept = json.loads(ev._trim_result("observe", json.dumps(state)))
    assert kept == {"time": 1, "lidars": {"c/l": {"rays": 3}}, "sensors": {"c/l_0": 0.5}}
