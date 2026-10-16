"""The agent eval's checks (scripts/agent_eval.py) on sessions recorded with Codex.

tests/fixtures/agent_eval/<task>/ holds one session of the 2026-10-17 baseline run (commit 2f614f8, gpt-6.1-sol
at medium effort) and the worlds it left, trimmed: no images, shell output or texts that no check reads, and the
run folder's path written as RUN. The expected verdicts are the ones read by hand from the full transcripts.
The script itself needs Codex, a login and the network; these tests need none of them.
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
    "arm_cube": (True, {"an arm": True, "a red object": True, "it ended in a box or bin": True}),
    "go2_turn": (True, None),
    # Asked to drive "a little", it drove 0.293 m; the prompt now asks for about half a meter.
    "urdf_rover": (False, {"a robot imported from my_rover.urdf": True, "moved >= 0.3 m": False}),
    "hexapod_build": (True, {"it is a hexapod": True}),
    "rover_rock": (True, None),
    # It opened lesson 6 and built the cart-pole, but never ran a controller.
    "learn_control": (False, {"opened the control lesson (lessons 6)": True, "ran a balance controller": False}),
    "fix_workshop": (True, None),
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
    assert next(c for c in outcome.criteria if c["name"].startswith("ended within"))["observed"] == "0.42 m"


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
    text = json.dumps(outcome.failures)
    assert str(tmp_path) not in text and str(Path.home()) not in text
    assert "RUN/tasks/urdf_rover/work/my_rover.urdf" in text and "~/x.urdf" in text


def test_a_robot_that_fell_at_the_end_does_not_pass(ev, tmp_path):
    """The same hexapod session, but its last move ends on its back."""
    import shutil

    shutil.copytree(FIXTURES / "hexapod_build", tmp_path / "hexapod_build")
    events = tmp_path / "hexapod_build" / "events.jsonl"
    text = events.read_text(encoding="utf-8")
    fell = text.replace('\\"heading_deg\\":-0.4,\\"upright\\":true', '\\"heading_deg\\":-0.4,\\"upright\\":false')
    assert fell != text
    events.write_text(fell, encoding="utf-8")
    outcome = ev.score(ev.TASKS["hexapod_build"], tmp_path / "hexapod_build")
    assert not outcome.passed
    assert next(c for c in outcome.criteria if c["name"] == "upright at the end")["ok"] is False


def test_a_session_that_ran_out_of_time_does_not_pass(ev, tmp_path):
    import shutil

    shutil.copytree(FIXTURES / "go2_turn", tmp_path / "go2_turn")
    (tmp_path / "go2_turn" / "meta.json").write_text(json.dumps({"wall_seconds": 600, "timed_out": True}))
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
