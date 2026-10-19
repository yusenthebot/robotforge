"""Suites of generated tasks (design/tasks/suite.py): made the same way every time, each world on disk with its
task, openable as a learning environment, and judged in the agent eval by the task's own check on what the session's
tool calls reported."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from robotforge.control.tasks.world_task import world_env
from robotforge.design.tasks.suite import load_suite, make_suite
from robotforge.design.world import World

SCRIPT = Path(__file__).parents[1] / "scripts" / "agent_eval.py"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    out = tmp_path_factory.mktemp("suite")
    cases = make_suite(5, seed=3, out=out)
    return out, cases


def test_a_suite_is_varied_and_every_world_on_disk_has_its_task(suite):
    out, cases = suite
    assert len(cases) == 5 and load_suite(out) == cases
    assert len({c.world for c in cases}) >= 4 and len({c.kind for c in cases}) >= 2
    for case in cases:
        world = World.from_yaml((out / "worlds" / case.name / "world.yaml").read_text(encoding="utf-8"))
        assert [t.name for t in world.tasks] == ["task"]
        task = world.tasks[0]
        assert task.robot in [r.name for r in world.robots] and task.kind == case.kind
        assert task.start.robot is not None


def test_the_same_count_and_seed_make_the_same_suite(suite, tmp_path):
    out, cases = suite
    again = make_suite(5, seed=3, out=tmp_path)
    assert again == cases
    for case in cases:
        a = (out / "worlds" / case.name / "world.yaml").read_text(encoding="utf-8")
        b = (tmp_path / "worlds" / case.name / "world.yaml").read_text(encoding="utf-8")
        assert a == b


def test_each_case_opens_as_a_learning_environment(suite):
    out, cases = suite
    env = world_env(out / "worlds" / cases[0].name)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and info["success"] is False
    _, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0)
    assert reward < 0 and not info["success"]


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("agent_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_eval"] = module
    spec.loader.exec_module(module)
    return module


def _call(ev, tool, args, result):
    return ev.Call("robotforge", tool, args, True, json.dumps(result))


def test_the_eval_judges_a_case_by_its_own_check_on_what_the_session_reported(suite, ev, tmp_path):
    out, cases = suite
    case = next(c for c in cases if c.kind in ("bring", "put"))
    home = tmp_path / "home"
    task = ev.suite_tasks(out)[f"suite_{case.name}"]
    task.setup(tmp_path, home)
    world = World.from_yaml((home / "worlds" / case.name / "world.yaml").read_text(encoding="utf-8"))
    target = next(o for o in world.objects if o.name == case.target)
    top = (target.pos[2] if len(target.pos) == 3 else 0.0) + target.size[2]
    opened = _call(
        ev,
        "open_world",
        {"name_or_path": case.name},
        {"world": case.name, "world_dir": str(home / "worlds" / case.name)},
    )

    def session(*calls):
        return ev.Session(calls=(opened, *calls), commands=(), messages=(), usage={})

    there = [target.pos[0], target.pos[1], top + 0.03]
    done = _call(
        ev, "pick_and_place", {"robot": case.robot, "object": case.object, "target": case.target},
        {"object": case.object, "object_position": there},
    )  # fmt: skip
    worlds = ev.load_worlds(home)
    passed = task.check(ev.Evidence(session(done), worlds, tmp_path, home))
    assert all(c.ok for c in passed), passed
    missed = _call(
        ev, "pick_and_place", {"robot": case.robot, "object": case.object, "target": case.target},
        {"object": case.object, "object_position": [there[0] + 3.0, there[1], 0.04]},
    )  # fmt: skip
    failed = task.check(ev.Evidence(session(missed), worlds, tmp_path, home))
    assert not failed[0].ok
    # moved there by editing the world: the check of where it is passes, the one of how it got there does not
    edit = _call(ev, "update_object", {"name": case.object, "pos": there}, {"world": case.name})
    cheated = task.check(ev.Evidence(session(edit, done), worlds, tmp_path, home))
    assert cheated[0].ok and not cheated[1].ok
    shutil.rmtree(home)
