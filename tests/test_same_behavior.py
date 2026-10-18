"""scripts/same_behavior.py's verdicts: identical, or from when the robots moved differently."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "same_behavior.py"


@pytest.fixture(scope="module")
def same():
    spec = importlib.util.spec_from_file_location("same_behavior", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["same_behavior"] = module
    spec.loader.exec_module(module)
    return module


def test_verdicts_name_where_the_states_part(same):
    before = {"a": ["h1", "h2", "h3"], "b": ["h1", "h2", "h3"], "c": ["h1", "h2"], "d": ["h1", "error: boom"]}
    now = {"a": ["h1", "h2", "h3"], "b": ["h1", "x2", "x3"], "c": ["h1", "h2", "h3"], "d": ["h1", "error: boom"]}
    verdicts = same.compare(now, before)
    assert verdicts["a"] == "identical"
    assert verdicts["b"] == "differs from mark 1 (0.50 s)"  # the second hash, taken after 0.5 s
    assert verdicts["c"] == "differs: 2 marks before, 3 now"
    assert verdicts["d"].startswith("identical [error: boom")  # failing the same way on both sides is shown


def test_one_scenario_hashes_the_same_twice(same, tmp_path, monkeypatch):
    """MuJoCo and the controllers are deterministic in one process: the check means something only if so."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "first"))
    first = same.worker(["dog_fresh"])
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "second"))
    second = same.worker(["dog_fresh"])
    assert len(first["dog_fresh"]) == 16 and first == second
