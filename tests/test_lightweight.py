"""Starting robotforge loads only what it needs (lightweight budget, docs/ROADMAP.md G7).

Mesh repair and decomposition (trimesh, manifold3d, coacd), image handling (Pillow) and the learning stack are
heavy and serve single features: they load when a feature needs them, not when the MCP server, the viewer or a
script starts. Each check runs in a fresh interpreter, since this one has long imported everything.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

HEAVY = ("trimesh", "manifold3d", "coacd", "PIL", "gymnasium", "torch", "stable_baselines3", "onnxruntime")


def _loaded_after(statement: str) -> set[str]:
    code = f"import json, sys\n{statement}\nprint(json.dumps(sorted({{m.split('.')[0] for m in sys.modules}})))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return set(json.loads(done.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("module", ["robotforge.mcp_server", "robotforge.cli", "robotforge.playground"])
def test_starting_up_loads_no_heavy_module(module):
    heavy = _loaded_after(f"import {module}") & set(HEAVY)
    assert not heavy, f"import {module} loaded {sorted(heavy)}"


def test_the_package_itself_loads_almost_nothing():
    """`import robotforge.<anything>` runs the package first: it must not drag in the command line's server stack."""
    loaded = _loaded_after("import robotforge") & {"mujoco", "uvicorn", "starlette", "mcp", "numpy", *HEAVY}
    assert not loaded, f"import robotforge loaded {sorted(loaded)}"


def test_a_world_with_robots_runs_without_heavy_modules(tmp_path):
    """Building a world with a template dog, an arm and objects and walking the dog needs none of them either."""
    statement = f"""
import os
os.environ["ROBOTFORGE_HOME"] = {str(tmp_path)!r}
from robotforge.playground import Playground
pg = Playground.with_demo()
pg.new_world("light")
pg.add_robot("dog", template="quadruped")
pg.add_robot("arm", template="arm", pos=[1.0, 0.0])
pg.add_object("cube", pos=[1.25, 0.0])
pg.behaviors.move("dog", 0.3, 0.0)
pg.run_fast(lambda: False, 0.5)
"""
    heavy = _loaded_after(statement) & set(HEAVY)
    assert not heavy, f"loaded {sorted(heavy)}"
