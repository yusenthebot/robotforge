"""Exact world-origin snap semantics, independent of browser or rendering."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_snap_ties_bounds_free_compatibility_and_input_ownership():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable for the pure viewer-coordinate module")
    source = Path(__file__).parents[1] / "src/robotforge/viewer/static/placement-coordinate.js"
    module = "data:text/javascript;base64," + base64.b64encode(source.read_bytes()).decode()
    examples = [
        ([0.05, -0.05, 0.75], "0.1", [0.1, -0.1, 0.75]),
        ([0.15, -0.15, 0], "0.1", [0.2, -0.2, 0]),
        ([0.04999999, -0.04999999, 0], "0.1", [0, 0, 0]),
        ([0.05000001, -0.05000001, 0], "0.1", [0.1, -0.1, 0]),
        ([0.125, -0.125, 0.7], "0.25", [0.25, -0.25, 0.7]),
        ([0.1249999, -0.1249999, 0.7], "0.25", [0, 0, 0.7]),
        ([-1.2, -3.6, 0], "0.1", [-1.2, -3.6, 0]),
        ([100, -100, 100], "0.25", [100, -100, 100]),
        ([0.123456, -1.876543, 0.74999], "free", [0.1235, -1.8765, 0.75]),
        ([100.0001, 0, 0], "0.1", None),
        ([0, -100.0001, 0], "free", None),
        ([0, 0, -100.0001], "0.25", None),
        ([0, 0, 0], "0.2", None),
        ([0, 0, 0], 0.1, None),
        ([True, 0, 0], "0.1", None),
        ([0, 0], "free", None),
    ]
    script = f"""
      const {{placementCoordinates: point}} = await import({json.dumps(module)});
      const examples = {json.dumps(examples)};
      const before = JSON.stringify(examples);
      const results = examples.map(([value, spacing]) => point(value, spacing));
      const invalid = [null, {{}}, [], [NaN,0,0], [0,Infinity,0], [0,0,-Infinity]];
      const negativeZero = point([-0.01,-0,0], '0.1').slice(0,2).some(x => Object.is(x,-0));
      const free = [.123456,-.00001,.749999];
      console.log(JSON.stringify({{results, invalid:invalid.map(x=>point(x,'0.1')),
        unchanged:before===JSON.stringify(examples), negativeZero,
        freeCompatibility:point(free,'free').every((v,i)=>Object.is(v,+free[i].toFixed(4)))}}));
    """
    result = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=10,
    )
    values = json.loads(result.stdout)
    assert values["results"] == [example[2] for example in examples]
    assert values["invalid"] == [None] * 6
    assert values["unchanged"] and values["freeCompatibility"] and not values["negativeZero"]
