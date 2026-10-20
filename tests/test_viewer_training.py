"""The Training panel's plain helpers (training.js) run in Node as the page loads them (skipped without Node)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "robotforge" / "viewer" / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def js(body: str) -> object:
    source = (
        f"import * as tv from {json.dumps((STATIC / 'training.js').as_uri())};\n"
        f"const out = await (async () => {{ {body} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    run = subprocess.run(
        [NODE, "--input-type=module"], input=source, capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@needs_node
def test_numbers_are_written_short_and_without_a_trailing_zero():
    assert js(
        "return [tv.formatSteps(50000), tv.formatSteps(1500), tv.formatSteps(250000), tv.formatSteps(2000000)];"
    ) == [
        "50k",
        "1.5k",
        "250k",
        "2M",
    ]
    assert js(
        "return [tv.formatValue(0.000123), tv.formatValue(6.8213), tv.formatValue(24), tv.formatValue(null)];"
    ) == [
        "1.2e-4",
        "6.82",
        "24",
        "-",
    ]
    assert js("return [tv.formatTime(45), tv.formatTime(322), tv.formatTime(7300)];") == [
        "45 s",
        "5 min 22 s",
        "2 h 1 min",
    ]


@needs_node
def test_the_table_shows_the_settings_that_differ_and_all_of_one_trials_when_nothing_does():
    trials = [
        {"settings": {"noise": 0.1, "step_size": 0.1, "directions": 8}},
        {"settings": {"noise": 0.3, "step_size": 0.1, "directions": 8}},
    ]
    assert js(f"return tv.settingColumns({json.dumps(trials)});") == ["noise"]
    assert js(f"return tv.settingColumns({json.dumps(trials[:1])});") == ["noise", "step_size", "directions"]
    wide = [{"settings": {f"s{i}": v for i in range(9)}} for v in (1, 2)]
    assert len(js(f"return tv.settingColumns({json.dumps(wide)}, 4);")) == 4


@needs_node
def test_trials_get_different_colors_however_many_there_are():
    colors = js("return Array.from({length: 24}, (_, i) => tv.trainingColor(i, false));")
    assert len(set(colors)) == 24
