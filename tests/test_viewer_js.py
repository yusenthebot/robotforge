"""The viewer's JavaScript parses as the ES modules the page loads (needs Node on PATH, skipped without it).

A merge once left lesson.js two closing braces short; Python tests and the smoke test never load the page,
so main shipped a viewer that drew nothing ("Unexpected token 'export'")."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "robotforge" / "viewer" / "static"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
@pytest.mark.parametrize("script", sorted(p.name for p in STATIC.glob("*.js")))
def test_the_viewer_script_parses_as_a_module(script):
    source = (STATIC / script).read_text(encoding="utf-8")
    run = subprocess.run(
        [NODE, "--input-type=module", "--check"], input=source, capture_output=True, text=True, encoding="utf-8"
    )
    assert run.returncode == 0, run.stderr
