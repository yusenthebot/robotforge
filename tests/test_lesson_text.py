"""Lesson text as the viewer shows it: numbers and signs a beginner reads in the live formula and the step text."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from robotforge.learn import balance, pd

LEARN = Path(pd.__file__).parent


@pytest.mark.parametrize("signed", [pd._signed, balance._signed], ids=["pd", "balance"])
def test_a_signed_value_that_rounds_to_zero_reads_plus_zero(signed):
    """The PID formula of an arm at rest read "Kp × e = 80 × -0.000 rad" and "-0.00 N·m"."""
    assert signed(-0.0004, 2) == "+0.00"
    assert signed(-0.00004, 3) == "+0.000"
    assert signed(-0.04, 2) == "-0.04"
    assert signed(0.3, 1) == "+0.3"
    assert signed(0.0, 2) == "+0.00"


def test_lesson_text_writes_plus_or_minus_as_one_sign():
    """The English said "within +-3 degrees" where the Chinese beside it said "±3°"."""
    found = [
        f"{path.name}:{n}"
        for path in sorted(LEARN.glob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'"[^"#]*\+-[^"]*"', line)  # inside a string, not in a comment
    ]
    assert not found, found
