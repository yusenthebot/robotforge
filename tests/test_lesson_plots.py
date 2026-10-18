"""Lesson plots as the viewer draws them: a robot holding still must not fill a plot with its last fractions."""

from __future__ import annotations

import re
from pathlib import Path

from robotforge.learn import LESSONS
from robotforge.learn.engine import Plot

LESSON_JS = Path(__file__).parents[1] / "src" / "robotforge" / "viewer" / "static" / "lesson.js"


def _viewer_unit_spans() -> set[str]:
    """The units the viewer has a smallest span for (UNIT_SPAN in lesson.js)."""
    found = re.search(r"const UNIT_SPAN = \{([^}]*)\}", LESSON_JS.read_text(encoding="utf-8"))
    assert found, "lesson.js no longer has UNIT_SPAN"
    return set(re.findall(r'"([^"]+)":', found.group(1))) | set(re.findall(r"(\w+):", found.group(1)))


def test_every_line_plot_has_a_smallest_span():
    """The IK lesson drew its arm's manipulability going from 131 to 132 (home pose = 100) as a jump across the whole
    plot: a plot whose unit the viewer has no smallest span for has to set its own."""
    units = _viewer_unit_spans()
    missing = [
        f"{lesson_id}:{plot.id}"
        for lesson_id, lesson in LESSONS.items()
        for plot in getattr(lesson(), "plots", ())
        if isinstance(plot, Plot) and plot.kind == "lines" and not plot.span and plot.unit not in units
    ]
    assert not missing, missing
