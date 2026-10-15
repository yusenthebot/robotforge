"""Interactive lessons that teach robotics ideas while you play in the viewer."""

from __future__ import annotations

from robotforge.learn.engine import Lesson, LessonError, LessonRunner
from robotforge.learn.pd import PDLesson

LESSONS: dict[str, type[Lesson]] = {cls.id: cls for cls in (PDLesson,)}

__all__ = ["LESSONS", "Lesson", "LessonError", "LessonRunner"]
