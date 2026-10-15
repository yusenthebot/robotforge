"""Interactive lessons that teach robotics ideas while you play in the viewer."""

from __future__ import annotations

from robotforge.learn.challenges import CourseChallenge, MazeChallenge, ShoveChallenge
from robotforge.learn.engine import Lesson, LessonError, LessonRunner
from robotforge.learn.ik import IKLesson
from robotforge.learn.pd import PDLesson
from robotforge.learn.rl import RLLesson
from robotforge.learn.sensors import SensorsLesson

LESSONS: dict[str, type[Lesson]] = {
    cls.id: cls
    for cls in (
        PDLesson,
        IKLesson,
        SensorsLesson,
        RLLesson,
        MazeChallenge,
        ShoveChallenge,
        CourseChallenge,
    )
}

__all__ = ["LESSONS", "Lesson", "LessonError", "LessonRunner"]
