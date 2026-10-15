"""Interactive lessons that teach robotics ideas while you play in the viewer."""

from __future__ import annotations

from robotforge.learn.challenges import CourseChallenge, MazeChallenge, ShoveChallenge
from robotforge.learn.compare import ArmFaceOff, HumanoidFaceOff
from robotforge.learn.engine import Lesson, LessonError, LessonRunner
from robotforge.learn.ik import IKLesson
from robotforge.learn.pd import PDLesson
from robotforge.learn.rl import RLLesson
from robotforge.learn.sensors import SensorsLesson
from robotforge.learn.twins import GaitTwins, LQRTwins, MPCTwins, PIDTwins

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
        ArmFaceOff,
        HumanoidFaceOff,
        PIDTwins,
        LQRTwins,
        MPCTwins,
        GaitTwins,
    )
}

__all__ = ["LESSONS", "Lesson", "LessonError", "LessonRunner"]
