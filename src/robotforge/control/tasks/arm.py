"""What the tasks that move an arm's tool share: finding the tool point, the arm's joints and their actuators."""

from __future__ import annotations

import mujoco
import numpy as np

from robotforge.control.tasks.base import Task, TaskError


class ArmTask(Task):
    """A task for an arm with a tool point named 'tcp' (any robot from the arm template, or a library arm)."""

    robot_hint = "an arm with a tool point named 'tcp' (any robot from the arm template)"

    def _find_arm(self, model: mujoco.MjModel, robot: str) -> None:
        """Set site, site_sensor, joints, actuators, qadr and dadr: the tool point and the arm's own joints."""
        self.site = self.robot.need(mujoco.mjtObj.mjOBJ_SITE, "tcp", self.robot_hint)
        self.site_sensor = self.robot.framepos_sensor(self.site)
        hinges = self.robot.hinge_joints()
        if not hinges:
            raise TaskError(f"robot {robot!r} has no hinge joints to move")
        # Only actuators that drive a joint: a tendon actuator (a Panda's gripper) keeps a tendon's index in the same
        # column, which would pass for a joint with that index and put the gripper in the arm's place.
        direct = int(mujoco.mjtTrn.mjTRN_JOINT)
        joint_to_act = {
            int(model.actuator_trnid[a, 0]): int(a) for a in self.actuators if int(model.actuator_trntype[a]) == direct
        }
        # Drive only the arm joints; a gripper does not help reach a point.
        self.joints = [j for j in hinges if j in joint_to_act]
        if not self.joints:
            raise TaskError(f"none of {robot!r}'s hinge joints has an actuator")
        self.actuators = np.array([joint_to_act[j] for j in self.joints])
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.joints])
